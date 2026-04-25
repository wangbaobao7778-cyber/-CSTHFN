import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

# Import your existing modules
from dataloader.VFTDataLoader import load_raw_data, augment_data_odd_even, DualModalityDataset, \
    load_excel_channel_data_dual
from models.DualBranchModel import DualBranchRecurrentModel
from PCMCI import compute_causal_prior_from_channels
from tool import set_seed, PLOT_FONT_CONFIG


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/VFT')
    parser.add_argument('--model_dir', type=str, default='./checkpoints/causal_EDL')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--hidden_dims', type=int, default=64)
    parser.add_argument('--head', type=int, default=2)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--k_memory', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--roi_mode', type=str, default='original')
    parser.add_argument('--keep_ratio', type=float, default=1.0)
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--disable_causal', type=bool, default=False)
    parser.add_argument('--edl_mode', type=bool, default=True)
    return parser.parse_args()


# Channel to grid coordinate mapping table
coords_map = {
    0: (0, 1), 1: (0, 3), 2: (0, 5), 3: (0, 7),
    4: (1, 0), 5: (1, 2), 6: (1, 4), 7: (1, 6), 8: (1, 8),
    9: (2, 1), 10: (2, 3), 11: (2, 5), 12: (2, 7),
    13: (3, 0), 14: (3, 2), 15: (3, 4), 16: (3, 6), 17: (3, 8),
    18: (4, 1), 19: (4, 3), 20: (4, 5), 21: (4, 7)
}

# Original ROI division
roi_mapping = [
    [0, 4, 5],  # ROI 0
    [1, 2],  # ROI 1
    [3, 7, 8],  # ROI 2
    [9, 13, 14, 18],  # ROI 3
    [6, 10, 11, 15, 19, 20],  # ROI 4
    [12, 16, 17, 21]  # ROI 5
]


def calculate_ece(confidences, accuracies, n_bins=10):
    """Calculate Expected Calibration Error (ECE)"""
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        if i == 0:
            in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        else:
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


def evaluate_and_get_edl(model, loader, device, A_causal_oxy=None, A_causal_dxy=None):
    """Forward pass and extract b, u, P, and predictions to calculate metrics"""
    model.eval()
    all_b, all_u, all_P = [], [], []
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for oxy, dxy, labels in loader:
            oxy, dxy, labels = oxy.to(device), dxy.to(device), labels.to(device)
            outputs = model(oxy, dxy, A_causal_oxy, A_causal_dxy, return_features=False)

            # EDL calculation logic
            evidence = outputs
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            probs = alpha / S

            max_probs, predicted = torch.max(probs, 1)  # Expected probability P
            uncertainty = 2 / S  # Uncertainty u (assuming 2 classes)
            belief = evidence / S
            # Extract model's Belief b for the predicted class
            pred_beliefs = torch.gather(belief, 1, predicted.unsqueeze(1)).squeeze(1)

            all_b.extend(pred_beliefs.cpu().numpy().flatten())
            all_u.extend(uncertainty.cpu().numpy().flatten())
            all_P.extend(max_probs.cpu().numpy().flatten())

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # Save probability of class 1 for AUC and NLL calculation
            all_probs.extend(probs[:, 1].cpu().numpy())

    return np.array(all_b), np.array(all_u), np.array(all_P), np.array(all_preds), np.array(all_labels), np.array(
        all_probs)


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading data from {args.data_path}...")
    X_oxy_all, X_dxy_all, y_all = load_raw_data(args)
    target_len = X_oxy_all.shape[1] + 1
    X_chan_oxy_all, X_chan_dxy_all = load_excel_channel_data_dual(args.data_path, target_len=target_len)
    X_chan_oxy_all = np.delete(X_chan_oxy_all, 0, axis=2)
    X_chan_dxy_all = np.delete(X_chan_dxy_all, 0, axis=2)

    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    # Initialize dictionary to save results of all folds
    results = {'original': {'b': [], 'u': [], 'P': [], 'correct': [], 'fold_metrics': []}}
    for i in range(len(roi_mapping)):
        results[f'roi_{i}'] = {'b': [], 'u': [], 'P': [], 'correct': [], 'fold_metrics': []}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_oxy_all)):
        print(f"\n{'=' * 20} Processing Fold {fold + 1} {'=' * 20}")

        # 1. Calculate causal mask for this fold
        X_train_chan_oxy = X_chan_oxy_all[train_idx]
        X_train_chan_dxy = X_chan_dxy_all[train_idx]
        y_train_raw = y_all[train_idx]

        A_causal_oxy, A_causal_dxy = None, None
        if not args.disable_causal:
            train_adhd_mask = (y_train_raw == 0)
            train_hc_mask = (y_train_raw == 1)

            A_oxy_adhd = compute_causal_prior_from_channels(X_train_chan_oxy[train_adhd_mask], roi_mapping).to(device)
            A_oxy_hc = compute_causal_prior_from_channels(X_train_chan_oxy[train_hc_mask], roi_mapping).to(device)
            A_causal_oxy = torch.where((A_oxy_adhd != 0) | (A_oxy_hc != 0), 1.0, 0.0)

            A_dxy_adhd = compute_causal_prior_from_channels(X_train_chan_dxy[train_adhd_mask], roi_mapping).to(device)
            A_dxy_hc = compute_causal_prior_from_channels(X_train_chan_dxy[train_hc_mask], roi_mapping).to(device)
            A_causal_dxy = torch.where((A_dxy_adhd != 0) | (A_dxy_hc != 0), 1.0, 0.0)

        # 2. Load trained model
        model_path = os.path.join(args.model_dir, f'best_model_fold_{fold}.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}. Please check if --model_dir is correct.")

        model = DualBranchRecurrentModel(
            embed_dim=args.hidden_dims, num_heads=args.head, depth=args.depth,
            k_memory=args.k_memory, num_classes=args.num_classes, drop=args.dropout,
            attn_drop=args.attn_drop, roi_mode=args.roi_mode, keep_ratio=args.keep_ratio,
            chunk_size=args.chunk_size, edl_mode=args.edl_mode
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # 3. Extract mean of HC samples in training set for masking
        train_hc_mask = (y_all[train_idx] == 1)
        mean_hc_oxy = np.mean(X_oxy_all[train_idx][train_hc_mask], axis=0)
        mean_hc_dxy = np.mean(X_dxy_all[train_idx][train_hc_mask], axis=0)

        # 4. Extract ADHD and HC samples in test set
        val_adhd_mask = (y_all[val_idx] == 0)
        val_hc_mask = (y_all[val_idx] == 1)

        val_adhd_oxy_raw = X_oxy_all[val_idx][val_adhd_mask]
        val_adhd_dxy_raw = X_dxy_all[val_idx][val_adhd_mask]
        val_adhd_y_raw = y_all[val_idx][val_adhd_mask]

        val_hc_oxy_raw = X_oxy_all[val_idx][val_hc_mask]
        val_hc_dxy_raw = X_dxy_all[val_idx][val_hc_mask]
        val_hc_y_raw = y_all[val_idx][val_hc_mask]

        # -------------------------------------------------------------
        # Baseline: Entire test set (ADHD + HC) unmasked state
        val_oxy_orig = np.concatenate([val_adhd_oxy_raw, val_hc_oxy_raw], axis=0)
        val_dxy_orig = np.concatenate([val_adhd_dxy_raw, val_hc_dxy_raw], axis=0)
        val_y_orig = np.concatenate([val_adhd_y_raw, val_hc_y_raw], axis=0)

        oxy_aug, dxy_aug, y_aug = augment_data_odd_even(val_oxy_orig, val_dxy_orig, val_y_orig)
        loader_orig = DataLoader(DualModalityDataset(oxy_aug, dxy_aug, y_aug), batch_size=args.batch_size,
                                 shuffle=False)
        b_orig, u_orig, P_orig, preds_orig, labels_orig, probs_orig = evaluate_and_get_edl(model, loader_orig, device,
                                                                                           A_causal_oxy, A_causal_dxy)

        adhd_idx = (labels_orig == 0)
        orig_is_correct = (preds_orig == labels_orig).astype(int)

        results['original']['b'].extend(b_orig[adhd_idx])
        results['original']['u'].extend(u_orig[adhd_idx])
        results['original']['P'].extend(P_orig[adhd_idx])
        results['original']['correct'].extend(orig_is_correct[adhd_idx])

        # Calculate Baseline metrics for this fold
        acc = accuracy_score(labels_orig, preds_orig)
        pre = precision_score(labels_orig, preds_orig, zero_division=0)
        rec = recall_score(labels_orig, preds_orig, zero_division=0)
        f1 = f1_score(labels_orig, preds_orig, zero_division=0)
        try:
            auc = roc_auc_score(labels_orig, probs_orig)
        except:
            auc = 0.5

        try:
            nll = log_loss(labels_orig, probs_orig, labels=[0, 1])
        except:
            nll = 0.0

        ece = calculate_ece(P_orig, orig_is_correct)

        mean_b, mean_u, mean_P = np.mean(b_orig), np.mean(u_orig), np.mean(P_orig)
        # Order: acc, f1, pre, rec, auc, nll, ece, mean_u, mean_b, mean_P
        results['original']['fold_metrics'].append([acc, f1, pre, rec, auc, nll, ece, mean_u, mean_b, mean_P])

        print(
            f"  -> [Unmasked Baseline] ACC: {acc * 100:.2f} | F1: {f1 * 100:.2f} | PRE: {pre * 100:.2f} | REC: {rec * 100:.2f} | AUC: {auc * 100:.2f} | NLL: {nll * 100:.2f} | ECE: {ece * 100:.2f}")
        print(
            f"                         mean u: {mean_u * 100:.2f} | mean b: {mean_b * 100:.2f} | mean P: {mean_P * 100:.2f}")

        # -------------------------------------------------------------
        # Masking: Iterate through 6 ROIs one by one to perform masking intervention
        for roi_idx, channels in enumerate(roi_mapping):
            val_oxy_masked_adhd = val_adhd_oxy_raw.copy()
            val_dxy_masked_adhd = val_adhd_dxy_raw.copy()

            for ch in channels:
                r, c = coords_map[ch]
                val_oxy_masked_adhd[:, :, r, c] = mean_hc_oxy[:, r, c]
                val_dxy_masked_adhd[:, :, r, c] = mean_hc_dxy[:, r, c]

            val_oxy_masked = np.concatenate([val_oxy_masked_adhd, val_hc_oxy_raw], axis=0)
            val_dxy_masked = np.concatenate([val_dxy_masked_adhd, val_hc_dxy_raw], axis=0)
            val_y_masked = np.concatenate([val_adhd_y_raw, val_hc_y_raw], axis=0)

            oxy_aug_m, dxy_aug_m, y_aug_m = augment_data_odd_even(val_oxy_masked, val_dxy_masked, val_y_masked)
            loader_m = DataLoader(DualModalityDataset(oxy_aug_m, dxy_aug_m, y_aug_m), batch_size=args.batch_size,
                                  shuffle=False)

            b_m, u_m, P_m, preds_m, labels_m, probs_m = evaluate_and_get_edl(model, loader_m, device, A_causal_oxy,
                                                                             A_causal_dxy)

            adhd_idx_m = (labels_m == 0)
            mask_is_correct = (preds_m == labels_m).astype(int)

            results[f'roi_{roi_idx}']['b'].extend(b_m[adhd_idx_m])
            results[f'roi_{roi_idx}']['u'].extend(u_m[adhd_idx_m])
            results[f'roi_{roi_idx}']['P'].extend(P_m[adhd_idx_m])
            results[f'roi_{roi_idx}']['correct'].extend(mask_is_correct[adhd_idx_m])

            acc_m = accuracy_score(labels_m, preds_m)
            pre_m = precision_score(labels_m, preds_m, zero_division=0)
            rec_m = recall_score(labels_m, preds_m, zero_division=0)
            f1_m = f1_score(labels_m, preds_m, zero_division=0)
            try:
                auc_m = roc_auc_score(labels_m, probs_m)
            except:
                auc_m = 0.5

            try:
                nll_m = log_loss(labels_m, probs_m, labels=[0, 1])
            except:
                nll_m = 0.0

            ece_m = calculate_ece(P_m, mask_is_correct)

            mean_b_m, mean_u_m, mean_P_m = np.mean(b_m), np.mean(u_m), np.mean(P_m)
            results[f'roi_{roi_idx}']['fold_metrics'].append(
                [acc_m, f1_m, pre_m, rec_m, auc_m, nll_m, ece_m, mean_u_m, mean_b_m, mean_P_m])

            print(
                f"  -> [Masked ROI {roi_idx}]      ACC: {acc_m * 100:.2f} | F1: {f1_m * 100:.2f} | PRE: {pre_m * 100:.2f} | REC: {rec_m * 100:.2f} | AUC: {auc_m * 100:.2f} | NLL: {nll_m * 100:.2f} | ECE: {ece_m * 100:.2f}")
            print(
                f"                         mean u: {mean_u_m * 100:.2f} | mean b: {mean_b_m * 100:.2f} | mean P: {mean_P_m * 100:.2f}")

    # ==========================================================
    # Output 5-fold cross-validation average metrics and standard deviation
    # ==========================================================
    print("\n" + "=" * 80)
    print("========= 5-Fold Cross Validation Average Evaluation Metrics ± Std =========")
    print("=" * 80)
    for key in results.keys():
        # Multiply each item of avg_metrics and std_metrics by 100
        avg_m = np.mean(results[key]['fold_metrics'], axis=0) * 100
        std_m = np.std(results[key]['fold_metrics'], axis=0) * 100

        # Order: [0:acc, 1:f1, 2:pre, 3:rec, 4:auc, 5:nll, 6:ece, 7:mean_u, 8:mean_b, 9:mean_P]
        print(f"[{key.upper()}]")
        print(
            f"  ACC: {avg_m[0]:.2f}±{std_m[0]:.2f} | F1: {avg_m[1]:.2f}±{std_m[1]:.2f} | PRE: {avg_m[2]:.2f}±{std_m[2]:.2f} | REC: {avg_m[3]:.2f}±{std_m[3]:.2f} | AUC: {avg_m[4]:.2f}±{std_m[4]:.2f} | NLL: {avg_m[5]:.2f}±{std_m[5]:.2f} | ECE: {avg_m[6]:.2f}±{std_m[6]:.2f}")
        print(
            f"  mean u: {avg_m[7]:.2f}±{std_m[7]:.2f} | mean b: {avg_m[8]:.2f}±{std_m[8]:.2f} | mean P: {avg_m[9]:.2f}±{std_m[9]:.2f}")
        print("-" * 80)

    # ==========================================================
    # 5. Save results and scatter plot visualization (ADHD only)
    # ==========================================================
    print("\n========== Start generating and saving scatter plot results ==========")

    base_out_dir = os.path.join(args.model_dir, 'causal_resultandimages')
    os.makedirs(base_out_dir, exist_ok=True)
    print(f"Main output directory created/exists: {base_out_dir}")

    font_family = PLOT_FONT_CONFIG['family']

    orig_b = np.array(results['original']['b'])
    orig_u = np.array(results['original']['u'])
    orig_P = np.array(results['original']['P'])
    orig_correct = np.array(results['original']['correct'])

    excel_save_path = os.path.join(base_out_dir, 'combined_roi_results.xlsx')

    try:
        with pd.ExcelWriter(excel_save_path) as writer:
            for roi_idx in range(len(roi_mapping)):
                roi_dir = os.path.join(base_out_dir, f'ROI_{roi_idx}')
                os.makedirs(roi_dir, exist_ok=True)

                mask_b = np.array(results[f'roi_{roi_idx}']['b'])
                mask_u = np.array(results[f'roi_{roi_idx}']['u'])
                mask_P = np.array(results[f'roi_{roi_idx}']['P'])
                mask_correct = np.array(results[f'roi_{roi_idx}']['correct'])

                df = pd.DataFrame({
                    'Original_b': orig_b,
                    'Original_u': orig_u,
                    'Original_P': orig_P,
                    'Original_Correct': orig_correct,
                    'Masked_b': mask_b,
                    'Masked_u': mask_u,
                    'Masked_P': mask_P,
                    'Masked_Correct': mask_correct
                })

                # 1. Save CSV
                csv_save_path = os.path.join(roi_dir, f'result_roi_{roi_idx}.csv')
                df.to_csv(csv_save_path, index_label='Sample_Index')

                # 2. Write to Excel Sheet
                df.to_excel(writer, sheet_name=f'ROI_{roi_idx}', index_label='Sample_Index')

                # --- Draw scatter plot ---
                plt.figure(figsize=(9, 7))
                plt.scatter(orig_b, orig_u, c='red', alpha=0.4, edgecolors='white', label='Original ADHD', s=60)
                plt.scatter(mask_b, mask_u, c='#1f77b4', alpha=0.8, edgecolors='white', label=f'Masked ROI {roi_idx}',
                            s=80)

                for i in range(len(orig_b)):
                    plt.arrow(orig_b[i], orig_u[i], mask_b[i] - orig_b[i], mask_u[i] - orig_u[i],
                              color='gray', alpha=0.25, width=0.0015, head_width=0.01)

                plt.xlabel('Belief (b)', fontdict={'family': font_family, 'size': 18})
                plt.ylabel('Uncertainty (u)', fontdict={'family': font_family, 'size': 18})
                plt.title(f'Causal Masking Shift: ROI {roi_idx} (ADHD Samples)',
                          fontdict={'family': font_family, 'size': 20})

                plt.xticks(fontname=font_family, fontsize=14)
                plt.yticks(fontname=font_family, fontsize=14)
                plt.legend(prop={'family': font_family, 'size': 14})
                plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout()

                png_save_path = os.path.join(roi_dir, f'u_b_scatter_roi_{roi_idx}.png')
                plt.savefig(png_save_path, dpi=300)
                plt.close()

                print(f"  - ROI {roi_idx} scatter plot and result files saved")

        print(f"\n  - Summary data of all ROIs has been written to independent Sheet files: {excel_save_path}")
        print("\n✅ All causal ablation metrics and plots saved!")

    except PermissionError as e:
        print("\n" + "❌" * 20)
        print("Save failed: File is in use!")
        print("Please check if you have the CSV or Excel file open in Excel or WPS.")
        print("Please close all related spreadsheet files and rerun this program.")
        print(f"Detailed error message: {e}")
        print("❌" * 20 + "\n")


if __name__ == '__main__':
    main()