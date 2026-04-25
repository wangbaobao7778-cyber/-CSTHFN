import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import sys
import datetime
import shutil
import matplotlib.pyplot as plt
from h5py.h5z import FLAG_SKIP_EDC
from torch.nn.functional import dropout
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from models.DualBranchModel import DualBranchRecurrentModel
from dataloader.VFTDataLoader import load_raw_data, augment_data_odd_even, DualModalityDataset, load_excel_channel_data_dual

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, log_loss
from torch.nn.functional import dropout, softmax

from PCMCI import compute_causal_prior_from_channels

from tool import (plot_mean_std_conf_matrix, set_seed, get_logger, close_logger,
                  log_hyperparameters, plot_loss_curve, EarlyStopping, plot_tsne,
                  edl_mse_loss, softplus_evidence, plot_edl_scatter, calculate_ece)
import pandas as pd

import torch.nn.functional as F


def calculate_metrics(all_labels, all_preds, all_probs):
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    nll = log_loss(all_labels, all_probs)
    ece = calculate_ece(all_labels, all_probs)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    try:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        auc = 0.5

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "cm": cm,
        "nll": nll,
        "ece": ece
    }


def train_one_epoch(model, loader, criterion, optimizer, device, A_causal_oxy, A_causal_dxy, epoch, args):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    for oxy, dxy, labels in loader:
        oxy, dxy, labels = oxy.to(device), dxy.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(oxy, dxy, A_causal_oxy, A_causal_dxy)

        if args.edl_mode:
            evidence = outputs
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            probs = alpha / S

            y_one_hot = F.one_hot(labels, num_classes=args.num_classes).float()
            loss = edl_mse_loss(softplus_evidence, y_one_hot, alpha, epoch, args.num_classes, args.annealing_step,
                                device)
            _, predicted = torch.max(probs, 1)
        else:
            loss = criterion(outputs, labels)
            probs = softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    return running_loss / len(loader.dataset), metrics


def evaluate(model, loader, criterion, device, A_causal_oxy, A_causal_dxy, args):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for oxy, dxy, labels in loader:
            oxy, dxy, labels = oxy.to(device), dxy.to(device), labels.to(device)
            outputs = model(oxy, dxy, A_causal_oxy, A_causal_dxy)

            if args.edl_mode:
                evidence = outputs
                alpha = evidence + 1
                S = torch.sum(alpha, dim=1, keepdim=True)
                probs = alpha / S

                y_one_hot = F.one_hot(labels, num_classes=args.num_classes).float()
                loss = edl_mse_loss(softplus_evidence, y_one_hot, alpha, args.annealing_step, args.num_classes,
                                    args.annealing_step, device)
                _, predicted = torch.max(probs, 1)
            else:
                loss = criterion(outputs, labels)
                probs = softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

            running_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    return running_loss / len(loader.dataset), metrics


def evaluate_with_features(model, loader, criterion, device, A_causal_oxy, A_causal_dxy, args):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs, all_features = [], [], [], []
    all_b, all_u, all_P = [], [], []

    with torch.no_grad():
        for oxy, dxy, labels in loader:
            oxy, dxy, labels = oxy.to(device), dxy.to(device), labels.to(device)
            outputs, features = model(oxy, dxy, A_causal_oxy, A_causal_dxy, return_features=True)

            if args.edl_mode:
                evidence = outputs
                alpha = evidence + 1
                S = torch.sum(alpha, dim=1, keepdim=True)
                probs = alpha / S

                max_probs, predicted = torch.max(probs, 1)
                uncertainty = args.num_classes / S
                belief = evidence / S
                pred_beliefs = torch.gather(belief, 1, predicted.unsqueeze(1)).squeeze(1)

                all_b.extend(pred_beliefs.cpu().numpy().flatten())
                all_u.extend(uncertainty.cpu().numpy().flatten())
                all_P.extend(max_probs.cpu().numpy().flatten())

                y_one_hot = F.one_hot(labels, num_classes=args.num_classes).float()
                loss = edl_mse_loss(softplus_evidence, y_one_hot, alpha, args.annealing_step, args.num_classes,
                                    args.annealing_step, device)
            else:
                loss = criterion(outputs, labels)
                probs = softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

            running_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_features.extend(features.cpu().numpy())

    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    return running_loss / len(loader.dataset), metrics, np.array(all_features), np.array(
        all_labels), all_b, all_u, all_P


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/VFT')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hidden_dims', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--head', type=int, default=2, help='Number of heads')
    parser.add_argument('--depth', type=int, default=1, help='Depth')
    parser.add_argument('--k_memory', type=int, default=10, help='Memory pool length')
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.4, help='Attention dropout ratio')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='L2 regularization coefficient')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--optim_patience', type=int, default=5, help='Halve lr every optim_patience epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--exp_name', type=str, default='dual_branch')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--roi_mode', type=str, default='original',
                        choices=('original', 'full', 'hemi_4_5', 'hemi_5_4', 'three_columns', 'grid_1x3'),
                        help='original: 6 ROIs, full: no partition, hemi_4_5: left 4 right 5, three_columns: three equal parts, grid_1x3: 15 1x3 grids')
    parser.add_argument('--keep_ratio', type=float, default=1.0, help='Ratio of strongest connections kept in causal matrix (Top-K)')
    parser.add_argument('--chunk_size', type=int, default=10, help='Sliding window size')
    parser.add_argument('--disable_causal', type=bool, default=False, help='Whether to ablate causal prior graph (disable causal mask)')
    parser.add_argument('--edl_mode', type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to enable EDL mode')
    parser.add_argument('--annealing_step', type=int, default=10, help='Epochs for KL divergence annealing in EDL')
    parser.add_argument('--special_note', type=str, default='')
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_folder_name = f"temp_{args.exp_name}_{current_time}_kfold_5metrics"
    temp_save_path = os.path.join(args.save_dir, temp_folder_name)
    final_folder_name = f"{args.exp_name}_{current_time}_kfold_metrics"
    final_save_path = os.path.join(args.save_dir, final_folder_name)

    os.makedirs(temp_save_path, exist_ok=True)
    logger = get_logger(os.path.join(temp_save_path, 'train.log'))

    try:
        logger.info(f"Start {args.k_folds}-Fold Cross Validation (Strict Subject Separation)")
        log_hyperparameters(logger, args)

        X_oxy_all, X_dxy_all, y_all = load_raw_data(args)
        target_len = X_oxy_all.shape[1] + 1
        X_chan_oxy_all, X_chan_dxy_all = load_excel_channel_data_dual(args.data_path, target_len=target_len)

        X_chan_oxy_all = np.delete(X_chan_oxy_all, 0, axis=2)
        X_chan_dxy_all = np.delete(X_chan_dxy_all, 0, axis=2)

        if args.roi_mode == 'grid_1x3':
            roi_mapping = [
                [0],
                [1, 2],
                [3],
                [4, 5],
                [6],
                [7, 8],
                [9],
                [10, 11],
                [12],
                [13, 14],
                [15],
                [16, 17],
                [18],
                [19, 20],
                [21]
            ]
        else:
            roi_mapping = [
                [0, 4, 5],
                [1, 2],
                [3, 7, 8],
                [9, 13, 14, 18],
                [6, 10, 11, 15, 19, 20],
                [12, 16, 17, 21]
            ]

        logger.info(f"Loaded raw data. Total subjects: {len(y_all)}")

        kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

        fold_final_metrics = []
        all_fold_cms = []
        device = torch.device(args.device)

        logger.info("\n>>> Start calculating global Oxy and Dxy causal prior based on all data (for observation only)...")

        all_adhd_mask = (y_all == 0)
        all_hc_mask = (y_all == 1)

        global_A_causal_oxy_adhd = compute_causal_prior_from_channels(
            fnirs_channel_data=X_chan_oxy_all[all_adhd_mask], roi_mapping=roi_mapping
        )
        global_A_causal_oxy_hc = compute_causal_prior_from_channels(
            fnirs_channel_data=X_chan_oxy_all[all_hc_mask], roi_mapping=roi_mapping
        )

        global_binary_A_causal_oxy_adhd = (global_A_causal_oxy_adhd != 0).any(dim=2).float()
        global_binary_A_causal_oxy_hc = (global_A_causal_oxy_hc != 0).any(dim=2).float()
        global_binary_A_causal_oxy = (
                    global_binary_A_causal_oxy_adhd.bool() | global_binary_A_causal_oxy_hc.bool()).float()

        logger.info(f"[Global Observation] binary_A_causal_oxy_adhd: \n{global_binary_A_causal_oxy_adhd}")
        logger.info(f"[Global Observation] binary_A_causal_oxy_hc: \n{global_binary_A_causal_oxy_hc}")
        logger.info(f"[Global Observation] binary_A_causal_oxy (Union): \n{global_binary_A_causal_oxy}")

        global_A_causal_dxy_adhd = compute_causal_prior_from_channels(
            fnirs_channel_data=X_chan_dxy_all[all_adhd_mask], roi_mapping=roi_mapping
        )
        global_A_causal_dxy_hc = compute_causal_prior_from_channels(
            fnirs_channel_data=X_chan_dxy_all[all_hc_mask], roi_mapping=roi_mapping
        )

        global_binary_A_causal_dxy_adhd = (global_A_causal_dxy_adhd != 0).any(dim=2).float()
        global_binary_A_causal_dxy_hc = (global_A_causal_dxy_hc != 0).any(dim=2).float()
        global_binary_A_causal_dxy = (
                    global_binary_A_causal_dxy_adhd.bool() | global_binary_A_causal_dxy_hc.bool()).float()

        logger.info(f"[Global Observation] binary_A_causal_dxy_adhd: \n{global_binary_A_causal_dxy_adhd}")
        logger.info(f"[Global Observation] binary_A_causal_dxy_hc: \n{global_binary_A_causal_dxy_hc}")
        logger.info(f"[Global Observation] binary_A_causal_dxy (Union): \n{global_binary_A_causal_dxy}")

        logger.info(">>> Global causal prior maps printed, starting 5-fold cross validation...\n")

        global_adhd_b = []
        global_adhd_u = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_oxy_all)):
            logger.info(f"\n{'=' * 20} Fold [{fold + 1}/{args.k_folds}] {'=' * 20}")
            logger.info(f"Train subjects: {len(train_idx)}, Val subjects: {len(val_idx)}")

            X_train_oxy_raw = X_oxy_all[train_idx]
            X_train_dxy_raw = X_dxy_all[train_idx]
            y_train_raw = y_all[train_idx]

            X_val_oxy_raw = X_oxy_all[val_idx]
            X_val_dxy_raw = X_dxy_all[val_idx]
            y_val_raw = y_all[val_idx]

            if not args.disable_causal:
                logger.info(">>> Start calculating Oxy and Dxy causal prior for current fold...")
                X_train_chan_oxy = X_chan_oxy_all[train_idx]
                X_train_chan_dxy = X_chan_dxy_all[train_idx]

                train_adhd_mask = (y_train_raw == 0)
                train_hc_mask = (y_train_raw == 1)

                A_causal_oxy_adhd = compute_causal_prior_from_channels(
                    fnirs_channel_data=X_train_chan_oxy[train_adhd_mask], roi_mapping=roi_mapping
                ).to(device)

                A_causal_oxy_hc = compute_causal_prior_from_channels(
                    fnirs_channel_data=X_train_chan_oxy[train_hc_mask], roi_mapping=roi_mapping
                ).to(device)

                A_causal_oxy = torch.where((A_causal_oxy_adhd != 0) | (A_causal_oxy_hc != 0), 1.0, 0.0)

                binary_A_causal_oxy_adhd = (A_causal_oxy_adhd != 0).any(dim=2).float()
                binary_A_causal_oxy_hc = (A_causal_oxy_hc != 0).any(dim=2).float()
                logger.info(f"binary_A_causal_oxy_adhd: \n{binary_A_causal_oxy_adhd}")
                logger.info(f"binary_A_causal_oxy_hc: \n{binary_A_causal_oxy_hc}")
                binary_A_causal_oxy = (binary_A_causal_oxy_adhd.bool() | binary_A_causal_oxy_hc.bool()).float()
                logger.info(f"binary_A_causal_oxy: \n{binary_A_causal_oxy}")

                A_causal_dxy_adhd = compute_causal_prior_from_channels(
                    fnirs_channel_data=X_train_chan_dxy[train_adhd_mask], roi_mapping=roi_mapping
                ).to(device)

                A_causal_dxy_hc = compute_causal_prior_from_channels(
                    fnirs_channel_data=X_train_chan_dxy[train_hc_mask], roi_mapping=roi_mapping
                ).to(device)

                A_causal_dxy = torch.where((A_causal_dxy_adhd != 0) | (A_causal_dxy_hc != 0), 1.0, 0.0)

                binary_A_causal_dxy_adhd = (A_causal_dxy_adhd != 0).any(dim=2).float()
                binary_A_causal_dxy_hc = (A_causal_dxy_hc != 0).any(dim=2).float()
                logger.info(f"binary_A_causal_dxy_adhd: \n{binary_A_causal_dxy_adhd}")
                logger.info(f"binary_A_causal_dxy_hc: \n{binary_A_causal_dxy_hc}")
                binary_A_causal_dxy = (binary_A_causal_dxy_adhd.bool() | binary_A_causal_dxy_hc.bool()).float()
                logger.info(f"binary_A_causal_dxy: \n{binary_A_causal_dxy}")

                logger.info(">>> Dual causal prior (ADHD & HC union) calculation completed!")

            else:
                logger.info(">>> [Ablation] Causal prior graph disabled, model degrades to full spatial attention!")
                A_causal_oxy = None
                A_causal_dxy = None

            train_oxy_aug, train_dxy_aug, train_y_aug = augment_data_odd_even(X_train_oxy_raw, X_train_dxy_raw,
                                                                              y_train_raw)
            val_oxy_aug, val_dxy_aug, val_y_aug = augment_data_odd_even(X_val_oxy_raw, X_val_dxy_raw, y_val_raw)

            logger.info(f"Augmented Train Samples: {len(train_y_aug)} | Augmented Val Samples: {len(val_y_aug)}")

            train_dataset = DualModalityDataset(train_oxy_aug, train_dxy_aug, train_y_aug)
            val_dataset = DualModalityDataset(val_oxy_aug, val_dxy_aug, val_y_aug)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)

            model = DualBranchRecurrentModel(
                embed_dim=args.hidden_dims,
                num_heads=args.head,
                depth=args.depth,
                k_memory=args.k_memory,
                num_classes=args.num_classes,
                drop=args.dropout,
                attn_drop=args.attn_drop,
                roi_mode=args.roi_mode,
                keep_ratio=args.keep_ratio,
                chunk_size=args.chunk_size,
                edl_mode=args.edl_mode
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                             patience=args.optim_patience)

            best_model_path = os.path.join(temp_save_path, f'best_model_fold_{fold}.pt')
            early_stopping = EarlyStopping(patience=args.patience, verbose=False, path=best_model_path)

            train_losses = []
            val_losses = []

            for epoch in range(args.epochs):
                train_loss, t_m = train_one_epoch(model, train_loader, criterion, optimizer, device, A_causal_oxy,
                                                  A_causal_dxy, epoch, args)
                val_loss, v_m = evaluate(model, val_loader, criterion, device, A_causal_oxy, A_causal_dxy, args)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    logger.info(f"Fold {fold + 1} Epoch [{epoch + 1}/{args.epochs}] "
                                f"T_Loss: {train_loss:.4f} | T_Acc: {t_m['acc']:.4f} T_Pre: {t_m['precision']:.4f} T_Rec: {t_m['recall']:.4f} T_F1: {t_m['f1']:.4f} T_AUC: {t_m['auc']:.4f} T_NLL: {t_m['nll']:.4f} T_ECE: {t_m['ece']:.4f}")
                    logger.info(
                        f"V_Loss: {val_loss:.4f} | V_Acc: {v_m['acc']:.4f} V_Pre: {v_m['precision']:.4f} V_Rec: {v_m['recall']:.4f} V_F1: {v_m['f1']:.4f} V_AUC: {v_m['auc']:.4f} V_NLL: {v_m['nll']:.4f} V_ECE: {v_m['ece']:.4f}")
                early_stopping(val_acc=v_m['acc'], val_loss=val_loss, model=model, logger=logger)
                if early_stopping.early_stop:
                    break

            plot_loss_curve(train_losses, val_losses, temp_save_path, fold)

            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path))
                f_loss, f_m, val_features, val_labels, val_b, val_u, val_P = evaluate_with_features(model, val_loader, criterion, device, A_causal_oxy, A_causal_dxy, args)
                plot_tsne(val_features, val_labels, temp_save_path, fold + 1)

                logger.info(f"Fold {fold + 1} t-SNE plot saved.")

                if args.edl_mode:
                    mean_b = np.mean(val_b)
                    mean_u = np.mean(val_u)
                    mean_P = np.mean(val_P)
                    logger.info(
                        f"Fold {fold + 1} Best Model EDL Metrics -> Mean b: {mean_b:.4f}, Mean u: {mean_u:.4f}, Mean Expected P: {mean_P:.4f}")

                    df_metrics = pd.DataFrame({
                        'True_Label': val_labels,
                        'Belief_b': val_b,
                        'Uncertainty_u': val_u,
                        'Expected_Prob_P': val_P
                    })
                    csv_name = os.path.join(temp_save_path, f'fold_{fold + 1}_edl_samples.csv')
                    df_metrics.to_csv(csv_name, index=False)
                    logger.info(f"Fold {fold + 1} EDL metrics for single samples saved to CSV: {csv_name}")

                    for lbl, b_val, u_val in zip(val_labels, val_b, val_u):
                        if lbl == 0:
                            global_adhd_b.append(b_val)
                            global_adhd_u.append(u_val)

                fold_final_metrics.append(f_m)
                all_fold_cms.append(f_m['cm'])
                logger.info(
                    f"Fold {fold + 1} BEST Result -> Acc: {f_m['acc']:.4f}, Pre: {f_m['precision']:.4f}, Rec: {f_m['recall']:.4f}, F1: {f_m['f1']:.4f}, AUC: {f_m['auc']:.4f}, NLL: {f_m['nll']:.4f}, ECE: {f_m['ece']:.4f}")
            else:
                fold_final_metrics.append({"acc": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0, "nll": 0, "ece": 0})


        if len(all_fold_cms) > 0:
            plot_mean_std_conf_matrix(all_fold_cms, temp_save_path)

        if args.edl_mode and len(global_adhd_b) > 0:
            plot_edl_scatter(global_adhd_b, global_adhd_u, temp_save_path)
            logger.info(f">>> b-u scatter plot for global ADHD samples (total {len(global_adhd_b)}) plotted and saved!")

        logger.info("\n" + "=" * 30)
        logger.info(f"Final {args.k_folds}-Fold CV Summary:")
        for m_name in ["acc", "precision", "recall", "f1", "auc", "nll", "ece"]:
            vals = [f[m_name] for f in fold_final_metrics]
            logger.info(f"{m_name.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        logger.info("=" * 30)

        close_logger(logger)
        if os.path.exists(temp_save_path):
            os.rename(temp_save_path, final_save_path)
            print(f"Saved to: {final_save_path}")

    except Exception as e:
        close_logger(logger)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()