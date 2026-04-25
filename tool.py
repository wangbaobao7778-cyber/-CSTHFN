import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import logging
import sys
import random
import json

from sklearn.manifold import TSNE
# ==========================================
# Added: EDL utility functions (placed at the end of tool.py)
# ==========================================
import torch.nn.functional as F
import pandas as pd


PLOT_FONT_CONFIG = {
    'family': 'Times New Roman',  # Unified font style
    'axis_label_size': 21,  # X-axis, Y-axis label text size
    'tick_label_size': 21,  # Axis tick numbers, Colorbar number size
    'annot_size': 30,  # Size of the 4 numbers inside the confusion matrix
    'legend_size': 18  # t-SNE legend text size
}


def plot_tsne(features, labels, save_path, fold_idx, title_prefix="Val Set"):
    """
    features: (N, D) model output features
    labels: (N,) true labels
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    # Assume 0 is ADHD, 1 is HC
    classes = ['ADHD', 'HC']
    colors = ['r', 'b']

    for i, class_name in enumerate(classes):
        mask = (labels == i)
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    c=colors[i], label=class_name, alpha=0.6, edgecolors='w')

    # Remove title (commented out)
    # plt.title(f'{title_prefix} t-SNE - Fold {fold_idx}')

    # --- Read config from dictionary ---
    font_family = PLOT_FONT_CONFIG['family']
    tick_size = PLOT_FONT_CONFIG['tick_label_size']
    legend_size = PLOT_FONT_CONFIG['legend_size']

    # Adjust font style and size of numbers on axes
    plt.xticks(fontname=font_family, fontsize=tick_size)
    plt.yticks(fontname=font_family, fontsize=tick_size)

    # Adjust legend font
    plt.legend(prop={'family': font_family, 'size': legend_size})
    plt.grid(True, linestyle='--', alpha=0.3)

    img_name = f'tsne_fold_{fold_idx}.png'
    plt.savefig(os.path.join(save_path, img_name))
    plt.close()


def plot_mean_std_conf_matrix(cm_list, save_path):
    """
    cm_list: List containing K confusion matrices [cm1, cm2, ..., cmK]
    """
    cms = np.array(cm_list)  # Shape: (K, 2, 2)

    # Calculate mean and standard deviation
    cm_mean = np.mean(cms, axis=0)
    cm_std = np.std(cms, axis=0)

    # Build text labels displayed in the grid (Mean ± Std)
    annot = np.empty_like(cm_mean).astype(str)
    rows, cols = cm_mean.shape
    for r in range(rows):
        for c in range(cols):
            annot[r, c] = f"{cm_mean[r, c]:.2f}\n±{cm_std[r, c]:.2f}"

    plt.figure(figsize=(8, 6))

    # --- Read config from dictionary ---
    font_family = PLOT_FONT_CONFIG['family']
    annot_size = PLOT_FONT_CONFIG['annot_size']
    axis_label_size = PLOT_FONT_CONFIG['axis_label_size']
    tick_size = PLOT_FONT_CONFIG['tick_label_size']

    # Adjust font style and size of numbers inside the confusion matrix plot
    ax = sns.heatmap(cm_mean, annot=annot, fmt="", cmap='Blues',
                     xticklabels=['ADHD', 'HC'], yticklabels=['ADHD', 'HC'],
                     annot_kws={"family": font_family, "size": annot_size})

    # Set font for axis labels
    plt.xlabel('Predicted', fontdict={'family': font_family, 'size': axis_label_size})
    plt.ylabel('Actual', fontdict={'family': font_family, 'size': axis_label_size})

    # Set font for XY tick labels
    plt.xticks(fontname=font_family, fontsize=tick_size)
    plt.yticks(fontname=font_family, fontsize=tick_size)

    # Get the Colorbar of the heatmap and modify the font of its numbers
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=tick_size)  # Modify Colorbar number size
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname(font_family)  # Modify Colorbar number font

    # Remove title (commented out)
    # plt.title('Average Confusion Matrix (Mean ± Std)')

    plt.savefig(os.path.join(save_path, 'cm_mean_std.png'))
    plt.close()

# ==========================================
# 1. Utility functions
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_logger(save_path):
    """
    Initialize logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear previous handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Console output
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File output
    if save_path:
        fh = logging.FileHandler(save_path, mode='w', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def close_logger(logger):
    """
    Manually close logger handlers to prevent errors when deleting folders
    """
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def log_hyperparameters(logger, args):
    """
    Print and save all hyperparameters to the log
    """
    logger.info("=" * 30)
    logger.info("Current Configuration (Hyperparameters):")
    # Convert args to dict and pretty print
    args_dict = vars(args)
    logger.info(json.dumps(args_dict, indent=4))
    logger.info("=" * 30)


def plot_loss_curve(train_losses, val_losses, save_path, fold_idx):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red', linestyle='--')
    plt.title(f'Fold {fold_idx} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'loss_curve_fold_{fold_idx}.png'))
    plt.close()



# ==========================================
# 3. Early Stopping
# ==========================================
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, path='checkpoint.pt'):
        """
        Args:
            patience (int): Stop if no improvement for given epochs
            verbose (bool): Whether to print log
            path (str): Model save path
        """
        self.patience = patience
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.early_stop = False

        # Record best metrics
        self.best_acc = -np.Inf
        self.best_loss = np.Inf

    def __call__(self, val_acc, val_loss, model, logger):
        """
        Logic:
        1. If current Acc > historical best Acc: save model
        2. If current Acc == historical best Acc:
             if current Loss < historical best Loss: save model
             else: counter +1
        3. If current Acc < historical best Acc: counter +1
        """

        # Case 1: Accuracy reaches new high
        if val_acc > self.best_acc:
            if self.verbose:
                logger.info(f'Validation Acc increased ({self.best_acc:.4f} --> {val_acc:.4f}). Saving model...')
            self.best_acc = val_acc
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0  # Reset counter

        # Case 2: Accuracy ties, check if Loss decreases
        elif val_acc == self.best_acc:
            if val_loss < self.best_loss:
                if self.verbose:
                    logger.info(
                        f'Acc same ({self.best_acc:.4f}), but Loss decreased ({self.best_loss:.4f} --> {val_loss:.4f}). Saving model...')
                self.best_loss = val_loss
                self.save_checkpoint(model)
                self.counter = 0  # Reset counter
            else:
                self.counter += 1
                if self.verbose:
                    logger.info(
                        f'EarlyStopping counter: {self.counter} out of {self.patience} (Acc same, Loss not improved)')

        # Case 3: Accuracy drops
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience} (Acc decreased)')

        # Check if early stopping is triggered
        if self.counter >= self.patience:
            self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def edl_mse_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device):
    """EDL mean squared error loss + KL divergence annealing"""
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    # Expected probability mean squared error
    A = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    # Prediction variance
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)

    # KL divergence annealing coefficient
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    alpha_tilde = y + (1 - y) * alpha
    KL = kl_divergence(alpha_tilde, num_classes, device=device)

    loss = (A + B) + annealing_coef * KL
    return loss.mean()


def plot_edl_scatter(b_list, u_list, save_path):
    """Plot Belief - Uncertainty scatter plot for ADHD samples"""
    plt.figure(figsize=(8, 6))

    # Use red scatter points for ADHD, medium size, with white edges
    plt.scatter(b_list, u_list, c='#D62728', alpha=0.7, edgecolors='white', s=80, label='ADHD Samples')

    font_family = PLOT_FONT_CONFIG['family']
    plt.xlabel('Belief (b)', fontdict={'family': font_family, 'size': PLOT_FONT_CONFIG['axis_label_size']})
    plt.ylabel('Uncertainty (u)', fontdict={'family': font_family, 'size': PLOT_FONT_CONFIG['axis_label_size']})

    plt.xticks(fontname=font_family, fontsize=PLOT_FONT_CONFIG['tick_label_size'])
    plt.yticks(fontname=font_family, fontsize=PLOT_FONT_CONFIG['tick_label_size'])

    plt.legend(prop={'family': font_family, 'size': PLOT_FONT_CONFIG['legend_size']})
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'adhd_edl_scatter.png'), dpi=300)
    plt.close()


def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error (ECE)"""
    # Get predicted class probability (confidence) and predicted labels
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece