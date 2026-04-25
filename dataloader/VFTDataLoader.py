# --- START OF FILE dataloader/VFTDataLoader.py ---

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
from sklearn.preprocessing import StandardScaler
import pandas as pd

class DualModalityDataset(Dataset):
    def __init__(self, oxy_data, dxy_data, labels):
        """
        Dual modality dataset
        """
        # Ensure inputs are Tensors
        if isinstance(oxy_data, np.ndarray):
            self.oxy_data = torch.from_numpy(oxy_data).float()
        else:
            self.oxy_data = oxy_data.float()

        if isinstance(dxy_data, np.ndarray):
            self.dxy_data = torch.from_numpy(dxy_data).float()
        else:
            self.dxy_data = dxy_data.float()

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.oxy_data[idx], self.dxy_data[idx], self.labels[idx]


def augment_data_odd_even(oxy, dxy, label):
    """
    Data augmentation function: odd-even sampling
    Input shape: (N, T, H, W)
    Output shape: (2N, T/2, H, W)
    """
    # 1. Check if the time dimension is even
    T = oxy.shape[1]
    if T % 2 != 0:
        # Handle silently, avoid too many prints
        oxy = oxy[:, :-1, ...]
        dxy = dxy[:, :-1, ...]

    # 2. Odd-even slicing
    oxy_even = oxy[:, 0::2, ...]
    dxy_even = dxy[:, 0::2, ...]

    oxy_odd = oxy[:, 1::2, ...]
    dxy_odd = dxy[:, 1::2, ...]

    # 3. Concatenate
    oxy_aug = np.concatenate((oxy_even, oxy_odd), axis=0)
    dxy_aug = np.concatenate((dxy_even, dxy_odd), axis=0)

    # 4. Duplicate labels
    label_aug = np.concatenate((label, label), axis=0)

    return oxy_aug, dxy_aug, label_aug


def load_raw_data(args):
    """
    Only responsible for loading raw numpy data, without splitting and augmentation.
    Returns: raw_oxy, raw_dxy, raw_labels (all samples)
    """
    data_path = args.data_path

    f_adhd_oxy = os.path.join(data_path, 'ADHD_grid_oxy.npy')
    f_adhd_dxy = os.path.join(data_path, 'ADHD_grid_dxy.npy')
    f_hc_oxy = os.path.join(data_path, 'HC_grid_oxy.npy')
    f_hc_dxy = os.path.join(data_path, 'HC_grid_dxy.npy')

    adhd_oxy = np.load(f_adhd_oxy)
    adhd_dxy = np.load(f_adhd_dxy)
    adhd_labels = np.zeros(len(adhd_oxy))

    hc_oxy = np.load(f_hc_oxy)
    hc_dxy = np.load(f_hc_dxy)
    hc_labels = np.ones(len(hc_oxy))

    # Concatenate
    X_oxy = np.concatenate((adhd_oxy, hc_oxy), axis=0)
    X_dxy = np.concatenate((adhd_dxy, hc_dxy), axis=0)
    y = np.concatenate((adhd_labels, hc_labels), axis=0)

    # Unified preprocessing: delete the first frame (if needed)
    # Assume original is 1600 -> delete first frame -> 1599 -> subsequent augment will truncate to 1598
    X_oxy = np.delete(X_oxy, 0, axis=1)
    X_dxy = np.delete(X_dxy, 0, axis=1)

    print(f"Raw Data Loaded. Shape: {X_oxy.shape}, Labels: {y.shape}")

    return X_oxy, X_dxy, y


def load_excel_channel_data_dual(data_path, target_len):
    """
    1:1 replica of the channel data extraction function from data2grid.py preprocessing logic (dual modality version)
    Extracts both Oxy and Dxy simultaneously, and performs independent standardization and length alignment.
    """
    adhd_dir = os.path.join(data_path, 'ADHD_xlsx')
    hc_dir = os.path.join(data_path, 'HC_xlsx')

    def read_dir(directory):
        # ⚠️ Extremely important: Must be the same as data2grid.py, do not use sorted()
        files = glob.glob(os.path.join(directory, "*.xlsx")) + \
                glob.glob(os.path.join(directory, "*.xls"))

        if not files:
            raise FileNotFoundError(f"No Excel files found in {directory}!")

        oxy_list, dxy_list = [], []
        for file_path in files:
            # --- Read Oxy ---
            df_oxy = pd.read_excel(file_path, sheet_name='oxyData', header=None)
            oxy_norm = StandardScaler().fit_transform(df_oxy.values[:, :22])

            # --- Read Dxy ---
            df_dxy = pd.read_excel(file_path, sheet_name='dxyData', header=None)
            dxy_norm = StandardScaler().fit_transform(df_dxy.values[:, :22])

            # --- Length alignment (Padding / Truncating) ---
            # Oxy alignment
            if oxy_norm.shape[0] < target_len:
                pad_width = target_len - oxy_norm.shape[0]
                oxy_norm = np.pad(oxy_norm, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
            elif oxy_norm.shape[0] > target_len:
                oxy_norm = oxy_norm[:target_len, :]

            # Dxy alignment
            if dxy_norm.shape[0] < target_len:
                pad_width = target_len - dxy_norm.shape[0]
                dxy_norm = np.pad(dxy_norm, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
            elif dxy_norm.shape[0] > target_len:
                dxy_norm = dxy_norm[:target_len, :]

            oxy_list.append(oxy_norm)
            dxy_list.append(dxy_norm)

        return oxy_list, dxy_list

    print("Extracting raw Oxy and Dxy channel data from Excel (strictly following preprocessing rules)...")
    adhd_oxy, adhd_dxy = read_dir(adhd_dir)
    hc_oxy, hc_dxy = read_dir(hc_dir)

    all_oxy = adhd_oxy + hc_oxy
    all_dxy = adhd_dxy + hc_dxy

    # Stack into a 3D array: (B, T, C) -> Transpose to (B, C, T)
    X_channel_oxy = np.transpose(np.stack(all_oxy, axis=0), (0, 2, 1))
    X_channel_dxy = np.transpose(np.stack(all_dxy, axis=0), (0, 2, 1))

    print(f"Excel extraction completed!")
    print(f"Oxy channel shape: {X_channel_oxy.shape}")
    print(f"Dxy channel shape: {X_channel_dxy.shape}")

    return X_channel_oxy, X_channel_dxy