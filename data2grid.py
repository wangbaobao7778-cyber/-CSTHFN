import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata


def process_single_sheet(file_path, sheet_name):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    except Exception as e:
        print(f"    - [Read failed] Sheet: {sheet_name} - {e}")
        return None

    data = df.values
    if data.shape[1] < 22:
        print(f"    - [Skipped] Insufficient columns: {sheet_name}")
        return None

    sensor_data = data[:, :22]

    scaler = StandardScaler()
    data_norm = scaler.fit_transform(sensor_data)

    coords_map = {
        0: (0, 1), 1: (0, 3), 2: (0, 5), 3: (0, 7),
        4: (1, 0), 5: (1, 2), 6: (1, 4), 7: (1, 6), 8: (1, 8),
        9: (2, 1), 10: (2, 3), 11: (2, 5), 12: (2, 7),
        13: (3, 0), 14: (3, 2), 15: (3, 4), 16: (3, 6), 17: (3, 8),
        18: (4, 1), 19: (4, 3), 20: (4, 5), 21: (4, 7)
    }
    points = np.array([coords_map[i] for i in range(22)])
    grid_x, grid_y = np.mgrid[0:5:1, 0:9:1]

    processed_matrices = []

    for row_values in data_norm:
        grid_z = griddata(points, row_values, (grid_x, grid_y), method='cubic')

        if np.isnan(grid_z).any():
            grid_z_nearest = griddata(points, row_values, (grid_x, grid_y), method='nearest')
            grid_z[np.isnan(grid_z)] = grid_z_nearest[np.isnan(grid_z)]

        grid_z[0, 0] = (grid_z[0, 1] + grid_z[1, 0]) / 2.0
        grid_z[0, 8] = (grid_z[0, 7] + grid_z[1, 8]) / 2.0
        grid_z[4, 0] = (grid_z[3, 0] + grid_z[4, 1]) / 2.0
        grid_z[4, 8] = (grid_z[3, 8] + grid_z[4, 7]) / 2.0

        processed_matrices.append(grid_z)

    return np.array(processed_matrices)


def process_single_sheet_zero(file_path, sheet_name):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    except Exception as e:
        print(f"    - [Read failed] Sheet: {sheet_name} - {e}")
        return None

    data = df.values
    if data.shape[1] < 22:
        print(f"    - [Skipped] Insufficient columns: {sheet_name}")
        return None

    sensor_data = data[:, :22]

    scaler = StandardScaler()
    data_norm = scaler.fit_transform(sensor_data)

    coords_map = {
        0: (0, 1), 1: (0, 3), 2: (0, 5), 3: (0, 7),
        4: (1, 0), 5: (1, 2), 6: (1, 4), 7: (1, 6), 8: (1, 8),
        9: (2, 1), 10: (2, 3), 11: (2, 5), 12: (2, 7),
        13: (3, 0), 14: (3, 2), 15: (3, 4), 16: (3, 6), 17: (3, 8),
        18: (4, 1), 19: (4, 3), 20: (4, 5), 21: (4, 7)
    }

    processed_matrices = []

    for row_values in data_norm:
        grid_z = np.zeros((5, 9))

        for i in range(22):
            if i in coords_map:
                r, c = coords_map[i]
                grid_z[r, c] = row_values[i]

        processed_matrices.append(grid_z)

    return np.array(processed_matrices)


def align_and_save(data_list, output_path, target_len, type_name="Data"):
    print(f"\n--- Processing {type_name} data ---")

    padded_data_list = []
    for d in data_list:
        current_len = d.shape[0]

        if current_len < target_len:
            pad_width = target_len - current_len
            d_padded = np.pad(d, ((0, pad_width), (0, 0), (0, 0)), mode='constant', constant_values=0)
            padded_data_list.append(d_padded)

        elif current_len > target_len:
            d_truncated = d[:target_len, :, :]
            padded_data_list.append(d_truncated)

        else:
            padded_data_list.append(d)

    final_dataset = np.array(padded_data_list)
    print(f"Shape: {final_dataset.shape}")
    np.save(output_path, final_dataset)
    print(f"Saved: {output_path}")


def batch_process_folder_dual(input_folder, output_base_name, fixed_length=None):
    print(f"--- Start processing folder: {input_folder} ---")
    all_files = glob.glob(os.path.join(input_folder, "*.xlsx")) + \
                glob.glob(os.path.join(input_folder, "*.xls"))

    if not all_files:
        print("Excel files not found!")
        return

    oxy_list = []
    dxy_list = []

    valid_count = 0

    for i, file_path in enumerate(all_files):
        file_name = os.path.basename(file_path)
        print(f"[{i + 1}/{len(all_files)}] Reading: {file_name}")

        oxy_data = process_single_sheet_zero(file_path, 'oxyData')
        dxy_data = process_single_sheet_zero(file_path, 'dxyData')

        if oxy_data is not None and dxy_data is not None:
            if oxy_data.shape[0] != dxy_data.shape[0]:
                print(
                    f"    [Warning] Inconsistent time steps (Oxy:{oxy_data.shape[0]}, Dxy:{dxy_data.shape[0]}), will truncate/pad individually.")

            oxy_list.append(oxy_data)
            dxy_list.append(dxy_data)
            valid_count += 1
            print(f"    -> Success (T={oxy_data.shape[0]})")
        else:
            print("    -> Skipped (A sheet failed to read)")

    if valid_count == 0:
        print("No valid data.")
        return

    if fixed_length is None:
        max_len_oxy = max([d.shape[0] for d in oxy_list])
        max_len_dxy = max([d.shape[0] for d in dxy_list])
        target_len = max(max_len_oxy, max_len_dxy)
        print(f"\n--- Auto-aligned length: {target_len} (Based on Max T) ---")
    else:
        target_len = fixed_length
        print(f"\n--- Forced aligned length: {target_len} ---")

    base, _ = os.path.splitext(output_base_name)

    out_oxy = f"{base}_oxy_zero.npy"
    out_dxy = f"{base}_dxy_zero.npy"

    align_and_save(oxy_list, out_oxy, target_len, "OxyData")
    align_and_save(dxy_list, out_dxy, target_len, "DxyData")

    print("\n--- All completed ---")


if __name__ == "__main__":
    my_input_folder = 'data/VFT/HC_xlsx'
    my_output_base = 'data/VFT/HC_grid.npy'

    if os.path.exists(my_input_folder):
        batch_process_folder_dual(my_input_folder, my_output_base, fixed_length=None)
    else:
        print(f"Folder not found: {my_input_folder}")