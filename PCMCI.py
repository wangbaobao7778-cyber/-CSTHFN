import numpy as np
import torch
from scipy.signal import decimate
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from statsmodels.stats.multitest import multipletests


def apply_global_signal_regression(fnirs_data):
    B, C, T = fnirs_data.shape
    cleaned_data = np.zeros_like(fnirs_data)

    for b in range(B):
        global_signal = np.mean(fnirs_data[b], axis=0)
        global_centered = global_signal - np.mean(global_signal)
        var_g = np.var(global_centered)

        for c in range(C):
            y = fnirs_data[b, c]
            y_centered = y - np.mean(y)

            if var_g == 0:
                cleaned_data[b, c] = y
            else:
                beta = np.cov(global_centered, y_centered)[0, 1] / var_g
                cleaned_data[b, c] = y - beta * global_centered

    return cleaned_data


def compute_causal_prior_from_channels(fnirs_channel_data, roi_mapping, tau_max=10, pc_alpha=0.05,
                                       effect_size_threshold=0.1):
    B, C, T = fnirs_channel_data.shape
    num_rois = len(roi_mapping)

    cleaned_channel_data = apply_global_signal_regression(fnirs_channel_data)

    roi_data = np.zeros((B, num_rois, T))
    for i, channels in enumerate(roi_mapping):
        roi_data[:, i, :] = np.mean(cleaned_channel_data[:, channels, :], axis=1)

    downsample_factor = 10
    roi_data_downsampled = roi_data[:, :, ::downsample_factor]


    data_for_tigramite = np.transpose(roi_data_downsampled, (0, 2, 1))
    var_names = [f'ROI_{i}' for i in range(num_rois)]

    dataframe = pp.DataFrame(
        data_for_tigramite,
        var_names=var_names,
        analysis_mode='multiple'
    )

    cond_ind_test = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)

    results = pcmci.run_pcmci(
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        max_conds_dim=2,
        max_conds_px=2,
        max_conds_py=2
    )

    p_matrix = results['p_matrix']
    val_matrix = results['val_matrix']

    p_flat = p_matrix.flatten()
    valid_mask = ~np.isnan(p_flat)
    q_flat = np.ones_like(p_flat)

    _, q_values, _, _ = multipletests(p_flat[valid_mask], alpha=pc_alpha, method='fdr_bh')
    q_flat[valid_mask] = q_values
    q_matrix = q_flat.reshape(p_matrix.shape)

    clean_causal_matrix = np.where(
        (q_matrix < pc_alpha) & (np.abs(val_matrix) > effect_size_threshold),
        val_matrix,
        0.0
    )

    clean_causal_matrix[:, :, 0] = 0.0

    return torch.tensor(clean_causal_matrix, dtype=torch.float32)