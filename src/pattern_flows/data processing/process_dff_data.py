# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pattern-flows",
# ]
#
# [tool.uv.sources]
# pattern-flows = {path = "../../src/pattern_flows", editable = true}
# ///

import os
import numpy as np
import torch
from torch.utils.data import random_split
from pathlib import Path
from scipy.signal import decimate
from scipy.ndimage import percentile_filter
from numpy import interp

import pattern_flows.utils as u
from pattern_flows.src.pattern_flows.data.dff_data import get_fluor_dataset
    
def segment_data(F_cells, window_size=20, step_size=20, num_subsets=20):
    num_cells, num_timepoints = F_cells.shape
    subset_size = num_cells // num_subsets
    windows_all = []

    for _ in range(num_subsets):
        # randomly pick subset_size cells
        subset_indices = np.random.choice(num_cells, subset_size, replace=False)
        F_subset = F_cells[subset_indices, :]

        # create sliding windows for subset
        windows = []
        for start in range(0, num_timepoints - window_size + 1, step_size):
            end = start + window_size
            windows.append(F_subset[::3, start:end])  # take every third cell

        windows = np.stack(windows, axis=0)  # shape: (num_windows, subset_size, window_size)
        windows_all.append(windows)

    num_windows = windows_all[0].shape[0]
    
    print("num_windows per subset:", num_windows)

    X = np.concatenate(windows_all, axis=0)  # shape: (num_windows * num_subsets, subset_size, window_size)
    return X

def dff(trace, window=1500, percentile=20, downsample=1):
    """
    Estimate delta-f / f_baseline with the option to downsample
    
    data : 1D numpy array
        Data to be processed
        
    window : int
        Window size for baseline estimation. If downsampling is used, window will shrink proportionally
        
    percentile : int
        Percentile used as baseline
    
    downsample : int
        Rate of downsampling used before estimating baseline. For no downsampling, use downsample=1.
    """

    data = trace

    if downsample == 1:
        baseline = percentile_filter(data, percentile=percentile, size=window)
    else:
        data_ds = decimate(data, downsample, ftype='iir', zero_phase=True)
        # using decimate with the default filter shifts the output by ~1-2% relative to the input. 
        # correct for baseline shift by adding a small constant to data_ds
        data_ds += data.min() - data_ds.min()
        baseline_ds = percentile_filter(data_ds, percentile=percentile, size=window//downsample)
        
        baseline = interp(range(0, len(data)), range(0, len(data), downsample), baseline_ds)
    
    return (data - baseline) / baseline

if __name__ == "__main__":
    # --- load parameters ---
    HERE = Path(__file__).parent
    cfg_path = HERE / "config.yaml"
    cfg = u.load_config(cfg_path)
    
    device        = cfg["device"]["type"]
    batch_size    = cfg["training"]["batch_size"]
    ds_dir        = cfg["paths"]["ds_dir"]
    data_dir      = cfg["fluor_data"]["fish1_dir"]
    window_size   = cfg["fluor_data"]["window_size"]
    train_frac    = cfg["training"]["train_frac"]

    F       = np.load(os.path.join(data_dir, "F.npy"), allow_pickle=True, mmap_mode="r")       # shape : (num_rois, num_frames)
    # Fneu   = np.load(os.path.join(data_dir, "Fneu.npy"), allow_pickle=True, mmap_mode="r")    # shape : (num_rois, num_frames)
    # spks   = np.load(os.path.join(data_dir, "spks.npy"), allow_pickle=True, mmap_mode="r")    # shape : (num_rois, num_frames)
    # stat   = np.load(os.path.join(data_dir, "stat.npy"), allow_pickle=True)
    # ops    = np.load(os.path.join(data_dir, "ops.npy"), allow_pickle=True)
    # ops    = ops.item()
    iscell = np.load(os.path.join(data_dir, "iscell.npy"), allow_pickle=True, mmap_mode="r")  # shape : (num_rois, 2)

    num_rois = F.shape[0]
    num_frames = F.shape[1]

    # --- select ROIs that are cells ---
    cell_idx = np.where(iscell[:, 0])[0]
    F_cells = F[cell_idx, :]

    num_cells, num_frames = F_cells.shape
    print(f"num_cells: {num_cells}, num_frames: {num_frames}")

    # --- compute dF/F of traces ---
    dff_traces = []
    print("starting dff calculation")

    for i, cell in enumerate(F_cells):
        dff_traces.append(dff(cell))

        if i == num_cells // 2:
            print("dff calculation halfway complete")

    dff_traces = np.array(dff_traces)

    # --- visualize cells ---
    # im = np.zeros((ops["Ly"], ops["Lx"]))
    # 
    # mean_image = ops["meanImg"]
    # num_planes = ops["nplanes"]
    # 
    # for n in range(0,num_cells):
    #     ypix = stat[n]["ypix"][~stat[n]["overlap"]]
    #     xpix = stat[n]["xpix"][~stat[n]["overlap"]]
    #     im[ypix, xpix] = n + 1  # Label cells starting from 1

    # plt.imshow(mean_image, cmap="gray")
    # plt.imshow(im, alpha=0.8)
    # plt.show()

    # --- separate early- and late-trial activity ---
    early_seg = dff_traces[:, :600]   # first 10 min
    late_seg  = dff_traces[:, -600:]  # last 10 min

    step_size   = 1
    num_subsets = 1

    X_early = segment_data(early_seg, window_size=window_size, step_size=step_size, num_subsets=num_subsets)  # shape: (num_windows * num_subsets, subset_size, window_size)
    X_late  = segment_data(late_seg, window_size=window_size, step_size=step_size, num_subsets=num_subsets)   # shape: (num_windows * num_subsets, subset_size, window_size)
    
    print(f"X_early shape: {X_early.shape}, X_late shape: {X_late.shape}")

    y_early = np.zeros(X_early.shape[0])  # class 0
    y_late  = np.ones(X_late.shape[0])    # class 1

    X_combined = np.concatenate([X_early, X_late], axis=0)
    y_combined = np.concatenate([y_early, y_late], axis=0)

    save_dir = u.get_save_dir(ds_dir)

    # --- create training & validation datasets ---
    ds = get_fluor_dataset(X_combined, y_combined)
    torch.save(ds, save_dir / "ds.pt")

    train_size = int(train_frac * len(ds))
    val_size = len(ds) - train_size

    train_ds, val_ds = random_split(ds, [train_size, val_size])

    print("total num samples:", len(ds))
    print("num training samples:", len(train_ds))
    print("num validation samples:", len(val_ds))

    splits_path = save_dir / "splits.pt"
    torch.save({
        "train_idx": train_ds.indices,
        "val_idx": val_ds.indices
    }, splits_path)

    print(f"processed data saved to {save_dir}")





