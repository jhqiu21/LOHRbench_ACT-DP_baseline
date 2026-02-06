#!/usr/bin/env python3
"""
DataLoader Bottleneck Diagnostic for LoHRbench Diffusion Policy
================================================================
Profiles every stage of the data pipeline to identify where GPU starvation
comes from. Run on the training machine (no GPU needed).

Usage:
  python benchmark_dataloader.py --data-root /data1/LoHRbench
  python benchmark_dataloader.py --data-root /data1/LoHRbench --quick
"""

from __future__ import annotations

import argparse
import gc
import os
import pathlib
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler

# ---------------------------------------------------------------------------
# Import from the training script (same directory)
# ---------------------------------------------------------------------------
from train_lohrbench import (
    TASK_TYPES,
    REQUIRED_H5_SUFFIX,
    TASK_INSTRUCTIONS,
    RGB_SIZE,
    STATE_DIM,
    ACTION_DIM,
    discover_lohrbench_h5_files,
    action_to_unit_range,
)
from diffusion_policy.utils import IterationBasedBatchSampler, worker_init_fn


# ============================================================================
# Helpers
# ============================================================================

def fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def pstats(times: List[float]) -> Dict[str, float]:
    """Return mean, median, p5, p95 from a list of times (seconds)."""
    if not times:
        return {"mean": 0, "median": 0, "p5": 0, "p95": 0, "std": 0}
    s = sorted(times)
    n = len(s)
    return {
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "p5": s[max(0, int(n * 0.05))],
        "p95": s[min(n - 1, int(n * 0.95))],
        "std": statistics.stdev(s) if n > 1 else 0.0,
    }


def print_header(title: str):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def print_row(label: str, value: str, indent: int = 2):
    print(f"{' ' * indent}{label:<35s} {value}")


# ============================================================================
# Test 1: Filesystem Analysis
# ============================================================================

def test_1_filesystem(data_root: str, file_list: List[Tuple[str, str, str]]):
    print_header("Test 1: Filesystem Analysis")

    # 1a. Mount point
    print_row("Data root:", data_root)
    print_row("Is mount point:", str(os.path.ismount(data_root)))

    # Walk up to find nearest mount
    p = pathlib.Path(data_root).resolve()
    while not p.is_mount() and p != p.parent:
        p = p.parent
    print_row("Nearest mount point:", str(p))

    # 1b. Filesystem type via findmnt
    try:
        r = subprocess.run(
            ["findmnt", "-n", "-o", "SOURCE,FSTYPE,OPTIONS", "--target", data_root],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split()
            print_row("Mount source:", parts[0] if len(parts) > 0 else "?")
            print_row("Filesystem type:", parts[1] if len(parts) > 1 else "?")
            if len(parts) > 2:
                print_row("Mount options:", " ".join(parts[2:])[:80])
        else:
            print_row("findmnt:", "(no output or command failed)")
    except Exception as e:
        print_row("findmnt:", f"(error: {e})")

    # 1c. Symlink analysis
    h5_paths = [fp for _, _, fp in file_list]
    symlink_count = 0
    dir_symlinks = set()
    for fp in h5_paths:
        if os.path.islink(fp):
            symlink_count += 1
        # Check parent dirs
        pp = pathlib.Path(fp)
        for parent in pp.parents:
            if parent == pathlib.Path(data_root).parent:
                break
            if os.path.islink(str(parent)):
                dir_symlinks.add(str(parent))

    print_row("H5 files scanned:", str(len(h5_paths)))
    print_row("File symlinks:", f"{symlink_count} / {len(h5_paths)}")
    print_row("Directory symlinks:", f"{len(dir_symlinks)}" + (f" {list(dir_symlinks)[:3]}" if dir_symlinks else ""))

    # Check resolved vs raw paths
    if h5_paths:
        raw = h5_paths[0]
        resolved = str(pathlib.Path(raw).resolve())
        print_row("Example raw path:", raw[-70:])
        if raw != resolved:
            print_row("Example resolved:", resolved[-70:])
            print_row("Paths differ:", "YES - symlinks in chain")
        else:
            print_row("Paths differ:", "No (direct, no symlinks)")

    # 1d. Sequential read speed
    if h5_paths:
        test_file = h5_paths[0]
        fsize = os.path.getsize(test_file)
        read_bytes = min(fsize, 100 * 1024 * 1024)  # up to 100 MB
        chunk = 1024 * 1024  # 1 MB

        t0 = time.perf_counter()
        total_read = 0
        with open(test_file, "rb") as fh:
            while total_read < read_bytes:
                data = fh.read(chunk)
                if not data:
                    break
                total_read += len(data)
        elapsed = time.perf_counter() - t0
        speed_mbps = (total_read / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        print_row("Sequential read test:", f"{fmt_bytes(total_read)} in {fmt_time(elapsed)}")
        print_row("Sequential read speed:", f"{speed_mbps:.1f} MB/s")
        print_row("  (tested file):", os.path.basename(test_file))

    # 1e. File open latency
    if h5_paths:
        n_open = min(10, len(h5_paths))
        open_times = []
        for fp in h5_paths[:n_open]:
            t0 = time.perf_counter()
            fh = h5py.File(fp, "r")
            fh.close()
            open_times.append(time.perf_counter() - t0)
        st = pstats(open_times)
        print_row("HDF5 open+close latency:", f"{fmt_time(st['mean'])} mean ({n_open} files)")


# ============================================================================
# Test 2: Raw HDF5 I/O Speed
# ============================================================================

def test_2_hdf5_io(data_root: str, file_list: List[Tuple[str, str, str]],
                   obs_horizon: int, num_samples: int):
    print_header("Test 2: Raw HDF5 I/O Speed")

    # Find first valid trajectory
    first_h5 = file_list[0][2]
    native_shape = None
    traj_key = None
    traj_len = 0

    with h5py.File(first_h5, "r") as f:
        for k in sorted(f.keys()):
            if not k.startswith("traj_"):
                continue
            try:
                shp = f[k]["obs"]["sensor_data"]["base_camera"]["rgb"].shape
                if len(shp) == 4:
                    native_shape = shp[1:]  # (H, W, C)
                    traj_key = k
                    traj_len = shp[0]
                    break
            except KeyError:
                continue

    if native_shape is None:
        print("  ERROR: Could not find valid trajectory in first H5 file")
        return

    print_row("Test file:", os.path.basename(first_h5))
    print_row("Trajectory:", f"{traj_key} (length={traj_len})")
    print_row("Native image shape:", f"{native_shape}")
    img_bytes = int(np.prod(native_shape))
    print_row("Single image size:", fmt_bytes(img_bytes))

    # 2a. Single image read
    times_single = []
    with h5py.File(first_h5, "r") as f:
        rgb_ds = f[traj_key]["obs"]["sensor_data"]["base_camera"]["rgb"]
        for _ in range(num_samples):
            idx = random.randint(0, traj_len - 1)
            t0 = time.perf_counter()
            _ = rgb_ds[idx]
            times_single.append(time.perf_counter() - t0)
    st = pstats(times_single)
    print_row("Single image read:",
              f"{fmt_time(st['mean'])} mean  (p5={fmt_time(st['p5'])}, p95={fmt_time(st['p95'])})")

    # 2b. Consecutive slice (obs_horizon frames)
    times_slice = []
    with h5py.File(first_h5, "r") as f:
        rgb_ds = f[traj_key]["obs"]["sensor_data"]["base_camera"]["rgb"]
        for _ in range(num_samples):
            idx = random.randint(0, traj_len - obs_horizon)
            t0 = time.perf_counter()
            _ = rgb_ds[idx:idx + obs_horizon]
            times_slice.append(time.perf_counter() - t0)
    st_s = pstats(times_slice)
    ratio = st_s["mean"] / st["mean"] if st["mean"] > 0 else 0
    print_row(f"{obs_horizon}-frame slice read:",
              f"{fmt_time(st_s['mean'])} mean  (p5={fmt_time(st_s['p5'])}, p95={fmt_time(st_s['p95'])})")
    print_row(f"  Slice vs {obs_horizon}x single:", f"{ratio:.2f}x  (1.0 = perfect contiguous benefit)")

    # 2c. Full per-sample I/O (4 images: 2 cameras x obs_horizon)
    # Collect valid (h5_path, traj_key, traj_len) tuples
    all_trajs = []
    for _, _, fp in file_list:
        with h5py.File(fp, "r") as f:
            for k in sorted(f.keys()):
                if not k.startswith("traj_"):
                    continue
                try:
                    shp = f[k]["obs"]["sensor_data"]["base_camera"]["rgb"].shape
                    if len(shp) == 4 and shp[0] >= obs_horizon + 1:
                        all_trajs.append((fp, k, shp[0]))
                except KeyError:
                    continue
        if len(all_trajs) >= 50:
            break

    print_row("Trajectories indexed:", f"{len(all_trajs)} (for random access test)")

    times_sample = []
    h5_cache: Dict[str, h5py.File] = {}
    try:
        for _ in range(num_samples):
            fp, tk, tl = random.choice(all_trajs)
            if fp not in h5_cache:
                h5_cache[fp] = h5py.File(fp, "r")
            f = h5_cache[fp]
            idx = random.randint(obs_horizon - 1, tl - 2)
            s = idx - (obs_horizon - 1)
            e = idx + 1

            t0 = time.perf_counter()
            _ = f[tk]["obs"]["sensor_data"]["base_camera"]["rgb"][s:e]
            _ = f[tk]["obs"]["sensor_data"]["hand_camera"]["rgb"][s:e]
            times_sample.append(time.perf_counter() - t0)
    finally:
        for fh in h5_cache.values():
            fh.close()

    st_samp = pstats(times_sample)
    print_row(f"Per-sample I/O (2 cam x {obs_horizon} frames):",
              f"{fmt_time(st_samp['mean'])} mean  (p5={fmt_time(st_samp['p5'])}, p95={fmt_time(st_samp['p95'])})")

    # 2d. Sequential vs random within one file
    times_seq = []
    times_rand = []
    with h5py.File(first_h5, "r") as f:
        rgb_ds = f[traj_key]["obs"]["sensor_data"]["base_camera"]["rgb"]
        n = min(100, traj_len)
        # Sequential
        t0 = time.perf_counter()
        for i in range(n):
            _ = rgb_ds[i]
        seq_total = time.perf_counter() - t0

        # Random
        indices = [random.randint(0, traj_len - 1) for _ in range(n)]
        t0 = time.perf_counter()
        for i in indices:
            _ = rgb_ds[i]
        rand_total = time.perf_counter() - t0

    print_row(f"Sequential {n} frames:", f"{fmt_time(seq_total)} total = {fmt_time(seq_total / n)}/frame")
    print_row(f"Random {n} frames:", f"{fmt_time(rand_total)} total = {fmt_time(rand_total / n)}/frame")
    if seq_total > 0:
        print_row("  Random access penalty:", f"{rand_total / seq_total:.2f}x")

    # 2e. Open+close overhead vs cached handle
    times_open = []
    for _ in range(min(50, len(all_trajs))):
        fp = random.choice(all_trajs)[0]
        t0 = time.perf_counter()
        fh = h5py.File(fp, "r")
        fh.close()
        times_open.append(time.perf_counter() - t0)
    st_open = pstats(times_open)
    print_row("H5 open+close overhead:", f"{fmt_time(st_open['mean'])} mean")

    return native_shape


# ============================================================================
# Test 3: Image Transform Speed
# ============================================================================

def test_3_transforms(native_shape: Tuple[int, ...], obs_horizon: int, num_samples: int):
    print_header("Test 3: Image Transform Speed (synthetic data, no disk I/O)")

    H, W, C = native_shape
    print_row("Native image size:", f"{H} x {W} x {C}")
    print_row("Target size:", f"{RGB_SIZE[0]} x {RGB_SIZE[1]}")

    resize_aa = T.Resize(RGB_SIZE, antialias=True)
    resize_no_aa = T.Resize(RGB_SIZE, antialias=False)

    # 3a. torch.from_numpy + permute
    img_np = np.random.randint(0, 255, (H, W, C), dtype=np.uint8)
    n_iter = min(1000, num_samples * 10)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = torch.from_numpy(img_np).permute(2, 0, 1)
    elapsed = time.perf_counter() - t0
    print_row(f"from_numpy+permute ({H}x{W}):", f"{fmt_time(elapsed / n_iter)}/image  [{n_iter} iters]")

    # 3b. Resize at different source sizes
    test_sizes = [(H, W), (256, 256), (128, 128), (RGB_SIZE[0], RGB_SIZE[1])]
    n_resize = num_samples
    for (th, tw) in test_sizes:
        t_in = torch.randint(0, 255, (C, th, tw), dtype=torch.uint8)
        t0 = time.perf_counter()
        for _ in range(n_resize):
            _ = resize_aa(t_in)
        elapsed = time.perf_counter() - t0
        label = "(no-op)" if (th, tw) == RGB_SIZE else ""
        print_row(f"Resize {th}x{tw} -> {RGB_SIZE[0]}x{RGB_SIZE[1]} {label}:",
                  f"{fmt_time(elapsed / n_resize)}/image  [{n_resize} iters]")

    # 3c. antialias=True vs antialias=False
    t_in = torch.randint(0, 255, (C, H, W), dtype=torch.uint8)
    t0 = time.perf_counter()
    for _ in range(n_resize):
        _ = resize_aa(t_in)
    time_aa = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_resize):
        _ = resize_no_aa(t_in)
    time_no_aa = time.perf_counter() - t0

    print()
    print_row("Resize antialias=True:", f"{fmt_time(time_aa / n_resize)}/image")
    print_row("Resize antialias=False:", f"{fmt_time(time_no_aa / n_resize)}/image")
    if time_no_aa > 0:
        print_row("  antialias=False speedup:", f"{time_aa / time_no_aa:.2f}x")

    # 3d. Full 4-image pipeline (Python loop, matching __getitem__)
    base_imgs = [np.random.randint(0, 255, (H, W, C), dtype=np.uint8) for _ in range(obs_horizon)]
    hand_imgs = [np.random.randint(0, 255, (H, W, C), dtype=np.uint8) for _ in range(obs_horizon)]

    times_loop = []
    for _ in range(n_resize):
        t0 = time.perf_counter()
        base_t, hand_t = [], []
        for i in range(obs_horizon):
            b = resize_aa(torch.from_numpy(base_imgs[i]).permute(2, 0, 1))
            h = resize_aa(torch.from_numpy(hand_imgs[i]).permute(2, 0, 1))
            base_t.append(b)
            hand_t.append(h)
        base_t = torch.stack(base_t, dim=0)
        hand_t = torch.stack(hand_t, dim=0)
        rgb = torch.cat([base_t, hand_t], dim=1)
        times_loop.append(time.perf_counter() - t0)
    st_loop = pstats(times_loop)
    n_images = obs_horizon * 2
    print()
    print_row(f"Full pipeline ({n_images} images, loop):",
              f"{fmt_time(st_loop['mean'])}/sample  (p5={fmt_time(st_loop['p5'])}, p95={fmt_time(st_loop['p95'])})")

    # 3e. Vectorized resize (batch all images, single resize call)
    times_vec = []
    for _ in range(n_resize):
        t0 = time.perf_counter()
        all_imgs = []
        for i in range(obs_horizon):
            all_imgs.append(torch.from_numpy(base_imgs[i]).permute(2, 0, 1))
            all_imgs.append(torch.from_numpy(hand_imgs[i]).permute(2, 0, 1))
        batch = torch.stack(all_imgs, dim=0)  # (4, 3, H, W)
        batch = resize_aa(batch)               # (4, 3, 224, 224)
        # Reshape back to (obs_horizon, 6, 224, 224)
        batch = batch.reshape(obs_horizon, 2, C, RGB_SIZE[0], RGB_SIZE[1])
        batch = batch.reshape(obs_horizon, 2 * C, RGB_SIZE[0], RGB_SIZE[1])
        times_vec.append(time.perf_counter() - t0)
    st_vec = pstats(times_vec)
    print_row(f"Full pipeline ({n_images} images, vectorized):",
              f"{fmt_time(st_vec['mean'])}/sample  (p5={fmt_time(st_vec['p5'])}, p95={fmt_time(st_vec['p95'])})")
    if st_vec["mean"] > 0:
        print_row("  Vectorized speedup:", f"{st_loop['mean'] / st_vec['mean']:.2f}x")

    # 3f. If images were already 224x224
    base_224 = [np.random.randint(0, 255, (RGB_SIZE[0], RGB_SIZE[1], C), dtype=np.uint8) for _ in range(obs_horizon)]
    hand_224 = [np.random.randint(0, 255, (RGB_SIZE[0], RGB_SIZE[1], C), dtype=np.uint8) for _ in range(obs_horizon)]
    times_noresize = []
    for _ in range(n_resize):
        t0 = time.perf_counter()
        base_t, hand_t = [], []
        for i in range(obs_horizon):
            base_t.append(torch.from_numpy(base_224[i]).permute(2, 0, 1))
            hand_t.append(torch.from_numpy(hand_224[i]).permute(2, 0, 1))
        base_t = torch.stack(base_t, dim=0)
        hand_t = torch.stack(hand_t, dim=0)
        rgb = torch.cat([base_t, hand_t], dim=1)
        times_noresize.append(time.perf_counter() - t0)
    st_nr = pstats(times_noresize)
    print_row("If pre-resized to 224x224 (skip resize):",
              f"{fmt_time(st_nr['mean'])}/sample")
    if st_nr["mean"] > 0:
        print_row("  Speedup vs current:", f"{st_loop['mean'] / st_nr['mean']:.1f}x")

    return st_loop["mean"], st_vec["mean"], st_nr["mean"]


# ============================================================================
# Test 4: Full __getitem__ Profiling
# ============================================================================

def _build_dataset_args(data_root: str, obs_horizon: int, pred_horizon: int) -> "TrainArgs":
    """Build a TrainArgs for dataset construction."""
    from train_lohrbench import Args as TrainArgs
    args = TrainArgs()
    args.data_root = data_root
    args.obs_horizon = obs_horizon
    args.pred_horizon = pred_horizon
    args.act_horizon = 8
    args.num_dataload_workers = 0
    args.preload_images = False
    args.clip_local_files_only = False
    return args


def _get_or_build_dataset(data_root: str, obs_horizon: int, pred_horizon: int,
                          _cache: dict = {}):
    """Build dataset once and cache it for reuse across tests."""
    key = (data_root, obs_horizon, pred_horizon)
    if key not in _cache:
        from train_lohrbench import LoHRbenchDiffusionDataset, compute_clip_text_embeddings
        args = _build_dataset_args(data_root, obs_horizon, pred_horizon)

        # Try building normally; if CLIP fails, monkey-patch with mock embeddings
        try:
            ds = LoHRbenchDiffusionDataset(args=args)
        except Exception as clip_err:
            print(f"  [WARN] Dataset init failed ({clip_err})")
            print(f"  [WARN] Retrying with mock CLIP embeddings for benchmarking...")
            import train_lohrbench as tlm
            _original = tlm.compute_clip_text_embeddings

            def _mock_clip(*a, **kw):
                clip_dim = 512
                embs = {name: torch.randn(clip_dim) for name in TASK_INSTRUCTIONS}
                return embs, clip_dim

            tlm.compute_clip_text_embeddings = _mock_clip
            try:
                ds = LoHRbenchDiffusionDataset(args=args)
            finally:
                tlm.compute_clip_text_embeddings = _original

        _cache[key] = ds
    return _cache[key]


def test_4_getitem_profile(data_root: str, obs_horizon: int, pred_horizon: int,
                           num_samples: int):
    print_header("Test 4: Full __getitem__ Profiling")

    dataset = _get_or_build_dataset(data_root, obs_horizon, pred_horizon)
    total_samples = len(dataset)
    print_row("Total samples:", str(total_samples))
    print_row("Trajectories:", str(len(dataset.trajs)))
    print_row("Profiling:", f"{num_samples} random __getitem__ calls")

    resize = T.Resize(RGB_SIZE, antialias=True)
    indices = random.sample(range(total_samples), min(num_samples, total_samples))

    timings = defaultdict(list)

    for idx in indices:
        traj_idx, t = dataset.slices[idx]
        h5_path, traj_key, L, task_name = dataset.trajs[traj_idx]

        obs_start = t - (obs_horizon - 1)
        obs_end = t + 1

        # Step 1: state/action slicing
        t0 = time.perf_counter()
        state_seq = dataset.states[traj_idx][obs_start:obs_end]
        act_start = t
        act_end = t + pred_horizon
        act_seq = dataset.actions[traj_idx][act_start:min(act_end, L)]
        need = pred_horizon - act_seq.shape[0]
        if need > 0:
            last = act_seq[-1] if act_seq.shape[0] > 0 else dataset.actions[traj_idx][-1]
            act_seq = torch.cat([act_seq, last.unsqueeze(0).repeat(need, 1)], dim=0)
        timings["1_state_action"].append(time.perf_counter() - t0)

        # Step 2: HDF5 image read
        t0 = time.perf_counter()
        f = dataset._get_h5(h5_path)
        g = f[traj_key]
        base_rgb = g["obs"]["sensor_data"]["base_camera"]["rgb"][obs_start:obs_end]
        hand_rgb = g["obs"]["sensor_data"]["hand_camera"]["rgb"][obs_start:obs_end]
        timings["2_hdf5_io"].append(time.perf_counter() - t0)

        # Step 3: to_tensor + permute
        t0 = time.perf_counter()
        base_tensors = []
        hand_tensors = []
        for i in range(base_rgb.shape[0]):
            base_tensors.append(torch.from_numpy(base_rgb[i]).permute(2, 0, 1))
            hand_tensors.append(torch.from_numpy(hand_rgb[i]).permute(2, 0, 1))
        timings["3_to_tensor"].append(time.perf_counter() - t0)

        # Step 4: Resize
        t0 = time.perf_counter()
        base_resized = [resize(bt) for bt in base_tensors]
        hand_resized = [resize(ht) for ht in hand_tensors]
        timings["4_resize"].append(time.perf_counter() - t0)

        # Step 5: stack + cat + normalize
        t0 = time.perf_counter()
        base_s = torch.stack(base_resized, dim=0)
        hand_s = torch.stack(hand_resized, dim=0)
        rgb = torch.cat([base_s, hand_s], dim=1)
        sm = dataset.stats["state_mean"][0]
        ss = dataset.stats["state_std"][0]
        state_seq = (state_seq - sm) / ss
        act_seq = action_to_unit_range(act_seq, dataset.stats)
        lang = dataset.task_embeddings[task_name]
        timings["5_stack_norm"].append(time.perf_counter() - t0)

    # Compute totals
    total_per_sample = []
    for i in range(len(indices)):
        t_sum = sum(timings[k][i] for k in timings)
        total_per_sample.append(t_sum)
    timings["TOTAL"] = total_per_sample

    # Print table
    labels = {
        "1_state_action": "State/action slice",
        "2_hdf5_io": "HDF5 image read",
        "3_to_tensor": "to_tensor + permute",
        "4_resize": "Resize (224x224)",
        "5_stack_norm": "Stack + cat + normalize",
        "TOTAL": "TOTAL __getitem__",
    }

    print()
    print(f"  {'Component':<25s} | {'Mean':>9s} | {'Median':>9s} | {'P5':>9s} | {'P95':>9s} | {'% total':>8s}")
    print(f"  {'-' * 25}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 8}")

    total_mean = pstats(timings["TOTAL"])["mean"]
    for key in ["1_state_action", "2_hdf5_io", "3_to_tensor", "4_resize", "5_stack_norm", "TOTAL"]:
        st = pstats(timings[key])
        pct = (st["mean"] / total_mean * 100) if total_mean > 0 else 0
        if key == "TOTAL":
            print(f"  {'-' * 25}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 8}")
        print(f"  {labels[key]:<25s} | {fmt_time(st['mean']):>9s} | {fmt_time(st['median']):>9s} | "
              f"{fmt_time(st['p5']):>9s} | {fmt_time(st['p95']):>9s} | {pct:>7.1f}%")

    # Identify bottleneck
    component_means = {k: pstats(v)["mean"] for k, v in timings.items() if k != "TOTAL"}
    bottleneck = max(component_means, key=component_means.get)
    print()
    print_row("BOTTLENECK:", f"{labels.get(bottleneck, bottleneck)} "
              f"({component_means[bottleneck] / total_mean * 100:.1f}% of total)")
    print_row("Single-threaded batch est:", f"{fmt_time(total_mean * 256)} (batch_size=256)")

    return total_mean


# ============================================================================
# Test 5: DataLoader Throughput
# ============================================================================

def test_5_dataloader_throughput(data_root: str, obs_horizon: int, pred_horizon: int,
                                batch_size: int, max_workers: int, quick: bool):
    print_header("Test 5: DataLoader Throughput")

    dataset = _get_or_build_dataset(data_root, obs_horizon, pred_horizon)
    total_samples = len(dataset)
    print_row("Total samples:", str(total_samples))
    print_row("Batch size:", str(batch_size))

    warmup_batches = 2
    measure_batches = 5 if quick else 10
    print_row("Per setting:", f"{warmup_batches} warmup + {measure_batches} measured batches")

    worker_counts = [w for w in [0, 1, 2, 4, 8, 12, 16] if w <= max_workers]

    print()
    print(f"  {'Workers':>7s} | {'Batches/s':>10s} | {'Samples/s':>10s} | {'s/batch':>9s} | {'Startup':>9s}")
    print(f"  {'-' * 7}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 9}-+-{'-' * 9}")

    results = {}

    for nw in worker_counts:
        try:
            total_needed = warmup_batches + measure_batches
            sampler = RandomSampler(dataset, replacement=False)
            b_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)
            ib_sampler = IterationBasedBatchSampler(b_sampler, total_needed)

            use_workers = nw > 0
            loader = DataLoader(
                dataset,
                batch_sampler=ib_sampler,
                num_workers=nw,
                worker_init_fn=lambda wid: worker_init_fn(wid, base_seed=42),
                pin_memory=True,
                persistent_workers=use_workers if nw > 0 else False,
                prefetch_factor=2 if use_workers else None,
            )

            # Startup
            t_start = time.perf_counter()
            it = iter(loader)

            # Warmup
            for _ in range(warmup_batches):
                _ = next(it)
            startup_time = time.perf_counter() - t_start

            # Measure
            batch_times = []
            for _ in range(measure_batches):
                t0 = time.perf_counter()
                _ = next(it)
                batch_times.append(time.perf_counter() - t0)

            del loader, it
            gc.collect()

            mean_bt = statistics.mean(batch_times)
            bps = 1.0 / mean_bt if mean_bt > 0 else 0
            sps = bps * batch_size

            mark = " <-- current" if nw == 8 else ""
            print(f"  {nw:>7d} | {bps:>10.2f} | {sps:>10.1f} | {fmt_time(mean_bt):>9s} | "
                  f"{fmt_time(startup_time):>9s}{mark}")
            results[nw] = {"batches_per_sec": bps, "sec_per_batch": mean_bt}

        except Exception as e:
            print(f"  {nw:>7d} | ERROR: {e}")

    # Identify optimal
    if results:
        best_nw = max(results, key=lambda k: results[k]["batches_per_sec"])
        print()
        print_row("Optimal num_workers:", f"{best_nw} ({results[best_nw]['batches_per_sec']:.2f} batches/s)")
        if 8 in results and best_nw != 8:
            ratio = results[best_nw]["batches_per_sec"] / results[8]["batches_per_sec"]
            print_row("vs current (8 workers):", f"{ratio:.2f}x")

    return results


# ============================================================================
# Test 6: Disk Space Check
# ============================================================================

def test_6_disk_space(data_root: str, file_list: List[Tuple[str, str, str]]):
    print_header("Test 6: Disk Space Check")

    # Per-task file sizes
    task_sizes: Dict[str, int] = defaultdict(int)
    task_files: Dict[str, int] = defaultdict(int)
    total_size = 0
    for ttype, task_name, fp in file_list:
        key = f"{ttype}/{task_name}"
        sz = os.path.getsize(fp)
        task_sizes[key] += sz
        task_files[key] += 1
        total_size += sz

    print("  Per-task breakdown:")
    for key in sorted(task_sizes.keys()):
        print(f"    {key:<50s} {fmt_bytes(task_sizes[key]):>10s}  ({task_files[key]} file(s))")
    print(f"    {'TOTAL':<50s} {fmt_bytes(total_size):>10s}  ({len(file_list)} files)")

    # Estimate uncompressed image data size
    # Quick sample: read first trajectory to estimate per-traj image size
    est_img_size = 0
    try:
        fp0 = file_list[0][2]
        with h5py.File(fp0, "r") as f:
            for k in sorted(f.keys()):
                if k.startswith("traj_"):
                    base_shp = f[k]["obs"]["sensor_data"]["base_camera"]["rgb"].shape
                    hand_shp = f[k]["obs"]["sensor_data"]["hand_camera"]["rgb"].shape
                    est_img_size = (np.prod(base_shp) + np.prod(hand_shp))
                    print(f"\n  Image data estimate (first traj): base={base_shp}, hand={hand_shp}")
                    print(f"    Single trajectory images: {fmt_bytes(est_img_size)}")
                    break
    except Exception:
        pass

    # Check local space
    print()
    for path_label, path in [("Script dir (.)", "."),
                              ("/dev/shm (RAM disk)", "/dev/shm"),
                              ("/tmp", "/tmp"),
                              ("Home dir", os.path.expanduser("~"))]:
        try:
            usage = shutil.disk_usage(path)
            feasible = "YES" if usage.free > total_size * 1.1 else "NO"
            print_row(f"{path_label}:", f"Free {fmt_bytes(usage.free)} / {fmt_bytes(usage.total)}  "
                      f"=> Copy feasible: {feasible}")
        except Exception as e:
            print_row(f"{path_label}:", f"(error: {e})")

    # RAM estimate for preloading
    try:
        fp0 = file_list[0][2]
        total_frames = 0
        sample_files = file_list[:min(3, len(file_list))]
        per_frame_bytes = 0
        for _, _, fp in sample_files:
            with h5py.File(fp, "r") as f:
                for k in sorted(f.keys()):
                    if k.startswith("traj_"):
                        shp = f[k]["obs"]["sensor_data"]["base_camera"]["rgb"].shape
                        per_frame_bytes = int(np.prod(shp[1:])) * 2  # 2 cameras
                        total_frames += shp[0]
                        break
        if per_frame_bytes > 0 and total_frames > 0:
            # Extrapolate
            avg_frames = total_frames / len(sample_files)
            # Count total trajectories across all files
            total_trajs = 0
            for _, _, fp in file_list:
                with h5py.File(fp, "r") as f:
                    total_trajs += sum(1 for k in f.keys() if k.startswith("traj_"))
            est_ram = total_trajs * avg_frames * per_frame_bytes
            print()
            print_row("Est. RAM for --preload-images:", f"{fmt_bytes(int(est_ram))} "
                      f"({total_trajs} trajs, ~{int(avg_frames)} frames/traj, "
                      f"{fmt_bytes(per_frame_bytes)}/frame)")
    except Exception:
        pass


# ============================================================================
# Test 7: Optimization Estimates
# ============================================================================

def test_7_optimizations(data_root: str, file_list: List[Tuple[str, str, str]],
                         obs_horizon: int, num_samples: int):
    print_header("Test 7: Optimization Estimates")

    # We need native shape
    native_shape = None
    first_h5 = file_list[0][2]
    with h5py.File(first_h5, "r") as f:
        for k in sorted(f.keys()):
            if k.startswith("traj_"):
                try:
                    shp = f[k]["obs"]["sensor_data"]["base_camera"]["rgb"].shape
                    native_shape = shp[1:]
                    break
                except KeyError:
                    continue
    if native_shape is None:
        print("  ERROR: Could not determine native image shape")
        return

    H, W, C = native_shape
    resize_aa = T.Resize(RGB_SIZE, antialias=True)
    resize_no_aa = T.Resize(RGB_SIZE, antialias=False)
    n_iter = num_samples

    # Helper: time a full per-sample transform pipeline
    def time_pipeline(base_imgs, hand_imgs, resize_fn, skip_resize=False):
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            base_t, hand_t = [], []
            for i in range(obs_horizon):
                b = torch.from_numpy(base_imgs[i]).permute(2, 0, 1)
                h = torch.from_numpy(hand_imgs[i]).permute(2, 0, 1)
                if not skip_resize:
                    b = resize_fn(b)
                    h = resize_fn(h)
                base_t.append(b)
                hand_t.append(h)
            _ = torch.cat([torch.stack(base_t), torch.stack(hand_t)], dim=1)
            times.append(time.perf_counter() - t0)
        return pstats(times)["mean"]

    # Generate synthetic data at native size and 224
    base_native = [np.random.randint(0, 255, (H, W, C), dtype=np.uint8) for _ in range(obs_horizon)]
    hand_native = [np.random.randint(0, 255, (H, W, C), dtype=np.uint8) for _ in range(obs_horizon)]
    base_224 = [np.random.randint(0, 255, (RGB_SIZE[0], RGB_SIZE[1], C), dtype=np.uint8) for _ in range(obs_horizon)]
    hand_224 = [np.random.randint(0, 255, (RGB_SIZE[0], RGB_SIZE[1], C), dtype=np.uint8) for _ in range(obs_horizon)]

    # Current baseline
    t_current = time_pipeline(base_native, hand_native, resize_aa)

    # Opt 1: Skip resize (pre-resized)
    t_preresize = time_pipeline(base_224, hand_224, resize_aa, skip_resize=True)

    # Opt 2: antialias=False
    t_noaa = time_pipeline(base_native, hand_native, resize_no_aa)

    # Opt 3: Vectorized resize
    def time_vectorized():
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            all_imgs = []
            for i in range(obs_horizon):
                all_imgs.append(torch.from_numpy(base_native[i]).permute(2, 0, 1))
                all_imgs.append(torch.from_numpy(hand_native[i]).permute(2, 0, 1))
            batch = torch.stack(all_imgs, dim=0)
            batch = resize_aa(batch)
            batch = batch.reshape(obs_horizon, 2 * C, RGB_SIZE[0], RGB_SIZE[1])
            times.append(time.perf_counter() - t0)
        return pstats(times)["mean"]
    t_vec = time_vectorized()

    # Opt 4: Preloaded (simulate RAM read vs HDF5 read)
    # Just measure the difference in read time: numpy indexing vs HDF5 slice
    preloaded_base = np.random.randint(0, 255, (200, H, W, C), dtype=np.uint8)
    preloaded_hand = np.random.randint(0, 255, (200, H, W, C), dtype=np.uint8)

    times_ram = []
    for _ in range(n_iter):
        idx = random.randint(0, 200 - obs_horizon)
        t0 = time.perf_counter()
        _ = preloaded_base[idx:idx + obs_horizon]
        _ = preloaded_hand[idx:idx + obs_horizon]
        times_ram.append(time.perf_counter() - t0)
    t_ram_read = pstats(times_ram)["mean"]

    # Measure HDF5 read for comparison
    times_h5 = []
    with h5py.File(first_h5, "r") as f:
        for k in sorted(f.keys()):
            if k.startswith("traj_"):
                tlen = f[k]["obs"]["sensor_data"]["base_camera"]["rgb"].shape[0]
                if tlen >= obs_horizon + 5:
                    base_ds = f[k]["obs"]["sensor_data"]["base_camera"]["rgb"]
                    hand_ds = f[k]["obs"]["sensor_data"]["hand_camera"]["rgb"]
                    for _ in range(n_iter):
                        idx = random.randint(0, tlen - obs_horizon)
                        t0 = time.perf_counter()
                        _ = base_ds[idx:idx + obs_horizon]
                        _ = hand_ds[idx:idx + obs_horizon]
                        times_h5.append(time.perf_counter() - t0)
                    break
    t_h5_read = pstats(times_h5)["mean"] if times_h5 else 0
    io_saving = t_h5_read - t_ram_read

    # Opt 5: Resize to 128x128 instead of 224x224 (matching PlainConv design)
    resize_128 = T.Resize((128, 128), antialias=True)
    t_128 = time_pipeline(base_native, hand_native, resize_128)

    # Combined best case: preloaded + pre-resized (no resize, no HDF5)
    t_best = t_preresize  # Already skip resize + synthetic (RAM-like) read

    # Print results
    print()
    print(f"  {'Optimization':<42s} | {'ms/sample':>10s} | {'Speedup':>8s}")
    print(f"  {'-' * 42}-+-{'-' * 10}-+-{'-' * 8}")

    opts = [
        ("Current pipeline (baseline)", t_current, 1.0),
        ("antialias=False (bilinear)", t_noaa, t_current / t_noaa if t_noaa > 0 else 0),
        ("Vectorized resize (batch)", t_vec, t_current / t_vec if t_vec > 0 else 0),
        ("Resize to 128x128 (PlainConv native)", t_128, t_current / t_128 if t_128 > 0 else 0),
        ("Skip resize (pre-resized 224x224)", t_preresize, t_current / t_preresize if t_preresize > 0 else 0),
    ]
    for label, t, spd in opts:
        print(f"  {label:<42s} | {t * 1000:>9.2f} | {spd:>7.2f}x")

    print()
    print_row("HDF5 read per sample:", fmt_time(t_h5_read))
    print_row("RAM read per sample:", fmt_time(t_ram_read))
    print_row("I/O saving with preload:", fmt_time(io_saving))

    print()
    print_row("Best combined estimate:",
              f"pre-resize + preload -> ~{fmt_time(t_preresize)}/sample transform + ~{fmt_time(t_ram_read)} I/O")
    if t_current > 0:
        est_best = t_preresize + t_ram_read
        print_row("  vs current:", f"{t_current / est_best:.1f}x speedup (transform only, excludes I/O gains)")


# ============================================================================
# Summary
# ============================================================================

def print_summary():
    print_header("SUMMARY & RECOMMENDATIONS")
    print("""
  Interpret the results above:

  1. If Test 4 shows Resize > 50% of __getitem__:
     -> RESIZE is the bottleneck. Pre-resize images offline or use 128x128.

  2. If Test 4 shows HDF5 I/O > 50%:
     -> DISK I/O is the bottleneck. Copy data to local SSD or /dev/shm.

  3. If Test 5 shows throughput plateaus at low num_workers:
     -> I/O saturation. Copy data to faster storage.

  4. If Test 5 shows throughput keeps increasing:
     -> CPU is the bottleneck. Add more workers or reduce per-sample CPU work.

  QUICK WINS (no code change):
    - Increase num_workers if Test 5 shows improvement
    - Copy dataset to local SSD or /dev/shm (if Test 6 shows space)
    - Use --preload-images flag (if RAM allows, see Test 6)

  MEDIUM EFFORT (code changes):
    - Switch to antialias=False (see Test 7 speedup)
    - Vectorize resize (batch all images, see Test 7)
    - Resize to 128x128 instead of 224x224 (PlainConv was designed for 128x128,
      uses AdaptiveMaxPool2d(1,1) so output is same; saves ~3x intermediate GPU
      compute AND reduces CPU resize cost)

  HIGH IMPACT (pipeline changes):
    - Pre-resize all images to target size offline (biggest win, see Test 7)
    - Store pre-resized images in a new HDF5 or as memory-mapped numpy arrays
    - Consider NVIDIA DALI for GPU-side decode+resize
""")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="DataLoader bottleneck diagnostic for LoHRbench DP")
    p.add_argument("--data-root", type=str, default="/data1/LoHRbench",
                   help="Path to LoHRbench dataset root")
    p.add_argument("--obs-horizon", type=int, default=2)
    p.add_argument("--pred-horizon", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-samples", type=int, default=100,
                   help="Number of random samples per sub-test")
    p.add_argument("--max-workers", type=int, default=16,
                   help="Maximum num_workers to sweep in Test 5")
    p.add_argument("--quick", action="store_true",
                   help="Run fewer samples/batches for faster turnaround")
    p.add_argument("--skip-dataloader", action="store_true",
                   help="Skip Test 5 (DataLoader throughput) to save time")
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.num_samples = min(args.num_samples, 30)
        args.max_workers = min(args.max_workers, 8)

    print("=" * 72)
    print("  DataLoader Bottleneck Diagnostic for LoHRbench Diffusion Policy")
    print("=" * 72)
    print(f"  Data root:    {args.data_root}")
    print(f"  Date:         {datetime.now().isoformat()}")
    print(f"  obs_horizon:  {args.obs_horizon}")
    print(f"  pred_horizon: {args.pred_horizon}")
    print(f"  batch_size:   {args.batch_size}")
    print(f"  num_samples:  {args.num_samples}")
    print(f"  quick mode:   {args.quick}")
    print()

    # Discover files
    file_list = discover_lohrbench_h5_files(args.data_root)
    if not file_list:
        print(f"ERROR: No HDF5 files found at {args.data_root} ending with '{REQUIRED_H5_SUFFIX}'")
        sys.exit(1)
    print(f"  Found {len(file_list)} H5 files across {len(set(t for t, _, _ in file_list))} task types")

    native_shape = None

    # Test 1
    try:
        test_1_filesystem(args.data_root, file_list)
    except Exception:
        print("  [ERROR] Test 1 failed:")
        traceback.print_exc()

    # Test 2
    try:
        native_shape = test_2_hdf5_io(args.data_root, file_list, args.obs_horizon, args.num_samples)
    except Exception:
        print("  [ERROR] Test 2 failed:")
        traceback.print_exc()

    # Fallback: discover native shape if Test 2 didn't return it
    if native_shape is None:
        try:
            with h5py.File(file_list[0][2], "r") as f:
                for k in sorted(f.keys()):
                    if k.startswith("traj_"):
                        native_shape = f[k]["obs"]["sensor_data"]["base_camera"]["rgb"].shape[1:]
                        break
        except Exception:
            native_shape = (480, 640, 3)  # reasonable fallback
            print(f"  [WARN] Using fallback native shape: {native_shape}")

    # Test 3
    try:
        test_3_transforms(native_shape, args.obs_horizon, args.num_samples)
    except Exception:
        print("  [ERROR] Test 3 failed:")
        traceback.print_exc()

    # Test 4
    try:
        test_4_getitem_profile(args.data_root, args.obs_horizon, args.pred_horizon,
                               args.num_samples)
    except Exception:
        print("  [ERROR] Test 4 failed:")
        traceback.print_exc()

    # Test 5
    if not args.skip_dataloader:
        try:
            test_5_dataloader_throughput(args.data_root, args.obs_horizon, args.pred_horizon,
                                         args.batch_size, args.max_workers, args.quick)
        except Exception:
            print("  [ERROR] Test 5 failed:")
            traceback.print_exc()
    else:
        print_header("Test 5: DataLoader Throughput (SKIPPED)")

    # Test 6
    try:
        test_6_disk_space(args.data_root, file_list)
    except Exception:
        print("  [ERROR] Test 6 failed:")
        traceback.print_exc()

    # Test 7
    try:
        test_7_optimizations(args.data_root, file_list, args.obs_horizon, args.num_samples)
    except Exception:
        print("  [ERROR] Test 7 failed:")
        traceback.print_exc()

    # Summary
    print_summary()

    print("=" * 72)
    print("  Diagnostic complete. Send this output for analysis.")
    print("=" * 72)


if __name__ == "__main__":
    main()
