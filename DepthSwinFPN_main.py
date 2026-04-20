# -*- coding: utf-8 -*-
"""
CNN-only supervised depth estimation (TUM-style RGB-D dataset)
- Supports associations.txt OR rgb.txt+depth.txt pairing
- Outputs metric depth in meters
- Saves best checkpoint by val loss

Run:
  python train_depth_cnn.py

Edit:
  SEQ_DIR = "D:/Projects/Underwater/Datasets/rgbd_dataset_freiburg3_cabinet"
"""

import os
from pathlib import Path
import random
import numpy as np
import cv2
import time
import json
import timm
import re
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# =========================
# Config
# =========================
SEQ_DIR = r"D:\Projects\Underwater\Datasets\redwood\02832"

SAVE_PATH = "best_depth_cnn.pth"

CKPT_ROOT = "depth_checkpoints"
SAVE_EVERY_EPOCH = 1  

# training
EPOCHS = 25
BATCH_SIZE = 8
LR = 2e-4
WEIGHT_DECAY = 1e-4
SEED = 123

# data
RESIZE_HW = (256, 320)   # (H,W). first run small, then scale up
NEAR = 0.20              # meters
FAR  = 3.00              # meters (match your scene / TSDF window)
MAX_DT = 0.05            # timestamp pairing threshold for rgb.txt/depth.txt fallback

# model
BASE_CH = 32
USE_GROUPNORM = True
MAX_DEPTH = FAR          # keep consistent

# performance
NUM_WORKERS = 0
PIN_MEMORY = True

# =========================
# Utilities
# =========================
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def infer_depth_scale_uint16(depth_raw: np.ndarray, near=0.2, far=3.0):
    """
    针对 uint16 深度图，自动推断 scale（像素值/scale = meters）。
    返回: best_scale, stats_dict
    """
    assert depth_raw.dtype == np.uint16
    d = depth_raw.astype(np.float32)
    nz = d[d > 0]
    if nz.size == 0:
        return 1000.0, {"reason": "all_zero"}  # 随便给个默认

    candidates = [1000.0, 5000.0, 1.0]  # mm, TUM, already meters
    best = None
    best_score = -1.0
    best_stats = None

    for s in candidates:
        dm = d / s
        m = np.isfinite(dm) & (dm > 0)
        if near is not None:
            m &= (dm >= near)
        if far is not None:
            m &= (dm <= far)
        valid_ratio = float(m.mean())

        vals = dm[m]
        if vals.size > 0:
            p50 = float(np.percentile(vals, 50))
            p95 = float(np.percentile(vals, 95))
        else:
            p50, p95 = float("inf"), float("inf")

        # “合理性”惩罚：如果有效像素太少，或分位数贴边/发散
        penalty = 0.0
        if valid_ratio < 0.05:
            penalty += 1.0
        if not (near <= p50 <= (far if far is not None else p50)):
            penalty += 0.5
        if far is not None and p95 > far:
            penalty += 0.5

        score = valid_ratio - penalty
        if score > best_score:
            best_score = score
            best = s
            best_stats = {"scale": s, "valid_ratio": valid_ratio, "p50": p50, "p95": p95, "score": score}

    return best, best_stats


# 建议：对每个序列/数据集只推断一次并缓存
_DEPTH_SCALE_CACHE = {}

def depth_to_meters_auto(depth_raw: np.ndarray, near=0.2, far=3.0, cache_key="default"):
    d = depth_raw.astype(np.float32)
    if d.size == 0:
        return d

    if depth_raw.dtype == np.uint16:
        if cache_key not in _DEPTH_SCALE_CACHE:
            scale, st = infer_depth_scale_uint16(depth_raw, near=near, far=far)
            _DEPTH_SCALE_CACHE[cache_key] = scale
            print(f"[depth] inferred scale={scale} (raw/scale=m), stats={st}")
        scale = _DEPTH_SCALE_CACHE[cache_key]
        return d / scale

    # float depth：默认认为已经是 meters
    return d

def _try_float(s):
    try:
        return float(s)
    except Exception:
        return None

def parse_associations(seq_dir: Path):
    """
    associations.txt: ts_rgb rgb/xxx ts_d depth/yyy
    """
    for name in ["associations.txt", "associate.txt"]:
        f = seq_dir / name
        if not f.exists():
            continue
        pairs = []
        with open(f, "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split()
                if len(parts) >= 4:
                    ts_rgb, rgb_rel, ts_d, dep_rel = parts[:4]
                    pairs.append((float(ts_rgb), seq_dir / rgb_rel, float(ts_d), seq_dir / dep_rel))
        if pairs:
            print(f"[io] use {name}, pairs={len(pairs)}")
            return pairs
    return None

def parse_time_file(path: Path):
    """
    rgb.txt / depth.txt: timestamp path
    """
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) >= 2:
                ts = _try_float(parts[0])
                if ts is None:
                    continue
                items.append((ts, parts[1]))
    return items

def pair_by_nearest(seq_dir: Path, max_dt=0.05):
    rgb_txt = seq_dir / "rgb.txt"
    dep_txt = seq_dir / "depth.txt"
    if not rgb_txt.exists() or not dep_txt.exists():
        raise FileNotFoundError("No associations.txt and missing rgb.txt/depth.txt")

    rgb_list = parse_time_file(rgb_txt)
    dep_list = parse_time_file(dep_txt)
    if not rgb_list or not dep_list:
        raise RuntimeError("rgb.txt or depth.txt parsed empty")

    dep_ts = np.array([t for t, _ in dep_list], dtype=np.float64)
    pairs = []
    for t_rgb, rgb_rel in rgb_list:
        j = int(np.argmin(np.abs(dep_ts - t_rgb)))
        if abs(dep_ts[j] - t_rgb) <= max_dt:
            t_d, dep_rel = dep_list[j]
            pairs.append((t_rgb, seq_dir / rgb_rel, t_d, seq_dir / dep_rel))

    if not pairs:
        raise RuntimeError("timestamp nearest pairing failed (max_dt too strict?)")

    print(f"[io] use rgb.txt+depth.txt nearest pairing, pairs={len(pairs)}, max_dt={max_dt}")
    return pairs

def make_gn(c, max_groups=8):
    g = min(max_groups, c)
    while g > 1 and (c % g) != 0:
        g -= 1
    return nn.GroupNorm(g, c)

# =========================
# Dataset
# =========================
class TUMRGBDDataset(Dataset):
    """
    returns:
      rgb  : (3,H,W) float32 [0,1]
      depth: (1,H,W) float32 meters
      mask : (1,H,W) bool valid depth
    """
    def __init__(self, seq_dir, resize_hw=(240,320), near=0.0, far=6.0, max_dt=0.05):
        self.seq_dir = Path(seq_dir)
        self.resize_hw = resize_hw
        self.near = float(near)
        self.far  = float(far)

        pairs = parse_associations(self.seq_dir)
        if pairs is None:
            pairs = pair_by_nearest(self.seq_dir, max_dt=max_dt)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        _, rgb_path, _, dep_path = self.pairs[idx]

        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        dep_raw = cv2.imread(str(dep_path), cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None or dep_raw is None:
            raise FileNotFoundError(f"read failed: {rgb_path} / {dep_path}")

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth_m = depth_to_meters_auto(dep_raw, near=self.near, far=self.far, cache_key=str(self.seq_dir))

        if self.resize_hw is not None:
            H, W = self.resize_hw
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            # depth: NEAREST is safest for metric values
            depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)

        mask = np.isfinite(depth_m) & (depth_m > 0)
        if self.near > 0:
            mask &= (depth_m >= self.near)
        if self.far > 0:
            mask &= (depth_m <= self.far)

        depth_m = np.where(mask, depth_m, 0.0).astype(np.float32)

        rgb_t = torch.from_numpy(rgb).permute(2,0,1).float() / 255.0
        depth_t = torch.from_numpy(depth_m).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        return rgb_t, depth_t, mask_t
    
class RedwoodRGBDDataset(Dataset):
    """
    Redwood RGB-D dataset loader.
    约定目录结构（常见两种）：
      A) seq_dir/color/*.jpg + seq_dir/depth/*.png
      B) seq_dir/rgb/*.jpg   + seq_dir/depth/*.png

    depth: uint16 in millimeters (mm) -> meters = /1000
    returns:
      rgb  : (3,H,W) float32 [0,1]
      depth: (1,H,W) float32 meters
      mask : (1,H,W) bool valid depth (after near/far)
    """
    def __init__(self, seq_dir, resize_hw=(256, 320), near=0.2, far=3.0,
                 rgb_dir_candidates=("color", "rgb"), depth_dir="depth",
                 rgb_exts=(".jpg", ".png", ".jpeg"), depth_ext=".png",
                 depth_scale=1000.0):
        self.seq_dir = Path(seq_dir)
        self.resize_hw = resize_hw
        self.near = float(near)
        self.far  = float(far)
        self.depth_scale = float(depth_scale)

        # find rgb dir
        rgb_dir = None
        for cand in rgb_dir_candidates:
            p = self.seq_dir / cand
            if p.exists() and p.is_dir():
                rgb_dir = p
                break
        if rgb_dir is None:
            raise FileNotFoundError(f"Redwood RGB folder not found under {self.seq_dir} (tried {rgb_dir_candidates})")

        dep_dir = self.seq_dir / depth_dir
        if not dep_dir.exists():
            raise FileNotFoundError(f"Redwood depth folder not found: {dep_dir}")

        def parse_name(p: Path):
            # 例如：0000002-000000033516 或 0000002_000000033516
            stem = p.stem
            parts = re.split(r"[-_]", stem)
            frame_id = int(parts[0]) if parts and parts[0].isdigit() else None
            ts = None
            if len(parts) >= 2 and parts[1].isdigit():
                ts = int(parts[1])
            else:
                nums = re.findall(r"\d+", stem)
                if nums:
                    frame_id = int(nums[0])
                    if len(nums) >= 2:
                        ts = int(nums[-1])
            return frame_id, ts

        # 递归搜文件（关键：防止子目录）
        rgb_files = []
        for ext in rgb_exts:
            rgb_files += sorted(rgb_dir.rglob(f"*{ext}"))
        dep_files = sorted(dep_dir.rglob(f"*{depth_ext}"))

        rgb_by_id = defaultdict(list)
        dep_by_id = defaultdict(list)

        for p in rgb_files:
            fid, ts = parse_name(p)
            if fid is not None:
                rgb_by_id[fid].append((ts if ts is not None else -1, p))

        for p in dep_files:
            fid, ts = parse_name(p)
            if fid is not None:
                dep_by_id[fid].append((ts if ts is not None else -1, p))

        pairs = []
        common_ids = sorted(set(rgb_by_id.keys()) & set(dep_by_id.keys()))
        for fid in common_ids:
            rgbs = sorted(rgb_by_id[fid], key=lambda x: x[0])
            deps = sorted(dep_by_id[fid], key=lambda x: x[0])

            for ts_r, rp in rgbs:
                if ts_r < 0 or all(ts_d < 0 for ts_d, _ in deps):
                    pairs.append((rp, deps[0][1]))
                else:
                    best = min(deps, key=lambda x: abs(x[0] - ts_r))
                    pairs.append((rp, best[1]))

        print(f"[io] Redwood rgb={len(rgb_files)} depth={len(dep_files)} matched_pairs={len(pairs)}")
        if common_ids:
            print(f"[dbg] id range: {common_ids[0]} .. {common_ids[-1]}")


        if not pairs:
            raise RuntimeError(f"Failed to pair RGB and depth in {self.seq_dir}. Check filenames in rgb/depth folders.")

        self.pairs = pairs
        print(f"[io] Redwood pairs={len(self.pairs)}  rgb_dir={rgb_dir.name}  depth_dir={dep_dir.name}  depth_scale=1/{self.depth_scale} m")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_path, dep_path = self.pairs[idx]

        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        dep_raw = cv2.imread(str(dep_path), cv2.IMREAD_UNCHANGED)

        if rgb_bgr is None or dep_raw is None:
            raise FileNotFoundError(f"read failed: {rgb_path} / {dep_path}")

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        # Redwood: uint16 depth in mm
        depth_m = dep_raw.astype(np.float32) / self.depth_scale

        if self.resize_hw is not None:
            H, W = self.resize_hw
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)

        mask = np.isfinite(depth_m) & (depth_m > 0)
        if self.near > 0:
            mask &= (depth_m >= self.near)
        if self.far > 0:
            mask &= (depth_m <= self.far)

        depth_m = np.where(mask, depth_m, 0.0).astype(np.float32)

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        depth_t = torch.from_numpy(depth_m).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        return rgb_t, depth_t, mask_t


# =========================
# Model (Swin-Transformer)
# =========================
def pad_to_multiple(x, m=32):
    B, C, H, W = x.shape
    pad_h = (m - H % m) % m
    pad_w = (m - W % m) % m
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    # pad: (left, right, top, bottom)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (0, pad_w, 0, pad_h)

def unpad(x, pad):
    l, r, t, b = pad
    if r == 0 and b == 0:
        return x
    return x[..., : x.shape[-2] - b, : x.shape[-1] - r]

class DepthSwinFPN(nn.Module):
    """
    Swin-Tiny backbone (features_only) + FPN + depth head
    输出: (B,1,H,W) depth in meters (0..max_depth)
    """
    def __init__(self, max_depth=6.0, pretrained=True, fpn_dim=256,
                 backbone_name="swin_tiny_patch4_window7_224",
                 input_hw=(256, 320)):
        super().__init__()
        self.max_depth = float(max_depth)
        self.input_hw = tuple(input_hw)

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=self.input_hw,
        )
        base = self.backbone.model if hasattr(self.backbone, "model") else self.backbone
        pe = getattr(base, "patch_embed", None)
        if pe is not None:
            # 不同 timm 版本字段不同，尽量兼容
            if hasattr(pe, "strict_img_size"):
                pe.strict_img_size = False
            if hasattr(pe, "img_size"):
                pe.img_size = None

        ch = self.backbone.feature_info.channels()
        self.ch = ch

        # lateral 1x1
        self.l4 = nn.Conv2d(ch[3], fpn_dim, 1)
        self.l3 = nn.Conv2d(ch[2], fpn_dim, 1)
        self.l2 = nn.Conv2d(ch[1], fpn_dim, 1)
        self.l1 = nn.Conv2d(ch[0], fpn_dim, 1)

        # smooth 3x3
        self.s4 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.s3 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.s2 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.s1 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)

        self.head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, 1, 1),
        )

    def forward(self, x):
        x0, pad = pad_to_multiple(x, 32)  # Swin/FPN 多级下采样要求尺寸对齐
        feats = self.backbone(x0)  # 可能是 NCHW，也可能是 NHWC

        # 统一成 NCHW
        fixed = []
        for f, c in zip(feats, self.ch):
            # 如果最后一维等于通道数，说明是 NHWC -> 转 NCHW
            if f.ndim == 4 and f.shape[-1] == c and f.shape[1] != c:
                f = f.permute(0, 3, 1, 2).contiguous()
            fixed.append(f)

        c1, c2, c3, c4 = fixed

        p4 = self.l4(c4)
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.l1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        p4 = self.s4(p4)
        p3 = self.s3(p3)
        p2 = self.s2(p2)
        p1 = self.s1(p1)

        if not hasattr(self, "_dbg"):
            print("[dbg] c1..c4:", c1.shape, c2.shape, c3.shape, c4.shape)
            self._dbg = True

        logits = self.head(p1)  # stride=4
        logits = F.interpolate(logits, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        logits = unpad(logits, pad)

        depth = F.softplus(logits).clamp(0.0, self.max_depth)
        return depth

# =========================
# Loss (masked)
# =========================
def masked_l1(pred, gt, mask):
    diff = (pred - gt).abs()
    return diff[mask].mean() if mask.any() else diff.mean() * 0.0

def grad_l1(pred, gt, mask):
    def dx(x): return x[..., :, 1:] - x[..., :, :-1]
    def dy(x): return x[..., 1:, :] - x[..., :-1, :]
    m = mask.float()
    mx = (m[..., :, 1:] * m[..., :, :-1]).bool()
    my = (m[..., 1:, :] * m[..., :-1, :]).bool()
    loss_x = (dx(pred) - dx(gt)).abs()[mx].mean() if mx.any() else pred.mean()*0.0
    loss_y = (dy(pred) - dy(gt)).abs()[my].mean() if my.any() else pred.mean()*0.0
    return 0.5 * (loss_x + loss_y)

def depth_loss(pred, gt, mask, w_grad=0.3):
    return masked_l1(pred, gt, mask) + w_grad * grad_l1(pred, gt, mask)

# =========================
# Train / Val
# =========================
def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total = 0.0
    n = 0
    for rgb, depth, mask in loader:
        rgb = rgb.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            pred = model(rgb)
            loss = depth_loss(pred, depth, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += float(loss.item())
        n += 1
    return total / max(1, n)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    tot = l1t = grt = mrt = 0.0
    n = 0
    for rgb, depth, mask in loader:
        rgb = rgb.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        pred = model(rgb)
        l1 = masked_l1(pred, depth, mask)
        gr = grad_l1(pred, depth, mask)
        loss = l1 + 0.3 * gr

        tot += float(loss.item())
        l1t += float(l1.item())
        grt += float(gr.item())
        mrt += float(mask.float().mean().item())
        n += 1

    tot /= max(1, n); l1t /= max(1, n); grt /= max(1, n); mrt /= max(1, n)
    print(f"  [val-detail] Total={tot:.5f}  L1={l1t:.5f}  Grad={grt:.5f}  mask_ratio={mrt:.3f}")
    return tot

def main():
    # =========================
    # Checkpoints
    # =========================
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(CKPT_ROOT) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / "best.pth"
    last_path = run_dir / "last.pth"

    # 可选：保存一份本次训练配置，方便复现实验
    cfg = {
        "SEQ_DIR": str(SEQ_DIR),
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "RESIZE_HW": RESIZE_HW,
        "NEAR": NEAR,
        "FAR": FAR,
        "MAX_DT": MAX_DT,
        "BASE_CH": BASE_CH,
        "USE_GROUPNORM": USE_GROUPNORM,
        "MAX_DEPTH": MAX_DEPTH,
        "SEED": SEED,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"[ckpt] run_dir = {run_dir}")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")

    # ds = TUMRGBDDataset(SEQ_DIR, resize_hw=RESIZE_HW, near=NEAR, far=FAR, max_dt=MAX_DT)
    ds = RedwoodRGBDDataset(SEQ_DIR, resize_hw=RESIZE_HW, near=NEAR, far=FAR)


    n = len(ds)
    if n < 10:
        raise RuntimeError(f"too few pairs: {n}")

    # split: by chunks to avoid near-duplicate leakage
    n = len(ds)
    chunk = 10  # 每 10 帧作为一个小段，可改 5/20
    chunks = [list(range(i, min(i+chunk, n))) for i in range(0, n, chunk)]

    rng = np.random.RandomState(SEED)
    rng.shuffle(chunks)
    idx_all = [i for ch in chunks for i in ch]

    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    idx_train = idx_all[:n_tr]
    idx_val   = idx_all[n_tr:n_tr + n_va]
    idx_test  = idx_all[n_tr + n_va:]

    tr_ds = torch.utils.data.Subset(ds, idx_train)
    va_ds = torch.utils.data.Subset(ds, idx_val)
    te_ds = torch.utils.data.Subset(ds, idx_test)

    # --- DataLoaders ---
    pin = (device.type == "cuda") and PIN_MEMORY
    tr_ld = DataLoader(
        tr_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin, drop_last=True
    )
    va_ld = DataLoader(
        va_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin
    )
    te_ld = DataLoader(
        te_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin
    )

    model = DepthSwinFPN(max_depth=MAX_DEPTH, pretrained=True, input_hw=RESIZE_HW).to(device)

    # 分组学习率：微调更稳
    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR * 0.1},
            {"params": head_params,     "lr": LR},
        ],
        weight_decay=WEIGHT_DECAY
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    best = 1e9
    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, tr_ld, optimizer, scaler, device)
        va_loss = validate(model, va_ld, device)

        print(f"[ep {ep:03d}] train={tr_loss:.5f}  val={va_loss:.5f}")

        # 组装 checkpoint
        ckpt = {
            "epoch": ep,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val": best,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "config": cfg,
        }

        # 1) 保存 last（每轮覆盖）
        torch.save(ckpt, str(last_path))

        # 2) 保存带轮次号的 checkpoint
        if (SAVE_EVERY_EPOCH is not None) and (SAVE_EVERY_EPOCH > 0) and (ep % SAVE_EVERY_EPOCH == 0):
            torch.save(ckpt, str(run_dir / f"epoch_{ep:03d}.pth"))

        # 3) 保存 best（当 val 更好时覆盖）
        if va_loss < best:
            best = va_loss
            ckpt["best_val"] = best
            torch.save(ckpt, str(best_path))
            print(f"  saved best: {best_path}  (best_val={best:.5f})")

    print("[done] best val =", best)

    best_ckpt = torch.load(str(best_path), map_location=device)
    model.load_state_dict(best_ckpt["model"], strict=True)

    te_loss = validate(model, te_ld, device)
    print(f"[test@best] loss={te_loss:.5f}")

if __name__ == "__main__":
    main()
