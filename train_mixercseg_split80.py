"""MixerCSeg (Zhao et al., CVPR 2026, arXiv 2603.01361) on the 16 Nov split (DATASETS/split_80_10_10).

    python train_mixercseg_split80.py train --data ~/split_80_10_10 --out ~/mixercseg_runs/split80
    python train_mixercseg_split80.py test  --data ~/split_80_10_10 --ckpt ~/mixercseg_runs/split80/best.pt --save-pred

The model is the authors' official code in this folder (models/, VMamba/), not touched. This script replaces their
main.py / engine.py / test.py for this dataset; the recipe is the official one (main.py defaults), and the deviations
are listed below.

Data       split_80_10_10/{train,val,test}/{images,masks}, pairs read from split_manifest.csv (made by
           DATASETS/split_dataset.py, seed 42). Masks: any format (png/jpg, L/RGB/1-bit), crack = value > 127.
Recipe     as official main.py: input resized to 512x512 (cubic, mask thresholded at 127 after resizing), images
           normalized with mean = std = 0.5, loss 0.87 * BCE + 0.13 * Dice, AdamW lr 5e-4, weight decay 0.01,
           PolyLR (power 1, per epoch) to 1e-6, 50 epochs, batch size 1, no augmentation (--augment adds flips/rot90).
Metrics    as official eval/evaluate.py, on 8-bit probability maps at thresholds 0.00 ... 0.99:
           mIoU (best threshold, per-image mean of crack and background IoU), ODS, OIS, and P/R/F1 pooled over all
           pixels. Also reported for the tiles that contain a crack ("crack_tiles"), because the official per-image
           F1 is 0 for a crack-free tile even when the prediction is empty.
Changed vs official code (on purpose)
  * The best epoch is chosen on val (official main.py chooses it on the test set).
  * Scores use sigmoid probabilities (official test code saves raw logits / max logit, without a sigmoid), so the
    fixed-threshold F1 is at probability 0.5 (official reports it at its first threshold, 0.0 on logit/max).
  * The DEGConv orientation histogram is computed without its Python loop (same numbers, much faster).
  * mmcv / mmseg / fvcore are only stubbed when missing (the model imports them but does not use them here).
Outputs in --out: last.pt (resume: rerun the same command), best.pt, history.csv, recipe.json, val_best.json,
test_metrics.json, test_per_image.csv, FINISHED.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
cv2.setNumThreads(0)

LEVELS = 256                                   # predictions are scored as 8-bit maps, as in the official eval
THRESHOLDS = np.arange(0.0, 1.0, 0.01)         # the official threshold list (100 values)
ABOVE = (np.arange(LEVELS)[None, :] / 255.0) > THRESHOLDS[:, None]   # [threshold, level]: level counts as crack
HALF = int(np.argmin(np.abs(THRESHOLDS - 0.5)))


# ----------------------------------------------------------------------------------------------------------------
# Official model, imported with stubs for packages it imports but does not need here
# ----------------------------------------------------------------------------------------------------------------
def _stub_missing_packages() -> None:
    def missing(name: str) -> bool:
        try:
            __import__(name)
            return False
        except Exception:
            return True

    if missing("mmseg.utils"):
        mmseg, utils = types.ModuleType("mmseg"), types.ModuleType("mmseg.utils")
        utils.get_root_logger = lambda *a, **k: logging.getLogger("mmseg")
        mmseg.utils = utils
        sys.modules.update({"mmseg": mmseg, "mmseg.utils": utils})
    if missing("mmcv.runner"):
        mmcv, runner = types.ModuleType("mmcv"), types.ModuleType("mmcv.runner")
        runner._load_checkpoint = lambda path, **k: torch.load(path, map_location="cpu")
        mmcv.runner = runner
        sys.modules.update({"mmcv": mmcv, "mmcv.runner": runner})
    if missing("fvcore.nn"):
        fvcore, nn_ = types.ModuleType("fvcore"), types.ModuleType("fvcore.nn")
        for name in ("FlopCountAnalysis", "flop_count_str", "flop_count", "parameter_count"):
            setattr(nn_, name, None)
        fvcore.nn = nn_
        sys.modules.update({"fvcore": fvcore, "fvcore.nn": nn_})
    if missing("triton") and not torch.cuda.is_available():
        # VMamba's csm_triton.py and mamba2/*.py use @triton.jit / autotune / heuristics at import time even when
        # Triton is absent; MixerCSeg uses their PyTorch path for CPU tensors anyway. So on CPU-only machines, a
        # permissive stub only while VMamba is imported (torch itself must not see it, hence timm/torchvision
        # first and removal afterwards).
        import timm.models.layers  # noqa: F401

        class Anything:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, *args, **kwargs):   # decorator use returns the function unchanged
                return args[0] if len(args) == 1 and callable(args[0]) and not kwargs else Anything()

            def __getattr__(self, name):
                if name.startswith("__"):
                    raise AttributeError(name)
                return Anything()

        def module_getattr(name: str):
            if name.startswith("__"):
                raise AttributeError(name)
            return Anything()
        triton, language = types.ModuleType("triton"), types.ModuleType("triton.language")
        triton.__version__ = "0.0.0"
        triton.__getattr__ = language.__getattr__ = module_getattr
        triton.language = language
        sys.modules.update({"triton": triton, "triton.language": language})
        try:
            import VMamba.models.csm_triton  # noqa: F401
        finally:
            del sys.modules["triton"], sys.modules["triton.language"]


def _fast_hog_feature(self, x: torch.Tensor) -> torch.Tensor:
    """HoGEdgeGateConv.get_hog_feature without the per-cell bincount loop; returns the same tensor (the official
    code groups each 'cell' with .view on the cropped map, kept exactly as is)."""
    x_mean = x.mean(dim=1, keepdim=True)
    b, _, h, w = x_mean.shape
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    dx = F.conv2d(x_mean.float(), sobel_x, padding=1)
    dy = F.conv2d(x_mean.float(), sobel_y, padding=1)
    direction = torch.abs(torch.atan2(dy, dx))
    cell_h, cell_w = self.cell_size
    h_cells, w_cells = int(h / cell_h), int(w / cell_w)
    dirs = direction[:, :, :h_cells * cell_h, :w_cells * cell_w].reshape(b, h_cells, w_cells, -1)
    bins = torch.clamp((dirs / (torch.pi / self.nbins)).floor().long(), 0, self.nbins - 1)
    counts = F.one_hot(bins, self.nbins).sum(dim=-2) / 64
    start = torch.pi / (2 * self.nbins)
    centres = torch.linspace(start, torch.pi - start, self.nbins, device=x.device)
    return (centres * counts).permute(0, 3, 1, 2)


def build_model(size: int, bce: float, dice: float, device: torch.device):
    _stub_missing_packages()
    from models import build_MixerCSeg
    from models.layers.hog_edge import HoGEdgeGateConv
    HoGEdgeGateConv.get_hog_feature = _fast_hog_feature
    cfg = argparse.Namespace(device=str(device), load_width=size, load_height=size,
                             BCELoss_ratio=bce, DiceLoss_ratio=dice)
    model, criterion = build_MixerCSeg(cfg)
    if device.type != "cuda":   # CPU runs (tests only): SyncBatchNorm and torch.cuda.device() need a GPU
        model = revert_sync_bn(model)
        import VMamba.models.csm_triton as csm
        import VMamba.models.vmamba as vmamba
        vmamba.cross_scan_fn = lambda x, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, \
            force_torch=False: csm.CrossScanF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)
        vmamba.cross_merge_fn = lambda y, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, \
            force_torch=False: csm.CrossMergeF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)
    return model.to(device), criterion


def revert_sync_bn(module: torch.nn.Module) -> torch.nn.Module:
    out = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        out = torch.nn.BatchNorm2d(module.num_features, module.eps, module.momentum, module.affine,
                                   module.track_running_stats)
        if module.affine:
            out.weight.data, out.bias.data = module.weight.data, module.bias.data
        out.running_mean, out.running_var = module.running_mean, module.running_var
        out.num_batches_tracked = module.num_batches_tracked
        out.training = module.training
    for name, child in module.named_children():
        out.add_module(name, revert_sync_bn(child))
    return out


# ----------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------
def load_split(root: Path, split: str, limit: int = 0) -> list[tuple[Path, Path]]:
    manifest = root / "split_manifest.csv"
    if not manifest.is_file():
        raise SystemExit(f"No split_manifest.csv in {root} (--data must be the split_80_10_10 folder)")
    with manifest.open(newline="", encoding="utf-8") as handle:
        pairs = [(root / split / "images" / row["image"], root / split / "masks" / row["mask"])
                 for row in csv.DictReader(handle) if row["split"] == split]
    missing = [str(p) for pair in pairs for p in pair if not p.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} files listed in the manifest are missing, e.g. {missing[0]}")
    if limit:
        pairs = pairs[::max(1, len(pairs) // limit)][:limit]   # spread over the (shuffled) split
    return pairs


class CrackSet(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], size: int, augment: bool = False):
        self.pairs, self.size, self.augment = pairs, size, augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Cannot read {image_path} or {mask_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        mask = (mask > 127).astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                image, mask = image[::-1], mask[::-1]
            k = random.randrange(4)
            image, mask = np.rot90(image, k), np.rot90(mask, k)
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)
        return image, torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0), index


def make_loader(pairs, size, batch_size, workers, shuffle=False, augment=False, seed=0) -> DataLoader:
    # DataLoader seeds python's random in each worker from this generator, so --augment is reproducible
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(CrackSet(pairs, size, augment), batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False, generator=generator)


# ----------------------------------------------------------------------------------------------------------------
# Metrics (official definitions, from per-image histograms of 8-bit probabilities)
# ----------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_histograms(model, loader, device, save_dir: Path | None = None, pairs=None):
    """Per image: counts of each probability level (0-255) on crack and on background pixels."""
    model.eval()
    hist = np.zeros((len(loader.dataset), 2, LEVELS), np.int64)
    for images, masks, index in loader:
        logits = model(images.to(device, non_blocking=True))
        level = torch.round(torch.sigmoid(logits.float()) * 255).long()       # B,1,H,W
        label = masks.to(device).long()
        b = level.shape[0]
        offset = torch.arange(b, device=device).view(b, 1, 1, 1) * (2 * LEVELS)
        counts = torch.bincount((level + LEVELS * label + offset).flatten(), minlength=b * 2 * LEVELS)
        hist[index.numpy()] = counts.view(b, 2, LEVELS)[:, [1, 0]].cpu().numpy()   # [crack, background]
        if save_dir is not None:
            for row, i in enumerate(index.tolist()):
                stem = pairs[i][0].stem
                cv2.imwrite(str(save_dir / f"{stem}_pre.png"), level[row, 0].byte().cpu().numpy())
                cv2.imwrite(str(save_dir / f"{stem}_lab.png"), (label[row, 0] * 255).byte().cpu().numpy())
    return hist


def counts_at_thresholds(hist: np.ndarray):
    """tp, fp, fn, tn per image and threshold: arrays [images, thresholds]."""
    crack, background = hist[:, 0], hist[:, 1]
    tp, fp = crack @ ABOVE.T, background @ ABOVE.T
    return tp, fp, crack.sum(1, keepdims=True) - tp, background.sum(1, keepdims=True) - fp


def image_f1(tp, fp, fn):
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp == 0, 1.0, tp / np.maximum(tp + fp, 1))
        r = np.where(tp + fn == 0, 0.0, tp / np.maximum(tp + fn, 1))
        return np.where(p + r == 0, 0.0, 2 * p * r / np.where(p + r == 0, 1, p + r))


def metrics(hist: np.ndarray, threshold_index: int | None = None) -> dict:
    """Official mIoU / ODS / OIS (best threshold on this set) and pooled P/R/F1 at 0.5, at the best pooled threshold
    and (if given) at threshold_index, e.g. the one chosen on val."""
    if len(hist) == 0:
        return {}
    tp, fp, fn, tn = counts_at_thresholds(hist)
    f1 = image_f1(tp, fp, fn)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou_crack = tp / np.maximum(tp + fp + fn, 1)
        iou_back = tn / np.maximum(tn + fp + fn, 1)
    miou = np.where(tp + fp + fn == 0, 0.0, (iou_crack + iou_back) / 2)   # official: 0 when no crack predicted or present
    TP, FP, FN = tp.sum(0), fp.sum(0), fn.sum(0)
    P = np.where(TP + FP == 0, 1.0, TP / np.maximum(TP + FP, 1))
    R = TP / np.maximum(TP + FN, 1)
    F = np.where(P + R == 0, 0.0, 2 * P * R / np.maximum(P + R, 1e-12))
    best = int(np.argmax(F))
    out = {"images": int(len(hist)),
           "mIoU": float(miou.mean(0).max()), "mIoU_threshold": float(THRESHOLDS[np.argmax(miou.mean(0))]),
           "ODS": float(f1.mean(0).max()), "ODS_threshold": float(THRESHOLDS[np.argmax(f1.mean(0))]),
           "OIS": float(f1.max(1).mean()),
           "P@0.5": float(P[HALF]), "R@0.5": float(R[HALF]), "F1@0.5": float(F[HALF]),
           "crack_IoU@0.5": float(TP[HALF] / max(TP[HALF] + FP[HALF] + FN[HALF], 1)),
           "F1_best": float(F[best]), "F1_best_threshold": float(THRESHOLDS[best])}
    if threshold_index is not None:
        t = threshold_index
        out.update({"threshold_from_val": float(THRESHOLDS[t]), "P@val_thr": float(P[t]), "R@val_thr": float(R[t]),
                    "F1@val_thr": float(F[t]), "crack_IoU@val_thr": float(TP[t] / max(TP[t] + FP[t] + FN[t], 1))})
    return out


def report(hist: np.ndarray, threshold_index: int | None = None) -> dict:
    has_crack = hist[:, 0].sum(1) > 0
    return {"all_tiles": metrics(hist, threshold_index), "crack_tiles": metrics(hist[has_crack], threshold_index)}


# ----------------------------------------------------------------------------------------------------------------
# Train / test
# ----------------------------------------------------------------------------------------------------------------
def poly_lr(epoch: int, args) -> float:
    """mmengine PolyLR (power 1, by epoch) as in official main.py."""
    return args.min_lr + (args.lr - args.min_lr) * (1 - min(epoch, args.epochs) / args.epochs)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args) -> None:
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "FINISHED").exists():
        print(f"{out} is finished already (delete FINISHED to rerun)")
        return
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    seed_everything(args.seed)
    if args.size % 512:
        raise SystemExit("--size must be a multiple of 512 (the encoder's DEGConv needs >= 8x8 px per view at stage 4)")
    data = Path(args.data).expanduser()
    pairs = {s: load_split(data, s, getattr(args, f"limit_{s}")) for s in ("train", "val", "test")}
    print({s: len(p) for s, p in pairs.items()}, "| device", device, flush=True)

    model, criterion = build_model(args.size, args.bce, args.dice, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"MixerCSeg params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)

    start, best, history = 0, {"epoch": -1, "mIoU": -1.0}, []
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best, history = state["epoch"] + 1, state["best"], state["history"]
        torch.set_rng_state(state["rng"]["torch"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
        print(f"resumed after epoch {state['epoch']}", flush=True)
    (out / "recipe.json").write_text(json.dumps({**vars(args), "paper": "arXiv 2603.01361", "train_images":
                                                 len(pairs["train"])}, indent=2, default=str))

    val_loader = make_loader(pairs["val"], args.size, args.eval_batch_size, args.workers)
    for epoch in range(start, args.epochs):
        for group in optimizer.param_groups:
            group["lr"] = poly_lr(epoch, args)
        loader = make_loader(pairs["train"], args.size, args.batch_size, args.workers, shuffle=True,
                             augment=args.augment, seed=args.seed * 1000 + epoch)
        model.train()
        t0, total, steps = time.time(), 0.0, 0
        for step, (images, masks, _) in enumerate(loader):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            loss = criterion(model(images), masks)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total, steps = total + loss.item(), steps + 1
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item():.4f} "
                      f"({(time.time() - t0) / (step + 1):.3f} s/step)", flush=True)
        train_time = time.time() - t0
        val = report(predict_histograms(model, val_loader, device))
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": total / max(steps, 1),
               **{f"val_{k}": v for k, v in val["all_tiles"].items() if k != "images"},
               **{f"val_crack_tiles_{k}": val["crack_tiles"].get(k) for k in ("mIoU", "ODS", "OIS")},
               "train_min": train_time / 60, "epoch_min": (time.time() - t0) / 60}
        history.append(row)
        print(f"epoch {epoch}: loss {row['train_loss']:.4f} | val mIoU {row['val_mIoU']:.4f} ODS {row['val_ODS']:.4f} "
              f"OIS {row['val_OIS']:.4f} F1@0.5 {row['val_F1@0.5']:.4f} | {row['epoch_min']:.1f} min", flush=True)
        if row["val_mIoU"] > best["mIoU"]:   # official criterion (mIoU), but on val
            best = {"epoch": epoch, "mIoU": row["val_mIoU"], "val": val}
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args), "val": val}, out / "best.pt")
            (out / "val_best.json").write_text(json.dumps(best, indent=2))
        with (out / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best": best,
                    "history": history, "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                                                "python": random.getstate()}}, out / "last.pt")

    print(f"best epoch {best['epoch']} (val mIoU {best['mIoU']:.4f}); scoring test", flush=True)
    state = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    evaluate(model, pairs["test"], data, out, args, device, state["val"])
    (out / "FINISHED").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def evaluate(model, test_pairs, data: Path, out: Path, args, device, val: dict | None) -> dict:
    """Test metrics; the pooled F1 is also given at the threshold that was best on val."""
    save_dir = None
    if args.save_pred:
        save_dir = out / "test_predictions"
        save_dir.mkdir(parents=True, exist_ok=True)
    loader = make_loader(test_pairs, args.size, args.eval_batch_size, args.workers)
    hist = predict_histograms(model, loader, device, save_dir, test_pairs)
    t_val = None
    if val and val["all_tiles"]:
        t_val = int(np.argmin(np.abs(THRESHOLDS - val["all_tiles"]["F1_best_threshold"])))
    result = {"data": str(data), "checkpoint_val": val, "test": report(hist, t_val)}
    (out / "test_metrics.json").write_text(json.dumps(result, indent=2))
    tp, fp, fn, _ = counts_at_thresholds(hist)
    f1 = image_f1(tp, fp, fn)
    t = HALF if t_val is None else t_val
    with (out / "test_per_image.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "crack_pixels", "threshold", "tp", "fp", "fn", "F1", "crack_IoU", "best_F1"])
        for i, (image, _) in enumerate(test_pairs):
            writer.writerow([image.name, int(hist[i, 0].sum()), float(THRESHOLDS[t]), int(tp[i, t]), int(fp[i, t]),
                             int(fn[i, t]), round(float(f1[i, t]), 4),
                             round(float(tp[i, t] / max(tp[i, t] + fp[i, t] + fn[i, t], 1)), 4),
                             round(float(f1[i].max()), 4)])
    a, c = result["test"]["all_tiles"], result["test"]["crack_tiles"]
    print(f"TEST all tiles ({a['images']}): mIoU {a['mIoU']:.4f} ODS {a['ODS']:.4f} OIS {a['OIS']:.4f} "
          f"F1@0.5 {a['F1@0.5']:.4f}" + (f" F1@val_thr {a['F1@val_thr']:.4f}" if t_val is not None else ""))
    if c:
        print(f"TEST crack tiles ({c['images']}): mIoU {c['mIoU']:.4f} ODS {c['ODS']:.4f} OIS {c['OIS']:.4f}")
    print(f"written to {out}", flush=True)
    return result


def test(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    data, ckpt = Path(args.data).expanduser(), Path(args.ckpt).expanduser()
    out = Path(args.out).expanduser() if args.out else ckpt.parent / f"test_{ckpt.stem}"
    out.mkdir(parents=True, exist_ok=True)
    model, _ = build_model(args.size, 0.87, 0.13, device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])   # our best.pt/last.pt and the authors' released checkpoint_best.pth
    evaluate(model, load_split(data, args.split, args.limit_test), data, out, args, device, state.get("val"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", required=True, help="split_80_10_10 folder (with split_manifest.csv)")
    common.add_argument("--size", type=int, default=512)
    common.add_argument("--workers", type=int, default=8)
    common.add_argument("--eval-batch-size", type=int, default=8)
    common.add_argument("--device", default="cuda")
    common.add_argument("--save-pred", action="store_true", help="write test <name>_pre.png / _lab.png")
    common.add_argument("--limit-test", type=int, default=0, help="quick runs: use only this many test images")

    t = sub.add_parser("train", parents=[common])
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch-size", type=int, default=1)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--min-lr", type=float, default=1e-6)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--bce", type=float, default=0.87, help="BCE weight (official main.py)")
    t.add_argument("--dice", type=float, default=0.13, help="Dice weight (official main.py)")
    t.add_argument("--augment", action="store_true", help="random flips and 90-degree rotations (official: none)")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--log-every", type=int, default=500)
    t.add_argument("--limit-train", type=int, default=0)
    t.add_argument("--limit-val", type=int, default=0)

    e = sub.add_parser("test", parents=[common])
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", default="", help="default: <ckpt folder>/test_<ckpt name>")
    e.add_argument("--split", default="test", choices=("train", "val", "test"))

    args = parser.parse_args()
    train(args) if args.command == "train" else test(args)


if __name__ == "__main__":
    main()
