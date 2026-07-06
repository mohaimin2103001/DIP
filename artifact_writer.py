from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _binary_u8(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.uint8) * 255


def _imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Could not write image: {path}")


def overlay_prediction(
    bgr: np.ndarray, pred: np.ndarray, gt: np.ndarray, fov: np.ndarray
) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    overlay = rgb.copy()
    pred = pred.astype(bool) & fov
    gt = gt.astype(bool) & fov

    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt

    overlay[tp] = 0.50 * overlay[tp] + 0.50 * np.array([255, 220, 0], dtype=np.float32)
    overlay[fp] = 0.45 * overlay[fp] + 0.55 * np.array([0, 150, 255], dtype=np.float32)
    overlay[fn] = 0.45 * overlay[fn] + 0.55 * np.array([255, 40, 40], dtype=np.float32)
    overlay[~fov] *= 0.30
    return cv2.cvtColor(np.clip(overlay, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def save_histogram_figure(
    preprocessed: Dict[str, np.ndarray], fov: np.ndarray, out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, key, color in zip(
        axes,
        ("green", "clahe", "enhanced"),
        ("green", "royalblue", "crimson"),
    ):
        values = preprocessed[key][fov].ravel()
        ax.hist(values, bins=256, color=color, alpha=0.85)
        ax.set_title(key.capitalize())
        ax.set_xlim(0, 255)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Pixels")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _show_panel(
    ax: plt.Axes, image: np.ndarray, title: str, cmap: str | None = None
) -> None:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
    else:
        ax.imshow(image, cmap=cmap or "gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_sample_comparison(
    bgr: np.ndarray,
    gt_mask: np.ndarray,
    gt_skeleton: np.ndarray,
    preprocessed: Dict[str, np.ndarray],
    method_artifacts: Dict[str, Dict[str, np.ndarray]],
    method_order: Sequence[str],
    out_path: Path,
) -> None:
    methods = [method for method in method_order if method in method_artifacts]
    columns = 2 + len(methods)
    fig, axes = plt.subplots(3, columns, figsize=(4 * columns, 10.5))

    _show_panel(axes[0, 0], bgr, "RGB image")
    _show_panel(axes[0, 1], _binary_u8(gt_mask), "Ground truth")
    for col, method in enumerate(methods, start=2):
        _show_panel(
            axes[0, col],
            _binary_u8(method_artifacts[method]["mask"]),
            f"{method} mask",
        )

    _show_panel(axes[1, 0], preprocessed["green"], "Green channel")
    _show_panel(axes[1, 1], _binary_u8(gt_skeleton), "GT skeleton")
    for col, method in enumerate(methods, start=2):
        _show_panel(
            axes[1, col],
            _binary_u8(method_artifacts[method]["skeleton"]),
            f"{method} skeleton",
        )

    _show_panel(axes[2, 0], preprocessed["enhanced"], "Enhanced")
    _show_panel(axes[2, 1], preprocessed["blackhat"], "Black-hat")
    for col, method in enumerate(methods, start=2):
        _show_panel(
            axes[2, col],
            method_artifacts[method]["overlay"],
            f"{method} overlay",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_preprocessing_outputs(
    output_root: Path,
    pair: Any,
    preprocessed: Dict[str, np.ndarray],
    fov: np.ndarray,
) -> None:
    base = output_root / "preprocessing" / pair.split / pair.image_id
    for key, image in preprocessed.items():
        _imwrite(base / f"{key}.png", image)
    _imwrite(base / "fov_mask.png", _binary_u8(fov))
    save_histogram_figure(preprocessed, fov, base / "histograms.png")


def save_method_outputs(
    output_root: Path,
    pair: Any,
    method: str,
    raw_mask: np.ndarray,
    morph: Dict[str, np.ndarray],
    skeleton: np.ndarray,
    overlay: np.ndarray,
    save_intermediates: bool,
) -> None:
    stem = pair.image_id + ".png"
    _imwrite(
        output_root / "masks" / pair.split / method / stem,
        _binary_u8(morph["cleaned"]),
    )
    _imwrite(output_root / "skeletons" / pair.split / method / stem, _binary_u8(skeleton))
    _imwrite(output_root / "overlays" / pair.split / method / stem, overlay)

    if save_intermediates:
        base = output_root / "morphology" / pair.split / method / pair.image_id
        _imwrite(base / "raw_threshold.png", _binary_u8(raw_mask))
        for key in ("opening", "closing", "filled", "cleaned"):
            _imwrite(base / f"{key}.png", _binary_u8(morph[key]))
