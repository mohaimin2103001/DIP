from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METHODS = ("global", "otsu", "adaptive")
BRANCHES = ("skeleton", "watershed", "edge")
FEATURE_KEYS = (
    "vessel_density_pct",
    "vessel_area_px",
    "largest_vessel_area_px",
    "smallest_vessel_area_px",
    "vessel_length_px",
    "average_width_px",
    "connected_components",
    "watershed_regions",
)
METRIC_KEYS = ("dice", "iou", "precision", "recall", "f1", "accuracy")


@dataclass(frozen=True)
class ImagePair:
    split: str
    image_id: str
    image_path: Path
    gt_path: Path


def natural_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.stem)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def resolve_dataset_root(dataset: Path) -> Path:
    dataset = dataset.resolve()
    if dataset.is_file() and dataset.suffix.lower() == ".zip":
        target = dataset.with_suffix("")
        if not target.exists():
            print(f"Extracting {dataset.name} to {target} ...")
            shutil.unpack_archive(str(dataset), str(target))
        dataset = target

    if (dataset / "train").exists() and (dataset / "test").exists():
        return dataset

    nested = dataset / "archive"
    if (nested / "train").exists() and (nested / "test").exists():
        return nested

    raise FileNotFoundError(
        f"Could not find FIVES train/test folders under {dataset}."
    )


def discover_pairs(dataset_root: Path, split: str, limit: int | None) -> List[ImagePair]:
    original_dir = dataset_root / split / "Original"
    gt_dir = dataset_root / split / "Ground truth"
    if not original_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f"Missing Original/Ground truth folders for {split}.")

    pairs: List[ImagePair] = []
    for image_path in sorted(original_dir.glob("*.png"), key=natural_key):
        gt_path = gt_dir / image_path.name
        if gt_path.exists():
            pairs.append(
                ImagePair(
                    split=split,
                    image_id=image_path.stem,
                    image_path=image_path,
                    gt_path=gt_path,
                )
            )
    return pairs[:limit] if limit else pairs


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def read_mask(path: Path, size: Tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read ground truth mask: {path}")
    if size is not None and (mask.shape[1], mask.shape[0]) != size:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return mask > 127


def binary_u8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Could not write image: {path}")


def odd_at_least(value: int, minimum: int = 3) -> int:
    value = max(int(value), minimum)
    return value if value % 2 == 1 else value + 1


def largest_component_mask(mask_u8: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    if num_labels <= 1:
        return mask_u8 > 0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = binary_u8(mask)
    inverted = cv2.bitwise_not(mask_u8)
    flood_filled = inverted.copy()
    h, w = mask_u8.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood_filled, flood_mask, (0, 0), 0)
    holes = flood_filled > 0
    return (mask | holes).astype(bool)


def create_fov_mask(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, raw = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    fov = largest_component_mask(raw.astype(np.uint8))
    fov = fill_holes(fov)
    if np.count_nonzero(fov) == 0:
        fov = np.ones(gray.shape, dtype=bool)
    return fov.astype(bool)


def contrast_stretch(
    image: np.ndarray,
    mask: np.ndarray | None,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> np.ndarray:
    values = image[mask] if mask is not None else image.reshape(-1)
    values = values.astype(np.float32)
    if values.size == 0:
        return np.zeros_like(image, dtype=np.uint8)
    lo, hi = np.percentile(values, [lower_pct, upper_pct])
    if hi <= lo:
        out = np.zeros_like(image, dtype=np.uint8)
    else:
        out = np.clip((image.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
        out = out.astype(np.uint8)
    if mask is not None:
        out = out.copy()
        out[~mask] = 0
    return out


def image_quality_metrics(green: np.ndarray, fov: np.ndarray) -> Dict[str, float]:
    values = green[fov].astype(np.float32)
    hist = cv2.calcHist([green], [0], binary_u8(fov), [256], [0, 256]).ravel()
    total = float(hist.sum())
    if total > 0:
        probabilities = hist / total
        nonzero = probabilities[probabilities > 0]
        entropy = float(-(nonzero * np.log2(nonzero)).sum())
        peak = int(np.argmax(hist))
    else:
        entropy = 0.0
        peak = 0
    return {
        "brightness_mean": float(values.mean()) if values.size else 0.0,
        "contrast_std": float(values.std()) if values.size else 0.0,
        "intensity_min": float(values.min()) if values.size else 0.0,
        "intensity_max": float(values.max()) if values.size else 0.0,
        "histogram_peak": float(peak),
        "histogram_entropy": entropy,
    }


def preprocess_image(
    bgr: np.ndarray,
    fov: np.ndarray,
    clahe_clip: float,
    blackhat_kernel_size: int,
    gaussian_kernel_size: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    green = bgr[:, :, 1].copy()
    quality = image_quality_metrics(green, fov)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    clahe_image = clahe.apply(green)
    gaussian_kernel_size = odd_at_least(gaussian_kernel_size)
    gaussian = cv2.GaussianBlur(clahe_image, (gaussian_kernel_size, gaussian_kernel_size), 0)

    blackhat_kernel_size = odd_at_least(blackhat_kernel_size)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (blackhat_kernel_size, blackhat_kernel_size)
    )
    blackhat = cv2.morphologyEx(gaussian, cv2.MORPH_BLACKHAT, kernel)
    enhanced = contrast_stretch(blackhat, fov)

    for image in (green, clahe_image, gaussian, blackhat, enhanced):
        image[~fov] = 0

    return (
        {
            "green": green,
            "clahe": clahe_image,
            "gaussian": gaussian,
            "blackhat": blackhat,
            "enhanced": enhanced,
        },
        quality,
    )


def threshold_image(
    enhanced: np.ndarray,
    fov: np.ndarray,
    method: str,
    global_percentile: float,
    adaptive_block_size: int,
    adaptive_c: float,
) -> Tuple[np.ndarray, float]:
    values = enhanced[fov]
    if values.size == 0:
        return np.zeros_like(enhanced, dtype=bool), 0.0

    if method == "global":
        threshold = float(np.percentile(values, global_percentile))
        mask = enhanced >= threshold
    elif method == "otsu":
        threshold, _ = cv2.threshold(
            values.reshape(-1, 1).astype(np.uint8),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        mask = enhanced >= float(threshold)
    elif method == "adaptive":
        block_size = odd_at_least(adaptive_block_size)
        source = enhanced.copy()
        source[~fov] = int(np.median(values))
        mask_u8 = cv2.adaptiveThreshold(
            source,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            adaptive_c,
        )
        mask = mask_u8 > 0
        threshold = float("nan")
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    mask &= fov
    return mask.astype(bool), float(threshold)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask_u8 = binary_u8(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    if num_labels <= 1:
        return mask.astype(bool)

    areas = stats[:, cv2.CC_STAT_AREA]
    keep = areas >= min_area
    keep[0] = False
    return keep[labels]


def refine_mask(mask: np.ndarray, fov: np.ndarray, min_area: int) -> Dict[str, np.ndarray]:
    kernel_open = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    opened = cv2.morphologyEx(binary_u8(mask), cv2.MORPH_OPEN, kernel_open) > 0
    closed = cv2.morphologyEx(binary_u8(opened), cv2.MORPH_CLOSE, kernel_close) > 0
    filled = fill_holes(closed)
    cleaned = remove_small_components(filled & fov, min_area)
    cleaned &= fov

    return {
        "opening": opened & fov,
        "closing": closed & fov,
        "filled": filled & fov,
        "cleaned": cleaned,
    }


def skeletonize(mask: np.ndarray) -> np.ndarray:
    image = binary_u8(mask)
    skeleton = np.zeros(image.shape, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while cv2.countNonZero(image) > 0:
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(image, opened)
        eroded = cv2.erode(image, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image = eroded

    return skeleton > 0


def watershed_analysis(
    mask: np.ndarray,
    base_gray: np.ndarray,
    distance_ratio: float = 0.35,
) -> Tuple[np.ndarray, int, np.ndarray]:
    mask_u8 = binary_u8(mask)
    if cv2.countNonZero(mask_u8) == 0:
        labels = np.zeros(mask.shape, dtype=np.int32)
        color = np.zeros((*mask.shape, 3), dtype=np.uint8)
        return labels, 0, color

    distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance <= 0:
        sure_fg = mask_u8
    else:
        sure_fg = (distance > distance_ratio * max_distance).astype(np.uint8)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers.astype(np.int32) + 1
    unknown = ((mask_u8 > 0) & (sure_fg == 0)).astype(np.uint8)
    markers[unknown > 0] = 0

    watershed_input = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(watershed_input, markers)
    labels = markers_ws.copy()
    labels[labels < 2] = 0
    region_count = int(len(np.unique(labels[labels > 0])))

    color_index = ((labels * 37) % 255).astype(np.uint8)
    color = cv2.applyColorMap(color_index, cv2.COLORMAP_TURBO)
    color[labels == 0] = 0
    color[markers_ws == -1] = (255, 255, 255)
    return labels, region_count, color


def edge_detection(mask: np.ndarray, fov: np.ndarray) -> Dict[str, np.ndarray]:
    source = binary_u8(mask)
    sobel_x = cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel = np.zeros_like(source)
    if float(magnitude.max()) > 0:
        sobel = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    canny = cv2.Canny(source, 50, 150)

    lap = cv2.Laplacian(source, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)
    laplacian = np.zeros_like(source)
    if float(lap_abs.max()) > 0:
        laplacian = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    for edge in (sobel, canny, laplacian):
        edge[~fov] = 0

    union = ((sobel > 0) | (canny > 0) | (laplacian > 0)) & fov
    return {
        "sobel": sobel,
        "canny": canny,
        "laplacian": laplacian,
        "union": binary_u8(union),
        "union_mask": union,
    }


def compute_metrics(prediction: np.ndarray, ground_truth: np.ndarray, fov: np.ndarray) -> Dict[str, float]:
    pred = prediction.astype(bool) & fov
    gt = ground_truth.astype(bool) & fov

    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt & fov))
    fn = int(np.count_nonzero(~pred & gt & fov))
    tn = int(np.count_nonzero(~pred & ~gt & fov))

    pred_count = tp + fp
    gt_count = tp + fn

    dice_den = 2 * tp + fp + fn
    union = tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn
    total = tp + fp + fn + tn

    dice = (2 * tp / dice_den) if dice_den else 1.0
    iou = (tp / union) if union else 1.0
    precision = (tp / precision_den) if precision_den else (1.0 if gt_count == 0 else 0.0)
    recall = (tp / recall_den) if recall_den else (1.0 if pred_count == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = ((tp + tn) / total) if total else 1.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def extract_features(
    mask: np.ndarray,
    fov: np.ndarray,
    watershed_regions: int,
    skeleton_mask: np.ndarray | None = None,
    width_support_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    roi_pixels = int(np.count_nonzero(fov))
    vessel_mask = mask.astype(bool) & fov
    area = int(np.count_nonzero(vessel_mask))
    density = (100.0 * area / roi_pixels) if roi_pixels else 0.0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_u8(vessel_mask), connectivity=8
    )
    component_areas = [
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, num_labels)
        if int(stats[label, cv2.CC_STAT_AREA]) > 0
    ]
    largest = max(component_areas) if component_areas else 0
    smallest = min(component_areas) if component_areas else 0

    if skeleton_mask is None:
        skeleton_mask = skeletonize(vessel_mask)
    skeleton_mask = skeleton_mask.astype(bool) & fov
    length = int(np.count_nonzero(skeleton_mask))

    support = vessel_mask if width_support_mask is None else (width_support_mask.astype(bool) & fov)
    distance = cv2.distanceTransform(binary_u8(support), cv2.DIST_L2, 5)
    widths = 2.0 * distance[skeleton_mask]
    average_width = float(widths.mean()) if widths.size else 0.0

    return {
        "vessel_density_pct": float(density),
        "vessel_area_px": float(area),
        "largest_vessel_area_px": float(largest),
        "smallest_vessel_area_px": float(smallest),
        "vessel_length_px": float(length),
        "average_width_px": float(average_width),
        "connected_components": float(len(component_areas)),
        "watershed_regions": float(watershed_regions),
    }


def prefixed(prefix: str, values: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def feature_differences(pred: Dict[str, float], gt: Dict[str, float]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for key in FEATURE_KEYS:
        diff = float(pred[key] - gt[key])
        output[f"diff_{key}"] = diff
        output[f"abs_diff_{key}"] = abs(diff)
    return output


def overlay_prediction(bgr: np.ndarray, pred: np.ndarray, gt: np.ndarray, fov: np.ndarray) -> np.ndarray:
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


def save_histogram_figure(preprocessed: Dict[str, np.ndarray], fov: np.ndarray, out_path: Path) -> None:
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


def show_panel(ax: plt.Axes, image: np.ndarray, title: str, cmap: str | None = None) -> None:
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
    gt_edges: Dict[str, np.ndarray],
    preprocessed: Dict[str, np.ndarray],
    method_artifacts: Dict[str, Dict[str, object]],
    out_path: Path,
) -> None:
    methods = [method for method in METHODS if method in method_artifacts]
    columns = 2 + len(methods)
    fig, axes = plt.subplots(4, columns, figsize=(4 * columns, 14))

    show_panel(axes[0, 0], bgr, "RGB image")
    show_panel(axes[0, 1], binary_u8(gt_mask), "Ground truth")
    for col, method in enumerate(methods, start=2):
        show_panel(axes[0, col], binary_u8(method_artifacts[method]["mask"]), f"{method} mask")

    show_panel(axes[1, 0], preprocessed["green"], "Green channel")
    show_panel(axes[1, 1], binary_u8(gt_skeleton), "GT skeleton")
    for col, method in enumerate(methods, start=2):
        show_panel(
            axes[1, col],
            binary_u8(method_artifacts[method]["skeleton"]),
            f"{method} skeleton",
        )

    show_panel(axes[2, 0], preprocessed["enhanced"], "Enhanced")
    show_panel(axes[2, 1], gt_edges["union"], "GT edges")
    for col, method in enumerate(methods, start=2):
        show_panel(axes[2, col], method_artifacts[method]["edges"]["union"], f"{method} edges")

    show_panel(axes[3, 0], preprocessed["blackhat"], "Black-hat")
    axes[3, 1].axis("off")
    axes[3, 1].set_title("Watershed labels", fontsize=9)
    for col, method in enumerate(methods, start=2):
        show_panel(
            axes[3, col],
            method_artifacts[method]["watershed_color"],
            f"{method} watershed",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_preprocessing_outputs(
    output_root: Path,
    pair: ImagePair,
    preprocessed: Dict[str, np.ndarray],
    fov: np.ndarray,
) -> None:
    base = output_root / "preprocessing" / pair.split / pair.image_id
    for key, image in preprocessed.items():
        imwrite(base / f"{key}.png", image)
    imwrite(base / "fov_mask.png", binary_u8(fov))
    save_histogram_figure(preprocessed, fov, base / "histograms.png")


def save_method_outputs(
    output_root: Path,
    pair: ImagePair,
    method: str,
    raw_mask: np.ndarray,
    morph: Dict[str, np.ndarray],
    skeleton: np.ndarray,
    watershed_color: np.ndarray,
    edges: Dict[str, np.ndarray],
    overlay: np.ndarray,
    save_intermediates: bool,
) -> None:
    stem = pair.image_id + ".png"
    imwrite(output_root / "masks" / pair.split / method / stem, binary_u8(morph["cleaned"]))
    imwrite(output_root / "skeletons" / pair.split / method / stem, binary_u8(skeleton))
    imwrite(output_root / "watershed" / pair.split / method / stem, watershed_color)
    imwrite(output_root / "overlays" / pair.split / method / stem, overlay)
    for edge_name in ("sobel", "canny", "laplacian", "union"):
        imwrite(output_root / "edges" / pair.split / method / edge_name / stem, edges[edge_name])

    if save_intermediates:
        base = output_root / "morphology" / pair.split / method / pair.image_id
        imwrite(base / "raw_threshold.png", binary_u8(raw_mask))
        for key in ("opening", "closing", "filled", "cleaned"):
            imwrite(base / f"{key}.png", binary_u8(morph[key]))


def process_pair(
    pair: ImagePair,
    args: argparse.Namespace,
    output_root: Path,
    sample_counts: Dict[str, int],
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], Dict[str, float]]:
    bgr = read_bgr(pair.image_path)
    height, width = bgr.shape[:2]
    gt_mask = read_mask(pair.gt_path, size=(width, height))
    fov = create_fov_mask(bgr)

    preprocessed, quality = preprocess_image(
        bgr=bgr,
        fov=fov,
        clahe_clip=args.clahe_clip,
        blackhat_kernel_size=args.blackhat_kernel,
        gaussian_kernel_size=args.gaussian_kernel,
    )

    gt_skeleton = skeletonize(gt_mask & fov)
    _, gt_watershed_regions, _ = watershed_analysis(gt_mask & fov, preprocessed["enhanced"])
    gt_edges = edge_detection(gt_mask & fov, fov)
    gt_features = extract_features(
        gt_mask,
        fov,
        gt_watershed_regions,
        skeleton_mask=gt_skeleton,
        width_support_mask=gt_mask,
    )

    gt_row: Dict[str, float] = {
        "split": pair.split,
        "image_id": pair.image_id,
        "image_path": str(pair.image_path),
        "gt_path": str(pair.gt_path),
        "fov_pixels": float(np.count_nonzero(fov)),
        **quality,
        **gt_features,
    }

    if args.save_images and args.save_intermediates:
        save_preprocessing_outputs(output_root, pair, preprocessed, fov)

    method_rows: List[Dict[str, float]] = []
    branch_rows: List[Dict[str, float]] = []
    sample_artifacts: Dict[str, Dict[str, object]] = {}

    for method in args.methods:
        method_start = time.perf_counter()
        raw_mask, threshold_value = threshold_image(
            preprocessed["enhanced"],
            fov,
            method,
            args.global_percentile,
            args.adaptive_block_size,
            args.adaptive_c,
        )
        morph = refine_mask(raw_mask, fov, args.min_object_area)
        refined = morph["cleaned"]

        skeleton_start = time.perf_counter()
        pred_skeleton = skeletonize(refined)
        skeleton_time = time.perf_counter() - skeleton_start

        watershed_start = time.perf_counter()
        _, watershed_regions, watershed_color = watershed_analysis(refined, preprocessed["enhanced"])
        watershed_time = time.perf_counter() - watershed_start

        edge_start = time.perf_counter()
        edges = edge_detection(refined, fov)
        edge_time = time.perf_counter() - edge_start

        pred_features = extract_features(
            refined,
            fov,
            watershed_regions,
            skeleton_mask=pred_skeleton,
            width_support_mask=refined,
        )
        segmentation_metrics = compute_metrics(refined, gt_mask, fov)
        processing_time = time.perf_counter() - method_start

        base = {
            "split": pair.split,
            "image_id": pair.image_id,
            "method": method,
            "threshold_value": threshold_value,
            "processing_time_sec": processing_time,
            "skeleton_time_sec": skeleton_time,
            "watershed_time_sec": watershed_time,
            "edge_time_sec": edge_time,
            **quality,
        }
        method_row = {
            **base,
            **segmentation_metrics,
            **prefixed("pred", pred_features),
            **prefixed("gt", gt_features),
            **feature_differences(pred_features, gt_features),
        }
        method_rows.append(method_row)

        branch_targets = {
            "skeleton": (pred_skeleton, gt_skeleton, skeleton_time),
            "watershed": (refined, gt_mask, watershed_time),
            "edge": (edges["union_mask"], gt_edges["union_mask"], edge_time),
        }
        for branch, (branch_pred, branch_gt, branch_time) in branch_targets.items():
            branch_metrics = compute_metrics(branch_pred, branch_gt, fov)
            branch_row = {
                **base,
                "branch": branch,
                "branch_processing_time_sec": branch_time,
                "branch_pred_pixels": float(np.count_nonzero(branch_pred & fov)),
                "branch_gt_pixels": float(np.count_nonzero(branch_gt & fov)),
                **branch_metrics,
                **prefixed("pred", pred_features),
                **prefixed("gt", gt_features),
                **feature_differences(pred_features, gt_features),
            }
            if branch == "edge":
                branch_row["pred_sobel_edge_pixels"] = float(np.count_nonzero(edges["sobel"] > 0))
                branch_row["pred_canny_edge_pixels"] = float(np.count_nonzero(edges["canny"] > 0))
                branch_row["pred_laplacian_edge_pixels"] = float(np.count_nonzero(edges["laplacian"] > 0))
            branch_rows.append(branch_row)

        if args.save_images:
            overlay = overlay_prediction(bgr, refined, gt_mask, fov)
            save_method_outputs(
                output_root,
                pair,
                method,
                raw_mask,
                morph,
                pred_skeleton,
                watershed_color,
                edges,
                overlay,
                args.save_intermediates,
            )

        sample_artifacts[method] = {
            "mask": refined,
            "skeleton": pred_skeleton,
            "watershed_color": watershed_color,
            "edges": edges,
        }

    if args.save_images and sample_counts.get(pair.split, 0) < args.sample_figures:
        sample_counts[pair.split] = sample_counts.get(pair.split, 0) + 1
        save_sample_comparison(
            bgr,
            gt_mask,
            gt_skeleton,
            gt_edges,
            preprocessed,
            sample_artifacts,
            output_root / "figures" / f"{pair.split}_{pair.image_id}_comparison.png",
        )

    return method_rows, branch_rows, gt_row


def flatten_columns(columns: Iterable[Tuple[str, str]]) -> List[str]:
    flattened = []
    for first, second in columns:
        flattened.append(first if not second else f"{first}_{second}")
    return flattened


def save_summary_tables(
    output_root: Path,
    method_df: pd.DataFrame,
    branch_df: pd.DataFrame,
) -> None:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    numeric_method_cols = [
        *METRIC_KEYS,
        "processing_time_sec",
        "pred_vessel_density_pct",
        "pred_vessel_area_px",
        "pred_vessel_length_px",
        "pred_average_width_px",
        "pred_connected_components",
        "pred_watershed_regions",
        "gt_vessel_density_pct",
        "gt_vessel_area_px",
        "gt_vessel_length_px",
        "gt_average_width_px",
        "gt_connected_components",
        "gt_watershed_regions",
        "abs_diff_vessel_density_pct",
        "abs_diff_vessel_area_px",
        "abs_diff_vessel_length_px",
        "abs_diff_average_width_px",
        "abs_diff_connected_components",
        "abs_diff_watershed_regions",
    ]
    available_method_cols = [col for col in numeric_method_cols if col in method_df.columns]
    method_summary = method_df.groupby(["split", "method"])[available_method_cols].agg(["mean", "std"])
    method_summary.columns = flatten_columns(method_summary.columns)
    method_summary = method_summary.reset_index()
    method_summary.to_csv(tables_dir / "method_summary.csv", index=False)

    branch_numeric_cols = [
        *METRIC_KEYS,
        "branch_processing_time_sec",
        "branch_pred_pixels",
        "branch_gt_pixels",
        "abs_diff_vessel_density_pct",
        "abs_diff_vessel_area_px",
        "abs_diff_vessel_length_px",
        "abs_diff_average_width_px",
        "abs_diff_connected_components",
        "abs_diff_watershed_regions",
    ]
    available_branch_cols = [col for col in branch_numeric_cols if col in branch_df.columns]
    branch_summary = branch_df.groupby(["split", "method", "branch"])[available_branch_cols].agg(
        ["mean", "std"]
    )
    branch_summary.columns = flatten_columns(branch_summary.columns)
    branch_summary = branch_summary.reset_index()
    branch_summary.to_csv(tables_dir / "branch_summary.csv", index=False)

    mean_df = method_df.groupby(["split", "method"], as_index=False).mean(numeric_only=True)
    best_rows: List[Dict[str, object]] = []
    for split, split_df in mean_df.groupby("split"):
        for metric in METRIC_KEYS:
            best = split_df.loc[split_df[metric].idxmax()]
            best_rows.append(
                {
                    "split": split,
                    "criterion": f"highest_{metric}",
                    "best_method": best["method"],
                    "value": float(best[metric]),
                }
            )
        fastest = split_df.loc[split_df["processing_time_sec"].idxmin()]
        best_rows.append(
            {
                "split": split,
                "criterion": "fastest_processing_time",
                "best_method": fastest["method"],
                "value": float(fastest["processing_time_sec"]),
            }
        )
    pd.DataFrame(best_rows).to_csv(tables_dir / "best_methods.csv", index=False)


def save_dashboard_figures(output_root: Path, method_df: pd.DataFrame) -> None:
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["dice", "iou", "precision", "recall", "f1", "accuracy"]
    error_cols = [
        "abs_diff_vessel_density_pct",
        "abs_diff_vessel_length_px",
        "abs_diff_average_width_px",
        "abs_diff_connected_components",
        "abs_diff_watershed_regions",
    ]

    for split, split_df in method_df.groupby("split"):
        metric_means = split_df.groupby("method")[metrics].mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        metric_means.plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean score")
        ax.set_title(f"{split}: average segmentation metrics")
        ax.legend(loc="lower right", ncol=2)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{split}_metric_summary.png", dpi=150)
        plt.close(fig)

        available_errors = [col for col in error_cols if col in split_df.columns]
        if available_errors:
            error_means = split_df.groupby("method")[available_errors].mean()
            fig, ax = plt.subplots(figsize=(10, 5))
            error_means.plot(kind="bar", ax=ax)
            ax.set_yscale("log")
            ax.set_ylabel("Mean absolute error, log scale")
            ax.set_title(f"{split}: feature difference from ground truth")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(figures_dir / f"{split}_feature_error_summary.png", dpi=150)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classical DIP pipeline for FIVES retinal vessel segmentation."
    )
    parser.add_argument("--dataset", type=Path, default=Path("archive"), help="Dataset folder or archive.zip.")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=["test"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Threshold methods to run.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit per split.")
    parser.add_argument("--global-percentile", type=float, default=92.0)
    parser.add_argument("--adaptive-block-size", type=int, default=201)
    parser.add_argument("--adaptive-c", type=float, default=-40.0)
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--gaussian-kernel", type=int, default=5)
    parser.add_argument("--blackhat-kernel", type=int, default=15)
    parser.add_argument("--min-object-area", type=int, default=50)
    parser.add_argument("--sample-figures", type=int, default=1, help="Comparison figures per split.")
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Also save preprocessing and morphology stage images.",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_false",
        dest="save_images",
        help="Write CSVs only, without mask/skeleton/edge/overlay PNG outputs.",
    )
    parser.set_defaults(save_images=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset)
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_pairs: List[ImagePair] = []
    for split in args.splits:
        split_pairs = discover_pairs(dataset_root, split, args.limit)
        print(f"{split}: found {len(split_pairs)} image/ground-truth pairs")
        all_pairs.extend(split_pairs)
    if not all_pairs:
        raise RuntimeError("No image pairs found.")

    config = vars(args).copy()
    config["dataset"] = str(dataset_root)
    config["output"] = str(output_root)
    with (output_root / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, default=str)

    all_method_rows: List[Dict[str, float]] = []
    all_branch_rows: List[Dict[str, float]] = []
    all_gt_rows: List[Dict[str, float]] = []
    sample_counts: Dict[str, int] = {}

    for pair in tqdm(all_pairs, desc="Processing FIVES"):
        method_rows, branch_rows, gt_row = process_pair(pair, args, output_root, sample_counts)
        all_method_rows.extend(method_rows)
        all_branch_rows.extend(branch_rows)
        all_gt_rows.append(gt_row)

    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    method_df = pd.DataFrame(all_method_rows)
    branch_df = pd.DataFrame(all_branch_rows)
    gt_df = pd.DataFrame(all_gt_rows)

    method_df.to_csv(tables_dir / "segmentation_metrics_and_features.csv", index=False)
    branch_df.to_csv(tables_dir / "branch_comparison_metrics.csv", index=False)
    gt_df.to_csv(tables_dir / "ground_truth_features.csv", index=False)
    save_summary_tables(output_root, method_df, branch_df)
    save_dashboard_figures(output_root, method_df)

    print(f"Done. Tables: {tables_dir}")
    if args.save_images:
        print(f"Visual outputs: {output_root}")


if __name__ == "__main__":
    main()
