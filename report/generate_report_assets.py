from __future__ import annotations

import sys
from pathlib import Path
from textwrap import fill

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import retinal_segmentation_pipeline as dip  # noqa: E402
from artifact_writer import overlay_prediction  # noqa: E402


ASSET_DIR = Path(__file__).resolve().parent / "assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

METHOD_COLORS = {
    "global": "#2563eb",
    "otsu": "#f97316",
    "adaptive": "#16a34a",
}

STYLES = {
    "dataset": ("#f6fff2", "#3f7f2c"),
    "prep": ("#eef6ff", "#245985"),
    "threshold": ("#fff6e8", "#c26a00"),
    "skeleton": ("#f3edff", "#6541a5"),
    "eval": ("#fffbea", "#a36a00"),
    "output": ("#ecfbf6", "#1f7a5a"),
    "note": ("#f8fafc", "#64748b"),
}


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def as_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def arrow(ax, start, end, color="#334155", rad=0.0, lw=0.9, zorder=2.2, alpha=1.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=lw,
            color=color,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            zorder=zorder,
        )
    )


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    style,
    body=None,
    title_size=10.5,
    body_size=8.2,
    title_color="#0f172a",
):
    fc, ec = STYLES[style]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.12",
        linewidth=0.75,
        edgecolor=ec,
        facecolor=fc,
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 1.05,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        weight="bold",
        color=title_color,
        zorder=4,
    )
    if body:
        ax.text(
            x + 0.35,
            y + h - 2.95,
            body,
            ha="left",
            va="top",
            fontsize=body_size,
            color="#1e293b",
            linespacing=1.04,
            zorder=4,
        )
    return patch


def put_image(ax, image, x, y, w, h, label=None, border="#cbd5e1"):
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor="white", edgecolor=border, linewidth=0.35, zorder=2)
    )
    ax.imshow(as_rgb(image), extent=(x, x + w, y, y + h), aspect="auto", zorder=3)
    if label:
        ax.text(
            x + w / 2,
            y - 0.18,
            label,
            ha="center",
            va="top",
            fontsize=7.2,
            color="#0f172a",
            zorder=4,
        )


def put_text(ax, x, y, text, width=25, size=8.8, weight="normal", color="#0f172a"):
    wrapped = "\n".join(fill(line, width=width) for line in text.split("\n"))
    ax.text(
        x,
        y,
        wrapped,
        ha="left",
        va="top",
        fontsize=size,
        weight=weight,
        color=color,
        linespacing=1.08,
        zorder=4,
    )


def mini_hist(ax, images, labels, colors, x, y, w, h):
    rounded_box(ax, x, y, w, h, "2.7 Histogram\nAnalysis", "prep", title_size=9.0)
    for idx, (image, label, color) in enumerate(zip(images, labels, colors)):
        values = image.reshape(-1)
        hist, bins = np.histogram(values, bins=70, range=(0, 255))
        hist = hist.astype(float)
        if hist.max() > 0:
            hist /= hist.max()
        x0 = x + 0.7
        x1 = x + w - 0.45
        y0 = y + 0.9 + idx * 1.65
        y1 = y0 + 1.1
        xs = np.linspace(x0, x1, len(hist))
        ys = y0 + hist * (y1 - y0)
        ax.fill_between(xs, y0, ys, color=color, alpha=0.8, zorder=3)
        ax.text(x + 0.35, y0 + 0.5, label, ha="left", va="center", fontsize=7.2, zorder=4)


def branch_letter(method: str) -> str:
    return {"global": "A", "otsu": "B", "adaptive": "C"}[method]


def method_panel(ax, method, x, y, w, h, raw, refined):
    color = METHOD_COLORS[method]
    letter = branch_letter(method)
    rounded_box(
        ax,
        x,
        y,
        w,
        h,
        f"3.{letter} {method.upper()}\nThresholding",
        "threshold",
        body={
            "global": "Percentile threshold on enhanced vessel image.",
            "otsu": "Automatic threshold from intensity distribution.",
            "adaptive": "Local Gaussian threshold for uneven illumination.",
        }[method],
        title_size=10.4,
        body_size=8.0,
        title_color=color,
    )
    put_image(ax, raw, x + 0.75, y + 0.8, w * 0.42, h * 0.48, "raw mask", color)
    put_image(ax, refined, x + w * 0.55, y + 0.8, w * 0.42, h * 0.48, "clean mask", color)


def morph_panel(ax, method, x, y, w, h):
    color = METHOD_COLORS[method]
    letter = branch_letter(method)
    rounded_box(
        ax,
        x,
        y,
        w,
        h,
        f"4.{letter} Morphological Refinement",
        "threshold",
        title_size=10.4,
        title_color=color,
    )
    steps = ["Opening", "Closing", "Fill holes", "Remove small\ncomponents", "Final mask"]
    step_w = (w - 1.1) / len(steps)
    for idx, step in enumerate(steps):
        sx = x + 0.45 + idx * step_w
        ax.add_patch(
            FancyBboxPatch(
                (sx, y + 2.15),
                step_w - 0.18,
                2.25,
                boxstyle="round,pad=0.012,rounding_size=0.06",
                linewidth=0.45,
                edgecolor=color,
                facecolor="#ffffff",
                zorder=2,
            )
        )
        ax.text(
            sx + (step_w - 0.18) / 2,
            y + 3.28,
            step,
            ha="center",
            va="center",
            fontsize=7.2,
            color="#0f172a",
            zorder=4,
        )
        if idx < len(steps) - 1:
            arrow(
                ax,
                (sx + step_w - 0.18, y + 3.28),
                (sx + step_w + 0.05, y + 3.28),
                color=color,
                lw=0.55,
                zorder=3.5,
            )
    ax.text(
        x + w / 2,
        y + 0.9,
        "Same refinement is applied after every threshold branch.",
        ha="center",
        va="center",
        fontsize=7.0,
        color="#334155",
        zorder=4,
    )


def skeleton_panel(ax, method, x, y, w, h, skeleton):
    color = METHOD_COLORS[method]
    letter = branch_letter(method)
    rounded_box(
        ax,
        x,
        y,
        w,
        h,
        f"5.{letter} Skeleton",
        "skeleton",
        title_size=9.4,
        title_color=color,
    )
    put_image(ax, skeleton, x + 0.75, y + 0.95, w * 0.38, h * 0.56, "skeleton", color)
    put_text(
        ax,
        x + w * 0.53,
        y + h - 2.75,
        "Features:\n- density, area\n- vessel length\n- average width\n- components",
        width=21,
        size=7.8,
    )


def prepare_artifacts():
    image_path = PROJECT_ROOT / "archive" / "test" / "Original" / "1_A.png"
    gt_path = PROJECT_ROOT / "archive" / "test" / "Ground truth" / "1_A.png"

    bgr = read_bgr(image_path)
    gt_mask = dip.read_mask(gt_path, size=(bgr.shape[1], bgr.shape[0]))
    fov = dip.create_fov_mask(bgr)
    pre, quality = dip.preprocess_image(
        bgr.copy(),
        fov,
        clahe_clip=2.0,
        blackhat_kernel_size=15,
        gaussian_kernel_size=5,
    )

    methods = {}
    for method in dip.METHODS:
        raw, _ = dip.threshold_image(pre["enhanced"], fov, method, 92.0, 201, -40.0)
        morph = dip.refine_mask(raw, fov, 50)
        refined = morph["cleaned"]
        skeleton = dip.skeletonize(refined)
        overlay = overlay_prediction(bgr, refined, gt_mask, fov)
        methods[method] = {
            "raw": raw,
            "refined": refined,
            "skeleton": skeleton,
            "overlay": overlay,
        }

    return {
        "bgr": bgr,
        "rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        "gt": gt_mask,
        "fov": fov,
        "pre": pre,
        "quality": quality,
        "methods": methods,
    }


def save_full_pipeline():
    data = prepare_artifacts()
    fig, ax = plt.subplots(figsize=(16, 9), dpi=360)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0.35, 99.65)
    ax.set_ylim(21.7, 69.65)
    ax.axis("off")

    ax.add_patch(Rectangle((0, 66.2), 100, 3.4, facecolor="#0f2a55", edgecolor="none", zorder=0))
    ax.text(
        50,
        68.45,
        "FIVES RETINAL VESSEL SEGMENTATION PROJECT",
        ha="center",
        va="center",
        fontsize=9.4,
        weight="bold",
        color="white",
    )
    ax.text(
        50,
        67.25,
        "CLASSICAL DIP PIPELINE - PREPROCESSING, THRESHOLDING, MORPHOLOGY, AND SKELETON FEATURES",
        ha="center",
        va="center",
        fontsize=6.1,
        color="#dbeafe",
    )

    # Top row: dataset and preprocessing.
    top_y, top_h = 55.7, 9.6
    xs = [1, 12.6, 24.2, 35.8, 47.4, 59.0, 70.6, 82.2]
    w = 10.2
    rounded_box(
        ax,
        xs[0],
        top_y,
        w,
        top_h,
        "1. Dataset (FIVES)",
        "dataset",
        body="200 test images\nfundus input\nmanual mask\nFOV ROI",
        title_size=8.6,
        body_size=7.0,
    )
    put_image(ax, data["rgb"], xs[0] + 0.55, top_y + 0.75, 4.35, 3.65, "input", "#3f7f2c")
    put_image(ax, data["gt"], xs[0] + 5.25, top_y + 0.75, 4.35, 3.65, "GT mask", "#3f7f2c")

    prep_items = [
        ("2.1 FOV\nMask", data["fov"], "retina ROI\nlargest component\nfill holes"),
        ("2.2 Green\nChannel", data["pre"]["green"], "best vessel\ncontrast"),
        ("2.3\nCLAHE", data["pre"]["clahe"], "local contrast\nenhancement"),
        ("2.4\nGaussian", data["pre"]["gaussian"], "noise smoothing\nkernel = 5"),
        ("2.5\nBlack-hat", data["pre"]["blackhat"], "dark vessel\nemphasis"),
        ("2.6 Contrast\nStretch", data["pre"]["enhanced"], "normalize FOV\nvessel response"),
    ]
    for idx, (title, image, body) in enumerate(prep_items, start=1):
        x = xs[idx]
        rounded_box(ax, x, top_y, w, top_h, title, "prep", body=body, title_size=8.3, body_size=6.8)
        put_image(ax, image, x + 0.95, top_y + 0.78, 8.3, 4.35, title.replace("\n", " "), "#245985")

    mini_hist(
        ax,
        [data["pre"]["green"], data["pre"]["clahe"], data["pre"]["enhanced"]],
        ["green", "CLAHE", "enhanced"],
        ["#22c55e", "#2563eb", "#ef4444"],
        xs[7],
        top_y,
        w + 6.5,
        top_h,
    )

    for x1, x2 in zip(xs[:6], xs[1:7]):
        arrow(ax, (x1 + w, top_y + 5.1), (x2, top_y + 5.1))
    arrow(ax, (xs[6] + w, top_y + 5.1), (xs[7], top_y + 5.1))

    # Threshold and morphology branches.
    branch_y, branch_h = 44.0, 8.7
    morph_y, morph_h = 33.6, 7.9
    skel_y, skel_h = 23.8, 7.7
    method_positions = {"global": 5.5, "otsu": 37.0, "adaptive": 68.5}
    method_w = 26.3
    enhanced_anchor = (xs[6] + w / 2, top_y)
    for method, x in method_positions.items():
        method_panel(
            ax,
            method,
            x,
            branch_y,
            method_w,
            branch_h,
            data["methods"][method]["raw"],
            data["methods"][method]["refined"],
        )
        morph_panel(ax, method, x, morph_y, method_w, morph_h)
        skeleton_panel(ax, method, x, skel_y, method_w, skel_h, data["methods"][method]["skeleton"])
        color = METHOD_COLORS[method]
        arrow(ax, enhanced_anchor, (x + method_w / 2, branch_y + branch_h), color=color, rad=0.08)
        arrow(ax, (x + method_w / 2, branch_y), (x + method_w / 2, morph_y + morph_h), color=color)
        arrow(ax, (x + method_w / 2, morph_y), (x + method_w / 2, skel_y + skel_h), color=color)

    fig.savefig(ASSET_DIR / "full_pipeline_architecture.png", dpi=360, facecolor="white")
    fig.savefig(ASSET_DIR / "full_pipeline_architecture.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    save_full_pipeline()
