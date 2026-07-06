from __future__ import annotations

import sys
from pathlib import Path
from textwrap import fill

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    title_size=4.8,
    body_size=4.0,
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
        y + h - 0.65,
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
            y + h - 1.75,
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
            fontsize=3.8,
            color="#0f172a",
            zorder=4,
        )


def put_text(ax, x, y, text, width=25, size=4.2, weight="normal", color="#0f172a"):
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
    rounded_box(ax, x, y, w, h, "2.7 Histogram Analysis", "prep", title_size=4.6)
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
        ax.text(x + 0.35, y0 + 0.5, label, ha="left", va="center", fontsize=3.6, zorder=4)


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
        f"3.{letter} {method.upper()} Thresholding",
        "threshold",
        body={
            "global": "Percentile threshold on enhanced vessel image.",
            "otsu": "Automatic threshold from intensity distribution.",
            "adaptive": "Local Gaussian threshold for uneven illumination.",
        }[method],
        title_size=4.6,
        body_size=3.75,
        title_color=color,
    )
    put_image(ax, raw, x + 0.55, y + 0.65, w * 0.38, h * 0.43, "raw mask", color)
    put_image(ax, refined, x + w * 0.57, y + 0.65, w * 0.38, h * 0.43, "clean mask", color)


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
        title_size=4.4,
        title_color=color,
    )
    steps = ["Opening", "Closing", "Fill holes", "Remove small\ncomponents", "Final mask"]
    step_w = (w - 1.1) / len(steps)
    for idx, step in enumerate(steps):
        sx = x + 0.45 + idx * step_w
        ax.add_patch(
            FancyBboxPatch(
                (sx, y + 1.35),
                step_w - 0.18,
                1.25,
                boxstyle="round,pad=0.012,rounding_size=0.06",
                linewidth=0.45,
                edgecolor=color,
                facecolor="#ffffff",
                zorder=2,
            )
        )
        ax.text(
            sx + (step_w - 0.18) / 2,
            y + 1.98,
            step,
            ha="center",
            va="center",
            fontsize=3.25,
            color="#0f172a",
            zorder=4,
        )
        if idx < len(steps) - 1:
            arrow(
                ax,
                (sx + step_w - 0.18, y + 1.98),
                (sx + step_w + 0.05, y + 1.98),
                color=color,
                lw=0.55,
                zorder=3.5,
            )
    ax.text(
        x + w / 2,
        y + 0.55,
        "Same refinement is applied after every threshold branch.",
        ha="center",
        va="center",
        fontsize=3.55,
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
        f"5.{letter} Skeleton + Features",
        "skeleton",
        title_size=4.4,
        title_color=color,
    )
    put_image(ax, skeleton, x + 0.55, y + 0.85, w * 0.34, h * 0.5, "skeleton", color)
    put_text(
        ax,
        x + w * 0.43,
        y + h - 1.55,
        "Computed features:\n- density (%)\n- area (px)\n- vessel length\n- average width\n- connected components",
        width=23,
        size=3.65,
    )


def table_text(df: pd.DataFrame) -> str:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"{row['method'].capitalize():8s} Dice {row['avg_dice']:.4f} | "
            f"IoU {row['avg_iou']:.4f} | F1 {row['avg_f1']:.4f} | "
            f"Acc {row['avg_accuracy']:.4f} | Time {row['avg_processing_time_sec']:.4f}s"
        )
    return "\n".join(rows)


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

    summary = pd.read_csv(
        PROJECT_ROOT
        / "results_test"
        / "visual_report"
        / "tables"
        / "single_easy_method_comparison_table.csv"
    )
    best = pd.read_csv(
        PROJECT_ROOT
        / "results_test"
        / "visual_report"
        / "tables"
        / "best_method_by_criterion_table.csv"
    )
    ranking = pd.read_csv(
        PROJECT_ROOT
        / "results_test"
        / "visual_report"
        / "tables"
        / "overall_method_ranking_table.csv"
    )
    sample_fig = read_bgr(PROJECT_ROOT / "results_test" / "figures" / "test_1_A_comparison.png")

    return {
        "bgr": bgr,
        "rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        "gt": gt_mask,
        "fov": fov,
        "pre": pre,
        "quality": quality,
        "methods": methods,
        "summary": summary,
        "best": best,
        "ranking": ranking,
        "sample_fig": sample_fig,
    }


def save_full_pipeline():
    data = prepare_artifacts()
    fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=360)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0.35, 99.65)
    ax.set_ylim(0.75, 69.65)
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
        "CLASSICAL DIP PIPELINE - COMPLETE WORKFLOW, BRANCHES, RESULTS, AND PROGRESS SUMMARY",
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
        body="Test split: 200 image pairs\nInput: fundus image\nTarget: manual vessel mask\nROI: field of view",
        title_size=4.7,
        body_size=3.6,
    )
    put_image(ax, data["rgb"], xs[0] + 0.6, top_y + 0.65, 4.1, 3.2, "input", "#3f7f2c")
    put_image(ax, data["gt"], xs[0] + 5.4, top_y + 0.65, 4.1, 3.2, "GT mask", "#3f7f2c")

    prep_items = [
        ("2.1 FOV Mask", data["fov"], "gray threshold\nmorph close/open\nlargest component\nfill holes"),
        ("2.2 Green Channel", data["pre"]["green"], "extract green\nbest vessel contrast"),
        ("2.3 CLAHE", data["pre"]["clahe"], "local contrast\nenhancement"),
        ("2.4 Gaussian", data["pre"]["gaussian"], "noise smoothing\nkernel = 5"),
        ("2.5 Black-hat", data["pre"]["blackhat"], "dark vessel\nemphasis"),
        ("2.6 Contrast Stretch", data["pre"]["enhanced"], "normalize inside FOV\nvessel-enhanced image"),
    ]
    for idx, (title, image, body) in enumerate(prep_items, start=1):
        x = xs[idx]
        rounded_box(ax, x, top_y, w, top_h, title, "prep", body=body, title_size=4.5, body_size=3.45)
        put_image(ax, image, x + 1.2, top_y + 0.75, 7.8, 4.0, title.split(" ", 1)[1], "#245985")

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

    # Evaluation, performance, and visual comparison.
    eval_y, eval_h = 12.9, 8.9
    rounded_box(
        ax,
        1.0,
        eval_y,
        22.0,
        eval_h,
        "6. Evaluation vs Ground Truth",
        "eval",
        body="Mask metrics:\nDice, IoU, Precision, Recall, F1, Accuracy\n\nSkeleton metrics:\ncenterline Dice, IoU, F1, pixels, time\n\nFeature errors:\ndensity, area, length, width, components",
        title_size=4.45,
        body_size=3.5,
    )
    rounded_box(
        ax,
        24.4,
        eval_y,
        31.0,
        eval_h,
        "7. Average Performance over 200 Test Images",
        "eval",
        title_size=4.45,
    )
    put_text(ax, 25.1, eval_y + eval_h - 1.55, table_text(data["summary"]), width=78, size=3.55)
    ranking_text = "\n".join(
        f"Rank {int(r.overall_position)}: {str(r.method).capitalize()} (avg rank {r.average_rank:.4f})"
        for _, r in data["ranking"].iterrows()
    )
    put_text(ax, 25.1, eval_y + 2.0, ranking_text, width=76, size=3.55, weight="bold")

    best_lines = [
        f"{row['criterion']}: {row['best_method']} ({row['value']:.4f})"
        for _, row in data["best"].head(10).iterrows()
    ]
    rounded_box(
        ax,
        56.8,
        eval_y,
        16.7,
        eval_h,
        "8. Best Result Summary",
        "eval",
        body="\n".join(best_lines),
        title_size=4.45,
        body_size=3.15,
    )
    rounded_box(ax, 74.8, eval_y, 24.2, eval_h, "9. Visual Comparison Sample", "eval", title_size=4.45)
    put_image(ax, data["sample_fig"], 75.6, eval_y + 0.7, 22.6, 6.55, "image 1_A pipeline comparison", "#a36a00")

    for method, x in method_positions.items():
        color = METHOD_COLORS[method]
        arrow(
            ax,
            (x + method_w / 2, skel_y),
            (12.0, eval_y + eval_h),
            color=color,
            rad=-0.08,
            lw=0.55,
            alpha=0.72,
        )
        arrow(
            ax,
            (x + method_w / 2, skel_y),
            (39.0, eval_y + eval_h),
            color=color,
            rad=0.0,
            lw=0.55,
            alpha=0.72,
        )
        arrow(
            ax,
            (x + method_w / 2, skel_y),
            (65.0, eval_y + eval_h),
            color=color,
            rad=0.07,
            lw=0.55,
            alpha=0.72,
        )
        arrow(
            ax,
            (x + method_w / 2, skel_y),
            (87.0, eval_y + eval_h),
            color=color,
            rad=0.12,
            lw=0.55,
            alpha=0.72,
        )

    # Bottom: progress, outputs, applications, conclusion.
    bottom_y, bottom_h = 2.0, 8.9
    rounded_box(
        ax,
        1.0,
        bottom_y,
        24.0,
        bottom_h,
        "10. What We Completed",
        "output",
        body="- Built full classical DIP pipeline\n- Removed edge and watershed parts\n- Kept skeletonization branch\n- Tested Global, Otsu, Adaptive\n- Evaluated all 200 test images\n- Generated masks, skeletons, overlays\n- Built CSV tables, dashboard, notebook, report",
        title_size=4.5,
        body_size=3.45,
    )
    rounded_box(
        ax,
        26.3,
        bottom_y,
        23.5,
        bottom_h,
        "11. Deliverables",
        "output",
        body="- Final binary vessel masks\n- Skeleton maps\n- Overlay error maps\n- Method comparison tables\n- Best-method ranking\n- HTML dashboard\n- Notebook visual figures\n- LaTeX report and PDF",
        title_size=4.5,
        body_size=3.45,
    )
    rounded_box(
        ax,
        51.1,
        bottom_y,
        22.5,
        bottom_h,
        "12. Why It Matters",
        "output",
        body="- Vessel density: amount of vessel area\n- Area: total detected vessel pixels\n- Skeleton length: vessel centerline length\n- Average width: estimated vessel thickness\n- Components: fragmentation/noise measure\n- Overlay: visual FP/FN diagnosis",
        title_size=4.5,
        body_size=3.45,
    )
    rounded_box(
        ax,
        74.9,
        bottom_y,
        24.1,
        bottom_h,
        "13. Final Decision and Legend",
        "note",
        body="Final ranking on this test set:\n1. Global thresholding\n2. Adaptive thresholding\n3. Otsu thresholding\n\nOverlay legend:\nYellow = true positive\nBlue = false positive\nRed = false negative\n\nBranch legend: blue=Global, orange=Otsu, green=Adaptive.",
        title_size=4.5,
        body_size=3.45,
    )

    arrow(ax, (12.5, eval_y), (12.5, bottom_y + bottom_h), color="#1f7a5a")
    arrow(ax, (39.0, eval_y), (39.0, bottom_y + bottom_h), color="#1f7a5a")
    arrow(ax, (65.0, eval_y), (65.0, bottom_y + bottom_h), color="#1f7a5a")
    arrow(ax, (87.0, eval_y), (87.0, bottom_y + bottom_h), color="#64748b")

    fig.savefig(ASSET_DIR / "full_pipeline_architecture.png", dpi=360, facecolor="white")
    fig.savefig(ASSET_DIR / "full_pipeline_architecture.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    save_full_pipeline()
