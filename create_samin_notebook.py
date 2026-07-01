import json
from pathlib import Path


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(True),
    }


cells = [
    md(
        r"""
# FIVES Retinal Vessel Segmentation - Step-by-Step Notebook

এই notebook-টি main Python project-এর একই concept follow করে:

- RGB retina image + ground truth mask
- preprocessing: green channel, CLAHE, Gaussian filter, black-hat, contrast stretching
- segmentation: Global, Otsu, Adaptive
- morphology: opening, closing, fill holes, connected-component cleanup
- advanced branches: skeleton, watershed, edge detection
- per-image comparison: Dice, IoU, Precision, Recall, F1, Accuracy
- feature comparison: predicted mask vs ground truth mask
- all 200 test images: summary tables, ranking, graphs, and representative examples

Notebook presentation idea:

1. First 3 selected images: full pipeline visual explanation.
2. Any image-wise report: change selected image IDs and rerun.
3. All 200 images: use summary tables and graphs instead of showing 200 images one by one.
"""
    ),
    code(
        r"""
from pathlib import Path
import sys
import re
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML, Image as IPImage

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "fives_retinal_pipeline.py").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fives_retinal_pipeline as dip
import visualize_results as viz

RESULTS_DIR = PROJECT_ROOT / "results_test"
DATASET_DIR = PROJECT_ROOT / "archive"
TABLES_DIR = RESULTS_DIR / "tables"
VISUAL_REPORT_DIR = RESULTS_DIR / "visual_report"
CHARTS_DIR = VISUAL_REPORT_DIR / "charts"
NOTEBOOK_FIG_DIR = PROJECT_ROOT / "sameinipynb" / "notebook_outputs"
NOTEBOOK_FIG_DIR.mkdir(parents=True, exist_ok=True)
for old_plot in NOTEBOOK_FIG_DIR.glob("*.png"):
    old_plot.unlink()

METHODS = ["global", "otsu", "adaptive"]
BRANCHES = ["skeleton", "watershed", "edge"]
METRICS = ["dice", "iou", "precision", "recall", "f1", "accuracy"]

PARAMS = {
    "clahe_clip": 2.0,
    "gaussian_kernel": 5,
    "blackhat_kernel": 15,
    "global_percentile": 92.0,
    "adaptive_block_size": 201,
    "adaptive_c": -40.0,
    "min_object_area": 50,
}

method_df = pd.read_csv(TABLES_DIR / "segmentation_metrics_and_features.csv")
branch_df = pd.read_csv(TABLES_DIR / "branch_comparison_metrics.csv")
gt_df = pd.read_csv(TABLES_DIR / "ground_truth_features.csv")

method_summary = viz.make_method_summary(method_df)
branch_summary = viz.make_branch_summary(branch_df)
best_methods = viz.make_best_methods(method_summary)
ranking = viz.make_overall_ranking(method_summary)
image_level_table = viz.make_image_comparison_table(method_df)

print("Project root:", PROJECT_ROOT)
print("Results folder:", RESULTS_DIR)
print("Total image-method rows:", len(method_df))
print("Unique test images:", method_df["image_id"].nunique())
"""
    ),
    md(
        r"""
## 1. Image Selection

Default হিসেবে first 3 test images select করা হয়েছে. Presentation-এর সময় শুধু `SELECTED_IMAGE_IDS` বদলালেই অন্য image-wise report দেখা যাবে.

যদি `ipywidgets` install থাকে, নিচের cell একটি multi-select option দেখাবে. না থাকলে list edit করে rerun করলেই হবে.
"""
    ),
    code(
        r"""
def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]

ALL_IMAGE_IDS = sorted(method_df["image_id"].unique().tolist(), key=natural_key)
SELECTED_IMAGE_IDS = ALL_IMAGE_IDS[:3]

try:
    import ipywidgets as widgets
    selector = widgets.SelectMultiple(
        options=ALL_IMAGE_IDS,
        value=tuple(SELECTED_IMAGE_IDS),
        rows=10,
        description="Images",
        disabled=False,
    )
    display(Markdown("Select 3 images, then run the next cells:"))
    display(selector)
    print("After selecting, run: SELECTED_IMAGE_IDS = list(selector.value)")
except Exception as exc:
    print("ipywidgets is not available. Edit SELECTED_IMAGE_IDS manually if needed.")

print("Current selected images:", SELECTED_IMAGE_IDS)
"""
    ),
    md(
        r"""
## 2. Dataset and Parameter Overview

এই section-এ dataset size, method list, এবং pipeline parameters দেখানো হয়েছে.
"""
    ),
    code(
        r"""
dataset_overview = pd.DataFrame([
    {"item": "Test images evaluated", "value": method_df["image_id"].nunique()},
    {"item": "Methods", "value": ", ".join(METHODS)},
    {"item": "Branches per method", "value": ", ".join(BRANCHES)},
    {"item": "Total method outputs", "value": len(method_df)},
    {"item": "Total branch outputs", "value": len(branch_df)},
])
display(dataset_overview)

param_table = pd.DataFrame([{"parameter": key, "value": value} for key, value in PARAMS.items()])
display(param_table)
"""
    ),
    md(
        r"""
## 3. Helper Functions for Image-wise Visualization

এই functions saved outputs এবং CSV result ব্যবহার করে:

- original / ground truth
- preprocessing stages
- threshold + morphology stages
- skeleton / watershed / edge / overlay outputs
- metrics and feature comparison tables
"""
    ),
    code(
        r"""
ARTIFACT_CACHE = {}

def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def bool_to_u8(mask):
    return (mask.astype(np.uint8) * 255)

def image_path(image_id):
    return DATASET_DIR / "test" / "Original" / f"{image_id}.png"

def gt_path(image_id):
    return DATASET_DIR / "test" / "Ground truth" / f"{image_id}.png"

def output_path(kind, method, image_id, edge_type=None):
    if kind == "edge":
        return RESULTS_DIR / "edges" / "test" / method / edge_type / f"{image_id}.png"
    return RESULTS_DIR / kind / "test" / method / f"{image_id}.png"

def plot_grid(items, cols=4, figsize=None, title=None, cmap="gray"):
    if not hasattr(plot_grid, "counter"):
        plot_grid.counter = 0
    plot_grid.counter += 1
    rows = math.ceil(len(items) / cols)
    if figsize is None:
        figsize = (4 * cols, 3.5 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for ax, item in zip(axes, items):
        label, image = item
        if image.ndim == 2:
            ax.imshow(image, cmap=cmap)
        else:
            ax.imshow(image)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    for ax in axes[len(items):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", title or "figure").strip("_")[:90]
    out_path = NOTEBOOK_FIG_DIR / f"{plot_grid.counter:03d}_{safe_title}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    display(IPImage(filename=str(out_path)))

def get_core_artifacts(image_id):
    if image_id in ARTIFACT_CACHE:
        return ARTIFACT_CACHE[image_id]

    bgr = dip.read_bgr(image_path(image_id))
    gt_mask = dip.read_mask(gt_path(image_id), size=(bgr.shape[1], bgr.shape[0]))
    fov = dip.create_fov_mask(bgr)
    preprocessed, quality = dip.preprocess_image(
        bgr=bgr.copy(),
        fov=fov,
        clahe_clip=PARAMS["clahe_clip"],
        blackhat_kernel_size=PARAMS["blackhat_kernel"],
        gaussian_kernel_size=PARAMS["gaussian_kernel"],
    )
    ARTIFACT_CACHE[image_id] = {
        "bgr": bgr,
        "rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        "gt_mask": gt_mask,
        "fov": fov,
        "preprocessed": preprocessed,
        "quality": quality,
    }
    return ARTIFACT_CACHE[image_id]

def metrics_table(image_id):
    cols = ["method", *METRICS, "processing_time_sec", "pred_vessel_density_pct", "gt_vessel_density_pct"]
    table = method_df[method_df["image_id"] == image_id][cols].copy()
    return table.round(4)

def feature_table(image_id):
    rows = []
    features = [
        ("vessel_density_pct", "Vessel density (%)"),
        ("vessel_area_px", "Vessel area (px)"),
        ("largest_vessel_area_px", "Largest vessel area"),
        ("smallest_vessel_area_px", "Smallest vessel area"),
        ("vessel_length_px", "Vessel length"),
        ("average_width_px", "Average width"),
        ("connected_components", "Connected components"),
        ("watershed_regions", "Watershed regions"),
    ]
    image_rows = method_df[method_df["image_id"] == image_id]
    for _, method_row in image_rows.iterrows():
        for key, label in features:
            rows.append({
                "method": method_row["method"],
                "feature": label,
                "predicted": method_row.get(f"pred_{key}", np.nan),
                "ground_truth": method_row.get(f"gt_{key}", np.nan),
                "absolute_error": method_row.get(f"abs_diff_{key}", np.nan),
            })
    return pd.DataFrame(rows).round(4)

def show_original_and_gt(image_id):
    artifacts = get_core_artifacts(image_id)
    plot_grid(
        [
            ("Original RGB", artifacts["rgb"]),
            ("Ground Truth Mask", bool_to_u8(artifacts["gt_mask"])),
            ("Field of View Mask", bool_to_u8(artifacts["fov"])),
        ],
        cols=3,
        title=f"{image_id}: Input and Ground Truth",
    )

def show_preprocessing_steps(image_id):
    artifacts = get_core_artifacts(image_id)
    pre = artifacts["preprocessed"]
    plot_grid(
        [
            ("Original RGB", artifacts["rgb"]),
            ("Green Channel", pre["green"]),
            ("CLAHE", pre["clahe"]),
            ("Gaussian Filter", pre["gaussian"]),
            ("Black-hat", pre["blackhat"]),
            ("Contrast Stretched Enhanced", pre["enhanced"]),
        ],
        cols=3,
        title=f"{image_id}: Preprocessing Pipeline",
    )
    display(pd.DataFrame([artifacts["quality"]]).round(4))

def recompute_method_stages(image_id, method):
    artifacts = get_core_artifacts(image_id)
    raw_mask, threshold_value = dip.threshold_image(
        artifacts["preprocessed"]["enhanced"],
        artifacts["fov"],
        method,
        PARAMS["global_percentile"],
        PARAMS["adaptive_block_size"],
        PARAMS["adaptive_c"],
    )
    morph = dip.refine_mask(raw_mask, artifacts["fov"], PARAMS["min_object_area"])
    return raw_mask, morph, threshold_value

def show_threshold_and_morphology(image_id):
    for method in METHODS:
        raw_mask, morph, threshold_value = recompute_method_stages(image_id, method)
        threshold_label = f"{threshold_value:.2f}" if not np.isnan(threshold_value) else "local"
        plot_grid(
            [
                (f"{method}: raw threshold ({threshold_label})", bool_to_u8(raw_mask)),
                ("Opening", bool_to_u8(morph["opening"])),
                ("Closing", bool_to_u8(morph["closing"])),
                ("Fill Holes", bool_to_u8(morph["filled"])),
                ("Final Cleaned Mask", bool_to_u8(morph["cleaned"])),
            ],
            cols=5,
            title=f"{image_id}: {method.upper()} Threshold + Morphological Refinement",
        )

def show_final_masks_and_overlays(image_id):
    artifacts = get_core_artifacts(image_id)
    mask_items = [("Ground Truth", bool_to_u8(artifacts["gt_mask"]))]
    overlay_items = [("Original RGB", artifacts["rgb"])]
    for method in METHODS:
        mask_items.append((f"{method} mask", read_rgb(output_path("masks", method, image_id))))
        overlay_items.append((f"{method} overlay", read_rgb(output_path("overlays", method, image_id))))
    plot_grid(mask_items, cols=4, title=f"{image_id}: Final Binary Mask Comparison")
    plot_grid(overlay_items, cols=4, title=f"{image_id}: Overlay Comparison with Ground Truth")

def show_advanced_branches(image_id):
    for method in METHODS:
        items = [
            (f"{method} final mask", read_rgb(output_path("masks", method, image_id))),
            (f"{method} skeleton", read_rgb(output_path("skeletons", method, image_id))),
            (f"{method} watershed", read_rgb(output_path("watershed", method, image_id))),
            (f"{method} edge union", read_rgb(output_path("edge", method, image_id, "union"))),
        ]
        plot_grid(items, cols=4, title=f"{image_id}: Advanced Branches for {method.upper()}")

def show_edge_details(image_id):
    for method in METHODS:
        items = [
            (f"{method} Sobel", read_rgb(output_path("edge", method, image_id, "sobel"))),
            (f"{method} Canny", read_rgb(output_path("edge", method, image_id, "canny"))),
            (f"{method} Laplacian", read_rgb(output_path("edge", method, image_id, "laplacian"))),
            (f"{method} Union", read_rgb(output_path("edge", method, image_id, "union"))),
        ]
        plot_grid(items, cols=4, title=f"{image_id}: Edge Detection Details for {method.upper()}")

def show_image_tables(image_id):
    display(Markdown(f"### {image_id}: Metric Comparison"))
    display(metrics_table(image_id))
    display(Markdown(f"### {image_id}: Feature Comparison Against Ground Truth"))
    display(feature_table(image_id))

def show_full_image_report(image_id):
    display(Markdown(f"# Image-wise Full Pipeline Report: `{image_id}`"))
    show_original_and_gt(image_id)
    show_preprocessing_steps(image_id)
    show_threshold_and_morphology(image_id)
    show_final_masks_and_overlays(image_id)
    show_advanced_branches(image_id)
    show_edge_details(image_id)
    show_image_tables(image_id)
"""
    ),
    md(
        r"""
## 4. Full Step-by-Step Pipeline for First 3 Selected Images

এই section presentation-এর visual proof:

- প্রতিটি selected image-এর জন্য preprocessing দেখাবে
- Global/Otsu/Adaptive threshold + morphology দেখাবে
- final mask, overlay, skeleton, watershed, edge map দেখাবে
- শেষে per-image metrics এবং feature comparison table দেখাবে
"""
    ),
    code(
        r"""
for selected_id in SELECTED_IMAGE_IDS:
    show_full_image_report(selected_id)
"""
    ),
    md(
        r"""
## 5. Single Image Quick Viewer

যেকোনো একটা image দ্রুত দেখতে চাইলে নিচের `IMAGE_ID_TO_VIEW` change করে cell run করো.
"""
    ),
    code(
        r"""
IMAGE_ID_TO_VIEW = SELECTED_IMAGE_IDS[0]
show_full_image_report(IMAGE_ID_TO_VIEW)
"""
    ),
    md(
        r"""
## 6. All 200 Images - Summary Tables

২০০ image presentation-এ একে একে দেখানো feasible না. তাই এখানে aggregate numerical comparison দেখানো হয়েছে.
"""
    ),
    code(
        r"""
display(Markdown("### Overall Method Ranking"))
display(viz.round_numeric(ranking))

display(Markdown("### Single Easy Method Comparison Table"))
display(viz.round_numeric(method_summary))

display(Markdown("### Best Method by Criterion"))
display(viz.round_numeric(best_methods))

display(Markdown("### Branch Comparison Summary"))
display(viz.round_numeric(branch_summary))
"""
    ),
    md(
        r"""
## 7. All 200 Images - Graphical Comparison

এই graphs সব ২০০ test image-এর CSV result থেকে তৈরি:

- metric comparison
- feature predicted vs ground truth
- feature error heatmap
- confusion pixel summary
- branch comparison
- processing time
- ranking
"""
    ),
    code(
        r"""
chart_files = [
    "test_segmentation_metric_comparison.png",
    "test_feature_values_predicted_vs_ground_truth.png",
    "test_feature_error_heatmap.png",
    "test_confusion_pixel_summary.png",
    "test_branch_metric_comparison.png",
    "test_processing_time.png",
    "test_overall_method_ranking.png",
]

chart_items = []
for name in chart_files:
    path = CHARTS_DIR / name
    if path.exists():
        chart_items.append((name.replace("_", " ").replace(".png", ""), read_rgb(path)))
plot_grid(chart_items, cols=2, figsize=(16, 5 * math.ceil(len(chart_items) / 2)), title="All 200 Images: Summary Charts")
"""
    ),
    md(
        r"""
## 8. Image-Level Comparison Table

এই table-এ ২০০ image-এর প্রতিটির Global/Otsu/Adaptive result আছে. নিচে first 20 rows দেখানো হলো.
"""
    ),
    code(
        r"""
display(viz.round_numeric(image_level_table).head(20))
print("Full image-level table rows:", len(image_level_table))
"""
    ),
    md(
        r"""
## 9. Representative Examples from 200 Images

Presentation-এ ২০০ image দেখানোর বদলে ৩টি representative case দেখানো ভালো:

- Best result
- Median/typical result
- Worst/failure case

নিচের example `adaptive` method-এর Dice score অনুযায়ী best/median/worst দেখায়. Method বদলাতে চাইলে `REPRESENTATIVE_METHOD` change করো.
"""
    ),
    code(
        r"""
def representative_image_ids(method="adaptive", metric="dice"):
    rows = method_df[method_df["method"] == method].sort_values(metric).reset_index(drop=True)
    worst = rows.iloc[0]
    median = rows.iloc[len(rows) // 2]
    best = rows.iloc[-1]
    return [
        ("Worst", worst["image_id"], worst[metric]),
        ("Median", median["image_id"], median[metric]),
        ("Best", best["image_id"], best[metric]),
    ]

def show_representative_cases(method="adaptive", metric="dice"):
    reps = representative_image_ids(method, metric)
    rows = []
    items = []
    for label, image_id, score in reps:
        rows.append({"case": label, "image_id": image_id, f"{method}_{metric}": score})
        artifacts = get_core_artifacts(image_id)
        items.extend([
            (f"{label}: {image_id} original", artifacts["rgb"]),
            (f"{label}: GT", bool_to_u8(artifacts["gt_mask"])),
            (f"{label}: {method} mask", read_rgb(output_path("masks", method, image_id))),
            (f"{label}: {method} overlay", read_rgb(output_path("overlays", method, image_id))),
        ])
    display(pd.DataFrame(rows).round(4))
    plot_grid(items, cols=4, figsize=(16, 10), title=f"Representative {method.upper()} Cases by {metric.upper()}")

REPRESENTATIVE_METHOD = "adaptive"
show_representative_cases(REPRESENTATIVE_METHOD, "dice")
"""
    ),
    md(
        r"""
## 10. Output Folder Map

সব saved image কোথায় আছে:
"""
    ),
    code(
        r"""
folders = []
for method in METHODS:
    folders.extend([
        {"output": f"{method} final masks", "folder": str(RESULTS_DIR / "masks" / "test" / method)},
        {"output": f"{method} skeletons", "folder": str(RESULTS_DIR / "skeletons" / "test" / method)},
        {"output": f"{method} watershed", "folder": str(RESULTS_DIR / "watershed" / "test" / method)},
        {"output": f"{method} overlays", "folder": str(RESULTS_DIR / "overlays" / "test" / method)},
        {"output": f"{method} Sobel edges", "folder": str(RESULTS_DIR / "edges" / "test" / method / "sobel")},
        {"output": f"{method} Canny edges", "folder": str(RESULTS_DIR / "edges" / "test" / method / "canny")},
        {"output": f"{method} Laplacian edges", "folder": str(RESULTS_DIR / "edges" / "test" / method / "laplacian")},
        {"output": f"{method} Union edges", "folder": str(RESULTS_DIR / "edges" / "test" / method / "union")},
    ])

folder_table = pd.DataFrame(folders)
folder_table["png_count"] = folder_table["folder"].apply(lambda p: len(list(Path(p).glob("*.png"))) if Path(p).exists() else 0)
display(folder_table)

display(Markdown("### Pipeline comparison sample figure"))
pipeline_fig = RESULTS_DIR / "figures" / "test_1_A_comparison.png"
if pipeline_fig.exists():
    plot_grid([("Pipeline comparison sample", read_rgb(pipeline_fig))], cols=1, figsize=(16, 10))
else:
    print("Pipeline comparison figure not found:", pipeline_fig)
"""
    ),
    md(
        r"""
## 11. Final Interpretation Template

Presentation/report-এ এই interpretation ব্যবহার করতে পারো:

- Global thresholding সাধারণত precision এবং accuracy ভালো দেয়, কিন্তু recall কম হতে পারে.
- Otsu thresholding recall বেশি দিতে পারে, কিন্তু false positive বেশি হলে precision কমে যায়.
- Adaptive thresholding local variation handle করে, তাই balanced Dice/F1 দিতে পারে.
- Accuracy একা final decision metric না, কারণ background pixel অনেক বেশি.
- Best method নির্বাচন করতে Dice, IoU, F1, feature error, এবং visual overlay একসাথে দেখতে হবে.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_dir = Path("sameinipynb")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "samin.ipynb"
out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(out_path.resolve())
