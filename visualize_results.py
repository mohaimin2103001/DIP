from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRICS = ["dice", "iou", "precision", "recall", "f1", "accuracy"]
FEATURE_PAIRS = [
    ("vessel_density_pct", "Vessel density (%)"),
    ("vessel_area_px", "Vessel area (px)"),
    ("vessel_length_px", "Vessel length (px)"),
    ("average_width_px", "Average width (px)"),
    ("connected_components", "Connected components"),
    ("watershed_regions", "Watershed regions"),
]
METHOD_ORDER = ["global", "otsu", "adaptive"]
BRANCH_ORDER = ["skeleton", "watershed", "edge"]
METHOD_COLORS = {
    "global": "#2563eb",
    "otsu": "#f97316",
    "adaptive": "#16a34a",
}


def default_results_dir() -> Path:
    for name in ("results_test", "results_full", "results_demo_tuned", "results_csv_smoke"):
        path = Path(name)
        if (path / "tables" / "segmentation_metrics_and_features.csv").exists():
            return path
    return Path("results_test")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def ensure_dirs(output: Path) -> Tuple[Path, Path]:
    charts_dir = output / "charts"
    tables_dir = output / "tables"
    charts_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return charts_dir, tables_dir


def ordered_methods(methods: Iterable[str]) -> List[str]:
    methods = list(dict.fromkeys(methods))
    known = [method for method in METHOD_ORDER if method in methods]
    unknown = sorted(method for method in methods if method not in METHOD_ORDER)
    return known + unknown


def ordered_branches(branches: Iterable[str]) -> List[str]:
    branches = list(dict.fromkeys(branches))
    known = [branch for branch in BRANCH_ORDER if branch in branches]
    unknown = sorted(branch for branch in branches if branch not in BRANCH_ORDER)
    return known + unknown


def available_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    return [column for column in columns if column in df.columns]


def round_numeric(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    numeric = out.select_dtypes(include=[np.number]).columns
    out[numeric] = out[numeric].round(digits)
    return out


def make_method_summary(method_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (split, method), group in method_df.groupby(["split", "method"], sort=False):
        row: Dict[str, object] = {
            "split": split,
            "method": method,
            "images": int(group["image_id"].nunique()),
        }
        for metric in METRICS:
            row[f"avg_{metric}"] = group[metric].mean()
        row["avg_processing_time_sec"] = group["processing_time_sec"].mean()

        for feature, _ in FEATURE_PAIRS:
            pred = f"pred_{feature}"
            gt = f"gt_{feature}"
            abs_diff = f"abs_diff_{feature}"
            if pred in group.columns:
                row[f"avg_pred_{feature}"] = group[pred].mean()
            if gt in group.columns:
                row[f"avg_gt_{feature}"] = group[gt].mean()
            if abs_diff in group.columns:
                row[f"avg_abs_error_{feature}"] = group[abs_diff].mean()
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["method"] = pd.Categorical(
        summary["method"], categories=ordered_methods(summary["method"]), ordered=True
    )
    return summary.sort_values(["split", "method"]).reset_index(drop=True)


def make_branch_summary(branch_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = available_columns(
        branch_df,
        [
            *METRICS,
            "branch_processing_time_sec",
            "branch_pred_pixels",
            "branch_gt_pixels",
            "abs_diff_vessel_density_pct",
            "abs_diff_vessel_area_px",
            "abs_diff_vessel_length_px",
            "abs_diff_average_width_px",
            "abs_diff_connected_components",
            "abs_diff_watershed_regions",
        ],
    )
    summary = (
        branch_df.groupby(["split", "method", "branch"], as_index=False)[numeric_cols]
        .mean()
        .rename(columns={column: f"avg_{column}" for column in numeric_cols})
    )
    summary["method"] = pd.Categorical(
        summary["method"], categories=ordered_methods(summary["method"]), ordered=True
    )
    summary["branch"] = pd.Categorical(
        summary["branch"], categories=ordered_branches(summary["branch"]), ordered=True
    )
    return summary.sort_values(["split", "method", "branch"]).reset_index(drop=True)


def make_image_comparison_table(method_df: pd.DataFrame) -> pd.DataFrame:
    value_cols = available_columns(
        method_df,
        [
            *METRICS,
            "processing_time_sec",
            "pred_vessel_density_pct",
            "gt_vessel_density_pct",
            "abs_diff_vessel_density_pct",
            "pred_vessel_length_px",
            "gt_vessel_length_px",
            "abs_diff_vessel_length_px",
            "pred_average_width_px",
            "gt_average_width_px",
            "abs_diff_average_width_px",
        ],
    )
    table = method_df[["split", "image_id", "method", *value_cols]].copy()
    pivot = table.pivot_table(
        index=["split", "image_id"],
        columns="method",
        values=value_cols,
        aggfunc="first",
    )
    pivot.columns = [f"{method}_{value}" for value, method in pivot.columns]
    return pivot.reset_index()


def make_best_methods(method_summary: pd.DataFrame) -> pd.DataFrame:
    criteria: List[Tuple[str, str, bool]] = [
        ("avg_dice", "Highest Dice", True),
        ("avg_iou", "Highest IoU", True),
        ("avg_precision", "Highest Precision", True),
        ("avg_recall", "Highest Recall", True),
        ("avg_f1", "Highest F1", True),
        ("avg_accuracy", "Highest Accuracy", True),
        ("avg_processing_time_sec", "Fastest Processing Time", False),
        ("avg_abs_error_vessel_density_pct", "Lowest Density Error", False),
        ("avg_abs_error_vessel_length_px", "Lowest Length Error", False),
        ("avg_abs_error_average_width_px", "Lowest Width Error", False),
    ]
    rows: List[Dict[str, object]] = []
    for split, group in method_summary.groupby("split", sort=False):
        for column, label, higher_is_better in criteria:
            if column not in group.columns or group[column].isna().all():
                continue
            idx = group[column].idxmax() if higher_is_better else group[column].idxmin()
            best = group.loc[idx]
            rows.append(
                {
                    "split": split,
                    "criterion": label,
                    "best_method": str(best["method"]),
                    "value": float(best[column]),
                }
            )
    return pd.DataFrame(rows)


def make_overall_ranking(method_summary: pd.DataFrame) -> pd.DataFrame:
    score_cols = available_columns(
        method_summary,
        [
            "avg_dice",
            "avg_iou",
            "avg_precision",
            "avg_recall",
            "avg_f1",
            "avg_accuracy",
            "avg_processing_time_sec",
            "avg_abs_error_vessel_density_pct",
            "avg_abs_error_vessel_length_px",
            "avg_abs_error_average_width_px",
            "avg_abs_error_connected_components",
            "avg_abs_error_watershed_regions",
        ],
    )
    lower_is_better = {
        "avg_processing_time_sec",
        "avg_abs_error_vessel_density_pct",
        "avg_abs_error_vessel_length_px",
        "avg_abs_error_average_width_px",
        "avg_abs_error_connected_components",
        "avg_abs_error_watershed_regions",
    }
    rows: List[pd.DataFrame] = []
    for split, group in method_summary.groupby("split", sort=False):
        ranked = group[["split", "method", "images"]].copy()
        rank_values = []
        for column in score_cols:
            ascending = column in lower_is_better
            rank_col = f"rank_{column}"
            ranked[rank_col] = group[column].rank(method="min", ascending=ascending)
            rank_values.append(rank_col)
        ranked["average_rank"] = ranked[rank_values].mean(axis=1)
        ranked["overall_position"] = ranked["average_rank"].rank(method="min", ascending=True)
        ranked = ranked.sort_values(["overall_position", "average_rank", "method"])
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_value_labels(ax: plt.Axes, digits: int = 3) -> None:
    for container in ax.containers:
        labels = []
        for value in container.datavalues:
            if np.isnan(value):
                labels.append("")
            elif abs(value) >= 100:
                labels.append(f"{value:.0f}")
            else:
                labels.append(f"{value:.{digits}f}")
        ax.bar_label(container, labels=labels, fontsize=8, padding=2)


def save_metric_chart(method_summary: pd.DataFrame, charts_dir: Path) -> List[Path]:
    paths = []
    for split, group in method_summary.groupby("split", sort=False):
        fig, ax = plt.subplots(figsize=(11, 5.5))
        methods = ordered_methods(group["method"].astype(str))
        x = np.arange(len(METRICS))
        width = 0.22
        offsets = np.linspace(-width, width, len(methods)) if len(methods) > 1 else [0]
        for offset, method in zip(offsets, methods):
            row = group[group["method"].astype(str) == method]
            values = [float(row[f"avg_{metric}"].iloc[0]) for metric in METRICS]
            ax.bar(
                x + offset,
                values,
                width=width,
                label=method,
                color=METHOD_COLORS.get(method),
            )
        ax.set_xticks(x)
        ax.set_xticklabels([metric.upper() if metric in {"iou", "f1"} else metric.title() for metric in METRICS])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Average score")
        ax.set_title(f"{split}: segmentation metric comparison")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        add_value_labels(ax)
        fig.tight_layout()
        path = charts_dir / f"{split}_segmentation_metric_comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def save_processing_time_chart(method_summary: pd.DataFrame, charts_dir: Path) -> List[Path]:
    paths = []
    for split, group in method_summary.groupby("split", sort=False):
        group = group.copy()
        group["method"] = group["method"].astype(str)
        group = group.set_index("method").loc[ordered_methods(group["method"])]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = [METHOD_COLORS.get(method) for method in group.index]
        ax.bar(group.index, group["avg_processing_time_sec"], color=colors)
        ax.set_ylabel("Average seconds per image")
        ax.set_title(f"{split}: processing time")
        ax.grid(axis="y", alpha=0.25)
        add_value_labels(ax)
        fig.tight_layout()
        path = charts_dir / f"{split}_processing_time.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def save_feature_value_charts(method_summary: pd.DataFrame, charts_dir: Path) -> List[Path]:
    paths = []
    for split, group in method_summary.groupby("split", sort=False):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
        axes = axes.ravel()
        methods = ordered_methods(group["method"].astype(str))
        group = group.set_index(group["method"].astype(str))
        for ax, (feature, label) in zip(axes, FEATURE_PAIRS):
            pred_col = f"avg_pred_{feature}"
            gt_col = f"avg_gt_{feature}"
            if pred_col not in group.columns or gt_col not in group.columns:
                ax.axis("off")
                continue
            pred = [group.loc[method, pred_col] for method in methods]
            gt = [group.loc[method, gt_col] for method in methods]
            x = np.arange(len(methods))
            ax.bar(x - 0.18, pred, width=0.36, label="Predicted", color="#38bdf8")
            ax.bar(x + 0.18, gt, width=0.36, label="Ground truth", color="#111827")
            ax.set_xticks(x)
            ax.set_xticklabels(methods)
            ax.set_title(label)
            ax.grid(axis="y", alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.suptitle(f"{split}: predicted features vs ground truth", y=1.02, fontsize=14)
        fig.tight_layout()
        path = charts_dir / f"{split}_feature_values_predicted_vs_ground_truth.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def save_feature_error_heatmap(method_summary: pd.DataFrame, charts_dir: Path) -> List[Path]:
    paths = []
    for split, group in method_summary.groupby("split", sort=False):
        methods = ordered_methods(group["method"].astype(str))
        group = group.set_index(group["method"].astype(str))
        columns = [
            (f"avg_abs_error_{feature}", label.replace(" (px)", "").replace(" (%)", ""))
            for feature, label in FEATURE_PAIRS
            if f"avg_abs_error_{feature}" in group.columns
        ]
        if not columns:
            continue
        data = np.array([[group.loc[method, column] for column, _ in columns] for method in methods], dtype=float)
        display_data = np.log10(data + 1.0)

        fig, ax = plt.subplots(figsize=(11, 4.8))
        image = ax.imshow(display_data, cmap="magma")
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels([label for _, label in columns], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_title(f"{split}: feature error heatmap (log scale)")
        for i in range(len(methods)):
            for j in range(len(columns)):
                value = data[i, j]
                text = f"{value:.2f}" if value < 100 else f"{value:.0f}"
                ax.text(j, i, text, ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=ax, label="log10(error + 1)")
        fig.tight_layout()
        path = charts_dir / f"{split}_feature_error_heatmap.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def save_confusion_chart(method_df: pd.DataFrame, charts_dir: Path) -> List[Path]:
    paths = []
    for split, group in method_df.groupby("split", sort=False):
        methods = ordered_methods(group["method"].astype(str))
        values = group.groupby("method")[["tp", "fp", "fn"]].mean().reindex(methods)
        fig, ax = plt.subplots(figsize=(9, 5))
        bottom = np.zeros(len(methods))
        colors = {"tp": "#22c55e", "fp": "#38bdf8", "fn": "#ef4444"}
        labels = {"tp": "TP", "fp": "FP", "fn": "FN"}
        for column in ["tp", "fp", "fn"]:
            ax.bar(methods, values[column], bottom=bottom, color=colors[column], label=labels[column])
            bottom += values[column].to_numpy()
        ax.set_ylabel("Average pixels")
        ax.set_title(f"{split}: mean TP/FP/FN pixels")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = charts_dir / f"{split}_confusion_pixel_summary.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def save_branch_chart(branch_summary: pd.DataFrame, charts_dir: Path) -> List[Path]:
    paths = []
    score_cols = ["avg_dice", "avg_iou", "avg_f1"]
    for split, group in branch_summary.groupby("split", sort=False):
        group = group.copy()
        group["label"] = group["method"].astype(str) + " / " + group["branch"].astype(str)
        fig, ax = plt.subplots(figsize=(13, 5))
        x = np.arange(len(group))
        width = 0.24
        for idx, column in enumerate(score_cols):
            if column not in group.columns:
                continue
            ax.bar(x + (idx - 1) * width, group[column], width=width, label=column.replace("avg_", "").upper())
        ax.set_xticks(x)
        ax.set_xticklabels(group["label"], rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Average score")
        ax.set_title(f"{split}: branch comparison")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = charts_dir / f"{split}_branch_metric_comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def save_ranking_chart(ranking: pd.DataFrame, charts_dir: Path) -> List[Path]:
    paths = []
    if ranking.empty:
        return paths
    for split, group in ranking.groupby("split", sort=False):
        group = group.sort_values("overall_position")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        methods = group["method"].astype(str).tolist()
        colors = [METHOD_COLORS.get(method) for method in methods]
        ax.bar(methods, group["average_rank"], color=colors)
        ax.set_ylabel("Average rank, lower is better")
        ax.set_title(f"{split}: overall method ranking")
        ax.grid(axis="y", alpha=0.25)
        add_value_labels(ax, digits=2)
        fig.tight_layout()
        path = charts_dir / f"{split}_overall_method_ranking.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def save_all_charts(
    method_df: pd.DataFrame,
    method_summary: pd.DataFrame,
    branch_summary: pd.DataFrame,
    ranking: pd.DataFrame,
    charts_dir: Path,
) -> List[Path]:
    paths: List[Path] = []
    paths.extend(save_metric_chart(method_summary, charts_dir))
    paths.extend(save_processing_time_chart(method_summary, charts_dir))
    paths.extend(save_feature_value_charts(method_summary, charts_dir))
    paths.extend(save_feature_error_heatmap(method_summary, charts_dir))
    paths.extend(save_confusion_chart(method_df, charts_dir))
    paths.extend(save_branch_chart(branch_summary, charts_dir))
    paths.extend(save_ranking_chart(ranking, charts_dir))
    return paths


def relative(path: Path, start: Path) -> str:
    return path.resolve().relative_to(start.resolve()).as_posix()


def df_to_html(df: pd.DataFrame, max_rows: int | None = None) -> str:
    display = df.head(max_rows).copy() if max_rows else df.copy()
    display = round_numeric(display, 4)
    return display.to_html(index=False, classes="data-table", border=0)


def make_html_report(
    output: Path,
    source_results: Path,
    method_summary: pd.DataFrame,
    branch_summary: pd.DataFrame,
    best_methods: pd.DataFrame,
    ranking: pd.DataFrame,
    image_table: pd.DataFrame,
    chart_paths: Sequence[Path],
) -> None:
    chart_html = "\n".join(
        f'<figure><img src="{relative(path, output)}" alt="{path.stem}"><figcaption>{path.stem.replace("_", " ").title()}</figcaption></figure>'
        for path in chart_paths
    )
    image_note = ""
    if len(image_table) > 80:
        image_note = f"<p>Showing first 80 rows out of {len(image_table)}. Full CSV is saved in the tables folder.</p>"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FIVES Result Dashboard</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #111827;
      background: #f8fafc;
    }}
    header {{
      background: #0f172a;
      color: white;
      padding: 24px 32px;
    }}
    main {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 24px;
    }}
    section {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 18px;
      overflow-x: auto;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    p {{
      line-height: 1.5;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
    }}
    figure {{
      margin: 0;
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 12px;
    }}
    img {{
      max-width: 100%;
      height: auto;
      display: block;
    }}
    figcaption {{
      font-size: 13px;
      color: #4b5563;
      margin-top: 8px;
    }}
    .data-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    .data-table th {{
      background: #e5e7eb;
      text-align: left;
      position: sticky;
      top: 0;
    }}
    .data-table th, .data-table td {{
      border: 1px solid #d1d5db;
      padding: 7px 9px;
      white-space: nowrap;
    }}
    .data-table tr:nth-child(even) {{
      background: #f9fafb;
    }}
    code {{
      background: #e5e7eb;
      padding: 2px 5px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>FIVES Retinal Vessel Segmentation Result Dashboard</h1>
    <p>Source result folder: <code>{source_results}</code></p>
  </header>
  <main>
    <section>
      <h2>Overall Method Ranking</h2>
      <p>Lower average rank is better. Scores use segmentation metrics, processing time, and feature errors.</p>
      {df_to_html(ranking)}
    </section>
    <section>
      <h2>Single Easy Method Comparison Table</h2>
      <p>This table combines the most important metric averages, feature averages, ground truth values, feature errors, and processing time.</p>
      {df_to_html(method_summary)}
    </section>
    <section>
      <h2>Best Method By Criterion</h2>
      {df_to_html(best_methods)}
    </section>
    <section>
      <h2>Branch Comparison Summary</h2>
      {df_to_html(branch_summary)}
    </section>
    <section>
      <h2>Charts</h2>
      <div class="grid">
        {chart_html}
      </div>
    </section>
    <section>
      <h2>Image-Level Comparison Table</h2>
      {image_note}
      {df_to_html(image_table, max_rows=80)}
    </section>
  </main>
</body>
</html>
"""
    (output / "index.html").write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create charts, combined tables, and an HTML dashboard from FIVES pipeline CSV outputs."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=default_results_dir(),
        help="Result folder created by fives_retinal_pipeline.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Dashboard output folder. Default: <results>/visual_report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = args.results.resolve()
    output = (args.output or (results / "visual_report")).resolve()
    charts_dir, tables_dir = ensure_dirs(output)

    source_tables = results / "tables"
    method_df = read_csv(source_tables / "segmentation_metrics_and_features.csv")
    branch_df = read_csv(source_tables / "branch_comparison_metrics.csv")

    method_summary = make_method_summary(method_df)
    branch_summary = make_branch_summary(branch_df)
    best_methods = make_best_methods(method_summary)
    ranking = make_overall_ranking(method_summary)
    image_table = make_image_comparison_table(method_df)

    round_numeric(method_summary).to_csv(tables_dir / "single_easy_method_comparison_table.csv", index=False)
    round_numeric(branch_summary).to_csv(tables_dir / "branch_comparison_summary_table.csv", index=False)
    round_numeric(best_methods).to_csv(tables_dir / "best_method_by_criterion_table.csv", index=False)
    round_numeric(ranking).to_csv(tables_dir / "overall_method_ranking_table.csv", index=False)
    round_numeric(image_table).to_csv(tables_dir / "image_level_method_comparison_table.csv", index=False)

    chart_paths = save_all_charts(method_df, method_summary, branch_summary, ranking, charts_dir)
    make_html_report(
        output=output,
        source_results=results,
        method_summary=method_summary,
        branch_summary=branch_summary,
        best_methods=best_methods,
        ranking=ranking,
        image_table=image_table,
        chart_paths=chart_paths,
    )

    print(f"Dashboard created: {output / 'index.html'}")
    print(f"Combined tables: {tables_dir}")
    print(f"Charts: {charts_dir}")


if __name__ == "__main__":
    main()
