from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


def flatten_columns(columns: Iterable[Tuple[str, str]]) -> List[str]:
    flattened = []
    for first, second in columns:
        flattened.append(first if not second else f"{first}_{second}")
    return flattened


def save_summary_tables(
    output_root: Path,
    method_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    metric_keys: Sequence[str],
) -> None:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    numeric_method_cols = [
        *metric_keys,
        "processing_time_sec",
        "pred_vessel_density_pct",
        "pred_vessel_area_px",
        "pred_vessel_length_px",
        "pred_average_width_px",
        "pred_connected_components",
        "gt_vessel_density_pct",
        "gt_vessel_area_px",
        "gt_vessel_length_px",
        "gt_average_width_px",
        "gt_connected_components",
        "abs_diff_vessel_density_pct",
        "abs_diff_vessel_area_px",
        "abs_diff_vessel_length_px",
        "abs_diff_average_width_px",
        "abs_diff_connected_components",
    ]
    available_method_cols = [
        col for col in numeric_method_cols if col in method_df.columns
    ]
    method_summary = method_df.groupby(["split", "method"])[available_method_cols].agg(
        ["mean", "std"]
    )
    method_summary.columns = flatten_columns(method_summary.columns)
    method_summary = method_summary.reset_index()
    method_summary.to_csv(tables_dir / "method_summary.csv", index=False)

    branch_numeric_cols = [
        *metric_keys,
        "branch_processing_time_sec",
        "branch_pred_pixels",
        "branch_gt_pixels",
    ]
    available_branch_cols = [
        col for col in branch_numeric_cols if col in branch_df.columns
    ]
    branch_summary = branch_df.groupby(["split", "method", "branch"])[
        available_branch_cols
    ].agg(["mean", "std"])
    branch_summary.columns = flatten_columns(branch_summary.columns)
    branch_summary = branch_summary.reset_index()
    branch_summary.to_csv(tables_dir / "branch_summary.csv", index=False)

    mean_df = method_df.groupby(["split", "method"], as_index=False).mean(
        numeric_only=True
    )
    best_rows: List[Dict[str, object]] = []
    for split, split_df in mean_df.groupby("split"):
        for metric in metric_keys:
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
