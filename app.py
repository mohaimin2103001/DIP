from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List

import report_dashboard
import retinal_segmentation_pipeline


ROOT = Path(__file__).resolve().parent
DATASET_DEFAULT = ROOT / "archive"
RESULTS_DEFAULT = ROOT / "results_test"
METHODS = ["global", "otsu", "adaptive"]


def ask_text(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def ask_int(prompt: str, default: int | None = None) -> int | None:
    while True:
        value = ask_text(prompt, "" if default is None else str(default))
        if value == "":
            return None
        try:
            return int(value)
        except ValueError:
            print("Please enter a whole number.")


def ask_float(prompt: str, default: float) -> float:
    while True:
        value = ask_text(prompt, str(default))
        try:
            return float(value)
        except ValueError:
            print("Please enter a number.")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    default_label = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{default_label}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer y or n.")


def ask_choice(prompt: str, options: Iterable[str], default: str) -> str:
    options = list(options)
    option_text = "/".join(options)
    while True:
        value = ask_text(f"{prompt} ({option_text})", default).lower()
        if value in options:
            return value
        print(f"Choose one of: {option_text}")


def ask_methods() -> List[str]:
    print("Available methods: global, otsu, adaptive")
    value = ask_text("Methods to run, comma-separated", ",".join(METHODS))
    methods = [part.strip().lower() for part in value.split(",") if part.strip()]
    invalid = [method for method in methods if method not in METHODS]
    if invalid:
        print(f"Ignoring invalid method(s): {', '.join(invalid)}")
    methods = [method for method in methods if method in METHODS]
    return methods or METHODS.copy()


def run_module_main(module_main, args: List[str]) -> None:
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0], *args]
    try:
        module_main()
    finally:
        sys.argv = old_argv


def run_pipeline(args: List[str]) -> None:
    print("\nRunning segmentation pipeline...\n")
    run_module_main(retinal_segmentation_pipeline.main, args)


def run_dashboard(args: List[str]) -> None:
    print("\nGenerating dashboard...\n")
    run_module_main(report_dashboard.main, args)


def pipeline_args(
    dataset: Path,
    output: Path,
    splits: List[str],
    methods: List[str],
    limit: int | None,
    save_images: bool,
    save_intermediates: bool,
    sample_figures: int,
    global_percentile: float = 92.0,
    adaptive_block_size: int = 201,
    adaptive_c: float = -40.0,
    clahe_clip: float = 2.0,
    gaussian_kernel: int = 5,
    blackhat_kernel: int = 15,
    min_object_area: int = 50,
) -> List[str]:
    args = [
        "--dataset",
        str(dataset),
        "--output",
        str(output),
        "--splits",
        *splits,
        "--methods",
        *methods,
        "--global-percentile",
        str(global_percentile),
        "--adaptive-block-size",
        str(adaptive_block_size),
        "--adaptive-c",
        str(adaptive_c),
        "--clahe-clip",
        str(clahe_clip),
        "--gaussian-kernel",
        str(gaussian_kernel),
        "--blackhat-kernel",
        str(blackhat_kernel),
        "--min-object-area",
        str(min_object_area),
        "--sample-figures",
        str(sample_figures),
    ]
    if limit is not None:
        args.extend(["--limit", str(limit)])
    if save_intermediates:
        args.append("--save-intermediates")
    if not save_images:
        args.append("--no-save-images")
    return args


def dashboard_args(results: Path, output: Path | None = None) -> List[str]:
    args = ["--results", str(results)]
    if output is not None:
        args.extend(["--output", str(output)])
    return args


def run_quick_smoke() -> None:
    output = ROOT / "results_smoke"
    run_pipeline(
        pipeline_args(
            dataset=DATASET_DEFAULT,
            output=output,
            splits=["test"],
            methods=METHODS,
            limit=1,
            save_images=False,
            save_intermediates=False,
            sample_figures=1,
        )
    )
    if ask_yes_no("Generate dashboard for smoke run?", True):
        run_dashboard(dashboard_args(output))


def run_test_set() -> None:
    output = Path(ask_text("Output folder", str(RESULTS_DEFAULT)))
    save_images = ask_yes_no("Save masks/skeletons/overlays?", True)
    run_pipeline(
        pipeline_args(
            dataset=DATASET_DEFAULT,
            output=output,
            splits=["test"],
            methods=METHODS,
            limit=None,
            save_images=save_images,
            save_intermediates=False,
            sample_figures=1,
        )
    )
    if ask_yes_no("Generate dashboard?", True):
        run_dashboard(dashboard_args(output))


def run_custom_pipeline() -> None:
    dataset = Path(ask_text("Dataset folder or zip", str(DATASET_DEFAULT)))
    output = Path(ask_text("Output folder", str(RESULTS_DEFAULT)))
    split_mode = ask_choice("Split", ["test", "train", "both"], "test")
    splits = ["train", "test"] if split_mode == "both" else [split_mode]
    methods = ask_methods()
    limit = ask_int("Image limit per split, blank for full", None)
    save_images = ask_yes_no("Save masks/skeletons/overlays?", True)
    save_intermediates = ask_yes_no("Save preprocessing and morphology intermediates?", False)
    sample_figures = ask_int("Sample comparison figures per split", 1) or 0

    tune = ask_yes_no("Tune advanced numeric parameters?", False)
    kwargs = {}
    if tune:
        kwargs = {
            "global_percentile": ask_float("Global percentile", 92.0),
            "adaptive_block_size": ask_int("Adaptive block size", 201) or 201,
            "adaptive_c": ask_float("Adaptive C", -40.0),
            "clahe_clip": ask_float("CLAHE clip limit", 2.0),
            "gaussian_kernel": ask_int("Gaussian kernel size", 5) or 5,
            "blackhat_kernel": ask_int("Black-hat kernel size", 15) or 15,
            "min_object_area": ask_int("Minimum object area", 50) or 50,
        }

    run_pipeline(
        pipeline_args(
            dataset=dataset,
            output=output,
            splits=splits,
            methods=methods,
            limit=limit,
            save_images=save_images,
            save_intermediates=save_intermediates,
            sample_figures=sample_figures,
            **kwargs,
        )
    )
    if ask_yes_no("Generate dashboard?", True):
        run_dashboard(dashboard_args(output))


def generate_report_only() -> None:
    results = Path(ask_text("Results folder", str(RESULTS_DEFAULT)))
    custom_output = ask_yes_no("Use custom dashboard output folder?", False)
    output = None
    if custom_output:
        output = Path(ask_text("Dashboard output folder", str(results / "visual_report")))
    run_dashboard(dashboard_args(results, output))


def show_project_guide() -> None:
    print(
        f"""
Project files
  retinal_segmentation_pipeline.py   core algorithms and experiment runner
  artifact_writer.py                 visual artifacts: masks, skeletons, overlays
  summary_tables.py                  summary CSV tables
  report_dashboard.py                HTML dashboard and charts
  notebooks/retinal_segmentation_notebook.ipynb

Typical full workflow
  1. Run option 2 for the full test set.
  2. Open results_test/visual_report/index.html.
  3. Use the notebook for presentation-style explanation.

Current default dataset
  {DATASET_DEFAULT}
"""
    )


def print_menu() -> None:
    print(
        """
FIVES Retinal Segmentation App

1. Quick smoke test (1 test image, CSV only)
2. Run full test set
3. Custom pipeline run
4. Generate dashboard/report only
5. Show project guide
0. Exit
"""
    )


def main() -> None:
    while True:
        print_menu()
        choice = ask_text("Choose an option", "1")
        try:
            if choice == "1":
                run_quick_smoke()
            elif choice == "2":
                run_test_set()
            elif choice == "3":
                run_custom_pipeline()
            elif choice == "4":
                generate_report_only()
            elif choice == "5":
                show_project_guide()
            elif choice == "0":
                print("Goodbye.")
                return
            else:
                print("Unknown option.")
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
