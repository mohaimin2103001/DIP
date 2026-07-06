# FIVES Retinal Vessel Segmentation DIP Pipeline

This project implements the complete classical Digital Image Processing pipeline for the FIVES retinal vessel dataset:

- Green channel extraction
- Image quality and histogram analysis
- CLAHE contrast enhancement
- Gaussian filtering
- Morphological black-hat transform
- Contrast stretching
- Global, Otsu, and adaptive thresholding
- Morphological opening, closing, hole filling, and connected-component cleanup
- Skeletonization
- Ground truth feature extraction
- Segmentation vs ground truth comparison using Dice, IoU, precision, recall, F1, and accuracy
- Dataset-wide summaries, best-method selection, CSV reports, overlays, and dashboard figures

## Dataset Layout

The script expects the extracted archive to look like this:

```text
archive/
  train/
    Original/
    Ground truth/
  test/
    Original/
    Ground truth/
```

`archive.zip` is also accepted with `--dataset archive.zip`; the script will extract it if needed.

## Quick Run

Run the guided CLI app:

```bash
python app.py
```

The app can run a smoke test, full test-set experiment, custom experiment, and dashboard generation from one menu.

## Direct Commands

Run a small verification on two test images:

```bash
python retinal_segmentation_pipeline.py --dataset archive --output results_demo --splits test --limit 2
```

Run the full test set:

```bash
python retinal_segmentation_pipeline.py --dataset archive --output results_test --splits test
```

Run the full train and test dataset:

```bash
python retinal_segmentation_pipeline.py --dataset archive --output results_full --splits train test
```

For CSV-only runs without writing visual PNG outputs:

```bash
python retinal_segmentation_pipeline.py --dataset archive --output results_csv --splits test --no-save-images
```

## Main Outputs

```text
results/
  masks/                         final refined binary masks
  skeletons/                     skeleton maps
  overlays/                      prediction vs ground truth overlays
  figures/                       sample comparison figures
  tables/
    segmentation_metrics_and_features.csv
    branch_comparison_metrics.csv
    ground_truth_features.csv
    method_summary.csv
    branch_summary.csv
    best_methods.csv
```

Overlay color meaning:

- Yellow: true positive vessel pixels
- Blue: false positive vessel pixels
- Red: false negative vessel pixels

## Useful Parameters

- `--global-percentile 92.0`: percentile used for global thresholding
- `--adaptive-block-size 201`: local window size for adaptive thresholding
- `--adaptive-c -40.0`: adaptive threshold constant
- `--blackhat-kernel 15`: kernel size for black-hat vessel enhancement
- `--min-object-area 50`: removes connected components smaller than this pixel area
- `--save-intermediates`: saves preprocessing and morphology stage images

The default split is `test` so that a normal run finishes faster. Use `--splits train test` for the complete dataset.

## Visualize CSV Results

After running the main pipeline, create an easy graphical dashboard from the CSV files:

```bash
python report_dashboard.py --results results_test
```

The dashboard is saved at:

```text
results_test/visual_report/index.html
```

It also creates combined CSV tables:

```text
results_test/visual_report/tables/
  single_easy_method_comparison_table.csv
  skeleton_branch_summary_table.csv
  best_method_by_criterion_table.csv
  overall_method_ranking_table.csv
  image_level_method_comparison_table.csv
```

## Notebook Presentation

A presentation-friendly notebook is available at:

```text
notebooks/retinal_segmentation_notebook.ipynb
```

It shows the first 3 selected images step by step, then summarizes all 200 test images with tables, charts, representative best/median/worst examples, and output folder locations.
