# Coverage

- Main: [coverage_refactored.ipynb](coverage_refactored.ipynb).
- Tutorial: [coverage_tutorial.ipynb](coverage_tutorial.ipynb).
- Kernel: **Python (crystal_seg)**
- Environment setup: [Conda example and kernel registration](../README.md#setup). Conda is optional.

## Main

- Images: `../data/mag2000/{old,new}/plXX/`; 36 plates / 287 images, up to eight per plate.
- Markdown guidance precedes each code cell.
- Cells 1–5: crop, six pixel features, three GMM methods, review figures.
- Cells 4–5 report every 20 images; Cell 5 displays one example and saves all figures.
- Cell 6: create a blank settings CSV for manual entry; preserve values if they are already entered.
  - after run this cell, **"manual label is needed"**
  - Inspect the output of Cell 5 `output/analysis_001/review/figures`
  - Edit `output/analysis_001/review/gmm_settings.csv` to select the clusters representing empty regions.
  - GMMlabel` = 0 (2 clusters), 1 (4 clusters), or 2 (2→2); `cluster` lists empty cluster IDs.
  - Alternatively use `fluction` for an empty fraction (0–1); cluster selections take precedence.
- Cells 7–8: coverage = 1 − empty fraction; plate summaries and photon-yield comparison. Cell 7 displays the first five image/plate rows; full results stay in CSV.
- `fluction` is manual empty-fraction input only; it stays blank when selecting clusters. Calculated results are in `empty_ratio` and `coverage`.
- Results: `output/analysis_001/`.

## Rules

- Blank selections are skipped. Plate/source summaries require at least two selected images.
- Optional plate-level manual empty fractions: `review/plate_manual_settings.csv` (`group`, `plate`, `fluction`). These are not copied to images or used in the cluster-based photon fit.
- Error ranges use leave-one-image-out means, not confidence intervals.
- To use new coverage in implement, supply matching `review/gmm_settings.csv` and `gmm/gmm_area_results.csv` from this run.
- Rerun downstream cells after editing selections; use a new run folder for different processing settings.

## Tutorial

- One cropped image at original resolution; six features, 2D PCA, EM snapshots, editable empty clusters.
- Feature calculations and all tutorial processing are written inside the notebook.
- Appendix: posterior map, confidence histograms, probability margins, feature distributions.
- Tutorial uses a full-covariance 2D GMM; main uses six-feature diagonal GMMs.
- Outputs: `output/tutorial_001/`.
