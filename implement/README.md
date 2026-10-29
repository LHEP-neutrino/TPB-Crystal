# Implementation

- Main: [superpixel_RF_GMM_implement_refactored.ipynb](superpixel_RF_GMM_implement_refactored.ipynb).
- Tutorial: [implement_tutorial.ipynb](implement_tutorial.ipynb).
- Kernel: **Python (crystal_seg)**; run from this folder or the project root.
- Environment setup: [Conda example and kernel registration](../README.md#setup). Conda is optional.

## Inputs and outputs

- Images: `../data/mag500/{old,new}/plXX/*.jpg`.
- Main uses all 36 plates / 299 images; `images_per_plate=None` keeps every image.
- Model: `../training/output/step1_only_001/export/implement_bundle/models/rf_step1.pkl`.
- Photon inputs and existing GMM-area results: `data/tables/`. Empty-area labels are entered in the run-local CSV; no historical selections are imported.
- Results: `outputs/all_plates_step1_001/`; [summary](outputs/all_plates_step1_001/README.md).

## Main cells

- 1–6: paths, crop/SLIC, model-reference color correction, 18 features, three-component GMM.
- 7: five-class step1 RF; majority vote within each GMM cluster. No child RFs.
- 8–10: optional corrections in `review/cluster_corrections.csv`. Blank `new_class` keeps RF; `reviewed=1` is not required.
- Cells 6, 7 and 10 report every 20 completed plates plus completion (counted by group/plate, not images).
- Cell 9 displays one image example and saves all review figures for the selected plate.
- 11: archive histogram example new/pl17 (four of eight images); stability plot through seven images, one color, no legend.
- 12: four category areas versus photon yield; hide means below 5% **in the plot only**.
- 13: uncorrected photon yield model.
- 14: coverage-corrected fit; first manually fill run-local empty-area selections.
- 15: stacked plate-mean area fractions; nearby bars shift only for display. Top frame at 1; new labels blue, old labels orange.

## Manual review and labeling

- RF inference with a trained model needs no new training labels.
- **Cells 8–10: corrections are optional:**
  - Cell 8 creates `outputs/<run>/review/cluster_corrections.csv` with blank `new_class`.
  - Cell 9 displays one image. Change group/plate to inspect others; all selected review figures are saved.
  - Enter `new_class` only for clusters you want to change. Blank entries keep RF; no `reviewed` column or flag is needed.
  - CSV image/cluster identities must still match the current run.
- **Cell 14 requires manual empty-area selections:**
  - `COVERAGE_RESULT_PATH` must point to GMM-area results for the coverage dataset being used. For new coverage images, run the coverage main first and point to its `gmm/gmm_area_results.csv`.
  - First execution creates blank `review/coverage_settings.csv` for available images in the selected plates.
  - Inspect the corresponding coverage figures and fill `GMMlabel` plus empty `cluster` IDs, or enter an empty fraction in `fluction`.
  - Blank rows are excluded. The corrected fit stops until at least one plate has two cluster-selected images; plates without coverage retain factor 1.
- Current `all_plates_step1_001` keeps its completed results and recorded selections. No review-completion marker is used.
- Use a new `RUN_DIR` for different inputs. New runs receive no historical labels.

## Interpretation

- Photon categories: RF 0=fine, 1=elongated, 2=plate-like, 3=empty; RF 4 is excluded without renormalizing.
- Images are weighted equally within each plate. Each plate contributes one model observation.
- Sampling ranges are not confidence intervals; fit R² is not held-out accuracy.
- `rf_corrected/area_ratios.csv` feeds the photon models; no manual edits means it matches `rf/`.

## Tutorial

- One-image 2D GMM demo, then eight full-resolution old/pl09 images with saved step1 RF.
- Per-class eight-image grids; optional additional GMM pools each RF class across images.
- Additional-GMM assignments ignore pre-RF GMM boundaries; this is tutorial-only.
- Appendix: posterior confidence and RF feature importance. Outputs: `outputs/tutorial_001/`.
- Tutorial uses `data/models/rf_step1.pkl`; the exported main step1 forest reproduces this model.
