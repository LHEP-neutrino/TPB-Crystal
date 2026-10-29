# Training

- Main: [superpixel_RF_GMM_training_refactored.ipynb](superpixel_RF_GMM_training_refactored.ipynb).
- Tutorial: [training_tutorial.ipynb](training_tutorial.ipynb).
- Kernel: **Python (crystal_seg)**; run from this folder or the project root.
- Environment setup: [Conda example and kernel registration](../README.md#setup). Conda is optional.

## Main: five-class step1 RF

- Inputs: all 166 images in `../data/mag500/training/`.
- Cells 1–5: shared implement preprocessing and features.
- Cell 6: create run-local GMM settings; manually enter a count for every image before fitting.
- Cells 7–9: inspect GMM groups; optionally enter extra GMM subdivisions and inspect them.
- Cell 10: create blank `labels/cluster_labels.csv`; existing entries are kept.
- Cells 11–13: require complete manual labels, then build/filter the dataset. No historical label table is imported.
- Annotation GMM subdivision remains necessary for those labels; it is not a step2 RF.
- Five groups: 11–14, 21–23, 31–33, 41–44, 51 → RF IDs 0–4.
- Probability cut: 0.70; manual label 0 excluded, label 24 unmapped. Archived sample order is preserved.
- Cells 14–16: RF-only holdout evaluation, final all-data refit, scores and error figures.
- Cell 17: compare RF-only and pre-GMM majority voting on the same held-out rows.
- Cell 18: export one RF and its color reference to `output/step1_only_001/export/implement_bundle/`.
- Results: [step1_only_001](output/step1_only_001/README.md). No step2/step3 RF models are used.

## Required manual input

- Cell 6 creates `output/<run>/review/gmm_settings.csv`. Fill `n_components` for every image and rerun; blank counts stop fitting.
- Inspect Cell 7 for every image. Optional subdivision: edit `review/gmm_subdivision_settings.csv` with `name,parent_cluster,n_components`, then run Cells 8–9. Header only means no subdivision.
- After Cell 10, inspect Cells 7/9 and edit `labels/cluster_labels.csv`:
  - Fill **every** `manual_label` with a nonnegative integer. Use `0` to exclude a group explicitly.
  - `stage1_cluster`/`stage2_cluster` identify GMM groups; do not edit these keys.
  - Blank or invalid labels stop dataset building and RF training before new results are written.
- The current `step1_only_001` contains the recorded GMM settings and 310 filled labels for reproduction. They are preserved; a new run starts blank.

## Evaluation and labels

- Split by source-image hash: identical files stay together; random seed 0, 25% holdout.
- Evaluation RF sees only training images; the exported RF is separately fitted on all eligible rows.
- New datasets: change `RUN_DIR`; old filled labels are never copied automatically.
- Existing entries are not overwritten. After edits, rerun Cells 11–18.
- Scores are superpixel-level agreement with human labels, not pixel-area accuracy.
- Old hierarchical code is in `archive2/`; required current intermediate data are stored inside the active run.

## Tutorial: eight training_sample images

- Inputs: `../data/mag500/training_sample/`; outputs: `output/tutorial_training_sample_001/`.
- Cells 2–5: one-image demo, then preparation of all eight images.
- Cell 6: edit per-image `GMM_COMPONENTS`; inspect GMM 0, 1, … . Full-resolution previews are saved as PDF pages.
- Cell 7: edit `MANUAL_LABELS` in component order; positive IDs are RF classes, 0/None means ignore. Existing labels are retained.
- Cell 8: fit the final eight-image RF using GMM probability ≥0.70.
- Appendix: class extracts, four folds (six train/two held out), RF-only versus majority-vote figures, feature importance.
- No accuracy scores are displayed. Rerun downstream cells after GMM/label changes.
