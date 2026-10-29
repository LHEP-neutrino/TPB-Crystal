# Dune: crystal image analysis

- Use **Python (crystal_seg)**; start Jupyter from this folder or a project folder.
- Run notebook cells in order. Source images stay in `data/`.
- Current RF workflow uses **step1 only: five classes**. Step2/step3 RF code is in `archive2/`.

## Workflows

- [Training](training/README.md): train/evaluate the five-class RF on 166 images.
- [Implementation](implement/README.md): classify 299 images from 36 plates; fit photon yield models.
- [Coverage](coverage/README.md): estimate coverage from 287 higher-magnification images.
- Tutorials: [implementation](implement/implement_tutorial.ipynb), [training](training/training_tutorial.ipynb), [coverage](coverage/coverage_tutorial.ipynb).

## Environment Setup

```bash
conda create -n crystal_seg python=3.10
conda activate crystal_seg
python --version
```

```bash
conda install -c conda-forge opencv scikit-image matplotlib numpy pandas scipy scikit-learn joblib
conda install -c conda-forge pygments
conda install -c conda-forge jupyter ipykernel nbformat nbclient
```

```bash
python -m ipykernel install --user --name crystal_seg --display-name "Python (crystal_seg)"
```

- Open the `.ipynb` file from the corresponding project folder and select **Python (crystal_seg)** as the kernel.

## Run and share

- Run training first, or include its exported model bundle: `training/output/step1_only_001/export/implement_bundle/`.
- Implementation and training share the archived-training 18-feature calculation and saved RF Lab reference.
- Keep `implement/`, `training/`, and `coverage/` together; modules and small input tables are shared.
- Main outputs: `implement/outputs/all_plates_step1_001/`, `training/output/step1_only_001/`, `coverage/output/analysis_001/`.
- Tutorial outputs use separate folders. Old experiments and test outputs have been removed.
- Keep source images, input models/tables and manual labels with the code, or provide them separately when sharing.
- Change the run folder when changing inputs/settings. Rerun downstream cells after edits.
- `archive/` and `archive2/` are historical code snapshots.
