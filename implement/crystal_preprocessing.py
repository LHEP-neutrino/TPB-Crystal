"""Crop and segment one image at a time."""
import gc
import json
import joblib
from pathlib import Path
import numpy as np
from skimage import io
from skimage.color import rgb2lab
from skimage.segmentation import slic
from crystal_pipeline import load_run


def preprocess_image(source, output_dir, config):
    """Keep large arrays inside this function and release them after each image."""
    original = io.imread(source)
    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError(f"Expected an RGB image: {source}")
    top, bottom, left, right = (config["crop"][key] for key in ("top", "bottom", "left", "right"))
    height, width = original.shape[:2]
    if min(top, bottom, left, right) < 0 or top + bottom >= height or left + right >= width:
        raise ValueError(f"Invalid crop for image: {source}")
    image = original[top:height-bottom, left:width-right].copy()
    del original
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "image.npy", image, allow_pickle=False)
    lab = rgb2lab(image)
    del image
    # Preserve the original notebook's SLIC calculation and default color handling.
    segments = slic(lab, n_segments=config["n_segments"], compactness=config["compactness"], sigma=1, start_label=1)
    labels = np.unique(segments)
    means = np.empty((len(labels), 3), dtype=np.float64)
    for index, label in enumerate(labels):
        means[index] = lab[segments == label].mean(axis=0)
    del lab
    np.save(output_dir / "segments.npy", segments, allow_pickle=False)
    np.savez(output_dir / "lab_means.npz", labels=labels, lab_means=means)
    return len(labels)


def run_preprocessing(project_root, run_dir):
    """Load saved settings and overwrite preprocessing outputs for selected images."""
    run_dir = Path(run_dir)
    config, records = load_run(run_dir)
    for index, record in enumerate(records, 1):
        output_dir = run_dir / "preprocessing" / f"image_{record['image_id']:04d}"
        count = preprocess_image(Path(project_root) / record["path"], output_dir, config)
        gc.collect()
        print(f"[{index}/{len(records)}] {record['name']}: {count} superpixels", flush=True)
    print(f"Output: {run_dir / 'preprocessing'}")


def run_color_correction(run_dir, reference_path):
    """Use the saved model reference and archived-training float64 Lab means."""
    run_dir = Path(run_dir)
    package = joblib.load(reference_path)
    reference = np.array(package["reference_mean"], dtype=np.float64, copy=True)
    del package
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError("reference_mean must contain three finite Lab values.")
    _, records = load_run(run_dir)
    output_dir = run_dir / "color_correction"
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, 1):
        image_name = f"image_{record['image_id']:04d}"
        with np.load(run_dir / "preprocessing" / image_name / "lab_means.npz", allow_pickle=False) as saved:
            means, labels = saved["lab_means"], saved["labels"]
        if means.shape != (len(labels), 3) or not len(labels) or not np.isfinite(means).all():
            raise ValueError(f"Invalid Lab means: {image_name}")
        if means.dtype != np.float64:
            # Upgrade preprocessing caches written with rounded float32 Lab means.
            directory = run_dir / "preprocessing" / image_name
            lab = rgb2lab(np.load(directory / "image.npy", allow_pickle=False))
            segments = np.load(directory / "segments.npy", mmap_mode="r", allow_pickle=False)
            means = np.array([lab[segments == label].mean(axis=0) for label in labels])
            del lab, segments
        correction = reference - means.mean(axis=0)
        corrected = means + correction
        np.savez(output_dir / f"{image_name}.npz", labels=labels, lab_means=corrected, correction=correction)
        print(f"[{index}/{len(records)}] {image_name}: saved", flush=True)
    settings = {"reference_file": str(reference_path), "reference_mean": reference.tolist(),
                "feature_calculation": "archived_training_v1"}
    (run_dir / "metadata" / "color_reference.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Output: {output_dir}")
