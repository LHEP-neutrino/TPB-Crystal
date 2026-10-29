"""Calculate the 18 RF features exactly as in archived training."""
import gc
from pathlib import Path
import numpy as np
from skimage.color import rgb2gray, rgb2lab
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import structure_tensor, structure_tensor_eigenvalues, local_binary_pattern, graycomatrix, graycoprops
from crystal_pipeline import load_run

FEATURE_CALCULATION = "archived_training_v1"

FEATURE_NAMES = ["Variance", "Gradient", "Entropy", "Edge density", "Coherence", "L", "a", "b",
                 "Brightness range", "LBP", "Coherence std", "Coherence p90", "GLCM contrast",
                 "GLCM homogeneity", "GLCM energy", "GLCM correlation", "Edge dominant angle", "Edge angle entropy"]


def calculate_image_features(image_dir, corrected_path):
    """Use weighted grayscale and correct pixel Lab maps before superpixel means."""
    image = np.load(image_dir / "image.npy", allow_pickle=False)
    segments = np.load(image_dir / "segments.npy", allow_pickle=False)
    with np.load(corrected_path, allow_pickle=False) as saved:
        corrected_sp_lab_means, corrected_labels = saved["lab_means"], saved["labels"]
        correction = saved["correction"]
    if image.shape[:2] != segments.shape or not np.array_equal(np.unique(segments), corrected_labels):
        raise ValueError(f"Image, segments, or corrected labels do not match: {image_dir}")
    edge_threshold = 0.05  # Same threshold as the original training features.
    # Preserve archived training's float32 pixel arithmetic and operation order.
    image_lab = rgb2lab(image).astype(np.float32)
    for channel in range(3):
        image_lab[:, :, channel] += correction[channel]
    gray = rgb2gray(image).astype(np.float32)
    del image
    gray_uint = np.clip(gray * 255, 0, 255).astype(np.uint8)
    entropy_img = entropy(gray_uint, disk(3)).astype(np.float32)
    gy, gx = np.gradient(gray)
    gradient_img = np.sqrt(gx**2 + gy**2).astype(np.float32)
    edge_angle_img = np.arctan2(gy, gx).astype(np.float32)
    A = structure_tensor(gray, sigma=2)
    l1, l2 = structure_tensor_eigenvalues(A)
    coherence_img = ((l1 - l2) / (l1 + l2 + 1e-10)).astype(np.float32)
    del A, l1, l2, gy, gx
    lbp_img = local_binary_pattern(gray_uint, P=8, R=1, method="uniform").astype(np.float32)
    unique_labels = np.unique(segments)
    if len(unique_labels) != len(corrected_sp_lab_means):
        raise ValueError("Superpixel count does not match corrected Lab means.")
    image_features = {name: np.zeros(len(unique_labels), dtype=np.float32) for name in FEATURE_NAMES}
    for sp_idx, label in enumerate(unique_labels):
        mask = segments == label
        n_pixels = np.sum(mask)
        if n_pixels < 2:
            continue
        values = gray[mask]
        coherence_values = coherence_img[mask]
        angles = edge_angle_img[mask]
        hist, bins = np.histogram(angles, bins=18, range=(-np.pi, np.pi))
        p = hist / (np.sum(hist) + 1e-10)
        ys, xs = np.where(mask)
        crop = gray_uint[ys.min():ys.max()+1, xs.min():xs.max()+1]
        if crop.size > 4:
            glcm = graycomatrix(crop, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            glcm_contrast = graycoprops(glcm, "contrast")[0, 0]
            glcm_homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
            glcm_energy = graycoprops(glcm, "energy")[0, 0]
            glcm_correlation = graycoprops(glcm, "correlation")[0, 0]
            del glcm
        else:
            glcm_contrast = glcm_homogeneity = glcm_energy = glcm_correlation = 0.0
        for channel, name in enumerate(("L", "a", "b")):
            image_features[name][sp_idx] = np.mean(image_lab[:, :, channel][mask])
        image_features["Brightness range"][sp_idx] = np.max(values) - np.min(values)
        image_features["Variance"][sp_idx] = np.var(values)
        image_features["Gradient"][sp_idx] = np.mean(gradient_img[mask])
        image_features["Entropy"][sp_idx] = np.mean(entropy_img[mask])
        image_features["Edge density"][sp_idx] = np.sum(gradient_img[mask] > edge_threshold) / n_pixels
        image_features["LBP"][sp_idx] = np.mean(lbp_img[mask])
        image_features["Coherence"][sp_idx] = np.mean(coherence_values)
        image_features["Coherence std"][sp_idx] = np.std(coherence_values)
        image_features["Coherence p90"][sp_idx] = np.percentile(coherence_values, 90)
        image_features["GLCM contrast"][sp_idx] = glcm_contrast
        image_features["GLCM homogeneity"][sp_idx] = glcm_homogeneity
        image_features["GLCM energy"][sp_idx] = glcm_energy
        image_features["GLCM correlation"][sp_idx] = glcm_correlation
        image_features["Edge dominant angle"][sp_idx] = bins[np.argmax(hist)]
        image_features["Edge angle entropy"][sp_idx] = -np.sum(p * np.log(p + 1e-10))

    features = np.column_stack([image_features[name] for name in FEATURE_NAMES])
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite feature values: {image_dir}")
    return unique_labels, features


def run_feature_extraction(run_dir):
    """Save archived-training features one image at a time."""
    run_dir = Path(run_dir)
    _, records = load_run(run_dir)
    output_dir = run_dir / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, 1):
        name = f"image_{record['image_id']:04d}"
        labels, features = calculate_image_features(run_dir / "preprocessing" / name, run_dir / "color_correction" / f"{name}.npz")
        np.savez(output_dir / f"{name}.npz", labels=labels, feature_names=np.array(FEATURE_NAMES), features=features,
                 feature_calculation=FEATURE_CALCULATION)
        print(f"[{index}/{len(records)}] {name}: {features.shape[0]} superpixels, {features.shape[1]} features", flush=True)
        del labels, features
        gc.collect()
    print(f"Output: {output_dir}")
