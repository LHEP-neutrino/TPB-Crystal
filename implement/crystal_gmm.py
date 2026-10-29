"""Group superpixels by their features, fitting a separate GMM for each image."""
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from crystal_features import FEATURE_NAMES
from crystal_pipeline import load_run, plate_progress_points


def fit_image_gmm(feature_path, n_components, pca_components, probability_cut):
    """Return cluster assignments for valid feature rows, preserving row indices."""
    with np.load(feature_path, allow_pickle=False) as saved:
        labels, features = saved["labels"], saved["features"]
        names = saved["feature_names"].tolist()
    if features.shape != (len(labels), len(names)) or names != FEATURE_NAMES:
        raise ValueError(f"Feature shape or order does not match: {feature_path}")
    return fit_gmm_features(labels, features, n_components, pca_components, probability_cut)


def fit_gmm_features(labels, features, n_components, pca_components, probability_cut):
    """Run the same calculation for a full image or a selected cluster."""
    X = features.astype(np.float64)
    valid_indices = np.where(np.all(np.isfinite(X), axis=1))[0]
    X = X[valid_indices]
    if len(X) < max(n_components, 2):
        raise ValueError('Not enough valid superpixels for GMM.')

    X_scaled = StandardScaler().fit_transform(X)
    n_pca = min(pca_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_pca, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=0, n_init=5, reg_covar=1e-5)
    gmm.fit(X_pca)
    probabilities = gmm.predict_proba(X_pca)
    confidence = probabilities.max(axis=1)
    return {"labels": labels, "valid_indices": valid_indices, "cluster_labels": probabilities.argmax(axis=1),
            "probabilities": probabilities, "max_probability": confidence, "high_mask": confidence >= probability_cut,
            "n_components": n_components, "pca_components": n_pca,
            "explained_variance": pca.explained_variance_ratio_.sum(), "probability_cut": probability_cut}


def run_gmm(run_dir, n_components=3, pca_components=15, probability_cut=0.90, report_every_plates=20):
    """Save one GMM result per image; rerunning overwrites the selected outputs."""
    if n_components < 1 or pca_components < 1 or not 0 <= probability_cut <= 1:
        raise ValueError("Component counts must be positive and probability_cut must be between 0 and 1.")
    run_dir = Path(run_dir)
    _, records = load_run(run_dir)
    progress = plate_progress_points(records, report_every_plates)
    output_dir = run_dir / "gmm"
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, 1):
        name = f"image_{record['image_id']:04d}"
        result = fit_image_gmm(run_dir / "features" / f"{name}.npz", n_components, pca_components, probability_cut)
        np.savez_compressed(output_dir / f"{name}.npz", **result)
        if index in progress:
            count, total = progress[index]
            print(f'GMM complete: {count}/{total} plates ({index}/{len(records)} images)', flush=True)
    print(f"Output: {output_dir}")
