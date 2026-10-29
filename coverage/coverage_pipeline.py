"""Disk-based pixel GMM coverage processing, matching the original six features."""
import gc
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io
from skimage.color import rgb2lab
from scipy.ndimage import gaussian_filter, uniform_filter
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


def load_run(run_dir):
    run_dir = Path(run_dir)
    return json.loads((run_dir / 'metadata/config.json').read_text()), pd.read_csv(run_dir / 'metadata/images.csv')


def prepare_run(project_root, run_dir, config):
    """Save an explicit manifest. Choose a new run directory when inputs change."""
    root, run_dir = Path(project_root), Path(run_dir)
    rows = []
    for group, plates in config['target_plates'].items():
        for plate in plates:
            folder = root / config['data_dir'] / group / plate
            if not folder.is_dir():
                raise FileNotFoundError(f'Image directory not found: {folder}')
            paths = sorted((p for p in folder.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png') and p.is_file()),
                           key=lambda p: [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', p.name)])
            limit = config['images_per_plate']
            if limit is not None and (not isinstance(limit, int) or limit < 1):
                raise ValueError('images_per_plate must be a positive integer or None.')
            paths = paths[:limit]
            if not paths:
                raise ValueError(f'No supported images found: {folder}')
            for p in paths:
                rows.append(dict(image_id=len(rows), group=group, plate=plate, name=p.stem, path=str(p.relative_to(root))))
    if not rows:
        raise ValueError('No images selected.')
    metadata = run_dir / 'metadata'
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    pd.DataFrame(rows).to_csv(metadata / 'images.csv', index=False)
    print(f'Selected {len(rows)} images. Metadata: {metadata}')


def crop_images(project_root, run_dir):
    config, images = load_run(run_dir)
    output = Path(run_dir) / 'images'
    output.mkdir(parents=True, exist_ok=True)
    crop = config['crop']
    for row in images.itertuples(index=False):
        image = io.imread(Path(project_root) / row.path)
        h, w = image.shape[:2]
        top, bottom, left, right = [crop[k] for k in ('top', 'bottom', 'left', 'right')]
        if min(top, bottom, left, right) < 0 or top + bottom >= h or left + right >= w:
            raise ValueError(f'Invalid crop for {row.name}: {image.shape}')
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f'Expected an 8-bit RGB image: {row.name}')
        np.save(output / f'image_{row.image_id:04d}.npy', image[top:h-bottom, left:w-right])
        del image
    print(f'Saved {len(images)} cropped images: {output}')


def feature_maps(image):
    lab = rgb2lab(image).astype(np.float32)
    lightness = lab[..., 0] / 100.0
    del lab
    brightness = (image.astype(np.float32) / 255.0).mean(axis=2)
    return [gaussian_filter(lightness, sigma=s, mode='reflect').astype(np.float32) for s in (1, 3)] + [
        uniform_filter(values, size=size, mode='reflect').astype(np.float32)
        for values in (lightness, brightness) for size in (5, 11)]


def predict_pixels(model, maps, scaler, chunk_size, indices=None):
    n = maps[0].size if indices is None else len(indices)
    labels = np.empty(n, dtype=np.uint8)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        selection = slice(start, end) if indices is None else indices[start:end]
        chunk = np.column_stack([a.ravel()[selection] for a in maps]).astype(np.float32)
        labels[start:end] = model.predict(scaler.transform(chunk).astype(np.float32))
    return labels


def fit_image(image, config):
    """Fit on sampled pixels; predict all pixels in bounded chunks, without PCA."""
    maps = feature_maps(image)
    n = image.shape[0] * image.shape[1]
    sample_size, chunk_size = config['max_sample_pixels'], config['predict_chunk_size']
    if sample_size < 4 or chunk_size < 1 or n < 4:
        raise ValueError('Need at least four pixels/sample pixels and a positive prediction chunk size.')
    indices = np.random.default_rng(config['random_state']).choice(n, min(sample_size, n), replace=False)
    sample = np.column_stack([a.ravel()[indices] for a in maps]).astype(np.float32)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(sample).astype(np.float32)
    def fit(values, count):
        return GaussianMixture(n_components=count, covariance_type='diag', random_state=config['random_state'],
                               n_init=1, max_iter=100).fit(values)
    parent = fit(scaled, 2)
    labels2 = predict_pixels(parent, maps, scaler, chunk_size)
    labels4 = predict_pixels(fit(scaled, 4), maps, scaler, chunk_size)
    sample_parent = parent.predict(scaled)
    hierarchical = np.empty(n, dtype=np.uint8)
    for k in range(2):
        child_sample = scaled[sample_parent == k]
        pixel_indices = np.flatnonzero(labels2 == k)
        if len(child_sample) < 2:
            hierarchical[pixel_indices] = k * 2
        else:
            child = fit(child_sample, 2)
            hierarchical[pixel_indices] = k * 2 + predict_pixels(child, maps, scaler, chunk_size, pixel_indices)
    shape = image.shape[:2]
    return [a.reshape(shape) for a in (labels2, labels4, hierarchical)]


def run_gmm(run_dir, report_every=20):
    """Save every image result; report progress at the requested image interval."""
    if not isinstance(report_every, int) or report_every < 1:
        raise ValueError('report_every must be a positive integer.')
    config, images = load_run(run_dir)
    output = Path(run_dir) / 'gmm'
    output.mkdir(parents=True, exist_ok=True)
    # Rebuild this small CSV on every run to avoid duplicate appended rows.
    columns = ['image_id', 'group', 'plate', 'name', 'gmm_label', 'cluster_id', 'area_ratio']
    csv_path = output / 'gmm_area_results.csv'
    pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
    for index, row in enumerate(images.itertuples(index=False), 1):
        image = np.load(Path(run_dir) / 'images' / f'image_{row.image_id:04d}.npy')
        labels = fit_image(image, config)
        np.savez_compressed(output / f'image_{row.image_id:04d}.npz', gmm_0=labels[0], gmm_1=labels[1], gmm_2=labels[2])
        records = []
        for method, (a, count) in enumerate(zip(labels, (2, 4, 4))):
            fractions = np.bincount(a.ravel(), minlength=count) / a.size
            records.extend(dict(image_id=row.image_id, group=row.group, plate=row.plate, name=row.name,
                                gmm_label=method, cluster_id=k, area_ratio=value) for k, value in enumerate(fractions))
        pd.DataFrame(records, columns=columns).to_csv(csv_path, mode='a', index=False, header=False)
        if index % report_every == 0 or index == len(images):
            print(f'GMM complete: {index}/{len(images)} images', flush=True)
        del image, labels
        gc.collect()
    return csv_path


def prepare_settings(run_dir):
    _, images = load_run(run_dir)
    path = Path(run_dir) / 'review/gmm_settings.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        settings = images[['image_id', 'group', 'plate', 'name']].rename(columns={'image_id': 'imageID'})
        settings['GMMlabel'], settings['cluster'], settings['fluction'] = '', '', ''
        settings.to_csv(path, index=False)
    manual_path = path.parent / 'plate_manual_settings.csv'
    if not manual_path.exists():
        pd.DataFrame(columns=['group', 'plate', 'fluction']).to_csv(manual_path, index=False)
    print('Inspect Cell 5, then enter GMMlabel and empty cluster IDs in the CSV. Rerun Cells 7–8 after editing.')
    print(f'Edit cluster (comma-separated IDs) or fluction (manual empty fraction, 0–1): {path}')
    print('GMMlabel: 0 = two clusters; 1 = four clusters; 2 = hierarchical 2 -> 2. Existing settings are preserved.')
    return path

