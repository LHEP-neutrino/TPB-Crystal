"""Training image selection using the implementation pipeline's metadata format."""
import json
from pathlib import Path
import pandas as pd
from crystal_pipeline import _natural_key


def prepare_training_run(project_root, run_dir, config):
    """Save naturally sorted image paths and settings without loading images."""
    project_root, run_dir = Path(project_root), Path(run_dir)
    data_dir = project_root / config['data_dir']
    paths = sorted((p for p in data_dir.glob('*.jpg') if p.is_file()), key=_natural_key)
    limit = config.get('max_images')
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError('max_images must be a positive integer or None.')
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f'No JPG images found in: {data_dir}')
    # Source group/plate information is not encoded in this flat training folder.
    records = [{'image_id': index, 'group': 'training', 'plate': '', 'name': path.stem,
                'path': path.relative_to(project_root).as_posix()} for index, path in enumerate(paths)]
    metadata_dir = run_dir / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(metadata_dir / 'images.csv', index=False)
    (metadata_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print(f'Selected images: {len(records)} | Metadata: {metadata_dir}')


def run_training_gmm(run_dir, settings_csv, pca_components=2, probability_cut=0.90):
    """Fit the shared GMM calculation with a component count for each image."""
    import numpy as np
    from crystal_gmm import fit_image_gmm
    from crystal_pipeline import load_run

    if pca_components < 1 or not 0 <= probability_cut <= 1:
        raise ValueError('PCA components must be positive and probability_cut must be between 0 and 1.')
    run_dir = Path(run_dir)
    _, records = load_run(run_dir)
    settings = pd.read_csv(settings_csv, dtype={'name': str})
    counts = pd.to_numeric(settings['n_components'], errors='raise')
    if not ((counts >= 1) & (counts % 1 == 0)).all():
        raise ValueError('n_components must contain positive integers.')
    selected = pd.DataFrame(records).merge(settings[['name', 'n_components']], on='name', how='left', validate='one_to_one')
    if selected.n_components.isna().any():
        raise ValueError('GMM settings are missing for one or more selected images.')
    output_dir = run_dir / 'gmm' / 'stage1'
    output_dir.mkdir(parents=True, exist_ok=True)
    selected[['image_id', 'name', 'n_components']].to_csv(output_dir / 'settings.csv', index=False)
    for index, record in enumerate(selected.to_dict('records'), 1):
        name = f"image_{record['image_id']:04d}"
        count = int(record['n_components'])
        result = fit_image_gmm(run_dir / 'features' / f'{name}.npz', count, pca_components, probability_cut)
        np.savez_compressed(output_dir / f'{name}.npz', **result)
        print(f"[{index}/{len(selected)}] {record['name']}: {count} clusters, {result['high_mask'].sum()} high confidence", flush=True)
    print(f'Output: {output_dir}')


def run_training_gmm_stage2(run_dir, settings_csv, pca_components=2, probability_cut=0.90):
    """Split selected stage-1 clusters; indices remain local to the full image."""
    import numpy as np
    from crystal_gmm import fit_gmm_features
    from crystal_features import FEATURE_NAMES
    from crystal_pipeline import load_run

    run_dir = Path(run_dir)
    _, records = load_run(run_dir)
    settings = pd.read_csv(settings_csv, dtype={'name': str})
    if settings.duplicated(['name', 'parent_cluster']).any():
        raise ValueError('Stage-2 settings contain duplicate image/cluster pairs.')
    selected = pd.DataFrame(records).merge(settings, on='name', how='inner', validate='one_to_many')
    output_dir = run_dir / 'gmm/stage2'
    output_dir.mkdir(parents=True, exist_ok=True)
    selected[['image_id', 'name', 'parent_cluster', 'n_components']].to_csv(output_dir / 'settings.csv', index=False)
    for row in selected.itertuples(index=False):
        name = f'image_{row.image_id:04d}'
        parent, count = int(row.parent_cluster), int(row.n_components)
        if parent != row.parent_cluster or count != row.n_components or parent < 0 or count < 1:
            raise ValueError('parent_cluster must be a nonnegative integer and n_components a positive integer.')
        with np.load(run_dir / 'features' / f'{name}.npz', allow_pickle=False) as saved:
            labels, features = saved['labels'], saved['features']
            if saved['feature_names'].tolist() != FEATURE_NAMES:
                raise ValueError(f'Unexpected feature order: {name}')
        with np.load(run_dir / 'gmm/stage1' / f'{name}.npz', allow_pickle=False) as saved:
            if not np.array_equal(labels, saved['labels']) or parent >= int(saved['n_components']):
                raise ValueError(f'Stage-1 labels or parent cluster do not match: {name}')
            indices = saved['valid_indices'][saved['cluster_labels'] == parent]
        result = fit_gmm_features(labels[indices], features[indices], count, pca_components, probability_cut)
        result['valid_indices'] = indices[result['valid_indices']]
        result['labels'] = labels
        result['parent_cluster'] = parent
        np.savez_compressed(output_dir / f'{name}_cluster_{parent:04d}.npz', **result)
        print(f'{name}, parent cluster {parent}: {len(result["valid_indices"])} SP -> {count} clusters', flush=True)
    print(f'Completed {len(selected)} stage-2 targets. Output: {output_dir}')
