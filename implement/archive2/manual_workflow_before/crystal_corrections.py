"""Apply manual cluster labels and recalculate areas one image at a time."""
from pathlib import Path
import numpy as np
import pandas as pd
from crystal_pipeline import load_run, image_area


def correct_image(run_dir, record, edits, clusters):
    """Return corrected arrays, areas, and an audit trail for one image."""
    name = f"image_{record['image_id']:04d}"
    with np.load(run_dir / 'rf' / f'{name}.npz', allow_pickle=False) as saved:
        result = {key: saved[key] for key in saved.files}
    labels = result['labels']
    manual = np.zeros(len(labels), dtype=bool)
    history = []
    with np.load(run_dir / 'gmm' / f'{name}.npz', allow_pickle=False) as saved:
        if not np.array_equal(labels, saved['labels']):
            raise ValueError(f'GMM labels do not match RF labels: {name}')
        valid, groups = saved['valid_indices'], saved['cluster_labels']
    for row in edits.itertuples(index=False):
        indices = valid[groups == row.cluster_id]
        if not len(indices):
            raise ValueError(f'Empty cluster: image {row.image_id}, cluster {row.cluster_id}')
        mask = (clusters['image_id'] == row.image_id) & (clusters['cluster_id'] == row.cluster_id)
        old_class = clusters.loc[mask, 'final_label'].iloc[0]
        result['final_labels'][indices] = row.new_class
        result['final_confidence'][indices] = 1.0  # Manual assignment, not model probability.
        manual[indices] = True
        clusters.loc[mask, 'final_label'] = row.new_class
        clusters.loc[mask, 'final_majority'] = 1.0
        history.append({'image_id': row.image_id, 'cluster_id': row.cluster_id, 'old_class': old_class,
                        'new_class': row.new_class, 'N_superpixels': len(indices)})
    result['manually_corrected'] = manual
    area = image_area(run_dir, record, labels, result['final_labels'], 'RF labels', result['final_confidence'])
    return result, area, history


def apply_manual_corrections(run_dir):
    """Always start from original RF outputs, so reruns do not accumulate edits."""
    run_dir = Path(run_dir)
    table = pd.read_csv(run_dir / 'review/cluster_corrections.csv', dtype=str, keep_default_na=False)
    edits = table.loc[table['new_class'].str.strip() != '', ['image_id', 'cluster_id', 'new_class']].copy()
    edits['new_class'] = edits['new_class'].str.strip()
    if not edits['new_class'].str.fullmatch(r'[0-9]{1,20}').all():
        raise ValueError('new_class must be a numeric class label of at most 20 characters.')
    for column in ('image_id', 'cluster_id'):
        edits[column] = pd.to_numeric(edits[column], errors='raise')
    clusters = pd.read_csv(run_dir / 'rf/cluster_results.csv', dtype={'stage1_label': str, 'stage2_label': str, 'final_label': str})
    keys = ['image_id', 'cluster_id']
    if edits.duplicated(keys).any() or not pd.MultiIndex.from_frame(edits[keys]).isin(pd.MultiIndex.from_frame(clusters[keys])).all():
        raise ValueError('Correction rows contain duplicate or unknown image/cluster IDs.')
    _, records = load_run(run_dir)
    output_dir = run_dir / 'rf_corrected'
    output_dir.mkdir(parents=True, exist_ok=True)
    areas, history = [], []
    for index, record in enumerate(records, 1):
        name = f"image_{record['image_id']:04d}"
        selected = edits[edits['image_id'] == record['image_id']]
        result, area, changes = correct_image(run_dir, record, selected, clusters)
        np.savez_compressed(output_dir / f'{name}.npz', **result)
        pd.DataFrame([area]).to_csv(output_dir / f'{name}_areas.csv', index=False)
        clusters[clusters['image_id'] == record['image_id']].to_csv(output_dir / f'{name}_clusters.csv', index=False)
        areas.append(area)
        history.extend(changes)
        print(f'[{index}/{len(records)}] {name}: {len(changes)} manual assignments', flush=True)
    pd.DataFrame(areas).to_csv(output_dir / 'area_ratios.csv', index=False)
    clusters.to_csv(output_dir / 'cluster_results.csv', index=False)
    pd.DataFrame(history, columns=['image_id', 'cluster_id', 'old_class', 'new_class', 'N_superpixels']).to_csv(output_dir / 'correction_history.csv', index=False)
    print(f'Applied {len(history)} assignments. Output: {output_dir}')
