"""Prepare editable ground-truth labels for the final training GMM groups."""
from pathlib import Path
import numpy as np
import pandas as pd
from crystal_pipeline import load_run


def prepare_training_labels(run_dir):
    """List unsplit stage-1 groups and split stage-2 children, without overlap."""
    run_dir = Path(run_dir)
    output = run_dir / 'labels' / 'cluster_labels.csv'
    if output.exists():
        print(f'Existing manual labels kept: {output}')
        return output
    _, records = load_run(run_dir)
    settings = pd.read_csv(run_dir / 'gmm/stage2/settings.csv')
    if settings.duplicated(['image_id', 'parent_cluster']).any():
        raise ValueError('Stage-2 settings contain duplicate image/parent pairs.')
    rows = []
    for record in records:
        name = f"image_{record['image_id']:04d}"
        with np.load(run_dir / 'gmm/stage1' / f'{name}.npz', allow_pickle=False) as saved:
            labels, valid, clusters = saved['labels'], saved['valid_indices'], saved['cluster_labels']
            count = int(saved['n_components'])
            probability = np.full(len(labels), np.nan)
            probability[valid] = saved['max_probability']
        seen = np.zeros(len(labels), dtype=int)

        def add_group(parent, child, indices, probability2):
            if not len(indices):
                return
            seen[indices] += 1
            p1 = probability[indices]
            rows.append({'image_id': record['image_id'], 'name': record['name'], 'stage1_cluster': parent,
                         'stage2_cluster': child, 'n_superpixels': len(indices),
                         'stage1_mean_probability': p1.mean(), 'stage2_mean_probability': np.mean(probability2),
                         'mean_gmm_probability': np.mean(p1 * probability2), 'manual_label': ''})

        parents = settings.loc[settings.image_id == record['image_id'], 'parent_cluster'].tolist()
        if any(parent not in range(count) for parent in parents):
            raise ValueError(f'Unknown stage-1 parent in stage-2 settings: {name}')
        for parent in range(count):
            original = valid[clusters == parent]
            if parent not in parents:
                add_group(parent, '', original, np.ones(len(original)))
                continue
            with np.load(run_dir / 'gmm/stage2' / f'{name}_cluster_{parent:04d}.npz', allow_pickle=False) as saved:
                indices, children, prob2 = saved['valid_indices'], saved['cluster_labels'], saved['max_probability']
                if not np.array_equal(labels, saved['labels']) or not np.isin(indices, original).all():
                    raise ValueError(f'Stage-2 labels do not match their stage-1 parent: {name}')
                for child in range(int(saved['n_components'])):
                    mask = children == child
                    add_group(parent, child, indices[mask], prob2[mask])
        if not np.all(seen[valid] == 1) or np.any(seen > 1):
            raise ValueError(f'Training groups overlap or omit stage-1 superpixels: {name}')
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f'Saved {len(rows)} groups from {len(records)} images: {output}')
    print('Fill every manual_label after reviewing the figures. Blank entries stop training; use 0 to exclude a group.')
    print('Cluster IDs are zero-based; an empty stage2_cluster means no second-stage split.')
    return output



def require_training_labels(run_dir):
    """Stop before dataset/model output if any human group label is missing."""
    path = Path(run_dir) / 'labels/cluster_labels.csv'
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    _, records = load_run(run_dir)
    if set(zip(table.image_id, table['name'])) != {(str(r['image_id']), r['name']) for r in records}:
        raise ValueError('Manual labels do not match the selected images. Use a new run for different inputs.')
    labels = table['manual_label'].str.strip()
    if table.empty or labels.eq('').any():
        raise ValueError(f'Manual labeling required: fill all manual_label entries in {path}. Use 0 to exclude a group; then rerun from Cell 11.')
    if not labels.str.fullmatch(r'[0-9]+').all():
        raise ValueError(f'manual_label must contain nonnegative integers: {path}')
    return path
