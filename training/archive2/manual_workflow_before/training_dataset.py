"""Build and inspect superpixel training tables from saved features and human labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from crystal_features import FEATURE_NAMES
from crystal_pipeline import load_run


def _image_rows(run_dir, record, groups, split_parents):
    name = f"image_{record['image_id']:04d}"
    with np.load(run_dir / 'features' / f'{name}.npz', allow_pickle=False) as saved:
        labels, features = saved['labels'], saved['features']
        if saved['feature_names'].tolist() != FEATURE_NAMES:
            raise ValueError(f'Unexpected feature order: {name}')
    with np.load(run_dir / 'gmm/stage1' / f'{name}.npz', allow_pickle=False) as saved:
        if not np.array_equal(labels, saved['labels']):
            raise ValueError(f'GMM and feature labels do not match: {name}')
        valid, clusters = saved['valid_indices'], saved['cluster_labels']
        p1 = np.full(len(labels), np.nan)
        p1[valid] = saved['max_probability']
        count = int(saved['n_components'])
    expected, tables, seen = set(), [], np.zeros(len(labels), dtype=int)
    for parent in range(count):
        indices = valid[clusters == parent]
        if parent in split_parents:
            with np.load(run_dir / 'gmm/stage2' / f'{name}_cluster_{parent:04d}.npz', allow_pickle=False) as saved:
                if not np.array_equal(labels, saved['labels']) or not np.isin(saved['valid_indices'], indices).all():
                    raise ValueError(f'Stage-2 regions do not match stage 1: {name}')
                children = [(child, saved['valid_indices'][saved['cluster_labels'] == child],
                             saved['max_probability'][saved['cluster_labels'] == child]) for child in range(int(saved['n_components']))]
        else:
            children = [(-1, indices, np.ones(len(indices)))]
        for child, indices, p2 in children:
            if not len(indices):
                continue
            expected.add((parent, child))
            row = groups[(groups.stage1_cluster == parent) & (groups.stage2_cluster == child)]
            if len(row) != 1 or int(row.iloc[0].n_superpixels) != len(indices):
                raise ValueError(f'Label table does not match GMM groups: {name}, {parent}/{child}')
            manual = str(row.iloc[0].manual_label).strip()
            if manual and (not manual.isascii() or not manual.isdecimal()):
                raise ValueError(f'manual_label must be a nonnegative integer or blank: {name}')
            seen[indices] += 1
            frame = pd.DataFrame(features[indices], columns=FEATURE_NAMES)
            metadata = {'image_id': record['image_id'], 'name': record['name'], 'path': record['path'],
                        'superpixel_index': indices, 'superpixel_label': labels[indices], 'stage1_cluster': parent,
                        'stage2_cluster': child, 'manual_label': int(manual) if manual else pd.NA,
                        'stage1_probability': p1[indices], 'stage2_probability': p2, 'gmm_probability': p1[indices] * p2}
            for key, value in metadata.items():
                frame[key] = value
            frame['manual_label'] = frame['manual_label'].astype('Int64')
            tables.append(frame)
    if set(zip(groups.stage1_cluster, groups.stage2_cluster)) != expected or not np.all(seen[valid] == 1):
        raise ValueError(f'Label groups overlap, omit regions, or contain stale rows: {name}')
    # Match archived row order: RF bootstrap samples depend on row positions.
    return pd.concat(tables, ignore_index=True).sort_values(
        ['stage1_cluster', 'superpixel_index'], kind='stable').reset_index(drop=True)


def build_training_dataset(run_dir):
    """Export all valid regions, preserving blank manual labels for later review."""
    run_dir = Path(run_dir)
    _, records = load_run(run_dir)
    groups = pd.read_csv(run_dir / 'labels/cluster_labels.csv', dtype=str, keep_default_na=False)
    for column in ('image_id', 'stage1_cluster'):
        groups[column] = pd.to_numeric(groups[column], errors='raise').astype(int)
    groups['stage2_cluster'] = pd.to_numeric(groups.stage2_cluster.replace('', '-1'), errors='raise').astype(int)
    if groups.duplicated(['image_id', 'stage1_cluster', 'stage2_cluster']).any():
        raise ValueError('Duplicate groups in cluster_labels.csv.')
    if not groups.image_id.isin([r['image_id'] for r in records]).all():
        raise ValueError('Label table contains unknown image IDs.')
    settings = pd.read_csv(run_dir / 'gmm/stage2/settings.csv')
    output = run_dir / 'datasets/superpixels.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    for index, record in enumerate(records):
        selected = groups[groups.image_id == record['image_id']]
        if not selected.name.eq(record['name']).all():
            raise ValueError('Image names in the label table do not match the manifest.')
        parents = settings.loc[settings.image_id == record['image_id'], 'parent_cluster'].tolist()
        frame = _image_rows(run_dir, record, selected, parents)
        frame.to_csv(output, mode='w' if index == 0 else 'a', header=index == 0, index=False)
        total += len(frame)
    print(f'Saved {total} superpixels from {len(records)} images: {output}')
    return output


def training_filter(data, probability_cut=0.70, excluded_labels=(0,)):
    if not 0 <= probability_cut <= 1:
        raise ValueError('probability_cut must be between 0 and 1.')
    status = pd.Series('eligible', index=data.index)
    status.loc[data.gmm_probability < probability_cut] = 'low_probability'
    status.loc[data.manual_label.isin(excluded_labels)] = 'excluded_label'
    status.loc[data.manual_label.isna()] = 'unlabeled'
    finite = np.isfinite(data[FEATURE_NAMES + ['gmm_probability']].to_numpy()).all(axis=1)
    status.loc[~finite] = 'invalid_features_or_probability'
    return status


def inspect_training_dataset(dataset_csv, probability_cut=0.70, excluded_labels=(0,)):
    data = pd.read_csv(dataset_csv)
    data['status'] = training_filter(data, probability_cut, excluded_labels)
    output_dir = Path(dataset_csv).parent
    summary = data.groupby('status').size().rename('n_superpixels').reset_index()
    summary.to_csv(output_dir / 'data_quality_summary.csv', index=False)
    data.groupby('manual_label', dropna=False).size().rename('n_superpixels').to_csv(output_dir / 'manual_label_counts.csv')
    data.groupby(['image_id', 'name', 'status']).size().rename('n_superpixels').to_csv(output_dir / 'image_status_counts.csv')
    print(summary.to_string(index=False))
    print(f"Images: {data.image_id.nunique()} | Probability cut: {probability_cut} | Excluded labels: {list(excluded_labels)}")


def group_training_labels(dataset_csv, groups, probability_cut=0.70, excluded_labels=(0,)):
    """Map manual classes to zero-based RF classes, following the original ordering."""
    data = pd.read_csv(dataset_csv)
    selected = data[training_filter(data, probability_cut, excluded_labels) == 'eligible'].copy()
    mapping = {}
    for rf_label, labels in enumerate(groups.values()):
        for label in labels:
            if label in mapping:
                raise ValueError(f'Manual label belongs to multiple RF groups: {label}')
            mapping[label] = rf_label
    output_dir = Path(dataset_csv).parent
    unmatched = selected[~selected.manual_label.isin(mapping)]
    unmatched.to_csv(output_dir / 'unmapped_labels.csv', index=False)
    selected = selected[selected.manual_label.isin(mapping)].copy()
    selected['rf_label'] = selected.manual_label.map(mapping).astype(int)
    output = output_dir / 'stage1_training.csv'
    selected.to_csv(output, index=False)
    settings = {'groups': groups, 'name_map': dict(enumerate(groups)), 'label_map': mapping,
                'feature_names': FEATURE_NAMES, 'probability_cut': probability_cut, 'excluded_labels': list(excluded_labels)}
    (output_dir / 'stage1_settings.json').write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    counts = selected.rf_label.value_counts()
    for rf_label, name in enumerate(groups):
        print(f'RF {rf_label} ({name}): {counts.get(rf_label, 0)} superpixels')
    print(f'Unmapped eligible rows: {len(unmatched)} | Saved: {output}')
    return output
