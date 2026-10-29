"""Classify GMM clusters with the original three-stage random forests."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from crystal_pipeline import load_run, image_area


def majority_vote(labels):
    values, counts = np.unique(labels, return_counts=True)
    index = counts.argmax()
    return str(values[index]), counts[index] / len(labels)


def classify_image(run_dir, record, models, feature_names, gmm_subdir="gmm"):
    """Process one image and return only small tables and superpixel arrays."""
    name = f"image_{record['image_id']:04d}"
    with np.load(run_dir / 'features' / f'{name}.npz', allow_pickle=False) as saved:
        labels, features = saved['labels'], saved['features']
        columns = saved['feature_names'].tolist()
    X = features[:, [columns.index(f) for f in feature_names]]
    with np.load(run_dir / gmm_subdir / f'{name}.npz', allow_pickle=False) as saved:
        if not np.array_equal(labels, saved['labels']):
            raise ValueError(f'GMM labels do not match features: {name}')
        valid, clusters = saved['valid_indices'], saved['cluster_labels']
        confidence, n_components = saved['max_probability'], int(saved['n_components'])
    probability = models[0]['all'].predict_proba(X)
    predictions = models[0]['all'].classes_[probability.argmax(axis=1)]
    rf_confidence = probability.max(axis=1)
    stages, cluster_rows = [], []
    for stage in range(3):
        results = {}
        for cluster in range(n_components):
            indices = valid[clusters == cluster]
            if not len(indices):
                continue
            if stage == 0:
                label, majority = majority_vote(predictions[indices])
                mean_confidence = rf_confidence[indices].mean()
            else:
                parent, majority, mean_confidence = stages[-1][cluster]
                model = models[stage].get(parent)
                label = parent
                if model is not None:
                    child, majority = majority_vote(model.predict(X[indices]))
                    label = parent + child
                    mean_confidence = model.predict_proba(X[indices]).max(axis=1).mean()
                elif stage == 1:
                    majority = 1.0  # Preserve the original stage-2 fallback.
            results[cluster] = (label, majority, mean_confidence)
        stages.append(results)
    arrays = [np.full(len(labels), '', dtype='<U20') for _ in range(3)]
    final_confidence = np.zeros(len(labels), dtype=np.float32)
    for cluster in stages[-1]:
        indices = valid[clusters == cluster]
        row = {'image_id': record['image_id'], 'cluster_id': cluster, 'N_superpixels': len(indices),
               'gmm_mean_confidence': confidence[clusters == cluster].mean()}
        for stage, prefix in enumerate(('stage1', 'stage2', 'final')):
            label, majority, mean_confidence = stages[stage][cluster]
            arrays[stage][indices] = label
            row.update({f'{prefix}_label': label, f'{prefix}_majority': majority, f'{prefix}_rf_confidence': mean_confidence})
        final_confidence[indices] = stages[-1][cluster][1]
        cluster_rows.append(row)
    area = image_area(run_dir, record, labels, arrays[-1], 'features', final_confidence)
    output = {'labels': labels, 'stage1_labels': arrays[0], 'stage2_labels': arrays[1],
              'final_labels': arrays[2], 'final_confidence': final_confidence}
    return output, area, cluster_rows


def run_rf_classification(run_dir, stage1_path, stage2_paths, stage3_paths):
    """Save per-image predictions and summary tables; rerunning overwrites outputs."""
    run_dir = Path(run_dir)
    paths = [{'all': stage1_path}, stage2_paths, stage3_paths]
    models, feature_names = [], None
    for stage_paths in paths:
        stage_models = {}
        for parent, path in stage_paths.items():
            package = joblib.load(path)
            names = list(package['feature_names'])
            if feature_names is None:
                feature_names = names
            if names != feature_names or package['model'].n_features_in_ != len(names):
                raise ValueError(f'Model feature names or order do not match: {path}')
            stage_models[str(parent)] = package['model']
        models.append(stage_models)
    _, records = load_run(run_dir)
    output_dir = run_dir / 'rf'
    output_dir.mkdir(parents=True, exist_ok=True)
    areas, clusters = [], []
    for index, record in enumerate(records, 1):
        name = f"image_{record['image_id']:04d}"
        output, area, rows = classify_image(run_dir, record, models, feature_names)
        np.savez_compressed(output_dir / f'{name}.npz', **output)
        pd.DataFrame([area]).to_csv(output_dir / f'{name}_areas.csv', index=False)
        pd.DataFrame(rows).to_csv(output_dir / f'{name}_clusters.csv', index=False)
        areas.append(area)
        clusters.extend(rows)
        print(f'[{index}/{len(records)}] {name}: {len(rows)} clusters classified', flush=True)
    pd.DataFrame(areas).to_csv(output_dir / 'area_ratios.csv', index=False)
    pd.DataFrame(clusters).to_csv(output_dir / 'cluster_results.csv', index=False)
    print(f'Output: {output_dir}')
