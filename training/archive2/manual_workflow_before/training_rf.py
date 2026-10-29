"""Evaluate stage-1 RF by image groups and save an implementation-compatible model."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from crystal_pipeline import load_run
from crystal_features import FEATURE_CALCULATION


def _image_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def train_stage1_rf(project_root, run_dir, training_csv, test_size=0.25, n_estimators=300, random_state=0, task=None):
    """Evaluate on held-out images, then refit the exported RF on all eligible rows.

    Byte-identical source images share a split group even if filenames differ.
    Blank labels and insufficient class/image coverage produce no trained model.
    """
    project_root, run_dir = Path(project_root), Path(run_dir)
    task = task or {}
    output_dir = run_dir / 'models' / task.get('model_subdir', 'stage1')
    output_dir.mkdir(parents=True, exist_ok=True)

    def stop(reason):
        (output_dir / 'training_status.json').write_text(json.dumps({'status': 'not_trained', 'reason': reason}, indent=2) + '\n')
        print(f'RF not trained: {reason}')

    data = pd.read_csv(training_csv)
    if data.empty or data.rf_label.nunique() < 2:
        return stop('At least two labeled RF classes are required. Fill manual_label and rerun Cells 11-14.')
    if not 0 < test_size < 1:
        raise ValueError('test_size must be between 0 and 1.')
    settings = json.loads((Path(training_csv).parent / task.get('settings_file', 'stage1_settings.json')).read_text())
    feature_names = settings['feature_names']
    X, y = data[feature_names].to_numpy(), data.rf_label.to_numpy(dtype=int)
    if not np.isfinite(X).all():
        raise ValueError('Training features contain non-finite values.')
    images = data[['image_id', 'name', 'path']].drop_duplicates('image_id').copy()
    images['split_group'] = [_image_hash(project_root / path) for path in images.path]
    data['split_group'] = data.image_id.map(images.set_index('image_id').split_group)
    minimum_images = 2 if task.get('reuse_stage1_split') else 3
    if data.split_group.nunique() < minimum_images or (data.groupby('rf_label').split_group.nunique() < 2).any():
        return stop(f'Need at least {minimum_images} distinct images, with each RF class present in two distinct images.')
    expected = set(range(len(settings['groups'])))
    if task.get('require_all_classes') and set(y) != expected:
        return stop('Not all configured classes have eligible labeled samples.')
    split = None
    if task.get('reuse_stage1_split'):
        saved_split = pd.read_csv(run_dir / 'models/stage1/image_split.csv')
        images = images.merge(saved_split[['image_id', 'partition', 'split_group']], on='image_id', how='left', suffixes=('', '_saved'), validate='one_to_one')
        if images.partition.isna().any() or not images.split_group.eq(images.split_group_saved).all():
            return stop('Source images do not match the saved stage-1 split.')
        partitions = data.image_id.map(images.set_index('image_id').partition)
        train, test = np.flatnonzero(partitions == 'train'), np.flatnonzero(partitions == 'test')
        if not len(train) or not len(test) or set(y[train]) != set(y) or set(y[test]) != set(y):
            return stop('The fixed stage-1 image split does not cover every child class on both sides.')
        split = train, test
    for train, test in ([] if split is not None else GroupShuffleSplit(n_splits=100, test_size=test_size, random_state=random_state).split(X, y, data.split_group)):
        if set(y[train]) == set(y) and set(y[test]) == set(y):
            split = train, test
            break
    if split is None:
        return stop('No image-group split covers all classes on both sides. Add labels/images or change test_size.')
    train, test = split
    if set(data.iloc[train].split_group) & set(data.iloc[test].split_group):
        raise ValueError('The image split leaks identical source files across partitions.')
    parameters = {'n_estimators': n_estimators, 'random_state': random_state, 'class_weight': 'balanced', 'n_jobs': -1}
    evaluation_model = RandomForestClassifier(**parameters).fit(X[train], y[train])
    predicted = evaluation_model.predict(X[test])
    metrics = {'test_accuracy': accuracy_score(y[test], predicted), 'test_balanced_accuracy': balanced_accuracy_score(y[test], predicted),
               'test_macro_f1': f1_score(y[test], predicted, average='macro', zero_division=0),
               'train_superpixels': len(train), 'test_superpixels': len(test),
               'train_images': int(data.iloc[train].image_id.nunique()), 'test_images': int(data.iloc[test].image_id.nunique()),
               'test_size': test_size, 'random_state': random_state, 'split_unit': 'source image SHA256',
               'exported_model_fit': 'all eligible rows, after separate held-out evaluation',
               'reuse_stage1_split': bool(task.get('reuse_stage1_split'))}
    name_map = {int(k): v for k, v in settings['name_map'].items()}
    classes = sorted(set(y))
    pd.DataFrame(confusion_matrix(y[test], predicted, labels=classes), index=classes, columns=classes).to_csv(output_dir / 'confusion_matrix.csv')
    predictions = data.iloc[test][['image_id', 'name', 'superpixel_index', 'superpixel_label', 'manual_label', 'rf_label']].copy()
    predictions['prediction'] = predicted
    predictions['confidence'] = evaluation_model.predict_proba(X[test]).max(axis=1)
    predictions.to_csv(output_dir / 'test_predictions.csv', index=False)
    images['partition'] = np.where(images.split_group.isin(data.iloc[test].split_group), 'test', 'train')
    images.to_csv(output_dir / 'image_split.csv', index=False)
    importances = pd.DataFrame({'feature': feature_names, 'importance': evaluation_model.feature_importances_})
    importances.sort_values('importance', ascending=False).to_csv(output_dir / 'feature_importance.csv', index=False)
    config, records = load_run(run_dir)
    reference = json.loads((run_dir / 'metadata/color_reference.json').read_text())
    corrections = []
    for record in records:
        with np.load(run_dir / 'color_correction' / f"image_{record['image_id']:04d}.npz", allow_pickle=False) as saved:
            corrections.append(saved['correction'])
    model = RandomForestClassifier(**parameters).fit(X, y)
    package = {'model': model, 'feature_set': 'All', 'feature_names': feature_names, **parameters,
               'gmm_probability_cut': settings['probability_cut'], 'groups': settings['groups'],
               'label_map': {int(k): v for k, v in settings['label_map'].items()}, 'name_map': name_map,
               'edge_threshold': 0.05, 'n_segments': config['n_segments'], 'compactness': config['compactness'],
               'crop': config['crop'], 'reference_mean': np.array(reference['reference_mean'], dtype=np.float64),
               'feature_calculation': FEATURE_CALCULATION, 'lab_corrections': np.array(corrections), 'evaluation': metrics, 'task': task}
    joblib.dump(package, output_dir / task.get('model_file', 'rf_step1.pkl'))
    joblib.dump({'model': evaluation_model, 'feature_names': feature_names, 'name_map': name_map}, output_dir / 'evaluation_model.pkl')
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'training_status.json').write_text(json.dumps({'status': 'trained', 'classes': [int(c) for c in classes]}, indent=2) + '\n')
    print(f"Held-out accuracy: {metrics['test_accuracy']:.4f} | Macro F1: {metrics['test_macro_f1']:.4f}")
    print(f'Model refitted on {len(data)} eligible superpixels. Saved: {output_dir}')
