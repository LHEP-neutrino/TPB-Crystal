"""Export compatible RF models, then evaluate the hierarchy on held-out images."""
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from crystal_features import FEATURE_NAMES
from crystal_pipeline import load_run
from crystal_rf import classify_image
from training_dataset import training_filter
from training_hierarchy import read_json, write_json, is_trained, class_routes
from training_rf import _image_hash


def _model_tasks(run_dir):
    base = read_json(run_dir / 'datasets/stage1_settings.json')
    tasks = [{'stage': 1, 'parent': 'all', 'model_subdir': 'stage1', 'model_file': 'rf_step1.pkl', 'n_classes': len(base['groups'])}]
    specifications = {}
    for stage in (2, 3):
        config = read_json(run_dir / f'datasets/stage{stage}/tasks.json')
        specifications[stage] = config['groups']
        for task in config['tasks']:
            tasks.append({**task, 'stage': stage, 'n_classes': len(config['groups'][task['parent']])})
    return base, specifications, tasks


def export_implement_bundle(run_dir):
    """Copy complete final models into a separate bundle; never overwrite implement."""
    run_dir = Path(run_dir)
    output_dir = run_dir / 'export/implement_bundle'
    write_json(output_dir / 'bundle_status.json', {'status': 'not_exported', 'reason': 'Validation not completed.'})
    base, specs, tasks = _model_tasks(run_dir)
    missing = [f"stage {t['stage']}, parent {t['parent']}" for t in tasks if not is_trained(run_dir / 'models' / t['model_subdir'])]
    if missing:
        write_json(output_dir / 'bundle_status.json', {'status': 'not_exported', 'missing_models': missing})
        print(f'Bundle not exported; missing trained models: {missing}')
        return
    packages = []
    for task in tasks:
        path = run_dir / 'models' / task['model_subdir'] / task['model_file']
        package = joblib.load(path)
        if package.get('test_only') or package['feature_names'] != FEATURE_NAMES:
            raise ValueError(f'Incompatible feature order or test-only model: {path}')
        if list(package['model'].classes_) != list(range(task['n_classes'])):
            write_json(output_dir / 'bundle_status.json', {'status': 'not_exported', 'reason': f'Incomplete class coverage: {path}'})
            print(f'Bundle not exported: incomplete configured class coverage in {path}')
            return
        if package['model'].n_features_in_ != len(FEATURE_NAMES):
            raise ValueError(f'Wrong feature count: {path}')
        packages.append((task, path, package))
    reference = packages[0][2]
    for _, path, package in packages:
        for key in ('n_segments', 'compactness', 'crop'):
            if package[key] != reference[key]:
                raise ValueError(f'Model preprocessing differs: {path}, {key}')
        if not np.array_equal(package['reference_mean'], reference['reference_mean']):
            raise ValueError(f'Model color reference differs: {path}')
    routes = class_routes(base, specs[2], specs[3])
    model_paths = {'stage1': '', 'stage2': {}, 'stage3': {}}
    names = {2: {'1': 'rf_step2_group2.pkl', '2': 'rf_step2_group3.pkl', '3': 'rf_step2_group4.pkl'},
             3: {'20': 'rf_step3_group31.pkl'}}
    (output_dir / 'models').mkdir(exist_ok=True)
    for task, path, package in packages:
        stage, parent = task['stage'], task['parent']
        filename = 'rf_step1.pkl' if stage == 1 else names[stage].get(parent, f'rf_step{stage}_parent_{parent}.pkl')
        shutil.copy2(path, output_dir / 'models' / filename)
        if stage == 1:
            model_paths['stage1'] = f'models/{filename}'
        else:
            model_paths[f'stage{stage}'][parent] = f'models/{filename}'
    joblib.dump({'reference_mean': reference['reference_mean'],
                 'feature_calculation': reference.get('feature_calculation', 'unknown')}, output_dir / 'models/color_reference.pkl')
    manifest = {'feature_calculation': reference.get('feature_calculation', 'unknown'), 'models': model_paths, 'color_reference': 'models/color_reference.pkl', 'feature_names': FEATURE_NAMES,
                'n_segments': reference['n_segments'], 'compactness': reference['compactness'], 'crop': reference['crop'],
                'stage1_groups': base['groups'], 'stage2_groups': specs[2], 'stage3_groups': specs[3],
                'manual_to_final_label': routes, 'source_run': str(run_dir.resolve()),
                'evaluation_policy': 'Cell 24 uses evaluation models, not these all-data models.'}
    write_json(output_dir / 'manifest.json', manifest)
    (output_dir / 'README.md').write_text('# Implementation model bundle\n\nUse the relative model paths and parent routing in manifest.json. Models were fitted on all eligible labeled rows after separate held-out evaluation. Cell 24 uses the separate evaluation models. No files were copied into the existing implement directory.\n', encoding='utf-8')
    write_json(output_dir / 'bundle_status.json', {'status': 'exported', 'n_models': len(packages)})
    print(f'Exported {len(packages)} compatible RF models and a color reference: {output_dir}')


def evaluate_hierarchy(project_root, run_dir):
    """Use implement's cluster-majority inference with evaluation-only models."""
    import matplotlib.pyplot as plt
    from IPython.display import Image, display

    project_root, run_dir = Path(project_root), Path(run_dir)
    output_dir = run_dir / 'models/hierarchy_evaluation'
    bundle = run_dir / 'export/implement_bundle'
    status_path = bundle / 'bundle_status.json'
    if not status_path.exists() or read_json(status_path).get('status') != 'exported':
        write_json(output_dir / 'evaluation_status.json', {'status': 'not_evaluated', 'reason': 'Complete the model bundle in Cell 23 first.'})
        print('Hierarchy not evaluated: a complete model bundle is not available.')
        return
    manifest = read_json(bundle / 'manifest.json')
    base, specs, tasks = _model_tasks(run_dir)
    if manifest['manual_to_final_label'] != {str(k): v for k, v in class_routes(base, specs[2], specs[3]).items()}:
        raise ValueError('The bundle class mapping is stale. Rerun Cell 23.')
    split = pd.read_csv(run_dir / 'models/stage1/image_split.csv')
    models = [{}, {}, {}]
    for task in tasks:
        model_dir = run_dir / 'models' / task['model_subdir']
        if not is_trained(model_dir):
            raise ValueError(f'An evaluation task is not trained: {model_dir}')
        local = pd.read_csv(model_dir / 'image_split.csv')
        aligned = local.merge(split[['image_id', 'partition', 'split_group']], on='image_id', suffixes=('', '_base'), validate='one_to_one')
        if len(aligned) != len(local) or not aligned.partition.eq(aligned.partition_base).all() or not aligned.split_group.eq(aligned.split_group_base).all():
            raise ValueError(f'Evaluation models do not share the stage-1 split: {model_dir}')
        package = joblib.load(model_dir / 'evaluation_model.pkl')
        if package['feature_names'] != FEATURE_NAMES:
            raise ValueError(f'Wrong evaluation feature order: {model_dir}')
        models[task['stage'] - 1][task['parent']] = package['model']
    raw = pd.read_csv(run_dir / 'datasets/superpixels.csv')
    data = raw[training_filter(raw, base['probability_cut'], base['excluded_labels']) == 'eligible'].copy()
    routes = {int(k): v for k, v in manifest['manual_to_final_label'].items()}
    data['true_final_label'] = data.manual_label.map(routes)
    data = data[data.true_final_label.notna() & data.image_id.isin(split.loc[split.partition == 'test', 'image_id'])]
    if data.empty:
        raise ValueError('No labeled held-out rows are available.')
    _, records = load_run(run_dir)
    rows, cluster_rows = [], []
    for record in records:
        selected = data[data.image_id == record['image_id']].copy()
        if selected.empty:
            continue
        expected = split.loc[split.image_id == record['image_id'], 'split_group'].iloc[0]
        if _image_hash(project_root / record['path']) != expected:
            raise ValueError('A held-out source image changed after the split.')
        output, _, clusters = classify_image(run_dir, record, models, FEATURE_NAMES, gmm_subdir='gmm/stage1')
        indices = selected.superpixel_index.to_numpy(dtype=int)
        if not np.array_equal(output['labels'][indices], selected.superpixel_label):
            raise ValueError('Held-out rows do not match current feature labels.')
        selected['predicted_final_label'] = output['final_labels'][indices]
        selected['predicted_stage1_label'] = output['stage1_labels'][indices]
        selected['predicted_stage2_label'] = output['stage2_labels'][indices]
        selected['final_majority'] = output['final_confidence'][indices]
        rows.append(selected)
        cluster_rows.extend(clusters)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / f"image_{record['image_id']:04d}.npz", **output)
    result = pd.concat(rows, ignore_index=True)
    truth, predicted = result.true_final_label.astype(str), result.predicted_final_label.astype(str)
    result['correct'] = truth == predicted
    result.to_csv(output_dir / 'test_predictions.csv', index=False)
    pd.DataFrame(cluster_rows).to_csv(output_dir / 'cluster_predictions.csv', index=False)
    metrics = {'accuracy': accuracy_score(truth, predicted), 'macro_f1': f1_score(truth, predicted, average='macro', zero_division=0),
               'n_test_superpixels': len(result), 'n_test_images': int(result.image_id.nunique()),
               'inference': 'implement cluster-majority routing over stage-1 GMM clusters',
               'models_used': 'evaluation models trained only on the stage-1 training partition',
               'scoring_rows': 'labeled, mapped superpixels passing the configured GMM probability cut'}
    write_json(output_dir / 'metrics.json', metrics)
    pd.DataFrame(classification_report(truth, predicted, output_dict=True, zero_division=0)).T.to_csv(output_dir / 'class_metrics.csv')
    fig, ax = plt.subplots(figsize=(9, 8))
    try:
        ConfusionMatrixDisplay.from_predictions(truth, predicted, ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
        ax.set_title(f"Held-out hierarchy | Accuracy: {metrics['accuracy']:.3f}, Macro F1: {metrics['macro_f1']:.3f}")
        fig.tight_layout()
        fig.savefig(output_dir / 'confusion_matrix.png', dpi=140)
        fig.savefig(output_dir / 'confusion_matrix.pdf')
    finally:
        plt.close(fig)
    display(Image(filename=str(output_dir / 'confusion_matrix.png')))
    write_json(output_dir / 'evaluation_status.json', {'status': 'evaluated'})
    print(f"Held-out hierarchy accuracy: {metrics['accuracy']:.4f} | Saved: {output_dir}")
