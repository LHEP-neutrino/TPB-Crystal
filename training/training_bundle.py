"""Export and evaluate the five-class step1 model only."""
import json
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
from training_rf import _image_hash
from training_labels import require_training_labels


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def is_trained(model_dir):
    path = Path(model_dir) / 'training_status.json'
    return path.is_file() and read_json(path).get('status') == 'trained'


def export_implement_bundle(run_dir):
    """Export one all-data RF; the original implement models remain preserved."""
    run_dir = Path(run_dir)
    require_training_labels(run_dir)
    model_dir = run_dir / 'models/stage1'
    if not is_trained(model_dir):
        raise ValueError('Train the step1 RF first.')
    package = joblib.load(model_dir / 'rf_step1.pkl')
    settings = read_json(run_dir / 'datasets/stage1_settings.json')
    if package['feature_names'] != FEATURE_NAMES or list(package['model'].classes_) != list(range(len(settings['groups']))):
        raise ValueError('Feature order or class coverage differs from the configured step1 task.')
    output_dir = run_dir / 'export/implement_bundle'
    models_dir = output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    if any(models_dir.glob('rf_step[23]*.pkl')):
        raise ValueError('Use a new run directory: this export contains old child models.')
    shutil.copy2(model_dir / 'rf_step1.pkl', models_dir / 'rf_step1.pkl')
    joblib.dump({'reference_mean': package['reference_mean'],
                 'feature_calculation': package['feature_calculation']}, models_dir / 'color_reference.pkl')
    write_json(output_dir / 'manifest.json', {
        'models': {'stage1': 'models/rf_step1.pkl', 'stage2': {}, 'stage3': {}},
        'color_reference': 'models/color_reference.pkl', 'stage1_groups': settings['groups'],
        'feature_names': FEATURE_NAMES, 'feature_calculation': package['feature_calculation'],
        'crop': package['crop'], 'n_segments': package['n_segments'], 'compactness': package['compactness'],
        'source_run': str(run_dir.resolve()),
        'evaluation_policy': 'Held-out metrics use evaluation_model.pkl; exported RF is fitted on all eligible rows.',
    })
    write_json(output_dir / 'bundle_status.json', {'status': 'exported', 'n_models': 1})
    (output_dir / 'README.md').write_text(
        '# Step1 model bundle\n\n'
        '- `models/rf_step1.pkl`: five-class all-data RF.\n'
        '- `models/color_reference.pkl`: matching Lab reference.\n'
        '- `manifest.json`: relative paths and preprocessing settings.\n'
        '- Use empty `stage2_paths` and `stage3_paths`; keep matching feature calculation.\n')
    print(f'Exported step1 RF and color reference: {output_dir}')


def evaluate_stage1_gmm(project_root, run_dir):
    """Compare raw RF and GMM voting on identical held-out rows, without child models."""
    import matplotlib.pyplot as plt
    from IPython.display import Image, display

    project_root, run_dir = Path(project_root), Path(run_dir)
    require_training_labels(run_dir)
    output_dir = run_dir / 'models/stage1/gmm_evaluation'
    base = read_json(run_dir / 'datasets/stage1_settings.json')
    model_dir = run_dir / 'models/stage1'
    if not is_trained(model_dir):
        raise ValueError('Train the step1 RF in Cell 14 first.')
    split = pd.read_csv(model_dir / 'image_split.csv')
    package = joblib.load(model_dir / 'evaluation_model.pkl')
    if package['feature_names'] != FEATURE_NAMES:
        raise ValueError('Evaluation feature names do not match.')
    models = [{'all': package['model']}, {}, {}]
    raw = pd.read_csv(run_dir / 'datasets/superpixels.csv')
    data = raw[training_filter(raw, base['probability_cut'], base['excluded_labels']) == 'eligible'].copy()
    routes = {int(k): str(v) for k, v in base['label_map'].items()}
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
               'inference': 'step1 RF majority over stage-1 GMM clusters; no child models',
               'models_used': 'evaluation models trained only on the stage-1 training partition',
               'scoring_rows': 'labeled, mapped superpixels passing the configured GMM probability cut'}
    raw_predictions = pd.read_csv(model_dir / 'test_predictions.csv')
    aligned = result.merge(raw_predictions[['image_id', 'superpixel_index', 'prediction']],
                           on=['image_id', 'superpixel_index'], validate='one_to_one')
    if len(aligned) != len(result) or len(aligned) != len(raw_predictions):
        raise ValueError('RF and GMM vote evaluations must cover identical held-out rows.')
    raw_truth = aligned.true_final_label.astype(int)
    comparison = pd.DataFrame([
        {'method': 'RF only', 'accuracy': accuracy_score(raw_truth, aligned.prediction),
         'macro_f1': f1_score(raw_truth, aligned.prediction, average='macro', zero_division=0)},
        {'method': 'RF + GMM vote', 'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1']},
    ])
    comparison.to_csv(output_dir / 'method_comparison.csv', index=False)
    print(comparison.to_string(index=False))
    write_json(output_dir / 'metrics.json', metrics)
    pd.DataFrame(classification_report(truth, predicted, output_dict=True, zero_division=0)).T.to_csv(output_dir / 'class_metrics.csv')
    fig, ax = plt.subplots(figsize=(9, 8))
    try:
        ConfusionMatrixDisplay.from_predictions(truth, predicted, ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
        ax.set_title(f"Held-out five-class GMM vote | Accuracy: {metrics['accuracy']:.3f}, Macro F1: {metrics['macro_f1']:.3f}")
        fig.tight_layout()
        fig.savefig(output_dir / 'confusion_matrix.png', dpi=140)
        fig.savefig(output_dir / 'confusion_matrix.pdf')
    finally:
        plt.close(fig)
    display(Image(filename=str(output_dir / 'confusion_matrix.png')))
    write_json(output_dir / 'evaluation_status.json', {'status': 'evaluated'})
    print(f"Held-out five-class GMM vote accuracy: {metrics['accuracy']:.4f} | Saved: {output_dir}")
