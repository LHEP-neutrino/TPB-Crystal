"""Display saved held-out RF evaluation and inspect misclassified regions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def _evaluation_dir(run_dir, model_subdir="stage1"):
    model_dir = Path(run_dir) / 'models' / model_subdir
    status_path = model_dir / 'training_status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get('status') != 'trained':
        print(f"No held-out evaluation available: {status.get('reason', 'Run Cell 14 after labeling data.')}")
        return None
    return model_dir


def _save_and_display(fig, path, pdf=None):
    import matplotlib.pyplot as plt
    from IPython.display import Image, display

    try:
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        if pdf is not None:
            pdf.savefig(fig, dpi=140)
    finally:
        plt.close(fig)
    display(Image(filename=str(path)))


def show_rf_evaluation(run_dir, top_features=18, model_subdir="stage1", settings_path=None):
    """Plot held-out confusion, per-class scores, and evaluation-model importance."""
    import matplotlib.pyplot as plt

    model_dir = _evaluation_dir(run_dir, model_subdir)
    if model_dir is None:
        return
    if top_features < 1:
        raise ValueError('top_features must be positive.')
    predicted = pd.read_csv(model_dir / 'test_predictions.csv')
    metrics = json.loads((model_dir / 'metrics.json').read_text())
    settings_path = Path(settings_path) if settings_path else Path(run_dir) / 'datasets/stage1_settings.json'
    settings = json.loads(settings_path.read_text())
    classes = sorted(set(predicted.rf_label) | set(predicted.prediction))
    names = [f"RF {c} ({settings['name_map'].get(str(c), c)})" for c in classes]
    matrix = confusion_matrix(predicted.rf_label, predicted.prediction, labels=classes)
    precision, recall, f1, support = precision_recall_fscore_support(predicted.rf_label, predicted.prediction, labels=classes, zero_division=0)
    scores = pd.DataFrame({'rf_label': classes, 'name': names, 'precision': precision, 'recall': recall, 'f1': f1, 'support': support})
    importance = pd.read_csv(model_dir / 'feature_importance.csv').sort_values('importance', ascending=False).head(top_features)
    output_dir = model_dir / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / 'class_metrics.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    try:
        axes[0].imshow(matrix, cmap='Blues')
        for (row, column), count in np.ndenumerate(matrix):
            axes[0].text(column, row, str(count), ha='center', va='center', color='white' if count > matrix.max()/2 else 'black')
        axes[0].set_xticks(range(len(classes)), names, rotation=35, ha='right')
        axes[0].set_yticks(range(len(classes)), names)
        axes[0].set(title='Held-out confusion matrix', xlabel='Predicted class', ylabel='True class')
        positions = np.arange(len(classes))
        for offset, column in zip((-0.25, 0, 0.25), ('precision', 'recall', 'f1')):
            axes[1].bar(positions + offset, scores[column], width=0.25, label=column)
        axes[1].set_xticks(positions, names, rotation=35, ha='right')
        axes[1].set(title='Held-out class scores', ylabel='Score', ylim=(0, 1.05))
        axes[1].legend()
        axes[2].barh(importance.feature.iloc[::-1], importance.importance.iloc[::-1])
        axes[2].set(title='Evaluation-model feature importance', xlabel='Impurity-based importance')
        prefix = 'DISPLAY TEST DATA | ' if metrics.get('test_only') else ''
        fig.suptitle(f"{prefix}Held-out accuracy: {metrics['test_accuracy']:.3f} | Macro F1: {metrics['test_macro_f1']:.3f}")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(output_dir / 'rf_evaluation.pdf')
    except Exception:
        plt.close(fig)
        raise
    _save_and_display(fig, output_dir / 'rf_evaluation.png')
    print(scores.to_string(index=False))
    print(f'Saved: {output_dir}')


def _plot_error_group(run_dir, row, predictions, source, test_only=False):
    import matplotlib.pyplot as plt

    name = f'image_{row.image_id:04d}'
    image = np.load(run_dir / 'preprocessing' / name / 'image.npy', allow_pickle=False)
    segments = np.load(run_dir / 'preprocessing' / name / 'segments.npy', allow_pickle=False)
    match = (predictions.image_id == row.image_id) & (predictions.stage1_cluster == row.stage1_cluster) & (predictions.stage2_cluster == row.stage2_cluster)
    tested = predictions[match]
    match = (source.image_id == row.image_id) & (source.stage1_cluster == row.stage1_cluster) & (source.stage2_cluster == row.stage2_cluster)
    group_labels = source.loc[match, 'superpixel_label'].to_numpy()
    if image.shape[:2] != segments.shape or not np.isin(tested.superpixel_label, np.unique(segments)).all():
        raise ValueError(f'Image regions do not match saved evaluation rows: {name}')
    region = np.full_like(image, 255)
    mask = np.isin(segments, group_labels)
    region[mask] = image[mask]
    result = np.full_like(image, 255)
    for correct, color in ((True, [0, 180, 0]), (False, [230, 40, 40])):
        selected = tested.rf_label.eq(tested.prediction) == correct
        mask = np.isin(segments, tested.loc[selected, 'superpixel_label'])
        result[mask] = (0.6 * image[mask] + 0.4 * np.array(color)).astype(image.dtype)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    try:
        for ax, array, title in zip(axes, (image, region, result), ('Cropped input', 'Entire labeled GMM group', 'Held-out SP only: green correct, red wrong')):
            ax.imshow(array)
            ax.set_title(title)
            ax.axis('off')
        child = 'unsplit' if row.stage2_cluster < 0 else str(row.stage2_cluster)
        prefix = 'DISPLAY TEST DATA | ' if test_only else ''
        fig.suptitle(f'{prefix}Image {row.image_id} | {row.name} | Parent {row.stage1_cluster}, child {child} | Errors: {row.n_errors}/{row.n_test}')
        return fig
    except Exception:
        plt.close(fig)
        raise


def show_rf_errors(run_dir, max_groups=3, model_subdir="stage1"):
    """Rank error groups and inspect held-out regions without running RF again."""
    from matplotlib.backends.backend_pdf import PdfPages

    run_dir = Path(run_dir)
    model_dir = _evaluation_dir(run_dir, model_subdir)
    if model_dir is None:
        return
    if max_groups < 1:
        raise ValueError('max_groups must be positive.')
    test_only = json.loads((model_dir / 'metrics.json').read_text()).get('test_only', False)
    source = pd.read_csv(run_dir / 'datasets/superpixels.csv')
    predictions = pd.read_csv(model_dir / 'test_predictions.csv')
    keys = ['image_id', 'superpixel_index']
    predictions = predictions.merge(source[keys + ['stage1_cluster', 'stage2_cluster']], on=keys, how='left', validate='one_to_one')
    if predictions[['stage1_cluster', 'stage2_cluster']].isna().any().any():
        raise ValueError('Evaluation rows do not match the current feature table.')
    predictions['incorrect'] = predictions.rf_label != predictions.prediction
    group_keys = ['image_id', 'name', 'stage1_cluster', 'stage2_cluster']
    groups = predictions.groupby(group_keys).incorrect.agg(n_test='size', n_errors='sum').reset_index()
    groups['error_ratio'] = groups.n_errors / groups.n_test
    groups = groups.sort_values(['n_errors', 'error_ratio'], ascending=False)
    output_dir = model_dir / 'evaluation/misclassifications'
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions[predictions.incorrect].to_csv(output_dir / 'misclassified_superpixels.csv', index=False)
    groups.to_csv(output_dir / 'group_error_rates.csv', index=False)
    selected = groups[groups.n_errors > 0].head(max_groups)
    if selected.empty:
        print(f'No misclassified held-out superpixels. Saved tables: {output_dir}')
        return
    with PdfPages(output_dir / 'review.pdf') as pdf:
        for row in selected.itertuples(index=False):
            fig = _plot_error_group(run_dir, row, predictions, source, test_only=test_only)
            child = 'unsplit' if row.stage2_cluster < 0 else f'child_{row.stage2_cluster}'
            output = output_dir / f'image_{row.image_id:04d}_parent_{row.stage1_cluster}_{child}.png'
            _save_and_display(fig, output, pdf=pdf)
    print(f'Saved {len(selected)} error-group figures: {output_dir}')
