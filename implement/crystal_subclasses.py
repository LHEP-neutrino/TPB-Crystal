"""Fit subclass GMMs and review their saved representative clusters separately."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from crystal_features import FEATURE_NAMES
from crystal_pipeline import load_run, image_area

RESULT_COLUMNS = ['config', 'image_id', 'group', 'plate', 'name', 'parent_class', 'cluster_id',
                  'N_superpixels', 'subclass', 'subclass_majority', 'subclass_mean_confidence']


def collect_class_features(run_dir, parent_class, clusters, records):
    """Pool only small feature arrays across images, not images or segment maps."""
    arrays, identifiers = [], []
    for record in records:
        rows = clusters[(clusters.image_id == record['image_id']) & clusters.final_label.str.startswith(str(parent_class))]
        if rows.empty:
            continue
        name = f"image_{record['image_id']:04d}"
        with np.load(run_dir / 'features' / f'{name}.npz', allow_pickle=False) as saved:
            labels, features = saved['labels'], saved['features'].astype(np.float64)
            if saved['feature_names'].tolist() != FEATURE_NAMES:
                raise ValueError(f'Unexpected feature order: {name}')
        with np.load(run_dir / 'gmm' / f'{name}.npz', allow_pickle=False) as saved:
            if not np.array_equal(labels, saved['labels']):
                raise ValueError(f'GMM labels do not match feature labels: {name}')
            mask = np.isin(saved['cluster_labels'], rows.cluster_id.to_numpy())
            indices, groups = saved['valid_indices'][mask], saved['cluster_labels'][mask]
        finite = np.isfinite(features[indices]).all(axis=1)
        indices, groups = indices[finite], groups[finite]
        if len(indices):
            arrays.append(features[indices])
            identifiers.append(np.column_stack([np.full(len(indices), record['image_id']), groups, indices]))
    if not arrays:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0, 3), dtype=int)
    return np.vstack(arrays), np.vstack(identifiers)


def run_subclass_gmm(run_dir, configs, pca_components=10):
    """Save per-superpixel assignments and cluster majorities without plotting."""
    run_dir = Path(run_dir)
    _, records = load_run(run_dir)
    clusters = pd.read_csv(run_dir / 'rf_corrected/cluster_results.csv', dtype={'final_label': str})
    for config in configs:
        output_dir = run_dir / 'subclass_gmm' / config['name']
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for parent, count in config['n_components'].items():
            X, identifiers = collect_class_features(run_dir, parent, clusters, records)
            if len(X) < max(count, 2):
                raise ValueError(f"Not enough superpixels for {config['name']}, class {parent}: {len(X)}")
            scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=min(pca_components, *scaled.shape), random_state=0)
            transformed = pca.fit_transform(scaled)
            model = GaussianMixture(n_components=count, covariance_type='full', random_state=0, n_init=5, reg_covar=1e-5)
            subclasses = model.fit_predict(transformed)
            probabilities = model.predict_proba(transformed)
            confidence = probabilities.max(axis=1)
            np.savez_compressed(output_dir / f'class_{parent}.npz', image_ids=identifiers[:, 0], cluster_ids=identifiers[:, 1],
                                sp_indices=identifiers[:, 2], subclass_labels=subclasses, probabilities=probabilities,
                                n_components=count, pca_components=pca.n_components_, explained_variance=pca.explained_variance_ratio_.sum())
            for record in records:
                rows = clusters[(clusters.image_id == record['image_id']) & clusters.final_label.str.startswith(str(parent))]
                for cluster_id in rows.cluster_id:
                    mask = (identifiers[:, 0] == record['image_id']) & (identifiers[:, 1] == cluster_id)
                    if not mask.any():
                        continue
                    values, counts = np.unique(subclasses[mask], return_counts=True)
                    results.append({'config': config['name'], **{k: record[k] for k in ('image_id', 'group', 'plate', 'name')},
                                    'parent_class': str(parent), 'cluster_id': cluster_id, 'N_superpixels': int(mask.sum()),
                                    'subclass': int(values[counts.argmax()]), 'subclass_majority': counts.max() / mask.sum(),
                                    'subclass_mean_confidence': confidence[mask].mean()})
            print(f"{config['name']} | Class {parent}: {len(X)} superpixels, {count} subclasses", flush=True)
        pd.DataFrame(results, columns=RESULT_COLUMNS).to_csv(output_dir / 'cluster_results.csv', index=False)
        print(f'Saved: {output_dir}')


def _plot_representatives(run_dir, page, title):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    try:
        for ax in axes.ravel():
            ax.axis('off')
        for ax, row in zip(axes.ravel(), page.itertuples(index=False)):
            name = f'image_{row.image_id:04d}'
            image = np.load(run_dir / 'preprocessing' / name / 'image.npy', allow_pickle=False)
            segments = np.load(run_dir / 'preprocessing' / name / 'segments.npy', allow_pickle=False)
            with np.load(run_dir / 'gmm' / f'{name}.npz', allow_pickle=False) as saved:
                selected = saved['labels'][saved['valid_indices'][saved['cluster_labels'] == row.cluster_id]]
            mask = np.isin(segments, selected)
            view = np.full_like(image, 255)
            view[mask] = image[mask]
            ax.imshow(view)
            ax.set_title(f'{row.group}/{row.plate} | Image {row.image_id}, cluster {row.cluster_id}\nN_SP={row.N_superpixels}, majority={row.subclass_majority:.3f}')
            del image, segments, mask, view
        fig.suptitle(title)
        fig.tight_layout()
        return fig
    except Exception:
        plt.close(fig)
        raise


def show_subclasses(run_dir, config_name, representative_n=20):
    """Read saved results and display original clusters ranked by subclass majority."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from IPython.display import Image, display

    if representative_n < 1:
        raise ValueError('representative_n must be positive.')
    run_dir = Path(run_dir)
    result_dir = run_dir / 'subclass_gmm' / config_name
    results = pd.read_csv(result_dir / 'cluster_results.csv', dtype={'parent_class': str})
    output_dir = result_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(result_dir / 'representatives.pdf') as pdf:
        for (parent, subclass), rows in results.groupby(['parent_class', 'subclass'], sort=False):
            rows = rows.sort_values(['subclass_majority', 'N_superpixels'], ascending=False).head(representative_n)
            for start in range(0, len(rows), 4):
                title = f'{config_name} | Class {parent}, subclass {subclass} | Original cluster representatives'
                fig = _plot_representatives(run_dir, rows.iloc[start:start+4], title)
                output = output_dir / f'class_{parent}_subclass_{subclass}_page_{start//4+1:02d}.png'
                try:
                    fig.savefig(output, dpi=140)
                    pdf.savefig(fig, dpi=140)
                finally:
                    plt.close(fig)
                display(Image(filename=str(output)))
    print(f"Figures: {output_dir}\nPDF: {result_dir / 'representatives.pdf'}")


def apply_image(run_dir, record, assignments):
    """Map cluster subclasses to superpixels, then count actual segment pixels."""
    name = f"image_{record['image_id']:04d}"
    with np.load(run_dir / 'rf_corrected' / f'{name}.npz', allow_pickle=False) as saved:
        labels, original = saved['labels'], saved['final_labels']
    final = original.astype('<U40')
    with np.load(run_dir / 'gmm' / f'{name}.npz', allow_pickle=False) as saved:
        if not np.array_equal(labels, saved['labels']):
            raise ValueError(f'GMM labels do not match corrected labels: {name}')
        valid, clusters = saved['valid_indices'], saved['cluster_labels']
    for row in assignments.itertuples(index=False):
        indices = valid[clusters == row.cluster_id]
        if not len(indices) or not np.all(original[indices] == row.parent_class):
            raise ValueError(f'Saved subclass results do not match corrected RF labels: {name}, cluster {row.cluster_id}')
        final[indices] = row.new_label
    area = image_area(run_dir, record, labels, final, 'corrected labels')
    return labels, final, area


def apply_subclasses(run_dir, config_name):
    """Save a separate set of final labels and area tables for a chosen config."""
    run_dir = Path(run_dir)
    result_dir = run_dir / 'subclass_gmm' / config_name
    assignments = pd.read_csv(result_dir / 'cluster_results.csv', dtype={'parent_class': str})
    keys = ['image_id', 'cluster_id']
    if assignments.duplicated(keys).any():
        raise ValueError('A cluster has multiple subclass assignments. Use non-overlapping parent classes.')
    assignments['new_label'] = assignments['parent_class'] + '_' + assignments['subclass'].astype(str)
    clusters = pd.read_csv(run_dir / 'rf_corrected/cluster_results.csv', dtype={'stage1_label': str, 'stage2_label': str, 'final_label': str})
    if not pd.MultiIndex.from_frame(assignments[keys]).isin(pd.MultiIndex.from_frame(clusters[keys])).all():
        raise ValueError('Subclass assignments contain unknown image/cluster IDs.')
    extra = ['parent_class', 'subclass', 'new_label', 'subclass_majority', 'subclass_mean_confidence']
    clusters = clusters.merge(assignments[keys + extra], on=keys, how='left', validate='one_to_one')
    clusters['original_final_label'] = clusters['final_label']
    clusters['final_label'] = clusters['new_label'].fillna(clusters['final_label'])
    clusters['new_label'] = clusters['final_label']
    _, records = load_run(run_dir)
    output_dir = result_dir / 'applied'
    output_dir.mkdir(parents=True, exist_ok=True)
    areas = []
    for index, record in enumerate(records, 1):
        name = f"image_{record['image_id']:04d}"
        selected = assignments[assignments.image_id == record['image_id']]
        labels, final, area = apply_image(run_dir, record, selected)
        np.savez_compressed(output_dir / f'{name}.npz', labels=labels, final_labels=final)
        pd.DataFrame([area]).to_csv(output_dir / f'{name}_areas.csv', index=False)
        clusters[clusters.image_id == record['image_id']].to_csv(output_dir / f'{name}_clusters.csv', index=False)
        areas.append(area)
        print(f'[{index}/{len(records)}] {name}: {len(selected)} subclass assignments', flush=True)
    pd.DataFrame(areas).to_csv(output_dir / 'area_ratios.csv', index=False)
    clusters.to_csv(output_dir / 'cluster_results.csv', index=False)
    print(f"Final labels: {sorted(clusters['final_label'].unique())}\nOutput: {output_dir}")
