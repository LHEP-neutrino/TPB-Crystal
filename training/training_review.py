"""Review saved training GMM results without repeating the calculation."""
from pathlib import Path
import numpy as np
from crystal_pipeline import load_run


def _draw_regions(ax, image, segments, labels, indices, title):
    mask = np.isin(segments, labels[indices])
    view = np.full_like(image, 255)
    view[mask] = image[mask]
    ax.imshow(view)
    ax.set_title(title)


def _plot_gmm_image(run_dir, record):
    import matplotlib.pyplot as plt

    name = f"image_{record['image_id']:04d}"
    image = np.load(run_dir / 'preprocessing' / name / 'image.npy', allow_pickle=False)
    segments = np.load(run_dir / 'preprocessing' / name / 'segments.npy', allow_pickle=False)
    with np.load(run_dir / 'gmm/stage1' / f'{name}.npz', allow_pickle=False) as saved:
        labels, valid = saved['labels'], saved['valid_indices']
        clusters, high = saved['cluster_labels'], saved['high_mask']
        count, threshold = int(saved['n_components']), float(saved['probability_cut'])
    if image.shape[:2] != segments.shape or not np.array_equal(labels, np.unique(segments)):
        raise ValueError(f'Image, segment labels, and GMM labels do not match: {name}')
    fig, axes = plt.subplots(2, count + 1, figsize=(5 * (count + 1), 10), squeeze=False)
    try:
        def draw(ax, indices, title):
            _draw_regions(ax, image, segments, labels, indices, title)

        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Cropped input image')
        draw(axes[1, 0], valid[high], f'All high-confidence regions\nProbability >= {threshold:.2f}: {high.sum()}/{len(high)} SP')
        for cluster in range(count):
            selected = clusters == cluster
            draw(axes[0, cluster + 1], valid[selected], f'Cluster {cluster}: all {selected.sum()} SP')
            draw(axes[1, cluster + 1], valid[selected & high], f'Cluster {cluster}: high-confidence {(selected & high).sum()} SP')
        for ax in axes.ravel():
            ax.axis('off')
        fig.suptitle(f"Image {record['image_id']} | {record['name']} | Stage 1 GMM")
        fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
        return fig
    except Exception:
        plt.close(fig)
        raise


def show_training_gmm(run_dir, image_ids):
    """Save PNGs and a PDF, display selected images, and close each figure."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from IPython.display import Image, display

    run_dir = Path(run_dir)
    _, records = load_run(run_dir)
    available = {r['image_id']: r for r in records}
    if not image_ids or any(i not in available for i in image_ids):
        raise ValueError('Choose image IDs from metadata/images.csv.')
    output_dir = run_dir / 'gmm/stage1/figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_dir / 'review.pdf') as pdf:
        for image_id in image_ids:
            fig = _plot_gmm_image(run_dir, available[image_id])
            output = output_dir / f'image_{image_id:04d}.png'
            try:
                fig.savefig(output, dpi=120)
                pdf.savefig(fig, dpi=120)
            finally:
                plt.close(fig)
            display(Image(filename=str(output)))
            print(f'Saved: {output}')
    print(f"PDF: {output_dir / 'review.pdf'}")


def _plot_stage2_image(run_dir, record, parent):
    import matplotlib.pyplot as plt

    name = f"image_{record['image_id']:04d}"
    image = np.load(run_dir / 'preprocessing' / name / 'image.npy', allow_pickle=False)
    segments = np.load(run_dir / 'preprocessing' / name / 'segments.npy', allow_pickle=False)
    with np.load(run_dir / 'gmm/stage1' / f'{name}.npz', allow_pickle=False) as saved:
        labels = saved['labels']
        selected = saved['cluster_labels'] == parent
        original = saved['valid_indices'][selected]
        original_high = saved['valid_indices'][selected & saved['high_mask']]
        threshold1 = float(saved['probability_cut'])
    with np.load(run_dir / 'gmm/stage2' / f'{name}_cluster_{parent:04d}.npz', allow_pickle=False) as saved:
        valid, clusters, high = saved['valid_indices'], saved['cluster_labels'], saved['high_mask']
        count, threshold2 = int(saved['n_components']), float(saved['probability_cut'])
        if not np.array_equal(labels, saved['labels']) or not np.isin(valid, original).all():
            raise ValueError(f'Stage-2 regions do not match their stage-1 parent: {name}')
    fig, axes = plt.subplots(2, count + 1, figsize=(5 * (count + 1), 10), squeeze=False)
    try:
        def draw(ax, indices, title):
            _draw_regions(ax, image, segments, labels, indices, title)

        draw(axes[0, 0], original, f'Stage 1 parent {parent}: all {len(original)} SP')
        draw(axes[1, 0], original_high, f'Stage 1 probability >= {threshold1:.2f}\n{len(original_high)} SP')
        for cluster in range(count):
            selected = clusters == cluster
            draw(axes[0, cluster + 1], valid[selected], f'Stage 2 cluster {cluster}: all {selected.sum()} SP')
            draw(axes[1, cluster + 1], valid[selected & high], f'Stage 2 probability >= {threshold2:.2f}\nCluster {cluster}: {(selected & high).sum()} SP')
        for ax in axes.ravel():
            ax.axis('off')
        fig.suptitle(f"Image {record['image_id']} | {record['name']} | Parent {parent} split into {count} clusters")
        fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
        return fig
    except Exception:
        plt.close(fig)
        raise


def show_training_gmm_stage2(run_dir, image_ids):
    """Compare saved stage-1 parents and stage-2 children for selected images."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from IPython.display import Image, display

    run_dir = Path(run_dir)
    _, records = load_run(run_dir)
    available = {r['image_id']: r for r in records}
    settings = pd.read_csv(run_dir / 'gmm/stage2/settings.csv')
    selected = settings[settings.image_id.isin(image_ids)]
    if selected.empty or any(i not in available for i in image_ids):
        raise ValueError('Choose image IDs with saved stage-2 results in gmm/stage2/settings.csv.')
    output_dir = run_dir / 'gmm/stage2/figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_dir / 'review.pdf') as pdf:
        for row in selected.itertuples(index=False):
            fig = _plot_stage2_image(run_dir, available[row.image_id], row.parent_cluster)
            output = output_dir / f'image_{row.image_id:04d}_cluster_{row.parent_cluster:04d}.png'
            try:
                fig.savefig(output, dpi=120)
                pdf.savefig(fig, dpi=120)
            finally:
                plt.close(fig)
            display(Image(filename=str(output)))
            print(f'Saved: {output}')
    print(f"PDF: {output_dir / 'review.pdf'}")
