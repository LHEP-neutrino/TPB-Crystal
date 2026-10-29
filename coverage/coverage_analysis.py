"""Review GMM maps, calculate coverage, and compare plate means with photon yield."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython.display import display
from coverage_pipeline import load_run


def show_gmm(run_dir, image_ids=None, max_display=None, report_every=20):
    if not isinstance(report_every, int) or report_every < 1:
        raise ValueError('report_every must be a positive integer.')
    _, images = load_run(run_dir)
    output = Path(run_dir) / 'review/figures'
    output.mkdir(parents=True, exist_ok=True)
    if image_ids is not None:
        missing = set(image_ids) - set(images.image_id)
        if missing:
            raise ValueError(f'Unknown image IDs: {missing}')
        images = images[images.image_id.isin(image_ids)]
    for index, row in enumerate(images.itertuples(index=False)):
        image = np.load(Path(run_dir) / 'images' / f'image_{row.image_id:04d}.npy')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        try:
            axes.flat[0].imshow(image)
            axes.flat[0].set_title('Original')
            with np.load(Path(run_dir) / 'gmm' / f'image_{row.image_id:04d}.npz') as saved:
                for method, count in enumerate((2, 4, 4)):
                    labels = saved[f'gmm_{method}']
                    fractions = np.bincount(labels.ravel(), minlength=count) / labels.size
                    ax = axes.flat[method + 1]
                    ax.imshow(image)
                    ax.imshow(labels, cmap=ListedColormap(plt.cm.tab10.colors[:count]), alpha=.35, vmin=0, vmax=count-1)
                    title = ('2 clusters', '4 clusters', 'Hierarchical 2 -> 2')[method]
                    ax.set_title(f'GMMlabel {method}: {title}\n' + ' '.join(f'C{k}: {v:.1%}' for k, v in enumerate(fractions)))
            for ax in axes.flat:
                ax.axis('off')
            fig.suptitle(f'Image {row.image_id}: {row.group}/{row.plate}/{row.name}')
            fig.tight_layout()
            fig.savefig(output / f'GMM_{row.image_id:04d}.png', dpi=120, bbox_inches='tight')
            if max_display is None or index < max_display:
                display(fig)
            if (index + 1) % report_every == 0 or index + 1 == len(images):
                print(f'Review figures saved: {index + 1}/{len(images)}', flush=True)
        finally:
            plt.close(fig)


def calculate_coverage(run_dir):
    """Keep legacy settings columns for compatibility with implement coverage tables."""
    _, manifest = load_run(run_dir)
    areas = pd.read_csv(Path(run_dir) / 'gmm/gmm_area_results.csv')
    settings = pd.read_csv(Path(run_dir) / 'review/gmm_settings.csv')
    for frame in (areas, settings):
        frame.columns = frame.columns.str.strip()
        for key in ('group', 'plate', 'name'):
            frame[key] = frame[key].astype(str).str.strip()
    if settings.imageID.duplicated().any() or areas.duplicated(['image_id', 'gmm_label', 'cluster_id']).any():
        raise ValueError('Duplicate image settings or GMM area rows.')
    records, skipped = [], []
    for row in settings.itertuples(index=False):
        info = dict(image_id=row.imageID, group=row.group, plate=row.plate, name=row.name)
        selected = manifest[manifest.image_id == row.imageID]
        if len(selected) != 1 or any(str(selected.iloc[0][k]) != str(info[k]) for k in ('group', 'plate', 'name')):
            raise ValueError(f'Settings do not match the current manifest: {info}')
        cluster_text = '' if pd.isna(row.cluster) else str(row.cluster).strip()
        if cluster_text:
            # pandas may infer a single selected cluster as a float when other entries are blank.
            values = [float(s.strip()) for s in cluster_text.split(',')]
            if any(not np.isfinite(v) or v != int(v) for v in values) or len(set(values)) != len(values):
                raise ValueError(f'Cluster IDs must be unique integers: {info}')
            clusters = [int(v) for v in values]
            if pd.isna(row.GMMlabel) or row.GMMlabel not in (0, 1, 2):
                raise ValueError(f'GMMlabel must be 0, 1 or 2: {info}')
            match = areas[(areas.image_id == row.imageID) & (areas.gmm_label == row.GMMlabel)
                          & (areas.group == row.group) & (areas.plate == row.plate) & (areas.name == row.name)
                          & areas.cluster_id.isin(clusters)]
            if set(match.cluster_id) != set(clusters):
                raise ValueError(f'Selected clusters not found in GMM results: {info}')
            empty, source = match.area_ratio.sum(), 'cluster'
        elif pd.notna(row.fluction):
            empty, source = float(row.fluction), 'fluction'
        else:
            skipped.append({**info, 'reason': 'No empty-area selection'})
            continue
        if not np.isfinite(empty) or not 0 <= empty <= 1:
            raise ValueError(f'Empty fraction must be between 0 and 1: {info}')
        records.append({**info, 'gmm_label': row.GMMlabel, 'cluster': cluster_text, 'fluction': row.fluction,
                        'empty_ratio': empty, 'coverage': 1 - empty, 'source': source})
    image_columns = ['image_id', 'group', 'plate', 'name', 'gmm_label', 'cluster', 'fluction', 'empty_ratio', 'coverage', 'source']
    images = pd.DataFrame(records, columns=image_columns)
    plates = []
    for (group, plate, source), rows in images.groupby(['group', 'plate', 'source']):
        values = rows.coverage.to_numpy()
        if len(values) < 2:
            continue
        leave_one_out = (values.sum() - values) / (len(values) - 1)
        plates.append(dict(group=group, plate=plate, source=source, coverage=values.mean(),
                           coverage_min=leave_one_out.min(), coverage_max=leave_one_out.max(), n_images=len(values)))
    manual_path = Path(run_dir) / 'review/plate_manual_settings.csv'
    if manual_path.exists():
        manual = pd.read_csv(manual_path)
        if manual.duplicated(['group', 'plate']).any():
            raise ValueError('Duplicate plate-level manual settings.')
        for row in manual.itertuples(index=False):
            if not np.isfinite(row.fluction) or not 0 <= row.fluction <= 1:
                raise ValueError('Plate-level empty fractions must be between 0 and 1.')
            plates.append(dict(group=row.group, plate=row.plate, source='plate_manual', coverage=1-row.fluction,
                               coverage_min=np.nan, coverage_max=np.nan, n_images=0))
    output = Path(run_dir) / 'coverage'
    output.mkdir(parents=True, exist_ok=True)
    images.to_csv(output / 'image_coverage.csv', index=False)
    pd.DataFrame(skipped, columns=['image_id', 'group', 'plate', 'name', 'reason']).to_csv(output / 'skipped_settings.csv', index=False)
    plate_csv = output / 'plate_coverage.csv'
    pd.DataFrame(plates, columns=['group', 'plate', 'source', 'coverage', 'coverage_min', 'coverage_max', 'n_images']).to_csv(plate_csv, index=False)
    print(f'Coverage: {len(images)} images; {len(plates)} plate/source groups; {len(skipped)} unselected images.')
    print('Plate summaries require at least two selected images per plate and source.')
    return plate_csv


def plot_photon_yield(run_dir, photon_csv):
    """Plot cluster-based plate means. Error bars show leave-one-image-out mean ranges."""
    output = Path(run_dir) / 'coverage'
    output.mkdir(parents=True, exist_ok=True)
    # Remove stale regression output before a rerun that may have too few data points.
    fit_csv = output / 'linear_fit.csv'
    if fit_csv.exists():
        fit_csv.unlink()
    plates = pd.read_csv(output / 'plate_coverage.csv')
    photon = pd.read_csv(photon_csv)
    photon.columns = photon.columns.str.strip()
    photon = photon.rename(columns={'Type': 'group'})
    photon['group'] = photon['group'].astype(str).str.strip()
    photon['plate'] = photon['plate'].astype(str).str.strip().apply(
        lambda s: f'pl{int(s[2:] if s.startswith("pl") else float(s)):02d}')
    photon['photon_yield'] = pd.to_numeric(photon['photon_yield'], errors='raise')
    if photon.duplicated(['group', 'plate']).any():
        raise ValueError('Duplicate photon yield rows for the same plate.')
    comparison = plates[plates.source == 'cluster'].merge(photon[['group', 'plate', 'photon_yield']], on=['group', 'plate'], how='left')
    for column in ('coverage', 'photon_yield'):
        comparison[column] = pd.to_numeric(comparison[column], errors='raise')
    comparison.to_csv(output / 'photon_comparison.csv', index=False)
    valid = comparison[np.isfinite(comparison.photon_yield) & np.isfinite(comparison.coverage)]
    print(f'Photon comparison: {len(valid)} plates; {len(comparison) - len(valid)} missing photon yields.')
    if valid.empty:
        print('No plot created. Select empty clusters for at least two images per plate, then rerun coverage.')
        for suffix in ('png', 'pdf'):
            path = output / f'coverage_vs_photon_yield.{suffix}'
            if path.exists():
                path.unlink()
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        for group, rows in valid.groupby('group'):
            error = np.maximum(0, np.vstack([rows.coverage - rows.coverage_min, rows.coverage_max - rows.coverage]))
            ax.errorbar(rows.coverage, rows.photon_yield, xerr=error, fmt='o', capsize=4, linestyle='none', label=f'{group} plate')
        x, y = valid.coverage.to_numpy(), valid.photon_yield.to_numpy()
        if len(x) >= 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            predicted = slope * x + intercept
            total = np.sum((y - y.mean()) ** 2)
            r2 = 1 - np.sum((y - predicted) ** 2) / total if total > 0 else np.nan
            grid = np.linspace(0, 1, 100)
            ax.plot(grid, slope * grid + intercept, 'k--', label=f'y = {slope:.1f}x + {intercept:.1f}')
            pd.DataFrame([dict(slope=slope, intercept=intercept, r2=r2, n_plates=len(x))]).to_csv(fit_csv, index=False)
            print(f'Linear fit: y = {slope:.4f}x + {intercept:.4f}; R² = {r2:.4f}')
        else:
            print('Linear fit skipped: need at least two distinct coverage values.')
        ax.set(xlabel='Coverage (image process estimation)', ylabel='Photon yield (10⁶ ADC unit)', xlim=(0, 1))
        ax.legend()
        fig.tight_layout()
        for suffix in ('png', 'pdf'):
            fig.savefig(output / f'coverage_vs_photon_yield.{suffix}', dpi=150, bbox_inches='tight')
        display(fig)
    finally:
        plt.close(fig)
