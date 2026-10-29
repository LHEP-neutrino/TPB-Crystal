"""Inspect sampling variability using all image combinations within each plate."""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t


def sample_means(values, n_use):
    return np.array([values[list(indices)].mean() for indices in combinations(range(len(values)), n_use)])


def _save_figure(fig, path):
    import matplotlib.pyplot as plt
    from IPython.display import Image, display

    try:
        fig.tight_layout()
        fig.savefig(path, dpi=140)
    finally:
        plt.close(fig)
    display(Image(filename=str(path)))


def analyze_sampling(area_csv, target_labels, class_name='fine crystal', group='new', plate='pl01', n_use=2, max_error=None, max_plot_images=7):
    """Save histogram, stability curves, and tables from small area-ratio records.

    Parent labels include underscore-separated subclasses, e.g. 0 includes 0_1.
    The leave-one-out t interval preserves the original calculation as a diagnostic;
    overlapping samples are dependent, so it is not an independent-sample 95% CI.
    max_error is an optional half-range threshold in percentage points.
    max_plot_images limits the stability plot only; saved summaries stay complete.
    """
    import matplotlib.pyplot as plt

    area_csv = Path(area_csv)
    data = pd.read_csv(area_csv)
    columns = [c for c in data if c.startswith('class_') and c.endswith('_area_ratio')]
    columns = [c for c in columns if any(c[6:-11] == str(label) or c[6:-11].startswith(str(label) + '_') for label in target_labels)]
    if not columns:
        raise ValueError(f'No area columns match target labels: {target_labels}')
    data['target_area_ratio'] = data[columns].fillna(0).sum(axis=1)
    selected = data[(data.group == group) & (data.plate == plate)]
    if not isinstance(n_use, int) or not 1 <= n_use <= len(selected):
        raise ValueError(f'n_use must be between 1 and {len(selected)} for {group}/{plate}.')
    if max_error is not None and max_error < 0:
        raise ValueError('max_error must be nonnegative or None.')
    if not isinstance(max_plot_images, int) or max_plot_images < 1:
        raise ValueError('max_plot_images must be a positive integer.')
    output_dir = area_csv.parent / 'sampling' / class_name.replace(' ', '_')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Area columns: {columns}')
    stability, leave_out = [], []
    for (current_group, current_plate), rows in data.groupby(['group', 'plate']):
        values = rows.target_area_ratio.to_numpy()
        n = len(values)
        if n < 2:
            continue
        for k in range(1, n):
            means = sample_means(values, k)
            stability.append({'group': current_group, 'plate': current_plate, 'n_use': k, 'n_images': n,
                              'mean': values.mean(), 'min': means.min(), 'max': means.max(), 'stability': np.ptp(means) / 2})
        # Keep the original leave-one-out summary, with explicit diagnostic names.
        means = sample_means(values, n - 1)
        std = means.std(ddof=1)
        half = t.ppf(0.975, df=n-2) * std / np.sqrt(n-1) if n > 2 else np.nan
        leave_out.append({'group': current_group, 'plate': current_plate, 'n_images': n, 'n_use': n-1,
                          'mean': means.mean(), 'std': std, 'diagnostic_t_low': means.mean()-half,
                          'diagnostic_t_high': means.mean()+half, 'diagnostic_t_half': half})
    stability = pd.DataFrame(stability)
    if stability.empty:
        raise ValueError('At least one plate must have two or more images.')
    required = []
    for (current_group, current_plate), rows in stability.groupby(['group', 'plate']):
        stable = rows[rows.stability <= max_error] if max_error is not None else rows.iloc[:0]
        required.append({**rows.iloc[-1].to_dict(), 'required_images': stable.n_use.iloc[0] if len(stable) else np.nan})
    stability.to_csv(output_dir / 'plate_mean_stability.csv', index=False)
    pd.DataFrame(required).to_csv(output_dir / 'plate_required_images.csv', index=False)
    pd.DataFrame(leave_out).to_csv(output_dir / 'leave_one_out_summary.csv', index=False)
    values = selected.target_area_ratio.to_numpy()
    means = sample_means(values, n_use)
    histogram = pd.DataFrame({'image_ids': [','.join(str(selected.image_id.iloc[i]) for i in indices) for indices in combinations(range(len(values)), n_use)], 'mean_area_ratio': means})
    histogram.to_csv(output_dir / f'histogram_{group}_{plate}_n{n_use}.csv', index=False)
    print(f'{group}/{plate}: {len(values)} images, choose {n_use}, {len(means)} combinations; half range: {np.ptp(means)/2:.3f} percentage points')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(means, bins=10, alpha=0.5, color='C2')
    ax.axvline(values.mean(), linestyle='--', color='C2', label=f'Mean of all {len(values)} images')
    ax.set(xlabel=f'{class_name} area ratio (%)', ylabel=f'Number of combinations ({len(values)} choose {n_use})', title=f'{group}/{plate}')
    ax.legend()
    _save_figure(fig, output_dir / f'histogram_{group}_{plate}_n{n_use}.png')
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = stability[stability.n_use <= max_plot_images]
    for _, rows in plotted.groupby(['group', 'plate']):
        ax.plot(rows.n_use, rows.stability, marker='o', color='C0', alpha=0.45, linewidth=1, markersize=4)
    if max_error is not None:
        ax.axhline(max_error, linestyle='--', color='black', label=f'Threshold: {max_error}')
    ax.set(xlabel='Number of images used', ylabel='(Max - Min) / 2 (percentage points)', title=f'Stability of plate mean: {class_name}')
    ax.set_xticks(range(1, max_plot_images + 1))
    ax.set_xlim(0.8, max_plot_images + 0.2)
    _save_figure(fig, output_dir / 'plate_mean_stability.png')
    print(f'Output: {output_dir}')
