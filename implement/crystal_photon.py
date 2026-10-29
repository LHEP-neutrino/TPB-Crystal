"""Plot four crystal area categories against photon yield in one 2-by-2 figure."""
from pathlib import Path
import numpy as np
import pandas as pd


def plot_area_photon(area_csv, photon_csv, class_groups, min_area=0.0):
    """Save plate means, leave-one-out ranges, and a four-panel figure.

    Horizontal bars show the min/max of leave-one-out means, not a confidence
    interval. They are calculated for plates with at least two images.
    """
    import matplotlib.pyplot as plt
    from IPython.display import Image, display

    if len(class_groups) != 4:
        raise ValueError('Exactly four class groups are required for a 2-by-2 figure.')
    area_csv = Path(area_csv)
    images = pd.read_csv(area_csv)
    photons = pd.read_csv(photon_csv, skipinitialspace=True)
    photons.columns = photons.columns.str.strip()
    photons['group'] = photons['Type'].str.strip()
    photons['plate'] = photons['plate'].map(lambda value: f'pl{int(value):02d}')
    photons['photon_yield'] = pd.to_numeric(photons['photon_yield'], errors='raise')
    columns = [c for c in images if c.startswith('class_') and c.endswith('_area_ratio')]
    rows = []
    for category, labels in class_groups.items():
        selected = [c for c in columns if any(c[6:-11] == str(label) or c[6:-11].startswith(str(label) + '_') for label in labels)]
        if not selected:
            raise ValueError(f'No area columns match {category}: {labels}')
        images[category] = images[selected].fillna(0).sum(axis=1)
        for (group, plate), data in images.groupby(['group', 'plate']):
            values = data[category].to_numpy()
            mean = values.mean()
            leave_out = np.array([np.delete(values, i).mean() for i in range(len(values))]) if len(values) > 1 else np.array([np.nan])
            rows.append({'category': category, 'group': group, 'plate': plate, 'n_images': len(values), 'mean': mean,
                         'leave_one_out_min': leave_out.min(), 'leave_one_out_max': leave_out.max(),
                         'half_range': np.ptp(leave_out) / 2})
    table = pd.DataFrame(rows).merge(photons[['group', 'plate', 'photon_yield']], on=['group', 'plate'], how='left', validate='many_to_one')
    if table.photon_yield.isna().any():
        missing = table.loc[table.photon_yield.isna(), ['group', 'plate']].drop_duplicates()
        raise ValueError(f'Missing photon yield for:\n{missing.to_string(index=False)}')
    output_dir = area_csv.parent / 'photon_yield'
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / 'plate_area_photon.csv', index=False)
    images[['image_id', 'group', 'plate', *class_groups]].to_csv(output_dir / 'image_category_areas.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    styles = {'old': ('C0', 'o'), 'new': ('C1', 's')}
    try:
        for ax, category in zip(axes.ravel(), class_groups):
            for group, data in table[(table.category == category) & (table['mean'] >= min_area)].groupby('group'):
                color, marker = styles.get(group, ('C2', '^'))
                low = np.maximum(0, data['mean'] - data.leave_one_out_min)
                high = np.maximum(0, data.leave_one_out_max - data['mean'])
                ax.errorbar(data['mean'], data.photon_yield, xerr=np.array([low, high]), fmt=marker, color=color, capsize=4, label=group)
            ax.set(title=category, xlabel='Mean area ratio (%)', ylabel='Photon yield', xlim=(-2, 102))
            ax.legend()
        fig.suptitle('Area ratio vs photon yield | bars: leave-one-out range')
        fig.tight_layout()
        fig.savefig(output_dir / 'area_vs_photon.png', dpi=160)
        fig.savefig(output_dir / 'area_vs_photon.pdf')
    finally:
        plt.close(fig)
    display(Image(filename=str(output_dir / 'area_vs_photon.png')))
    print(f'Saved {len(table)} plate/category results: {output_dir}')


def _separated_bar_positions(values, spacing):
    """Least-squares horizontal shifts with ordered centers at least spacing apart."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(np.diff(values) < 0) or spacing <= 0:
        raise ValueError('Bar values must be finite, sorted, and spacing positive.')
    # Isotonic regression of x[i] - spacing*i using pooled adjacent violators.
    blocks = []
    for value in values - spacing * np.arange(len(values)):
        blocks.append([float(value), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            right, left = blocks.pop(), blocks.pop()
            count = left[1] + right[1]
            blocks.append([(left[0] * left[1] + right[0] * right[1]) / count, count])
    fitted = np.concatenate([np.full(count, mean) for mean, count in blocks])
    return fitted + spacing * np.arange(len(values))


def plot_stacked_area_photon(plate_csv, categories, bar_width=2.4, gap=0.6):
    """Stack plate-mean whole-image fractions against measured photon yield.

    Horizontal shifts affect display only. Every image within a plate receives
    equal weight, matching the photon model. Missing RF categories remain zero;
    any fraction outside the four categories is shown as Other, not renormalized.
    """
    import matplotlib.pyplot as plt
    from IPython.display import Image, display

    if bar_width <= 0 or gap < 0:
        raise ValueError('bar_width must be positive and gap nonnegative.')
    plate_csv = Path(plate_csv)
    table = pd.read_csv(plate_csv)
    keys = ['group', 'plate']
    if table.duplicated(keys + ['category']).any():
        raise ValueError('Plate/category records must be unique.')
    if table.groupby(keys).photon_yield.nunique().ne(1).any():
        raise ValueError('Each plate must have one measured photon yield.')
    fractions = table.pivot(index=keys, columns='category', values='mean').reindex(columns=categories)
    if fractions.empty or fractions.isna().any().any():
        raise ValueError('All selected categories must be present for every plate.')
    fractions = fractions / 100
    if not np.isfinite(fractions.to_numpy()).all() or (fractions < 0).any().any() or (fractions.sum(axis=1) > 1 + 1e-8).any():
        raise ValueError('Area fractions must be finite, nonnegative and sum to at most one.')
    data = fractions.join(table.groupby(keys).agg(photon_yield=('photon_yield', 'first'), n_images=('n_images', 'first'))).reset_index()
    data = data.sort_values(['photon_yield', 'group', 'plate'], kind='stable').reset_index(drop=True)
    data['display_photon_yield'] = _separated_bar_positions(data.photon_yield, bar_width + gap)
    data['horizontal_shift'] = data.display_photon_yield - data.photon_yield
    data['other_fraction'] = np.maximum(0, 1 - data[categories].sum(axis=1))
    output_dir = plate_csv.parent
    data.to_csv(output_dir / 'stacked_area_photon.csv', index=False)
    colors = {'fine': 'tab:blue', 'elongated': 'tab:orange', 'plate-like': 'tab:green', 'empty': 'tab:red'}
    names = {'fine': 'fine crystal', 'elongated': 'elongated crystal', 'plate-like': 'plate-like crystal', 'empty': 'empty'}
    fig, ax = plt.subplots(figsize=(17, 7))
    try:
        bottom = np.zeros(len(data))
        for category in categories:
            values = data[category].to_numpy()
            ax.bar(data.display_photon_yield, values, width=bar_width, bottom=bottom,
                   color=colors.get(category), edgecolor='white', linewidth=0.4, label=names.get(category, category))
            bottom += values
        if data.other_fraction.max() > 1e-8:
            ax.bar(data.display_photon_yield, data.other_fraction, width=bar_width, bottom=bottom,
                   color='0.7', edgecolor='white', linewidth=0.4, label='other (RF 4 / unclassified)')
        for row in data.itertuples(index=False):
            ax.text(row.display_photon_yield, 1.018, row.plate, rotation=90,
                    ha='center', va='bottom', fontsize=7,
                    color={'new': 'tab:blue', 'old': 'tab:orange'}.get(row.group, 'black'),
                    clip_on=False)
        ax.set(xlabel='Photon yield', ylabel='Area ratio', ylim=(0, 1), yticks=np.arange(0, 1.01, 0.2))
        ax.set_xlim(min(data.display_photon_yield.min(), data.photon_yield.min()) - 6,
                    max(data.display_photon_yield.max(), data.photon_yield.max()) + 6)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
        ax.spines[['top', 'right']].set_visible(True)
        fig.text(0.1, 0.015, 'One bar per plate; mean over all images. Nearby bars are shifted horizontally for visibility.', fontsize=9)
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        fig.savefig(output_dir / 'stacked_area_photon.png', dpi=180)
        fig.savefig(output_dir / 'stacked_area_photon.pdf')
    finally:
        plt.close(fig)
    display(Image(filename=str(output_dir / 'stacked_area_photon.png')))
    print(f'{len(data)} plates; maximum display shift: {data.horizontal_shift.abs().max():.2f}. True yields and shifts: {output_dir / "stacked_area_photon.csv"}')
