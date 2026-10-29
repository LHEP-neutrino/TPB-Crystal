"""Calculate plate coverage from selected GMM cluster areas."""
from pathlib import Path
import numpy as np
import pandas as pd
from crystal_photon_model import fit_photon_model


def prepare_coverage(settings_csv, result_csv, output_dir):
    """Save image coverage and plate means; use cluster-based rows for fitting."""
    settings, areas = pd.read_csv(settings_csv), pd.read_csv(result_csv)
    for frame in (settings, areas):
        frame.columns = frame.columns.str.strip()
        for column in ('group', 'plate', 'name'):
            frame[column] = frame[column].astype(str).str.strip()
    records, skipped = [], []
    for _, row in settings.iterrows():
        info = {'image_id': row.imageID, **{key: row[key] for key in ('group', 'plate', 'name')}}
        cluster_text = str(row.cluster).strip() if pd.notna(row.cluster) else ''
        if cluster_text:
            if pd.isna(row.GMMlabel):
                skipped.append({**info, 'reason': 'Missing GMMlabel'})
                continue
            cluster_ids = [int(value.strip()) for value in cluster_text.split(',')]
            match = areas[(areas.image_id == row.imageID) & (areas.group == row.group) & (areas.plate == row.plate)
                          & (areas.name == row['name']) & (areas.gmm_label == row.GMMlabel) & areas.cluster_id.isin(cluster_ids)]
            if set(match.cluster_id) != set(cluster_ids):
                skipped.append({**info, 'reason': 'Selected clusters not found'})
                continue
            if match.cluster_id.duplicated().any():
                raise ValueError(f'Duplicate cluster areas for {info}')
            empty, source = match.area_ratio.sum(), 'cluster'
        elif pd.notna(row.fluction):
            empty, source = float(row.fluction), 'fluction'
        else:
            skipped.append({**info, 'reason': 'No coverage selection'})
            continue
        if not np.isfinite(empty) or not 0 <= empty <= 1:
            raise ValueError(f'Empty area ratio must be a fraction between 0 and 1: {info}')
        records.append({**info, 'coverage': 1 - empty, 'source': source})
    images = pd.DataFrame(records, columns=['image_id', 'group', 'plate', 'name', 'coverage', 'source'])
    if images.duplicated(['group', 'plate', 'name']).any():
        raise ValueError('Multiple coverage settings exist for the same image.')
    plates = []
    for (group, plate), rows in images[images.source == 'cluster'].groupby(['group', 'plate']):
        values = rows.coverage.to_numpy()
        if len(values) < 2:
            continue
        means = np.array([np.delete(values, i).mean() for i in range(len(values))])
        plates.append({'group': group, 'plate': plate, 'coverage': values.mean(), 'coverage_min': means.min(),
                       'coverage_max': means.max(), 'n_images': len(values)})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images.to_csv(output_dir / 'image_coverage.csv', index=False)
    pd.DataFrame(skipped, columns=['image_id', 'group', 'plate', 'name', 'reason']).to_csv(output_dir / 'skipped_settings.csv', index=False)
    path = output_dir / 'plate_coverage.csv'
    pd.DataFrame(plates, columns=['group', 'plate', 'coverage', 'coverage_min', 'coverage_max', 'n_images']).to_csv(path, index=False)
    print(f'Coverage: {len(images)} images, {len(plates)} plates; skipped settings: {len(skipped)}')
    return path


def fit_coverage_model(plate_csv, settings_csv, result_csv, categories, bounds=(0, 500)):
    """Scale empty area and its area error by coverage, as in the original model.

    Coverage variation is saved for inspection but is not propagated into the
    model sensitivity bars. Plates without coverage retain their original area.
    """
    output_dir = Path(plate_csv).parent / 'model_coverage'
    coverage_csv = prepare_coverage(settings_csv, result_csv, output_dir)
    fit_photon_model(plate_csv, categories, bounds=bounds, coverage_csv=coverage_csv)
