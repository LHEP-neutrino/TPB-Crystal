"""Fit the original bounded weighted photon-yield model from saved plate data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.metrics import r2_score, mean_squared_error


def _fit(X, y, scale, bounds):
    result = lsq_linear(X / scale[:, None], y / scale, bounds=bounds)
    if not result.success:
        raise RuntimeError(f'Photon model fit failed: {result.message}')
    return result.x


def fit_photon_model(plate_csv, categories, bounds=(0, 500), coverage_csv=None):
    """Use leave-one-image-out half-ranges as weights and inspect plate omission.

    Area percentages are always divided by 100. Error bars follow the original
    sensitivity calculation and are not calibrated prediction intervals.
    R2 and RMSE describe the fit to the same plates used for training.
    """
    import matplotlib.pyplot as plt
    from IPython.display import Image, display

    plate_csv = Path(plate_csv)
    table = pd.read_csv(plate_csv)
    keys = ['group', 'plate']
    means = table.pivot(index=keys, columns='category', values='mean')[categories]
    errors = table.pivot(index=keys, columns='category', values='half_range')[categories]
    photons = table.groupby(keys).photon_yield.agg(['first', 'nunique'])
    if not (photons['nunique'] == 1).all():
        raise ValueError('Photon yield differs between categories for the same plate.')
    X, area_error = means.to_numpy() / 100, errors.to_numpy() / 100
    y = photons.reindex(means.index)['first'].to_numpy()
    coverage = np.ones(len(y))
    coverage_used = np.zeros(len(y), dtype=bool)
    if coverage_csv is not None:
        saved = pd.read_csv(coverage_csv).set_index(keys)
        if not saved.index.is_unique:
            raise ValueError('Coverage has duplicate group/plate records.')
        aligned = saved.reindex(means.index)['coverage']
        coverage_used = aligned.notna().to_numpy()
        coverage = aligned.fillna(1.0).to_numpy()
        if not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
            raise ValueError('Coverage must be a fraction between 0 and 1.')
        empty = categories.index('empty')
        X[:, empty] *= coverage
        area_error[:, empty] *= coverage
    if len(y) < 3 or not all(np.isfinite(a).all() for a in (X, area_error, y)):
        raise ValueError('At least three plates with finite area means, errors, and photon yields are required.')
    scale = np.linalg.norm(area_error, axis=1)
    positive = scale[scale > 0]
    scale[scale == 0] = positive.min() if len(positive) else 1.0
    coefficients = _fit(X, y, scale, bounds)
    predicted = X @ coefficients
    loo_coefficients, loo_predictions = [], []
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        fitted = _fit(X[keep], y[keep], scale[keep], bounds)
        loo_coefficients.append(fitted)
        loo_predictions.append(X[index] @ fitted)
    loo_coefficients = np.array(loo_coefficients)
    coefficient_error = np.ptp(loo_coefficients, axis=0) / 2
    error_area = np.linalg.norm(area_error * coefficients, axis=1)
    error_coef = np.linalg.norm(X * coefficient_error, axis=1)
    total_error = np.hypot(error_area, error_coef)
    output_dir = plate_csv.parent / ('model_coverage' if coverage_csv is not None else 'model')
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = means.index.to_frame(index=False)
    predictions['measured_photon_yield'] = y
    predictions['predicted_photon_yield'] = predicted
    predictions['leave_one_plate_out_prediction'] = loo_predictions
    predictions['error_from_area'] = error_area
    predictions['error_from_coefficients'] = error_coef
    predictions['sensitivity_error'] = total_error
    predictions['weight'] = 1 / scale**2
    if coverage_csv is not None:
        predictions['coverage_factor'] = coverage
        predictions['coverage_used'] = coverage_used
        print(f'Coverage applied to {coverage_used.sum()}/{len(y)} plates; missing coverage uses factor 1.')
    for index, name in enumerate(categories):
        predictions[f'{name}_mean_fraction'] = X[:, index]
        predictions[f'{name}_area_error'] = area_error[:, index]
        predictions[f'{name}_contribution'] = X[:, index] * coefficients[index]
        predictions[f'{name}_contribution_ratio'] = np.divide(X[:, index] * coefficients[index], predicted, out=np.full(len(y), np.nan), where=predicted != 0)
    coefficient_table = pd.DataFrame({'category': categories, 'coefficient': coefficients, 'loo_half_range': coefficient_error,
                                      'loo_min': loo_coefficients.min(axis=0), 'loo_max': loo_coefficients.max(axis=0)})
    predictions.to_csv(output_dir / 'plate_predictions.csv', index=False)
    coefficient_table.to_csv(output_dir / 'coefficients.csv', index=False)
    loo = pd.DataFrame(loo_coefficients, columns=categories)
    loo.insert(0, 'omitted_plate', means.index.get_level_values('plate'))
    loo.insert(0, 'omitted_group', means.index.get_level_values('group'))
    loo.to_csv(output_dir / 'leave_one_plate_out_coefficients.csv', index=False)
    metrics = {'training_r2': r2_score(y, predicted), 'training_rmse': float(np.sqrt(mean_squared_error(y, predicted))),
               'leave_one_plate_out_rmse': float(np.sqrt(mean_squared_error(y, loo_predictions))),
               'n_plates': len(y), 'design_rank': int(np.linalg.matrix_rank(X)), 'bounds': list(bounds),
               'error_bar_definition': 'Original area and coefficient sensitivity calculation; not a prediction interval.'}
    if coverage_csv is not None:
        metrics['coverage_file'] = str(coverage_csv)
        metrics['coverage_uncertainty_included'] = False
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    print(coefficient_table.to_string(index=False))
    print(f"Training R2: {metrics['training_r2']:.4f} | Training RMSE: {metrics['training_rmse']:.4f} | Leave-one-plate-out RMSE: {metrics['leave_one_plate_out_rmse']:.4f}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    try:
        for group in predictions.group.unique():
            selected = predictions.group == group
            axes[0].errorbar(y[selected], predicted[selected], yerr=total_error[selected], fmt='o', capsize=4, label=group)
        limit = max(y.max(), (predicted + total_error).max()) * 1.05
        axes[0].plot([0, limit], [0, limit], '--', color='black', label='y = x')
        axes[0].set(xlabel='Measured photon yield', ylabel='Fitted photon yield', title='Training fit | sensitivity bars', xlim=(0, limit), ylim=(0, limit))
        axes[0].legend()
        axes[1].errorbar(np.arange(len(categories)), coefficients, yerr=coefficient_error, fmt='o', capsize=5)
        axes[1].set_xticks(np.arange(len(categories)), categories)
        axes[1].set(ylabel='Photon yield coefficient (area fraction)', title='Coefficients | plate-omission half-range')
        fig.tight_layout()
        fig.savefig(output_dir / 'photon_model.png', dpi=160)
        fig.savefig(output_dir / 'photon_model.pdf')
    finally:
        plt.close(fig)
    display(Image(filename=str(output_dir / 'photon_model.png')))
    print(f'Output: {output_dir}')
