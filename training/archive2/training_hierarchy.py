"""Prepare, train, and review child RF models using the stage-1 image split."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from training_dataset import training_filter
from training_rf import train_stage1_rf


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def is_trained(model_dir):
    path = Path(model_dir) / 'training_status.json'
    return path.is_file() and read_json(path).get('status') == 'trained'


def class_routes(stage1_settings, stage2_groups=None, stage3_groups=None):
    """Translate manual labels into the concatenated class paths used by implement."""
    routes = {int(label): str(value) for label, value in stage1_settings['label_map'].items()}
    for stage in (stage2_groups or {}, stage3_groups or {}):
        updated, assigned = {}, set()
        for parent, groups in stage.items():
            if not groups or len(groups) > 10:
                raise ValueError('Each parent must define 1-10 child groups.')
            for child, labels in enumerate(groups.values()):
                for label in labels:
                    if label in assigned or routes.get(label) != str(parent):
                        raise ValueError(f'Manual label {label} does not uniquely belong to parent {parent}.')
                    assigned.add(label)
                    updated[label] = str(parent) + str(child)
        # A configured parent must cover all of its manual labels.
        missing = [label for label, parent in routes.items() if parent in stage and label not in assigned]
        if missing:
            raise ValueError(f'Child groups omit manual labels: {missing}')
        routes.update(updated)
    return routes


def prepare_hierarchy_stage(run_dir, stage, groups):
    """Build truth-routed child datasets; model predictions never select training rows."""
    if stage not in (2, 3):
        raise ValueError('stage must be 2 or 3.')
    run_dir = Path(run_dir)
    base = read_json(run_dir / 'datasets/stage1_settings.json')
    stage2 = read_json(run_dir / 'datasets/stage2/tasks.json')['groups'] if stage == 3 else groups
    parent_routes = class_routes(base, stage2 if stage == 3 else None)
    class_routes(base, stage2, groups if stage == 3 else None)  # Validate every configured mapping.
    raw = pd.read_csv(run_dir / 'datasets/superpixels.csv')
    data = raw[training_filter(raw, base['probability_cut'], base['excluded_labels']) == 'eligible'].copy()
    data['parent_label'] = data.manual_label.map(parent_routes)
    output_dir = run_dir / 'datasets' / f'stage{stage}'
    tasks, counts = [], []
    for parent, children in groups.items():
        selected = data[data.parent_label == str(parent)].copy()
        mapping = {label: child for child, labels in enumerate(children.values()) for label in labels}
        selected['rf_label'] = selected.manual_label.map(mapping).astype(int)
        task_dir = output_dir / f'parent_{parent}'
        task_dir.mkdir(parents=True, exist_ok=True)
        selected.to_csv(task_dir / 'training.csv', index=False)
        settings = {**base, 'groups': children, 'label_map': mapping, 'name_map': dict(enumerate(children)), 'stage': stage, 'parent_label': str(parent)}
        write_json(task_dir / 'settings.json', settings)
        task = {'parent': str(parent), 'model_subdir': f'stage{stage}/parent_{parent}',
                'dataset': f'datasets/stage{stage}/parent_{parent}/training.csv',
                'settings_file': 'settings.json', 'model_file': 'rf_model.pkl',
                'reuse_stage1_split': True, 'require_all_classes': True}
        tasks.append(task)
        for child, name in enumerate(children):
            counts.append({'parent': parent, 'rf_label': child, 'name': name, 'n_superpixels': int((selected.rf_label == child).sum())})
        print(f'Stage {stage}, parent {parent}: {len(selected)} eligible superpixels')
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'tasks.json', {'stage': stage, 'groups': groups, 'tasks': tasks})
    pd.DataFrame(counts).to_csv(output_dir / 'class_counts.csv', index=False)
    print(f'Saved: {output_dir}')


def train_hierarchy_stage(project_root, run_dir, stage, n_estimators=300, random_state=0):
    """Train each child task on the fixed stage-1 partitions, or explain missing data."""
    run_dir = Path(run_dir)
    tasks = read_json(run_dir / f'datasets/stage{stage}/tasks.json')['tasks']
    for task in tasks:
        model_dir = run_dir / 'models' / task['model_subdir']
        if not is_trained(run_dir / 'models/stage1'):
            reason = 'Stage 1 has no trained evaluation model and fixed image split. Complete Cell 14 first.'
            write_json(model_dir / 'training_status.json', {'status': 'not_trained', 'reason': reason})
            print(f"Parent {task['parent']}: {reason}")
            continue
        print(f"Training stage {stage}, parent {task['parent']}")
        train_stage1_rf(project_root, run_dir, run_dir / task['dataset'], n_estimators=n_estimators, random_state=random_state, task=task)


def show_hierarchy_stage(run_dir, stage, max_error_groups=1):
    """Reuse the stage-1 evaluation plots for every trained child task."""
    from training_evaluation import show_rf_evaluation, show_rf_errors

    run_dir = Path(run_dir)
    for task in read_json(run_dir / f'datasets/stage{stage}/tasks.json')['tasks']:
        print(f"Stage {stage}, parent {task['parent']}")
        settings = (run_dir / task['dataset']).parent / 'settings.json'
        show_rf_evaluation(run_dir, model_subdir=task['model_subdir'], settings_path=settings)
        show_rf_errors(run_dir, max_groups=max_error_groups, model_subdir=task['model_subdir'])
