"""Image selection and run metadata for crystal segmentation."""
import json
import re
from pathlib import Path
import pandas as pd


def prepare_run(project_root, run_dir, config):
    """Save selected image paths and settings; images are not loaded."""
    data_dir = Path(project_root) / config["data_dir"]
    limit = config["images_per_plate"]
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("images_per_plate must be a positive integer or None for all images.")
    records = []
    for group, plates in config["target_plates"].items():
        for plate in plates:
            folder = data_dir / group / plate
            paths = sorted(folder.glob("*.jpg"), key=_natural_key)[:limit]
            if not paths:
                raise FileNotFoundError(f"No JPG images found in: {folder}")
            for path in paths:
                records.append({"image_id": len(records), "group": group, "plate": plate,
                                "name": path.stem, "path": path.relative_to(project_root).as_posix()})
    if not records:
        raise ValueError("No images selected. Check target_plates.")
    metadata_dir = Path(run_dir) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(metadata_dir / "images.csv", index=False)
    (metadata_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Selected images: {len(records)} | Metadata: {metadata_dir}")


def _natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def load_run(run_dir):
    """Read the small metadata files used by each processing step."""
    metadata_dir = Path(run_dir) / "metadata"
    config = json.loads((metadata_dir / "config.json").read_text(encoding="utf-8"))
    records = pd.read_csv(metadata_dir / "images.csv").to_dict("records")
    return config, records


def plate_progress_points(records, report_every=20):
    """Report after every N completed group/plate pairs, plus completion."""
    if not isinstance(report_every, int) or isinstance(report_every, bool) or report_every < 1:
        raise ValueError('report_every_plates must be a positive integer.')
    last = {(r['group'], r['plate']): i for i, r in enumerate(records, 1)}
    total = len(last)
    return {index: (count, total) for count, index in enumerate(sorted(last.values()), 1)
            if count % report_every == 0 or count == total}


def image_area(run_dir, record, labels, final_labels, label_source, confidence=None):
    """Count segment pixels by final class, optionally summarizing confidence."""
    import numpy as np

    name = f"image_{record['image_id']:04d}"
    segments = np.load(run_dir / 'preprocessing' / name / 'segments.npy', mmap_mode='r', allow_pickle=False)
    pixel_labels, counts = np.unique(segments, return_counts=True)
    if not np.array_equal(pixel_labels, labels):
        raise ValueError(f'Segment labels do not match {label_source}: {name}')
    area = {key: record[key] for key in ('image_id', 'group', 'plate', 'name')}
    area['image_path'] = record['path']
    for label in np.unique(final_labels):
        if label == '':
            continue
        mask = final_labels == label
        pixels = counts[mask].sum()
        area.update({f'class_{label}_area_ratio': pixels / segments.size * 100,
                     f'class_{label}_area_pixels': pixels, f'class_{label}_N': mask.sum()})
    if confidence is not None:
        selected = confidence[confidence > 0]
        area['mean_confidence'] = selected.mean() if len(selected) else np.nan
        area['median_confidence'] = np.median(selected) if len(selected) else np.nan
    return area
