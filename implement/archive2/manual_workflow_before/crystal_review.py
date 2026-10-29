"""Prepare editable cluster labels for manual review."""
from pathlib import Path
import pandas as pd


def prepare_corrections(run_dir):
    """Create a CSV once; keep existing manual edits on subsequent runs."""
    run_dir = Path(run_dir)
    output = run_dir / "review" / "cluster_corrections.csv"
    if output.exists():
        print(f"Existing correction file kept: {output}")
        return output
    clusters = pd.read_csv(run_dir / "rf" / "cluster_results.csv", dtype={"stage1_label": str, "stage2_label": str, "final_label": str})
    images = pd.read_csv(run_dir / "metadata" / "images.csv")
    table = clusters.merge(images[["image_id", "group", "plate", "name"]], on="image_id", how="left", validate="many_to_one")
    columns = ["image_id", "group", "plate", "name", "cluster_id", "N_superpixels", "gmm_mean_confidence",
               "stage1_label", "stage2_label", "final_label", "final_majority", "final_rf_confidence"]
    table = table[columns].sort_values(["image_id", "cluster_id"])
    table["new_class"] = ""
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"Saved {len(table)} clusters: {output}")
    print("Fill new_class only for clusters you want to change; leave other rows blank.")
    print("Edits will be applied in a later cell. RF predictions are unchanged.")
    return output


def _save_cluster_figure(run_dir, record, cluster_table, output_dir):
    """Keep image arrays and the figure local to one image."""
    import numpy as np
    import matplotlib.pyplot as plt

    name = f"image_{record['image_id']:04d}"
    image = np.load(run_dir / "preprocessing" / name / "image.npy", allow_pickle=False)
    segments = np.load(run_dir / "preprocessing" / name / "segments.npy", allow_pickle=False)
    with np.load(run_dir / "gmm" / f"{name}.npz", allow_pickle=False) as saved:
        labels, valid, clusters = saved["labels"], saved["valid_indices"], saved["cluster_labels"]
        count = int(saved["n_components"])
    fig, axes = plt.subplots((count + 2) // 2, 2, figsize=(14, 5 * ((count + 2) // 2)), squeeze=False)
    try:
        axes = axes.ravel()
        axes[0].imshow(image)
        axes[0].set_title("Cropped input image")
        for cluster in range(count):
            selected = labels[valid[clusters == cluster]]
            mask = np.isin(segments, selected)
            view = np.full_like(image, 255)
            view[mask] = image[mask]
            rows = cluster_table[(cluster_table["image_id"] == record["image_id"]) & (cluster_table["cluster_id"] == cluster)]
            title = f"Cluster {cluster} | {len(selected)} superpixels"
            if not rows.empty:
                row = rows.iloc[0]
                title += f"\nRF: {row['stage1_label']} -> {row['stage2_label']} -> {row['final_label']} | Majority: {row['final_majority']:.3f}"
            axes[cluster + 1].imshow(view)
            axes[cluster + 1].set_title(title)
            del view, mask
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Image {record['image_id']} | {record['group']}/{record['plate']}/{record['name']}")
        fig.tight_layout()
        output = output_dir / f"{name}.png"
        fig.savefig(output, dpi=140)
    finally:
        plt.close(fig)
    return output


def show_clusters(run_dir, group="old", plate="pl01"):
    """Save and display review figures for one plate, without editing predictions."""
    from IPython.display import Image, display
    from crystal_pipeline import load_run

    run_dir = Path(run_dir)
    _, records = load_run(run_dir)
    selected = [r for r in records if r["group"] == group and r["plate"] == plate]
    if not selected:
        raise ValueError(f"No selected images for {group}/{plate}. Check Cell 1.")
    clusters = pd.read_csv(run_dir / "rf" / "cluster_results.csv", dtype={"stage1_label": str, "stage2_label": str, "final_label": str})
    output_dir = run_dir / "review" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in selected:
        output = _save_cluster_figure(run_dir, record, clusters, output_dir)
        display(Image(filename=str(output)))
        print(f"Saved: {output}")
