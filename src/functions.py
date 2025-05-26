"""Functions used in the analyses."""

import os
import re
from fnmatch import fnmatch
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import nilearn.masking
import numpy as np
import pandas as pd
import seaborn as sns


def get_data(root, mask_img, ids=None):
    """
    Get feature maps in ON and OFF states.

    Args:
        root (str or Path): Path to the root directory containing the data
        mask_img (str or Path): Path to the mask image
        ids (list, optional): List of IDs to filter the data

    Returns:
        two arrays containing the on and off feature vectors for all subjects

    """
    pattern_on = "*ON*.nii"
    pattern_off = "*OFF*.nii"
    on_state_paths = []
    off_state_paths = []

    # Walk through the directory to find ON and OFF files
    for path,_,files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern_on):
                on_state_paths.append(Path(path) / name)
            if fnmatch(name, pattern_off):
                off_state_paths.append(Path(path) / name)

    # Filter out paths containing '105'
    on_state_paths = [path for path in on_state_paths if "105" not in str(path)]
    off_state_paths = [path for path in off_state_paths if "105" not in str(path)]

    # Sort paths based on numerical values in filenames
    on_paths = sorted(
        on_state_paths,
        key=lambda path: int(re.search(r"(\d+)", str(path).split("/")[-1]).group())
        if re.search(r"(\d+)", str(path).split("/")[-1])
        else 0,
    )
    off_paths = sorted(
        off_state_paths,
        key=lambda path: int(re.search(r"(\d+)", str(path).split("/")[-1]).group())
        if re.search(r"(\d+)", str(path).split("/")[-1])
        else 0,
    )

    # If 'ids' is provided, filter the paths by subject ID
    if ids is not None:
        subject_numbers = [int(Path.path.split("_")[0]) for path in on_paths]
        on_paths = [
            path
            for path, num in zip(on_paths, subject_numbers, strict=True)
            if f"SUBJECT_{num:03d}" in ids["Subject"].to_numpy()
        ]
        off_paths = [
            path
            for path, num in zip(off_paths, subject_numbers, strict=True)
            if f"SUBJECT_{num:03d}" in ids["Subject"].to_numpy()
        ]

    # Load scans
    on_scans = np.array([nib.load(path) for path in on_paths])
    off_scans = np.array([nib.load(path) for path in off_paths])

    # Apply mask to the scans
    on_features = nilearn.masking.apply_mask(on_scans, mask_img, smoothing_fwhm=None)
    off_features = nilearn.masking.apply_mask(off_scans, mask_img, smoothing_fwhm=None)

    return on_features, off_features


def plot_auc_heatmap(file_path, model_list):
    """
    Generate a square-like heatmap plot of AUC scores for each model-feature map pair.

    Args:
        file_path (str or Path): Path to the AUC results. Each sheet should
                                 represent one dataset and contain model names in the first column
                                 and corresponding AUC scores in the second column
        model_list (list of str): List of model names

    Returns:
        None: display and save the plot.

    """
    xls = pd.ExcelFile(file_path)

    dataset_list = []
    auc_list = []

    # Get perfrormences per feature map
    for sheet_name in xls.sheet_names:
        df_results = pd.read_excel(xls, sheet_name=sheet_name)

        # Get model names
        models = df_results.iloc[:, 0]

        # Get AUC scores
        mean_auc = df_results.iloc[:, 1]

        # Add data to the lists
        dataset_list.extend([sheet_name] * len(models))
        auc_list.extend(mean_auc)

    # Results export
    auc_long = pd.DataFrame({"Model": model_list, "Dataset": dataset_list, "AUC": auc_list})

    # Remove underscores from names
    auc_long["Dataset"] = auc_long["Dataset"].str.replace("_", " ", regex=False)

    # Plot figure heatmap
    plt.figure(figsize=(13, 8))
    ax = sns.scatterplot(
        data=auc_long,
        x="Dataset",
        y="Model",
        size="AUC",
        hue="AUC",
        palette="RdYlBu",
        marker="s",
        sizes=(1000, 2400),
        legend=False,
    )

    # Add text annotations
    for i in range(len(auc_long)):
        ax.text(
            x=auc_long["Dataset"][i],
            y=auc_long["Model"][i],
            s=f"{auc_long['AUC'][i]:.2f}",
            color="black",
            ha="center",
            va="center",
            fontsize=13,
        )

    norm = plt.Normalize(vmin=auc_long["AUC"].min(), vmax=auc_long["AUC"].max())
    sm = plt.cm.ScalarMappable(cmap="RdYlBu", norm=norm)
    sm.set_array([])

    # Add the colorbar with min/max ticks
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("AUC Score", fontsize=15)

    # Round the min and max AUC values
    min_auc_rounded = round(auc_long["AUC"].min(), 2)
    max_auc_rounded = round(auc_long["AUC"].max(), 2)

    # Set ticks to show the rounded min and max values
    cbar.set_ticks([min_auc_rounded, max_auc_rounded])
    cbar.ax.tick_params(labelsize=15)

    plt.xticks(fontsize=15, rotation=45)
    plt.yticks(fontsize=15)

    plt.tight_layout()
    plt.savefig("heatmap.svg")
    plt.show()


def plot_metrics(file_path, metrics, model_names):
    """
    Plot metrics across feature maps and models.

    Args:
        file_path (str or Path): Path to results (.xlsx). Each sheet should contain metric values
                                 for each model, with model names in the first column.
        metrics (list of str): List of metric names to plot (e.g., ["AUC", "F1"]).
        model_names (list of str): Ordered list of model names to display on the x-axis.

    Returns:
        None: display and save the plot.

    """
    # Read data
    xls = pd.ExcelFile(file_path)
    datasets = xls.sheet_names

    data_for_plot = {metric: {} for metric in metrics}

    # Get data
    for dataset in datasets:
        df_map = xls.parse(dataset)
        df_map.set_index(df_map.columns[0])
        df_map.index = model_names

        # Extract values for each metric
        for metric in metrics:
            if metric in df_map.columns:
                data_for_plot[metric][dataset] = df_map[metric]

    # Create subplots for each metric
    _, axes = plt.subplots(len(metrics), 1, figsize=(7, 5 * len(metrics)))

    # Loop through each metric and plot it
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for dataset in datasets:
            metric_values = data_for_plot[metric].get(dataset)
            if metric_values is not None:
                x_pos = range(len(model_names))
                ax.scatter(
                    x_pos, metric_values, label=dataset.replace("_", " "), s=100, edgecolor="black", linewidth=1
                )
                ax.plot(x_pos, metric_values, linestyle="-", alpha=0.6)

        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, fontsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_ylabel(metric)

        ax.set_ylim(0, 1)
        ax.legend(title="Datasets", loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig("dotplot.svg")
    plt.show()
