"""Functions used in the analyses."""

import matplotlib.pyplot as plt

# import nibabel as nib
# import nilearn.masking
# import numpy as np
import pandas as pd
import seaborn as sns
from aeon.visualisation import plot_critical_difference, plot_significance
from scipy.stats import friedmanchisquare


def plot_auc_heatmap(file_path, model_list):
    xls = pd.ExcelFile(file_path)

    dataset_list = []
    auc_list = []

    # Get performances per feature map
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # Get model names
        models = df.iloc[:, 0]

        # Get AUC scores
        mean_auc = df.iloc[:, 1]

        # Add data to the lists
        dataset_list.extend([sheet_name] * len(models))
        auc_list.extend(mean_auc)

    # Results export
    auc_long = pd.DataFrame({"Model": model_list, "Dataset": dataset_list, "AUC": auc_list})

    # Remove underscores from names
    auc_long["Dataset"] = auc_long["Dataset"].str.replace("_", " ", regex=False)

    # Plot figure heatmap
    fig = plt.figure(figsize=(13, 8))
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
    plt.close(fig)
    
    return fig


def plot_metrics(file_path, metrics, model_names):
    # Read data
    xls = pd.ExcelFile(file_path)
    datasets = xls.sheet_names

    data_for_plot = {metric: {} for metric in metrics}

    # Get data
    for dataset in datasets:
        df = xls.parse(dataset)
        df = df.set_index(df.columns[0])
        df.index = model_names

        # Extract values for each metric
        for metric in metrics:
            if metric in df.columns:
                data_for_plot[metric][dataset] = df[metric]

    # Create subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7, 5.3 * len(metrics)))

    # Loop through each metric and plot it
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        # for i, dataset in enumerate(datasets):
        for dataset in datasets:
            metric_values = data_for_plot[metric].get(dataset)
            if metric_values is not None:
                x_pos = range(len(model_names))
                ax.scatter(
                    x_pos, metric_values, label=dataset.replace("_", " "), s=100, edgecolor="black", linewidth=1
                )
                ax.plot(x_pos, metric_values, linestyle="-", alpha=0.6)

        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, fontsize=19)
        ax.tick_params(axis="y", labelsize=19)
        ax.set_ylabel(metric)

        ax.set_ylim(0, 1)
        ax.legend(title="Datasets", loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.close(fig)
    
    return fig

def plot_critical_differences(file_path, output_path, model_names, save, figures_dir):
    
    all_sheets = pd.read_excel(file_path, sheet_name=None, index_col=0)

    auc_df = pd.DataFrame({sheet: df["ROC_AUC"].values for sheet, df in all_sheets.items()}).T  

    auc_df.columns = model_names

    friedman_stat, p_value = friedmanchisquare(*auc_df.T.values)
    open(output_path, "w").write(f"Friedman test statistic: {friedman_stat:.4f}\nP-value: {p_value:.6e}\n")

    print(f"Friedman test statistic: {friedman_stat:.4f}, p-value = {p_value:.4e}")

    fig = plt.figure(figsize=(16, 12))
    plot_critical_difference(
        auc_df.values,
        auc_df.columns,
        lower_better=False,
        correction='holm',
        alpha=0.05
    )
    plt.savefig(figures_dir + 'critical_difference_diagram_'+save)
    plt.close(fig)
    
    fig2 = plt.figure(figsize=(16, 12))
    plot_significance(np.array(auc_df), model_names)
    plt.savefig(figures_dir + 'significance_'+save)
    plt.close(fig2)    
    
    return fig, fig2

def plot_boxplots(file_path, model_names):
    
    all_sheets = pd.read_excel(file_path, sheet_name=None, index_col=0)

    df_long = (
        pd.DataFrame({sheet: df["ROC_AUC"].values for sheet, df in all_sheets.items()})
        .T.set_axis(model_names, axis=1)
        .melt(var_name="Classifier", value_name="AUC")
    )
 
    fig = plt.figure(figsize=(8, 3.5))

    sns.violinplot(
        data=df_long, 
        x="Classifier", 
        y="AUC", 
        inner=None,  
        linewidth=1.5,
        color='lightgray')

    sns.boxplot(
        data=df_long, 
        x="Classifier", 
        y="AUC", 
        width=0.12,  
        showcaps=False,  
        showfliers=False,
        boxprops={'facecolor':'black', 'edgecolor':'None', 'linewidth':0},  
        whiskerprops={'color':'black', 'linewidth':1},  
        medianprops={'color':'white', 'linewidth':3}  
    )

    plt.xticks(rotation=45, size=12)
    plt.yticks( size=12)
    plt.title("AUC Distribution Across Classifiers")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0.16, 1.13)
    plt.close(fig)
    
    return fig

def get_feature_maps(feature_importances_path, measures, mask, output):

    for measure in measures:
        print(f"Processing feature importance for measure: {measure}")
        df_feat = pd.read_excel(feature_importances_path, sheet_name=measure)

        for model_name in df_feat.columns:
            importance_scores = df_feat[model_name].values

            tmp_img = nilearn.masking.unmask(importance_scores, mask_img)

            nii_data = tmp_img.get_fdata()
            nii_affine = tmp_img.affine
            importance_nii = nib.Nifti1Image(nii_data, affine=nii_affine)
            nib.save(importance_nii, output + f'{measure}_{model_name}_importance_map.nii.gz')
            
    return
