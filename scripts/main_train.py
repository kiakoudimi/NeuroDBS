import random

import nibabel as nib
import pandas as pd
import xlsxwriter
from functions import *
from nilearn.input_data import NiftiLabelsMasker
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer, recall_score
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Paths and parameters
data_path_train = "../data/DBS15T/"
mask_img = nib.load("../data/msk/sum_80_bin.nii")
measures = ["ALFF", "fALFF", "ECM_add", "ECM_deg", "ECM_norm", "ECM_rank", "GCOR", "ICC", "IHC", "LCOR"]

# Models
models = {
    "Logistic Regression": LogisticRegression(random_state=124),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=124),
    "Random Forest": RandomForestClassifier(random_state=124),
    "XGBoost": XGBClassifier(random_state=124),
    "Gradient Boosting": GradientBoostingClassifier(random_state=124),
    "SVC": SVC(probability=True, random_state=124),
    "LDA": LinearDiscriminantAnalysis(),
}

random.seed(124)

# Performance metrics
scoring = {
    "AUC": "roc_auc",
    "Accuracy": make_scorer(accuracy_score),
    "Recall": make_scorer(recall_score),
    "F1": make_scorer(f1_score),
}

# Save results
writer = pd.ExcelWriter("classification_performance_train.xlsx", engine="xlsxwriter")

# Classification of ON-OFF states per feature map
for measure in measures:
    print(f"Processing measure: {measure}")

    # Load data
    on_features, off_features = get_data(data_path_train + measure, mask_img)
    X = np.concatenate((on_features, off_features), axis=0)
    y = np.concatenate([np.zeros(len(on_features)), np.ones(len(off_features))])
    groups = np.concatenate((range(len(on_features)), range(len(off_features))))

    # LOGOCV
    logo = LeaveOneGroupOut()

    # Results df
    columns = [
        "Model",
        "AUC_mean",
        "AUC_std",
        "Accuracy_mean",
        "Accuracy_std",
        "Recall_mean",
        "Recall_std",
        "F1_mean",
        "F1_std",
    ]

    measure_results = pd.DataFrame(columns=columns)

    # Classification of ON-OFF states per feature map
    for model_name, model in models.items():
        print(f"Evaluating model: {model_name}")

        # CV
        scores = cross_validate(model, X, y, cv=logo, scoring=scoring, groups=groups, n_jobs=-1)

        # Compute mean & std
        stats = {}
        for metric in scoring:
            test_scores = scores[f"test_{metric}"]
            stats[f"{metric}_mean"] = np.mean(test_scores)
            stats[f"{metric}_std"] = np.std(test_scores)

        # Final export results
        row = pd.DataFrame([
            {
                "Model": model_name,
                "AUC_mean": stats["AUC_mean"],
                "AUC_std": stats["AUC_std"],
                "Accuracy_mean": stats["Accuracy_mean"],
                "Accuracy_std": stats["Accuracy_std"],
                "Recall_mean": stats["Recall_mean"],
                "Recall_std": stats["Recall_std"],
                "F1_mean": stats["F1_mean"],
                "F1_std": stats["F1_std"],
            }
        ])

        measure_results = pd.concat([measure_results, row], ignore_index=True)

    # Save sheet
    measure_results.to_excel(writer, sheet_name=measure, index=False)
    print(f"Finished processing measure: {measure}\n")

# Save and close writer
writer.close()
