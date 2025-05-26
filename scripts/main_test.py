"""Run classification on validation cohort."""

# Libraries
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import random

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from src.functions import get_data
from xgboost import XGBClassifier

# File paths and parameteres
data_path_train = "../data/DBS15T/"
data_path_test = "../data/DBS15T2/"
mask_img = nib.load("../data/msk/sum_80_bin.nii")
measures = ["ALFF", "fALFF", "ECM_add", "ECM_deg", "ECM_norm", "ECM_rank", "GCOR", "ICC", "IHC", "LCOR"]

# Define models
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

# Set seed
random.seed(124)

# Save results
writer = pd.ExcelWriter("classification_performance_test.xlsx", engine="xlsxwriter")
feature_writer = pd.ExcelWriter("classification_features_importances_test.xlsx", engine="xlsxwriter")

# Classification of ON-OFF states per feature map
for measure in measures:
    print(f"Processing measure: {measure}")

    # Fetch features for ON and OFF states
    on_features_train, off_features_train = get_data(data_path_train + measure, mask_img)
    on_features_test, off_features_test = get_data(data_path_test + measure, mask_img)

    X_tr = np.concatenate((on_features_train, off_features_train), axis=0)
    y_tr = np.concatenate([np.zeros(len(on_features_train)), np.ones(len(off_features_train))])
    X_ts = np.concatenate((on_features_test, off_features_test), axis=0)
    y_ts = np.concatenate([np.zeros(len(on_features_test)), np.ones(len(off_features_test))])

    results = []
    feature_importances = {}

    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training and evaluating model: {name}")

        clf = model.fit(X_tr, y_tr)

        y_pred = clf.predict(X_ts)
        y_proba = clf.predict_proba(X_ts)[:, 1]

        auc = roc_auc_score(y_ts, y_proba)
        acc = accuracy_score(y_ts, y_pred)
        prec = precision_score(y_ts, y_pred)
        rec = recall_score(y_ts, y_pred)
        f1 = f1_score(y_ts, y_pred)

        results.append({"Model": name, "ROC_AUC": auc, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})

        # Get feature importances
        if hasattr(clf, "coef_"):
            importance = clf.coef_[0]
            feature_importances[name] = importance
        elif hasattr(clf, "feature_importances_"):
            importance = clf.feature_importances_
            feature_importances[name] = importance

    # Save performance metrics
    df_results = pd.DataFrame(results)
    df_results.to_excel(writer, sheet_name=measure, index=False)

    # Save feature importances
    if feature_importances:
        df_feat = pd.DataFrame(feature_importances)
        df_feat.to_excel(feature_writer, sheet_name=measure, index=False)

    print(f"Finished processing measure: {measure}\n")

# Save and close writers
writer.close()
feature_writer.close()
