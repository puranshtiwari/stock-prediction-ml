import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from features import FEATURE_COLS

def get_models(n_est):
    return {
        "Logistic Regression": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=10, weights="distance"),
        "Random Forest"      : RandomForestClassifier(n_estimators=n_est, max_depth=6, n_jobs=-1, random_state=42),
        "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=n_est, learning_rate=0.05, max_depth=4, random_state=42),
        "AdaBoost"           : AdaBoostClassifier(n_estimators=n_est, learning_rate=0.5, random_state=42),
        "XGBoost"            : xgb.XGBClassifier(n_estimators=n_est, learning_rate=0.05, max_depth=4, subsample=0.8, eval_metric="logloss", verbosity=0, random_state=42),
        "SVM (RBF)"          : SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
    }

def train_models(df, n_splits, n_est, selected_models):
    X = df[FEATURE_COLS].values
    y = df["target"].values
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    models = {k:v for k,v in get_models(n_est).items() if k in selected_models}

    all_true  = []
    all_preds = {name: [] for name in models}

    progress = st.progress(0, text="Training models...")
    total_fits = len(models) * n_splits

    for fold_i, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)
        all_true.extend(y_val.tolist())

        for m_i, (name, model) in enumerate(models.items()):
            model.fit(X_tr_sc, y_tr)
            all_preds[name].extend(model.predict(X_val_sc).tolist())
            done = fold_i * len(models) + m_i + 1
            progress.progress(done / total_fits, text=f"Fold {fold_i+1}/{n_splits} — {name}")

    progress.empty()

    y_true = np.array(all_true)
    records = []
    for name, preds in all_preds.items():
        yp = np.array(preds)
        records.append({
            "Model"    : name,
            "Accuracy" : round(accuracy_score (y_true, yp), 4),
            "Precision": round(precision_score(y_true, yp, zero_division=0), 4),
            "Recall"   : round(recall_score   (y_true, yp, zero_division=0), 4),
            "F1 Score" : round(f1_score        (y_true, yp, zero_division=0), 4),
            "ROC-AUC"  : round(roc_auc_score  (y_true, yp), 4),
            "_preds"   : yp,
        })

    results = pd.DataFrame(records).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    return results, y_true, all_preds