import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

st.set_page_config(
    page_title="Wildfire Occurrence Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).parent

MODEL_FILES = {
    "Logistic Regression": BASE / "model" / "logistic_regression.joblib",
    "Decision Tree": BASE / "model" / "decision_tree.joblib",
    "KNN": BASE / "model" / "knn.joblib",
    "Gaussian NB": BASE / "model" / "naive_bayes.joblib",
    "SVM": BASE / "model" / "svm.joblib"
}

TARGET = "occured"
@st.cache_resource
def get_models():
    return {name: joblib.load(path) for name, path in MODEL_FILES.items()}


@st.cache_data
def get_predictions(X_scaled):
    results = {}
    for name, model in get_models().items():
        results[name] = (
            model.predict(X_scaled),
            model.predict_proba(X_scaled)[:, 1],
        )
    return results

# Loading the data

scaler = joblib.load(BASE / "model" / "scaler.joblib")

with open(BASE / "model" / "featureOrd.json") as f:
    FEATURES = json.load(f)


# User


st.title("Wildfire Occurrence Dashboard")

st.write(
    """
    Upload a dataset and evaluate different machine learning models
    for wildfire occurrence prediction.
    """
)

# Sidebar 

st.sidebar.header("Options")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset (CSV)",
    type=["csv"]
)

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_FILES.keys())
)

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

if uploaded_file is None:
    st.info("Please upload a CSV dataset.")
    st.stop()

# The Dataset

df = pd.read_csv(uploaded_file)

missing_columns = []

for col in FEATURES + [TARGET]:

    if col not in df.columns:
        missing_columns.append(col)

if len(missing_columns) > 0:

    st.error(
        f"Missing columns: {missing_columns}"
    )

    st.stop()


# Data visibility

X = df[FEATURES]
y = df[TARGET].astype(int)

X_scaled = scaler.transform(X)


# The models

loaded_models = get_models()
raw_results = get_predictions(X_scaled)

all_results = {}
for name, (predictions, probabilities) in raw_results.items():
    if abs(threshold - 0.50) > 1e-9:
        predictions = (probabilities >= threshold).astype(int)
    all_results[name] = (predictions, probabilities)

y_pred, y_prob = all_results[selected_model]


# Evaluating the results

accuracy = accuracy_score(y, y_pred)

auc = roc_auc_score(y, y_prob)

precision = precision_score(
    y,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y,
    y_pred,
    zero_division=0
)

mcc = matthews_corrcoef(
    y,
    y_pred
)


# Tabs to differentiate the workflow

tab1, tab2, tab3, tab4= st.tabs([
    "Overview",
    "Performance",
    "Model Comparison",
    "Single Prediction"
])


# Overview for the user = TAB 1

with tab1:

    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", len(df))
    c2.metric("Features", len(FEATURES))
    c3.metric("Wildfire Cases", int(y.sum()))
    c4.metric(
        "Non-Wildfire Cases",
        len(y) - int(y.sum())
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing_df = pd.DataFrame(
        df.isnull().sum(),
        columns=["Missing Values"]
    )

    st.dataframe(
        missing_df,
        use_container_width=True
    )


# Performance of the metrics = TAB 2

with tab2:

    st.subheader(
        f"Performance Metrics - {selected_model}"
    )

    m1, m2, m3 = st.columns(3)
    m4, m5, m6 = st.columns(3)

    m1.metric("Accuracy", round(accuracy, 4))
    m2.metric("AUC", round(auc, 4))
    m3.metric("Precision", round(precision, 4))

    m4.metric("Recall", round(recall, 4))
    m5.metric("F1 Score", round(f1, 4))
    m6.metric("MCC", round(mcc, 4))

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Fire", "Fire"],
        yticklabels=["No Fire", "Fire"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    st.subheader("Classification Report")

    st.text(
        classification_report(
            y,
            y_pred,
            target_names=[
                "No Fire",
                "Fire"
            ]
        )
    )

    results = df.copy()

    results["predicted"] = y_pred
    results["fire_probability"] = y_prob.round(4)

    st.download_button(
        "Download Predictions",
        results.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )


# Model comparision to identify the models = TAB 3

with tab3:

    st.subheader("Model Comparison")

    comparison_rows = []

    for model_name in MODEL_FILES:

        pred, prob = all_results[model_name]

        comparison_rows.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y, pred),
            "AUC": roc_auc_score(y, prob),
            "Precision": precision_score(
                y,
                pred,
                zero_division=0
            ),
            "Recall": recall_score(
                y,
                pred,
                zero_division=0
            ),
            "F1 Score": f1_score(
                y,
                pred,
                zero_division=0
            )
        })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=comparison_df,
        x="Model",
        y="Accuracy",
        ax=ax
    )

    plt.xticks(rotation=20)

    st.pyplot(fig)

    best_model = comparison_df.loc[
        comparison_df["Accuracy"].idxmax()
    ]

    st.success(
        f"Best Model: {best_model['Model']} "
        f"(Accuracy = {best_model['Accuracy']:.4f})"
    )

    st.subheader("ROC Curves")

    fig, ax = plt.subplots(figsize=(6, 5))

    for model_name in MODEL_FILES:

        _, prob = all_results[model_name]

        fpr, tpr, _ = roc_curve(y, prob)

        ax.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC={roc_auc_score(y, prob):.3f})"
        )

    ax.plot([0, 1], [0, 1], "--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.legend()

    st.pyplot(fig)


# The prediction of a single row = TAB 4

with tab4:

    st.subheader("Predict One Observation")

    values = {}

    cols = st.columns(3)

    for i, feature in enumerate(FEATURES):

        values[feature] = cols[i % 3].number_input(
            feature,
            value=float(df[feature].median())
        )

    if st.button("Predict"):

        row = pd.DataFrame(
            [values]
        )[FEATURES]

        row_scaled = scaler.transform(row)

        probability = loaded_models[
            selected_model
        ].predict_proba(row_scaled)[0, 1]

        st.metric(
            "Fire Probability",
            f"{probability:.1%}"
        )

        if probability >= threshold:
            st.error("Prediction: Fire")
        else:
            st.success("Prediction: No Fire")

