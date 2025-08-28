import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import joblib, os, gc

# ---------------------------
# 🎨 App Config
# ---------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("🛡️ Fraud Detection with Machine Learning")

st.sidebar.header("⚙️ Controls")

# ---------------------------
# 📂 File Upload or Sample Data
# ---------------------------
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
use_sample = st.sidebar.checkbox("👉 Use sample dataset (demo)")

df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom dataset uploaded!")
elif use_sample:
    try:
        df = pd.read_csv("sample_fraud_data.csv")  # keep this file small (<5-10MB)
        st.success("✅ Loaded sample dataset!")
    except FileNotFoundError:
        st.error("❌ sample_fraud_data.csv not found. Please add it to the repo.")
else:
    st.info("👆 Upload a dataset or use the sample dataset to get started.")

# ---------------------------
# 🚀 If dataset is available
# ---------------------------
if df is not None:
    # Limit dataset size for memory
    MAX_ROWS = 50000
    if df.shape[0] > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=42)
        st.warning(f"⚠️ Dataset was too large. Using a random {MAX_ROWS} rows for demo.")

    # ---------------------------
    # Let user pick target column
    # ---------------------------
    target_col = st.sidebar.selectbox("Select the target column", df.columns)

    if target_col:
        # Drop obvious non-features (like IDs)
        drop_cols = [c for c in df.columns if "id" in c.lower()]
        df = df.drop(columns=drop_cols, errors="ignore")

        # Preprocess
        X = pd.get_dummies(df.drop(target_col, axis=1), drop_first=True)
        y = df[target_col]

        # Train-test split
        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Model selection
        model_choice = st.sidebar.selectbox("Select Model", ["Random Forest (light)", "Logistic Regression"])

        if model_choice == "Random Forest (light)":
            n_estimators = st.sidebar.slider("n_estimators", 10, 100, 50)
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=10,   # limit depth to reduce memory
                random_state=42
            )
        else:
            model = LogisticRegression(max_iter=500)  # lighter than default 1000

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/fraud_model.joblib")

        # ---------------------------
        # 📊 Tabs for Navigation
        # ---------------------------
        tab1, tab2, tab3 = st.tabs(["📂 Data & EDA", "📈 Model Performance", "🔮 Predictions"])

        # ---------------------------
        # 📂 Data & EDA Tab
        # ---------------------------
        with tab1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            st.write("Shape:", df.shape)

            st.subheader("Summary Statistics")
            st.write(df.describe())

            st.subheader("Correlation Heatmap")
            if X.shape[1] <= 30:  # avoid giant heatmaps
                corr = df.corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.info("⚠️ Too many features to plot correlation heatmap.")

        # ---------------------------
        # 📈 Model Performance Tab
        # ---------------------------
        with tab2:
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(
                y_test, y_pred,
                average="binary" if len(y.unique()) == 2 else "macro",
                zero_division=0
            )
            rec = recall_score(
                y_test, y_pred,
                average="binary" if len(y.unique()) == 2 else "macro",
                zero_division=0
            )
            f1 = f1_score(
                y_test, y_pred,
                average="binary" if len(y.unique()) == 2 else "macro",
                zero_division=0
            )

            st.subheader("📈 Model Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{acc:.2f}")
            col2.metric("Precision", f"{prec:.2f}")
            col3.metric("Recall", f"{rec:.2f}")
            col4.metric("F1 Score", f"{f1:.2f}")

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            st.success("✅ Model trained and evaluated automatically!")

            # Free memory
            del X_train, X_test, y_train, y_test
            gc.collect()

        # ---------------------------
        # 🔮 Predictions Tab
        # ---------------------------
        with tab3:
            st.subheader("Custom Prediction")

            user_input = {}
            for col in X.columns:
                user_input[col] = st.number_input(f"{col}", value=float(X[col].mean()))

            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]

            st.write("Prediction:", "⚠️ Fraud" if prediction == 1 else "✅ Not Fraud")
