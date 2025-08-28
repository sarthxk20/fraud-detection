import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib, os, json

st.title("Fraud Detection with Machine Learning")

# Upload dataset
uploaded_file = st.file_uploader("Upload fraud_data.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    if "isFraud" not in df.columns:
        st.error("❌ Dataset must contain an 'isFraud' column as target variable.")
    else:
        # Split data
        X = df.drop("isFraud", axis=1)
        y = df["isFraud"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # Save model
        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/fraud_model.joblib")

        meta = {"features": list(X.columns), "threshold": 0.5}
        with open("model/meta.json", "w") as f:
            json.dump(meta, f)

        st.success("✅ Model trained and saved successfully!")

        # Evaluate
        y_pred = model.predict(X_test)
        st.write("### Model Performance")
        st.text(classification_report(y_test, y_pred))

        # Custom input for prediction
        st.write("### Try Custom Prediction")
        user_input = {}
        for col in X.columns:
            user_input[col] = st.number_input(f"Enter {col}", value=float(X[col].mean()))

        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)
        st.write("Prediction:", "Fraud" if prediction[0] == 1 else "Not Fraud")
