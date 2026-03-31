# neural_network.py — TensorFlow neural network classifier for fraud detection
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load data ──────────────────────────────────────────────────────────────────
# Replace with your actual data loading
df = pd.read_csv("sample_fraud_data.csv")

target_col = "fraud"  # replace with your actual target column name
drop_cols  = [c for c in df.columns if "id" in c.lower()]
df = df.drop(columns=drop_cols, errors="ignore")

X = pd.get_dummies(df.drop(target_col, axis=1), drop_first=True)
y = df[target_col]

# ── Preprocessing ──────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Baseline: Random Forest ────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

rf_results = {
    "Model":     "Random Forest",
    "Accuracy":  round(accuracy_score(y_test, rf_pred), 4),
    "Precision": round(precision_score(y_test, rf_pred, zero_division=0), 4),
    "Recall":    round(recall_score(y_test, rf_pred, zero_division=0), 4),
    "F1":        round(f1_score(y_test, rf_pred, zero_division=0), 4),
    "AUC":       round(roc_auc_score(y_test, rf_proba), 4),
}

# ── Neural Network: TensorFlow / Keras ────────────────────────────────────────
nn_model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

nn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

history = nn_model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

nn_pred       = (nn_model.predict(X_test_scaled) > 0.5).astype(int).flatten()
nn_proba      = nn_model.predict(X_test_scaled).flatten()

nn_results = {
    "Model":     "Neural Network (TensorFlow)",
    "Accuracy":  round(accuracy_score(y_test, nn_pred), 4),
    "Precision": round(precision_score(y_test, nn_pred, zero_division=0), 4),
    "Recall":    round(recall_score(y_test, nn_pred, zero_division=0), 4),
    "F1":        round(f1_score(y_test, nn_pred, zero_division=0), 4),
    "AUC":       round(roc_auc_score(y_test, nn_proba), 4),
}

# ── Model Comparison ───────────────────────────────────────────────────────────
comparison = pd.DataFrame([rf_results, nn_results])
print("\n=== MODEL COMPARISON ===")
print(comparison.to_string(index=False))

# ── Training History Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"], label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Training vs Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["auc"], label="Train AUC")
axes[1].plot(history.history["val_auc"], label="Val AUC")
axes[1].set_title("Training vs Validation AUC")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
print("Training history saved to training_history.png")

# ── Save model ─────────────────────────────────────────────────────────────────
nn_model.save("models/fraud_neural_network.keras")
print("Neural network model saved to models/fraud_neural_network.keras")
