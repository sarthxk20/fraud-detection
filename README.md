# 🛡️ Fraud Detection with Machine Learning & Streamlit

This project is an **end-to-end fraud detection system** built using **Python, Scikit-learn, Pandas, and Streamlit**.  
It provides an interactive web application where users can upload transaction datasets, train a fraud detection model, evaluate performance, and make predictions on new data.

---

## ✨ Features
- 📂 **Dataset Upload**: Upload custom transaction data (`CSV` format).
- 📊 **Exploratory Data Analysis**: Preview and summarize dataset.
- 🧠 **Machine Learning Model**: Train a **Random Forest Classifier** (can be extended to other models).
- 📈 **Model Evaluation**: View accuracy, classification report, and confusion matrix.
- 🔮 **Custom Predictions**: Enter transaction details and predict whether it is **Fraudulent** or **Not Fraudulent**.
- 🌍 **Deployment**: Fully deployable on **Streamlit Cloud**.

---

## Model Comparison

This project compares two approaches for fraud detection:

| Model | Approach |
|---|---|
| Random Forest | Ensemble of decision trees — fast, interpretable baseline |
| Neural Network | 3-layer TensorFlow/Keras network with BatchNorm and Dropout |

To run the comparison:
```bash
pip install tensorflow
python neural_network.py
```

Results are printed to console and the training history is saved as `training_history.png`.
```

---

**Step 4 — Update fraud detection tag line**

Add to your project description wherever you list the tech stack:

`Python · Scikit-Learn · TensorFlow · Keras · Random Forest · Logistic Regression · Streamlit`

---

**Step 5 — Add this bullet to fraud detection on your resume**

Once you've run the script and have actual numbers, use this template — fill in your real results:

> "Implemented and compared a TensorFlow neural network (3-layer Keras architecture with BatchNorm and Dropout) against a Random Forest baseline — evaluating across accuracy, precision, recall, F1, and AUC on imbalanced fraud data."

---

**One thing before you run it** — your fraud detection app uses uploaded CSVs so there's no fixed dataset. For this script to produce real, reportable numbers you need to run it against a standard dataset. The Kaggle Credit Card Fraud dataset is the benchmark — it's free, widely used, and will give you publishable results. Download it from:
```
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


## 🛠️ Tech Stack
- **Python** (3.11+)
- **Pandas** – Data manipulation
- **NumPy** – Numerical computations
- **Scikit-learn** – Machine learning
- **Joblib** – Model persistence
- **Streamlit** – Interactive web app

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/sarthxk20/fraud-detectionproject.git

cd fraud-detection-streamlit
