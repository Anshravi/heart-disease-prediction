import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from flask import Flask, request, jsonify
import pickle

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
df = pd.read_csv(url, names=columns, na_values="?")

# Handle missing values
df.dropna(inplace=True)

# Convert categorical data
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)  # Convert target to binary (0: No Disease, 1: Disease)
df = pd.get_dummies(df, columns=["cp", "restecg", "slope", "thal"], drop_first=True)  # One-hot encoding

# Split data into features and labels
X = df.drop("target", axis=1)
y = df["target"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate multiple models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    results.append([name, accuracy, class_report["0"]["precision"], class_report["0"]["recall"], class_report["1"]["precision"], class_report["1"]["recall"]])
    
    # Save the best model
    if name == "Random Forest":
        with open("heart_disease_model.pkl", "wb") as f:
            pickle.dump(model, f)

    # Visualizing confusion matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Print comparison table
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision (No Disease)", "Recall (No Disease)", "Precision (Disease)", "Recall (Disease)"])
print("\nModel Comparison:")
print(results_df.sort_values(by="Accuracy", ascending=False))

# Flask API for deployment
app = Flask(__name__)

# Load trained model
with open("heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
