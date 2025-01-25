import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Step 1: Load the Dataset (Replace 'heart.csv' with the actual dataset path)
data = pd.read_csv('C:\Users\anshr\OneDrive\Desktop')

# Step 2: Preprocess the Data
# Assuming the dataset has a 'target' column as the label
y = data['target']
X = data.drop(columns=['target'])

# Step 3: Standardize the Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 4: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Tune Hyperparameters for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 500, 1000]
}

grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best Parameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Step 6: Train the Optimized Logistic Regression Model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = best_model.predict(X_test)

# Step 8: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
