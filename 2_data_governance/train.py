import numpy as np
import pandas as pd
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

X_train_res = pd.read_csv("X_train_res.csv")
y_train_res = pd.read_csv("y_train_res.csv")
X_test_scaled = pd.read_csv("X_test_scaled.csv")
y_test = pd.read_csv("y_test_encoded.csv")

# Logistic regression after implementing SMOTE
lr = LogisticRegression()
lr.fit(X_train_res, y_train_res.values.ravel())

y_pred_lr = lr.predict(X_test_scaled)

# Compute metrics
conf_matrix = confusion_matrix(y_test, y_pred_lr).tolist()  # Convert to list for JSON serialization
class_report = classification_report(y_test, y_pred_lr, output_dict=True)  # Convert to dict for JSON serialization
accuracy = accuracy_score(y_test, y_pred_lr)

# Save metrics to a JSON file for DVC
metrics = {
    "accuracy": accuracy,
    "confusion_matrix": conf_matrix,
    "classification_report": class_report
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("Metrics saved to metrics.json")



