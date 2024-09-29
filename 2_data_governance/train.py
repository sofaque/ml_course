import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train_res = pd.read_csv("X_train_res.csv", index_col=0)
y_train_res = pd.read_csv("y_train_res.csv", index_col=0)
X_test_scaled = pd.read_csv("X_test_scaled.csv", index_col=0)
y_test = pd.read_csv("y_test_encoded.csv", index_col=0)

# Logistic regression after implementing SMOTE
lr = LogisticRegression()
lr.fit(X_train_res, y_train_res.values.ravel())

y_pred_lr = lr.predict(X_test_scaled)

# Compute simplified metrics
accuracy = accuracy_score(y_test, y_pred_lr)
precision = precision_score(y_test, y_pred_lr, average='macro')
recall = recall_score(y_test, y_pred_lr, average='macro')
f1 = f1_score(y_test, y_pred_lr, average='macro')

# Save metrics to a JSON file for DVC
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved to metrics.json")




