import numpy as np
import pandas as pd
import matplotlib
import yaml
from imblearn.over_sampling import SMOTE

# Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

X_train_scaled = pd.read_csv("X_train_scaled.csv")
y_train = pd.read_csv("y_train_encoded.csv")

# implementing SMOTE
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))
  
sm = SMOTE(random_state = params.split.random_state)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train.values.ravel())

X_train_res = pd.DataFrame(X_train_res)
X_train_res.to_csv("X_train_res.csv", index=False)

y_train_res = pd.DataFrame(y_train_res)
y_train_res.to_csv("y_train_res.csv", index=False)

