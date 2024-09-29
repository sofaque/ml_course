import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE

# Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

X_train_scaled = pd.read_csv("X_train_scaled.csv", index_col=0)
y_train = pd.read_csv("y_train_encoded.csv", index_col=0)
  
sm = SMOTE(random_state = params['split']['random_state'])
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train.values.ravel())

X_train_res = pd.DataFrame(X_train_res)
X_train_res.to_csv("X_train_res.csv")

y_train_res = pd.DataFrame(y_train_res)
y_train_res.to_csv("y_train_res.csv")

