import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv("X_train_encoded.csv", index_col=0)
X_test = pd.read_csv("X_test_encoded.csv", index_col=0)

# scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train)
X_train_scaled.to_csv("X_train_scaled.csv")

X_test = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test)
X_test_scaled.to_csv("X_test_scaled.csv")
