import numpy as np
import pandas as pd
import matplotlib
from sklearn import preprocessing

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

categorical = X_train.select_dtypes(exclude=[np.number])
for feature in categorical:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

X_train = pd.DataFrame(X_train)
X_train.to_csv("X_train_encoded.csv")
y_train = pd.DataFrame(y_train)
y_train.to_csv("y_train_encoded.csv")
X_test = pd.DataFrame(X_test)
X_test.to_csv("X_test_encoded.csv")
y_test = pd.DataFrame(y_test)
y_test.to_csv("y_test_encoded.csv")