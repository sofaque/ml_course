import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data_cleaned.csv")

#train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(labels=['Class'], axis=1),
    df['Class'],
    test_size = params['split']['test_size'],
    random_state = params['split']['random_state'])

X_train = pd.DataFrame(X_train)
X_train.to_csv("X_train.csv")

y_train = pd.DataFrame(y_train)
y_train.to_csv("y_train.csv")

X_test = pd.DataFrame(X_test)
X_test.to_csv("X_test.csv")

y_test = pd.DataFrame(y_test)
y_test.to_csv("y_test.csv")
