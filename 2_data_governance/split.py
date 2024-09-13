import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from box import ConfigBox
from sklearn.model_selection import train_test_split

yaml = YAML(typ="safe")

with open("params.yaml", "r", encoding="utf-8") as f:
    params = ConfigBox(yaml.load(f))

df = pd.read_csv("data_cleaned.csv")

#train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(labels=['Class'], axis=1),
    df['Class'],
    test_size = params.split.test_size,
    random_state = params.split.random_state)

X_train = pd.DataFrame(X_train)
X_train.to_csv("X_train.csv")

y_train = pd.DataFrame(y_train)
y_train.to_csv("y_train.csv")

X_test = pd.DataFrame(X_test)
X_test.to_csv("X_test.csv")

y_test = pd.DataFrame(y_test)
y_test.to_csv("y_test.csv")