import numpy as np
import pandas as pd

df = pd.read_csv('dataset_57_hypothyroid.csv', sep=',', na_values='?')

df = df.drop(columns=['TBG', 'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured'])

df = df.dropna()

df = df.drop_duplicates()

df = df[df['age']<=100]


df['Class'] = df['Class'].replace(['negative', 'compensated_hypothyroid', 'primary_hypothyroid', 'secondary_hypothyroid'], [0, 1, 1, 1])

df.to_csv("data_cleaned.csv")