#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
sns.set_style(style = 'whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[ ]:


df = pd.read_csv('dataset_57_hypothyroid.csv', sep=',', na_values='?')


# In[ ]:


df = df.dropna()


# In[ ]:


df = df.drop_duplicates()


# In[ ]:


df = df[df['age']<=100]


# In[ ]:


df_2 = df
df_2['Class'] = df_2['Class'].replace(['negative', 'compensated_hypothyroid', 'primary_hypothyroid', 'secondary_hypothyroid'], [0, 1, 1, 1])

