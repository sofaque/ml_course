#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#train/test split
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    df_2.drop(labels=['Class'], axis=1),
    df_2['Class'],
    test_size=0.3,
    random_state=0)


# In[ ]:


# encoding categorical data
categorical = categorical_data = X_train_2.select_dtypes(exclude=[np.number])
for feature in categorical:
        le = preprocessing.LabelEncoder()
        X_train_2[feature] = le.fit_transform(X_train_2[feature])
        X_test_2[feature] = le.transform(X_test_2[feature])
y_train_2 = le.fit_transform(y_train_2)
y_test_2 = le.fit_transform(y_test_2)


# In[ ]:


# scaling data
scaler = StandardScaler()
X_train_S_2 = scaler.fit_transform(X_train_2)

X_test_S_2 = scaler.transform(X_test_2)

