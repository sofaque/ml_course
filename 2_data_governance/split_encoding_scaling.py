{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "062355cc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((1848, 21), (793, 21))"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#train/test split\n",
    "X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(\n",
    "    df_2.drop(labels=['Class'], axis=1),\n",
    "    df_2['Class'],\n",
    "    test_size=0.3,\n",
    "    random_state=0)\n",
    "X_train_2.shape, X_test_2.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "00625fed",
   "metadata": {},
   "outputs": [],
   "source": [
    "# encoding categorical data\n",
    "categorical = categorical_data = X_train_2.select_dtypes(exclude=[np.number])\n",
    "for feature in categorical:\n",
    "        le = preprocessing.LabelEncoder()\n",
    "        X_train_2[feature] = le.fit_transform(X_train_2[feature])\n",
    "        X_test_2[feature] = le.transform(X_test_2[feature])\n",
    "y_train_2 = le.fit_transform(y_train_2)\n",
    "y_test_2 = le.fit_transform(y_test_2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "a2a2ead6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# scaling data\n",
    "scaler = StandardScaler()\n",
    "X_train_S_2 = scaler.fit_transform(X_train_2)\n",
    "\n",
    "X_test_S_2 = scaler.transform(X_test_2)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
