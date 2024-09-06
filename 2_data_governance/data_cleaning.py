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
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "361a2b9f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix:\n",
      "[[728   0]\n",
      " [ 32  33]]\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.96      1.00      0.98       728\n",
      "           1       1.00      0.51      0.67        65\n",
      "\n",
      "    accuracy                           0.96       793\n",
      "   macro avg       0.98      0.75      0.83       793\n",
      "weighted avg       0.96      0.96      0.95       793\n",
      "\n",
      "Accuracy: 0.9596469104665826\n"
     ]
    }
   ],
   "source": [
    "# log regression without SMOTE\n",
    "\n",
    "# logistic regression object\n",
    "lr = LogisticRegression()\n",
    "  \n",
    "# train the model on train set\n",
    "lr.fit(X_train_S_2, y_train_2.ravel())\n",
    "  \n",
    "y_pred_2 = lr.predict(X_test_S_2)\n",
    "  \n",
    "# print metrics results\n",
    "result = confusion_matrix(y_test_2, y_pred_2)\n",
    "print(\"Confusion Matrix:\")\n",
    "print(result)\n",
    "result1 = classification_report(y_test_2, y_pred_2)\n",
    "print(\"Classification Report:\",)\n",
    "print (result1)\n",
    "result2 = accuracy_score(y_test_2,y_pred_2)\n",
    "print(\"Accuracy:\",result2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bd3598b7",
   "metadata": {},
   "source": [
    "The recall of the minority class in very less. It proves that the model is more biased towards majority class.\n",
    "So it proves that this is not the best model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "78fc7b33",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Before OverSampling, counts of label '1': 151\n",
      "Before OverSampling, counts of label '0': 1697 \n",
      "\n",
      "After OverSampling, the shape of train_X: (3394, 21)\n",
      "After OverSampling, the shape of train_y: (3394,) \n",
      "\n",
      "After OverSampling, counts of label '1': 1697\n",
      "After OverSampling, counts of label '0': 1697\n"
     ]
    }
   ],
   "source": [
    "# implementing SMOTE\n",
    "\n",
    "print(\"Before OverSampling, counts of label '1': {}\".format(sum(y_train_2 == 1)))\n",
    "print(\"Before OverSampling, counts of label '0': {} \\n\".format(sum(y_train_2 == 0)))\n",
    "  \n",
    "# import SMOTE module from imblearn library\n",
    "# pip install imblearn (if you don't have imblearn in your system)\n",
    "from imblearn.over_sampling import SMOTE\n",
    "sm = SMOTE(random_state = 2)\n",
    "X_train_res, y_train_res = sm.fit_resample(X_train_S_2, y_train_2.ravel())\n",
    "  \n",
    "print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))\n",
    "print('After OverSampling, the shape of train_y: {} \\n'.format(y_train_res.shape))\n",
    "  \n",
    "print(\"After OverSampling, counts of label '1': {}\".format(sum(y_train_res == 1)))\n",
    "print(\"After OverSampling, counts of label '0': {}\".format(sum(y_train_res == 0)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "2bc34ea4",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix:\n",
      "[[714  14]\n",
      " [  4  61]]\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.99      0.98      0.99       728\n",
      "           1       0.81      0.94      0.87        65\n",
      "\n",
      "    accuracy                           0.98       793\n",
      "   macro avg       0.90      0.96      0.93       793\n",
      "weighted avg       0.98      0.98      0.98       793\n",
      "\n",
      "Accuracy: 0.9773013871374527\n"
     ]
    }
   ],
   "source": [
    "# log regression after implementing SMOTE\n",
    "\n",
    "lr1 = LogisticRegression()\n",
    "# train the model on train set\n",
    "lr1.fit(X_train_res, y_train_res.ravel())\n",
    "  \n",
    "y_pred_res = lr1.predict(X_test_S_2)\n",
    "  \n",
    "# print metrics results\n",
    "result = confusion_matrix(y_test_2, y_pred_res)\n",
    "print(\"Confusion Matrix:\")\n",
    "print(result)\n",
    "result1 = classification_report(y_test_2, y_pred_res)\n",
    "print(\"Classification Report:\",)\n",
    "print (result1)\n",
    "result2 = accuracy_score(y_test_2, y_pred_res)\n",
    "print(\"Accuracy:\",result2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d21f7583",
   "metadata": {},
   "source": [
    "As we can see all metrics show better result after classes were balanced. Recall for minority class significantly improved.\n",
    "Lets continue and try to tune hyperparameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "c147275d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Num Features: 3\n",
      "Selected Features: [False False  True False False False False  True False False False False\n",
      " False False False  True False False False False False]\n",
      "Feature Ranking: [12 10  1 15  9 18  4  1  3 19 17 16  5 13 11  1  7  8 14  2  6]\n"
     ]
    }
   ],
   "source": [
    "# implementing RFE with cross-validation to define optimal number of features\n",
    "\n",
    "from sklearn.feature_selection import RFECV\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "# load data\n",
    "\n",
    "# feature extraction\n",
    "rfecv = RFECV(estimator= lr, step = 1, cv = 5, scoring=\"accuracy\")\n",
    "fit = rfecv.fit(X_train_res, y_train_res)\n",
    "\n",
    "X_train_selected = rfecv.transform(X_train_res)\n",
    "X_test_selected = rfecv.transform(X_test_S_2)\n",
    "\n",
    "print(\"Num Features: %d\" % fit.n_features_)\n",
    "print(\"Selected Features: %s\" % fit.support_)\n",
    "print(\"Feature Ranking: %s\" % fit.ranking_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "7f0a892b",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix:\n",
      "[[724   4]\n",
      " [  1  64]]\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      0.99      1.00       728\n",
      "           1       0.94      0.98      0.96        65\n",
      "\n",
      "    accuracy                           0.99       793\n",
      "   macro avg       0.97      0.99      0.98       793\n",
      "weighted avg       0.99      0.99      0.99       793\n",
      "\n",
      "Accuracy: 0.9936948297604036\n"
     ]
    }
   ],
   "source": [
    "# log regression after implementing SMOTE and feature selection\n",
    "\n",
    "lr2 = LogisticRegression()\n",
    "# train the model on train set\n",
    "lr2.fit(X_train_selected, y_train_res.ravel())\n",
    "  \n",
    "y_pred_2 = lr2.predict(X_test_selected)\n",
    "  \n",
    "# print metrics results\n",
    "result = confusion_matrix(y_test_2, y_pred_2)\n",
    "print(\"Confusion Matrix:\")\n",
    "print(result)\n",
    "result1 = classification_report(y_test_2, y_pred_2)\n",
    "print(\"Classification Report:\",)\n",
    "print (result1)\n",
    "result2 = accuracy_score(y_test_2, y_pred_2)\n",
    "print(\"Accuracy:\",result2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "9401b11e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "from sklearn.metrics import roc_curve\n",
    "\n",
    "fpr, tpr, thresholds = roc_curve(y_test_2, y_pred_2)\n",
    "plt.plot([0, 1], [0, 1], 'k--')\n",
    "plt.plot(fpr, tpr)\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('ROC')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "39ceb6e3",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.metrics import precision_recall_curve\n",
    "from sklearn.metrics import PrecisionRecallDisplay\n",
    "\n",
    "precision, recall, threshold = precision_recall_curve(y_test_2, y_pred_2)\n",
    "prd = PrecisionRecallDisplay(precision, recall)\n",
    "prd.plot()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "13ddc683",
   "metadata": {},
   "source": [
    "After defining optimal number of features metrics improved.\n",
    "Final accuracy is >99% so i'm happy with final results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "7f02e6e6",
   "metadata": {},
   "outputs": [],
   "source": [
    "#----------------- IMPLEMENTING KNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "38148db5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((1848, 21), (793, 21))"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(\n",
    "    df_2.drop(labels=['Class'], axis=1),\n",
    "    df_2['Class'],\n",
    "    test_size=0.3,\n",
    "    random_state=0)\n",
    "X_train_knn.shape, X_test_knn.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "3c3b9fe6",
   "metadata": {},
   "outputs": [],
   "source": [
    "categorical = categorical_data = X_train_knn.select_dtypes(exclude=[np.number])\n",
    "for feature in categorical:\n",
    "        le = preprocessing.LabelEncoder()\n",
    "        X_train_knn[feature] = le.fit_transform(X_train_knn[feature])\n",
    "        X_test_knn[feature] = le.transform(X_test_knn[feature])\n",
    "y_train_knn = le.fit_transform(y_train_knn)\n",
    "y_test_knn = le.fit_transform(y_test_knn)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "789aac65",
   "metadata": {},
   "outputs": [],
   "source": [
    "# scaling data\n",
    "scaler = StandardScaler()\n",
    "X_train_S_knn = scaler.fit_transform(X_train_knn)\n",
    "\n",
    "X_test_S_knn = scaler.transform(X_test_knn)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "1c2a29f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# implementing KNN for unbalanced data\n",
    "\n",
    "knn_clf=KNeighborsClassifier()\n",
    "knn_clf.fit(X_train_S_knn, y_train_knn)\n",
    "ypred=knn_clf.predict(X_test_S_knn) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "9f424890",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix:\n",
      "[[728   0]\n",
      " [ 44  21]]\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.94      1.00      0.97       728\n",
      "           1       1.00      0.32      0.49        65\n",
      "\n",
      "    accuracy                           0.94       793\n",
      "   macro avg       0.97      0.66      0.73       793\n",
      "weighted avg       0.95      0.94      0.93       793\n",
      "\n",
      "Accuracy: 0.9445145018915511\n"
     ]
    }
   ],
   "source": [
    "result = confusion_matrix(y_test_knn, ypred)\n",
    "print(\"Confusion Matrix:\")\n",
    "print(result)\n",
    "result1 = classification_report(y_test_knn, ypred)\n",
    "print(\"Classification Report:\",)\n",
    "print (result1)\n",
    "result2 = accuracy_score(y_test_knn, ypred)\n",
    "print(\"Accuracy:\",result2)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f3f2059f",
   "metadata": {},
   "source": [
    "Despite the fact that accuracy shows pretty good results the recall of the minority class in very less"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "6efa2453",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Before OverSampling, counts of label '1': 151\n",
      "Before OverSampling, counts of label '0': 1697 \n",
      "\n",
      "After OverSampling, the shape of train_X: (3394, 21)\n",
      "After OverSampling, the shape of train_y: (3394,) \n",
      "\n",
      "After OverSampling, counts of label '1': 1697\n",
      "After OverSampling, counts of label '0': 1697\n"
     ]
    }
   ],
   "source": [
    "# implementing SMOTE\n",
    "\n",
    "print(\"Before OverSampling, counts of label '1': {}\".format(sum(y_train_knn == 1)))\n",
    "print(\"Before OverSampling, counts of label '0': {} \\n\".format(sum(y_train_knn == 0)))\n",
    "  \n",
    "# import SMOTE module from imblearn library\n",
    "# pip install imblearn (if you don't have imblearn in your system)\n",
    "from imblearn.over_sampling import SMOTE\n",
    "sm = SMOTE(random_state = 2)\n",
    "X_train_res_knn, y_train_res_knn = sm.fit_resample(X_train_S_knn, y_train_knn.ravel())\n",
    "  \n",
    "print('After OverSampling, the shape of train_X: {}'.format(X_train_res_knn.shape))\n",
    "print('After OverSampling, the shape of train_y: {} \\n'.format(y_train_res_knn.shape))\n",
    "  \n",
    "print(\"After OverSampling, counts of label '1': {}\".format(sum(y_train_res_knn == 1)))\n",
    "print(\"After OverSampling, counts of label '0': {}\".format(sum(y_train_res_knn == 0)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "d5a53009",
   "metadata": {},
   "outputs": [],
   "source": [
    "knn_clf=KNeighborsClassifier()\n",
    "knn_clf.fit(X_train_res_knn, y_train_res_knn)\n",
    "ypred=knn_clf.predict(X_test_S_knn) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "a9c107ac",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix:\n",
      "[[648  80]\n",
      " [ 20  45]]\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.97      0.89      0.93       728\n",
      "           1       0.36      0.69      0.47        65\n",
      "\n",
      "    accuracy                           0.87       793\n",
      "   macro avg       0.67      0.79      0.70       793\n",
      "weighted avg       0.92      0.87      0.89       793\n",
      "\n",
      "Accuracy: 0.8738965952080706\n"
     ]
    }
   ],
   "source": [
    "result = confusion_matrix(y_test_knn, ypred)\n",
    "print(\"Confusion Matrix:\")\n",
    "print(result)\n",
    "result1 = classification_report(y_test_knn, ypred)\n",
    "print(\"Classification Report:\",)\n",
    "print (result1)\n",
    "result2 = accuracy_score(y_test_knn, ypred)\n",
    "print(\"Accuracy:\",result2)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "880b1243",
   "metadata": {},
   "source": [
    "After balancing data recall for minority class improved. Unfortunately all other metrics for minority class and accuracy results decreased significantly\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "5be25bc0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 10 folds for each of 19 candidates, totalling 190 fits\n"
     ]
    }
   ],
   "source": [
    "# defining optimal parameters for balanced set using GridSearchCV\n",
    "\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "k_range = list(range(1, 20))\n",
    "param_grid = dict(n_neighbors=k_range)\n",
    "  \n",
    "# defining parameter range\n",
    "grid = GridSearchCV(knn_clf, param_grid, cv=10, scoring='accuracy', return_train_score=False, verbose=1)\n",
    "  \n",
    "# fitting the model for grid search\n",
    "grid_search=grid.fit(X_train_res_knn, y_train_res_knn)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "fb262efe",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9631788998785353\n",
      "{'n_neighbors': 2}\n",
      "KNeighborsClassifier(n_neighbors=2)\n"
     ]
    }
   ],
   "source": [
    "print(grid.best_score_)\n",
    "print(grid.best_params_)\n",
    "print(grid.best_estimator_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "54305c19",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# vizualizing work of GridSearchCV\n",
    "grid_mean_scores = grid.cv_results_['mean_test_score']\n",
    "plt.plot(k_range, grid_mean_scores)\n",
    "plt.xlabel('Value of K for KNN')\n",
    "plt.ylabel('Cross-Validated Accuracy')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "ee1a22ce",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix:\n",
      "[[689  39]\n",
      " [ 25  40]]\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.96      0.95      0.96       728\n",
      "           1       0.51      0.62      0.56        65\n",
      "\n",
      "    accuracy                           0.92       793\n",
      "   macro avg       0.74      0.78      0.76       793\n",
      "weighted avg       0.93      0.92      0.92       793\n",
      "\n",
      "Accuracy: 0.9192938209331651\n"
     ]
    }
   ],
   "source": [
    "knn_clf=KNeighborsClassifier(n_neighbors=1)\n",
    "knn_clf.fit(X_train_res_knn, y_train_res_knn)\n",
    "ypred=knn_clf.predict(X_test_S_knn) \n",
    "\n",
    "result = confusion_matrix(y_test_knn, ypred)\n",
    "print(\"Confusion Matrix:\")\n",
    "print(result)\n",
    "result1 = classification_report(y_test_knn, ypred)\n",
    "print(\"Classification Report:\",)\n",
    "print (result1)\n",
    "result2 = accuracy_score(y_test_knn, ypred)\n",
    "print(\"Accuracy:\",result2)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "4428b4ec",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fpr, tpr, thresholds = roc_curve(y_test_knn, ypred)\n",
    "# Plot ROC curve\n",
    "plt.plot([0, 1], [0, 1], 'k--')\n",
    "plt.plot(fpr, tpr)\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('ROC')\n",
    "plt.show()"
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
