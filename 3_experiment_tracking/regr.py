#!/usr/bin/env python
# coding: utf-8

# Dataset is about bike rental service. It contains bike sharing counts aggregated on hourly basis. Total number of records: 17379 hours (2 years of rental services). There are following features:
# - instant: record index
# - dteday : date of record
# - hr : hour (0 to 23) of record
# - season : season (1:springer, 2:summer, 3:fall, 4:winter)
# - yr : year (0: 2011, 1:2012)
# - mnth : month ( 1 to 12)
# - holiday : whether day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
# - weekday : day of the week
# - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
# + weathersit : weather situation in the current timeframe
#     - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
#     - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#     - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#     - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# - temp : Normalized temperature in Celsius. The values are divided to 41 (max)
# - atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
# - hum: Normalized humidity. The values are divided to 100 (max)
# - windspeed: Normalized wind speed. The values are divided to 67 (max)
# - casual: count of casual users (target 1)
# - registered: count of registered users (target 2)
# - cnt: count of total rental bikes including both casual and registered (target total)


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

import mlflow
mlflow.set_tracking_uri("http://mlflow:5000")

df = pd.read_csv('hour.csv')

df = df.drop(columns=['instant'])

df["dteday"] =  pd.to_datetime(df["dteday"]).dt.day

# From Time column we can take only Hour value
df.rename(columns={"dteday": "day"}, inplace=True)

# dropping corelated columns
df = df.drop(columns=['season'])

# dropping values with humidity = 0
df = df[df['hum'] > 0.1]

#train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(labels=['casual', 'registered', 'cnt'], axis=1), #dropping target and highly correlated features
    df['cnt'],
    test_size=0.3,
    random_state=0)
print('train shape:', X_train.shape)
print('test shape:', X_test.shape)

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

knn = KNeighborsRegressor()
dt = DecisionTreeRegressor()
rf =  RandomForestRegressor()
xgb = XGBRegressor()
lgb = LGBMRegressor()
        
def model_fit(parameters, x_train, x_test, Y_train, Y_test, model_name):
    start_time = timer(None)  # Start timer

    # Start MLFlow run with a unique name for each model
    with mlflow.start_run(run_name=model_name):
        print(f'######-Model => {model_name}')
        model = m()

        # Grid search for best parameters
        grid = GridSearchCV(model, parameters, cv=5, verbose=1, n_jobs=-1, scoring='neg_mean_absolute_error')
        grid.fit(x_train, Y_train)

        # Retrain with best parameters
        model_ = m(**grid.best_params_)
        model_.fit(x_train, Y_train)

        y_pred = model_.predict(x_test)
        y_pred_train = model_.predict(x_train)

        # Log parameters and metrics
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("MAE_train", mean_absolute_error(Y_train, y_pred_train))
        mlflow.log_metric("MSE_train", mean_squared_error(Y_train, y_pred_train))
        mlflow.log_metric("RMSE_train", np.sqrt(mean_squared_error(Y_train, y_pred_train)))
        mlflow.log_metric("R2_train", r2_score(Y_train, y_pred_train))

        mlflow.log_metric("MAE_test", mean_absolute_error(Y_test, y_pred))
        mlflow.log_metric("MSE_test", mean_squared_error(Y_test, y_pred))
        mlflow.log_metric("RMSE_test", np.sqrt(mean_squared_error(Y_test, y_pred)))
        mlflow.log_metric("R2_test", r2_score(Y_test, y_pred))

        # Log the model
        mlflow.sklearn.log_model(model_, "model")

'''Calculate train_test_split score of differnt models and plot them.'''
models = [knn, dt, rf, xgb, lgb]

params_knn = dict(n_neighbors=list(range(3, 20)))

params_dt = {
           "max_depth" : [None,10,20],
           "min_samples_leaf":[1,2,4],
           "min_weight_fraction_leaf":[0.0,0.1,0.3,0.6],
           "max_features":["auto","log2","sqrt"],
           "max_leaf_nodes":[None,10,20,30,50]}

params_rf  = {
             'max_depth': [None,10,20],
             'max_features': [1, 'sqrt'],
             'min_samples_leaf': [1, 2, 4],
             'min_samples_split': [2, 5, 10],
             'n_estimators': [100, 200]}
                  
params_xgb  = {
             'max_depth': [5, 8 ,10, 12],
             'n_estimators': [100, 200, 300, 400, 500],
             'learning_rate': [0.4, 0.3, 0.2, 0.1, 0.01],
             'min_child_weight': [1, 2],
             'subsample': [0.5, 1]}
                             
params_lgb  = {
             'max_depth': [5, 6, 7, 8, 9, 10],
             'learning_rate': [0.4, 0.3, 0.2, 0.1, 0.01],
             'min_child_weight': [1, 2],
             'subsample': [0.5, 1]}

models = [KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor, XGBRegressor, LGBMRegressor]
parameters = [params_knn, params_dt, params_rf, params_xgb, params_lgb]

for m, p in zip(models, parameters):
    model_name = m.__name__  # Use the model class name as the identifier
    model_fit(p, X_train_scaled, X_test_scaled, y_train, y_test, model_name)