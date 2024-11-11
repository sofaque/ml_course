from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Default arguments for Airflow DAG
default_args = {
    'owner': 'airflow',
    'retries': 1,
}

# Initialization of DAG
with DAG('data_preprocessing_pipeline', 
         default_args=default_args, 
         description='Preprocessing pipeline: load, clean, split, and scale data',
         schedule_interval=None, 
         start_date=None,
         catchup=False) as dag:

    # Task 1: Load and combine data
    def load_and_combine_data():
        df1 = pd.read_csv('/opt/airflow/data/hour.csv')
        df2 = pd.read_csv('/opt/airflow/data/weather.csv')
        df = pd.concat([df1, df2], axis=0, ignore_index=True)
        return df
    
    load_and_combine_data_task = PythonOperator(
        task_id='load_and_combine_data',
        python_callable=load_and_combine_data
    )

    # Task 2: Feature modification (e.g., date transformation, renaming columns)
    def feature_modification(df):
        df["dteday"] = pd.to_datetime(df["dteday"]).dt.day
        df.rename(columns={"dteday": "day"}, inplace=True)
        return df
    
    feature_modification_task = PythonOperator(
        task_id='feature_modification',
        python_callable=feature_modification,
        provide_context=True
    )

    # Task 3: Data cleaning (drop unnecessary columns and filter data)
    def data_cleaning(df):
        df = df.drop(columns=['instant', 'season'])
        df = df[df['hum'] > 0.1]
        return df
    
    data_cleaning_task = PythonOperator(
        task_id='data_cleaning',
        python_callable=data_cleaning,
        provide_context=True
    )

    # Task 4: Save cleaned data
    def save_cleaned_data(df):
        df.to_csv('/opt/airflow/data/cleaned_data.csv', index=False)
    
    save_cleaned_data_task = PythonOperator(
        task_id='save_cleaned_data',
        python_callable=save_cleaned_data,
        provide_context=True
    )

    # Task 5: Split data into features and target
    def split_data():
        df = pd.read_csv('/opt/airflow/data/cleaned_data.csv')
        X = df.drop(columns=['casual', 'registered', 'cnt'])
        y = df['cnt']
        return X, y
    
    split_data_task = PythonOperator(
        task_id='split_data',
        python_callable=split_data
    )

    # Task 6: Split data into train/test sets
    def train_test_split_data(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        return X_train, X_test, y_train, y_test
    
    train_test_split_task = PythonOperator(
        task_id='train_test_split',
        python_callable=train_test_split_data
    )

    # Task 7: Save train/test splits
    def save_splits(X_train, X_test, y_train, y_test):
        X_train.to_csv('/opt/airflow/data/X_train.csv', index=False)
        X_test.to_csv('/opt/airflow/data/X_test.csv', index=False)
        y_train.to_csv('/opt/airflow/data/y_train.csv', index=False)
        y_test.to_csv('/opt/airflow/data/y_test.csv', index=False)
    
    save_splits_task = PythonOperator(
        task_id='save_splits',
        python_callable=save_splits
    )

    # Task 8: Scale data
    def scale_data():
        X_train = pd.read_csv('/opt/airflow/data/X_train.csv')
        X_test = pd.read_csv('/opt/airflow/data/X_test.csv')
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    scale_data_task = PythonOperator(
        task_id='scale_data',
        python_callable=scale_data
    )

    # Task 9: Save scaled data
    def save_scaled_data(X_train_scaled, X_test_scaled):
        pd.DataFrame(X_train_scaled).to_csv('/opt/airflow/data/X_train_scaled.csv', index=False)
        pd.DataFrame(X_test_scaled).to_csv('/opt/airflow/data/X_test_scaled.csv', index=False)
    
    save_scaled_data_task = PythonOperator(
        task_id='save_scaled_data',
        python_callable=save_scaled_data
    )

    # Define task dependencies
    load_and_combine_data_task >> feature_modification_task >> data_cleaning_task >> save_cleaned_data_task
    save_cleaned_data_task >> split_data_task >> train_test_split_task >> save_splits_task
    save_splits_task >> scale_data_task >> save_scaled_data_task
