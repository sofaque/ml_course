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
         start_date=datetime(2023, 1, 1), 
         catchup=False) as dag:

    def load_data():
        # Load the raw dataset and perform initial cleaning
        df = pd.read_csv('hour.csv')
        df = df.drop(columns=['instant'])
        df["dteday"] = pd.to_datetime(df["dteday"]).dt.day
        df.rename(columns={"dteday": "day"}, inplace=True)
        df = df.drop(columns=['season'])
        df = df[df['hum'] > 0.1]
        df.to_csv('/tmp/cleaned_data.csv', index=False)  # Save the cleaned data

    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data
    )

    def split_data():
        # Load cleaned data, split into train/test, and save the splits
        df = pd.read_csv('/tmp/cleaned_data.csv')
        X = df.drop(columns=['casual', 'registered', 'cnt'])
        y = df['cnt']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        X_train.to_csv('/tmp/X_train.csv', index=False)
        X_test.to_csv('/tmp/X_test.csv', index=False)
        y_train.to_csv('/tmp/y_train.csv', index=False)
        y_test.to_csv('/tmp/y_test.csv', index=False)

    split_data_task = PythonOperator(
        task_id='split_data',
        python_callable=split_data
    )

    def scale_data():
        # Scale the training and test data
        X_train = pd.read_csv('/tmp/X_train.csv')
        X_test = pd.read_csv('/tmp/X_test.csv')
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        pd.DataFrame(X_train_scaled).to_csv('/tmp/X_train_scaled.csv', index=False)
        pd.DataFrame(X_test_scaled).to_csv('/tmp/X_test_scaled.csv', index=False)

    scale_data_task = PythonOperator(
        task_id='scale_data',
        python_callable=scale_data
    )

    # Define task dependencies
    load_data_task >> split_data_task >> scale_data_task