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
        df1 = pd.read_csv('/opt/airflow/data/hour.csv', sep=';')
        df2 = pd.read_csv('/opt/airflow/data/weather.csv', sep=';')
        df = pd.merge(df1, df2, on='instant')
        df.to_csv('/opt/airflow/data/combined_data.csv', index=False)
    
    load_and_combine_data_task = PythonOperator(
        task_id='load_and_combine_data',
        python_callable=load_and_combine_data
    )

    # Task 2: Feature modification (e.g., date transformation, renaming columns)
    def feature_modification():
        df = pd.read_csv('/opt/airflow/data/combined_data.csv')
        df["dteday"] = pd.to_datetime(df["dteday"]).dt.day
        df.rename(columns={"dteday": "day"}, inplace=True)
        df.to_csv('/opt/airflow/data/modified_data.csv', index=False)
    
    feature_modification_task = PythonOperator(
        task_id='feature_modification',
        python_callable=feature_modification
    )

    # Task 3: Data cleaning (drop unnecessary columns and filter data)
    def data_cleaning():
        df = pd.read_csv('/opt/airflow/data/modified_data.csv')
        df = df.drop(columns=['instant', 'season'])
        df = df[df['hum'] > 0.1]
        df.to_csv('/opt/airflow/data/cleaned_data.csv', index=False)
    
    data_cleaning_task = PythonOperator(
        task_id='data_cleaning',
        python_callable=data_cleaning
    )

    # Task 4: Split data into features and target
    def split_data():
        df = pd.read_csv('/opt/airflow/data/cleaned_data.csv')
        X = df.drop(columns=['casual', 'registered', 'cnt'])
        y = df['cnt']
        X.to_csv('/opt/airflow/data/X.csv', index=False)
        y.to_csv('/opt/airflow/data/y.csv', index=False)
    
    split_data_task = PythonOperator(
        task_id='split_data',
        python_callable=split_data
    )

    # Task 5: Split data into train/test sets
    def train_test_split_data():
        X = pd.read_csv('/opt/airflow/data/X.csv')
        y = pd.read_csv('/opt/airflow/data/y.csv')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        X_train.to_csv('/opt/airflow/data/X_train.csv', index=False)
        X_test.to_csv('/opt/airflow/data/X_test.csv', index=False)
        y_train.to_csv('/opt/airflow/data/y_train.csv', index=False)
        y_test.to_csv('/opt/airflow/data/y_test.csv', index=False)
    
    train_test_split_task = PythonOperator(
        task_id='train_test_split',
        python_callable=train_test_split_data
    )

    # Task 6: Scale data
    def scale_data():
        X_train = pd.read_csv('/opt/airflow/data/X_train.csv')
        X_test = pd.read_csv('/opt/airflow/data/X_test.csv')
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        pd.DataFrame(X_train_scaled).to_csv('/opt/airflow/data/X_train_scaled.csv', index=False)
        pd.DataFrame(X_test_scaled).to_csv('/opt/airflow/data/X_test_scaled.csv', index=False)
    
    scale_data_task = PythonOperator(
        task_id='scale_data',
        python_callable=scale_data
    )

    # Define task dependencies
    load_and_combine_data_task >> feature_modification_task >> data_cleaning_task >> split_data_task
    split_data_task >> train_test_split_task >> scale_data_task
