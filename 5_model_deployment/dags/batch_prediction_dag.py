from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "uladzislau",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "batch_prediction",
    default_args=default_args,
    description="Batch prediction DAG",
    schedule_interval="0 * * * *",  # Каждые 60 минут
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    batch_task = BashOperator(
        task_id="batch_predict",
        bash_command=(
            "python /opt/airflow/src/batch_predict.py "
            "/opt/airflow/data/input.csv /opt/airflow/data/output.csv"
        ),
    )
