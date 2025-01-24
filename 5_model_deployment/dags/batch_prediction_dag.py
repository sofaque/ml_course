from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime

default_args = {
    "owner": "uladzislau",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    "batch_prediction",
    default_args=default_args,
    description="Batch prediction DAG",
    schedule_interval="0 * * * *",  # Every hour
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    batch_task = DockerOperator(
        task_id="batch_predict",
        image="5_model_deployment-api",  # Replace with the API image name
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",  # Use the host's Docker
        network_mode="bridge",
        command="python src/toxicity_detection/batch_predict.py",
    )