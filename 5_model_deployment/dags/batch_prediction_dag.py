from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from datetime import datetime
import os

# Get the data path from environment variables
data_path = os.getenv("DATA_PATH")

# Default arguments for the DAG
default_args = {
    "owner": "uladzislau",  # DAG owner
    "depends_on_past": False,  # Do not wait for previous runs to succeed
    "email_on_failure": False,  # Disable failure notifications
    "email_on_retry": False,  # Disable retry notifications
}

# Define the DAG
with DAG(
    "batch_prediction",  # DAG name
    default_args=default_args,
    description="Batch prediction DAG",
    schedule_interval="0 * * * *",  # Run every hour
    start_date=datetime(2025, 1, 1),  # Start date
    catchup=False,  # Do not backfill past runs
) as dag:

    # Define a DockerOperator task
    batch_task = DockerOperator(
        task_id="batch_predict",  # Unique ID for the task
        image="5_model_deployment-api",  # Docker image name
        api_version="auto",  # Automatically detect Docker API version
        auto_remove="success",  # Remove the container after execution
        docker_url="unix://var/run/docker.sock",  # Use Docker socket on the host
        network_mode="bridge",  # Use bridge network mode
        command="python src/toxicity_detection/batch_predict.py",  # Command to execute inside the container
        mount_tmp_dir=False,  # Do not use a temporary directory
        mounts=[
            Mount(
                source=data_path,  # Host directory for input/output data
                target='/data',  # Mount point inside the container
                type='bind'  # Bind mount (link host and container directories)
            )
        ]
    )
