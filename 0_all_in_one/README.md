# ml_course_test
# Machine Learning Course Project

This repository contains all tasks from the ML course, including containerization, data governance, experiment tracking, data pipelines, and model deployment. Below are step-by-step instructions for setting up and running each task.

## Clone the Repository

The estimated docker-compose size is approximately **9GB**.

```bash
git clone https://github.com/sofaque/ml_course.git
cd ml_course/0_all_in_one/
```

## Prerequisites

### 1. Containerization Task

Optional:
Before running the container, set the necessary environment variables:

```bash
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)
```

- If the Jupyter service is running as root (without passing environment variables), no password protection is enabled.
- For non-root users, the password is **"password"**.

### 2. Data Governance Task

- Add the required credentials JSON file to the folder `ml_course/2_data_governance`. File can be found in comment section for "Data Governance" task on Learn platform

### 3. Pipelines and Model Deployment Tasks

- Ensure required directories exist:

```bash
mkdir -p ../5_model_deployment/logs ../5_model_deployment/plugins
```

- Place all DAGs in one folder:

```bash
cp -r ../4_pipelines/dags/* ../5_model_deployment/dags/
```

## Setting Up Environment Variables

```bash
echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env
echo "DATA_PATH=$(realpath ../5_model_deployment/data)" >> .env
echo "DOCKER_GID=$(getent group docker | cut -d: -f3)" >> .env
```

## Start Services

To start all services:

```bash
docker-compose up
```

Wait for the message `✅✅✅ All services are ready!` from the logger service before proceeding.

If running in detached mode:

```bash
docker-compose up -d
```

Check logs to ensure services are up:

```bash
docker-compose logs logger
```

---

## Task Instructions

### 1. Containerization

By default, the container runs as root, so no authentication is needed. If running as a non-root user, the password is **"password"**.

To verify password/tocken run:

```bash
docker exec -it 0_all_in_one-jupyter_main-1 jupyter server list
```

**Access Jupyter Notebook:**

In browser open:

```bash
http://localhost:8888
```

### 2. Data Governance

DVC is used with a custom Google Cloud project for Google Drive storage.

#### 2.1 Run the container

Navigate to `0_all_in_one` and start a terminal inside the container:

```bash
docker-compose run jupyter_main /bin/bash
```

#### 2.2 Initialize DVC

```bash
dvc init --no-scm
```

#### 2.3 Navigate to the task folder

```bash
cd 2_data_governance
```

#### 2.4 Set up DVC remote storage

```bash
dvc remote add -d myremote gdrive://1yIM8fxvqOoAH47avtxBRTfW2AcvE11qE
dvc remote modify myremote gdrive_use_service_account true
dvc remote modify myremote --local gdrive_service_account_json_file_path "credentials.json"
```

#### 2.5 Pull dataset from DVC

```bash
dvc pull dataset_57_hypothyroid.csv
```

#### 2.6 Reproduce the pipeline

```bash
dvc repro
```

#### 2.7 Show DVC metrics

```bash
dvc metrics show
```

#### 2.8 Container terminal can be closed

> Ctrl + D

---

### 3. Experiment Tracking

#### 3.1 Access Jupyter Notebook

In web browser go to:

```bash
http://localhost:8888/tree/3_experiment_tracking/regr.ipynb
```
Run all cells to execute the experiment.

#### 3.2 Check MLflow UI

In web browser go to:

```bash
http://localhost:5000
```
Verify that models and artifacts have been logged successfully.

---

### 4. Pipelines

#### 4.1 Access Airflow UI

In web browser go to:

```bash
http://localhost:8080
```

Login credentials:

- **Username:** `airflow`
- **Password:** `airflow`

#### 4.2 Trigger "data_preprocessing_pipeline" DAG

Manually trigger the DAG and monitor its progress.

#### 4.3 Verify Output

Ensure processed files are available in:

```bash
ml_course/5_model_deployment/data
```

---

### 5. Model Deployment

#### 5.1 Run API Tests

```bash
docker exec -it 0_all_in_one-api-1 pytest /tests
```

#### 5.2 Verify Flask API

Access the web UI:

```bash
http://localhost:5001/predict_form
```

Enter a comment, submit, and verify classification results.

#### 5.3 Monitor Airflow DAG Execution

Check the Airflow web UI:

```bash
http://localhost:8080
```

Login credentials:

- **Username:** `airflow`
- **Password:** `airflow`

Trigger the DAG and confirm the output is saved in:

```bash
ml_course/5_model_deployment/data
