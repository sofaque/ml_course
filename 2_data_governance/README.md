# Prerequisites:
- Use the container from the first task.
- Add the required credentials JSON file to the folder containing the second task (ml_course/2_data_governance).

# Instructions
## 1 run the container with terminal opened
- If the container is already built, run:
```bash
docker-compose run jupyter /bin/bash
```
- Or, if it's already running:
```bash
docker-compose exec jupyter bash
```
- If the container is not built yet:
>Clone the repository:
```bash
git clone https://github.com/sofaque/ml_course.git
```
>Build the container:
```bash
docker-compose build
```
>Start the container:
```bash
docker-compose run jupyter /bin/bash
```

## 2. From the container terminal initialize DVC:

Initialize DVC:
```bash
dvc init --no-scm
```
## 3 Navigate to the folder containing the second task:
```bash
cd ml_course/2_data_governance
```
## 4 Set up the DVC remote storage:
```bash
dvc remote add -d myremote gdrive://1yIM8fxvqOoAH47avtxBRTfW2AcvE11qE

dvc remote modify myremote gdrive_use_service_account true

dvc remote modify myremote --local gdrive_service_account_json_file_path "credentials.json"
```
## 5 Pull the dataset from DVC:
```bash
dvc pull dataset_57_hypothyroid.csv
```
## 6 Reproduce the pipeline:
```bash
dvc repro
```
## 7 Show the metrics:
```bash
dvc metrics show
```
## 8 Exit the container:

> Ctrl+D

```bash
docker-compose down
```
docker-compose down
