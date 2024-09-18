Prerequisites:
- Use the container from the first task.
- Add the required credentials JSON file to the folder containing the second task (ml_course/2_data_governance).

Instructions
1. If the container is already built, run:

>>docker-compose run jupyter /bin/bash

Or, if it's already running:

docker-compose exec jupyter bash

If the container is not built yet:

Clone the repository:

git clone https://github.com/sofaque/ml_course.git

Build the container:

docker-compose build

Start the container:

docker-compose run jupyter /bin/bash

In the container terminal:

Initialize DVC:

dvc init --no-scm

Navigate to the folder containing the second task:

cd ml_course/2_data_governance

Set up the DVC remote storage:

dvc remote add -d myremote gdrive://1yIM8fxvqOoAH47avtxBRTfW2AcvE11qE

dvc remote modify myremote gdrive_use_service_account true

dvc remote modify myremote --local gdrive_service_account_json_file_path data-434516-ab78982a0f07.json

Pull the dataset from DVC:

dvc pull dataset_57_hypothyroid.csv

Reproduce the pipeline:

dvc repro

Show the metrics:

dvc metrics show

Exit the container:

exit

To shut down the container:

docker-compose down
