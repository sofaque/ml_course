# Data Preprocessing Pipeline

This project sets up an Airflow DAG to preprocess data with tasks including cleaning, merging, and transformation. 

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone git@github.com:sofaque/ml_course.git

2. **Navigate to the project directory**
   ```bash
   cd ml_course/4_pipelines

3. **Prepare the necessary directories**
   ```bash
   mkdir ./logs ./plugins

4. **Set environment variables**
   ```bash
   echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env

5. **Build and start Docker containers**
   ```bash
   docker-compose build
   docker-compose up -d

6. **Access Airflow**
   Open your browser and go to:
   ```bash
   http://localhost:8080
   ```
   Login:
      ```bash
      airflow
      ```
   Password:
      ```bash
      airflow
      ```

8. **Trigger DAG**
   
   In the Airflow UI, manually trigger the "data_preprocessing_pipeline" DAG and monitor progress.
   Ensure that all tasks complete with a "success" status.

9. **Verify output**

    Confirm that processed files were created and are available in
    ml_course/4_pipelines/data

9. **Shutdown**

    When finished, stop and remove volumes:
    ```bash
    docker-compose down --volumes
