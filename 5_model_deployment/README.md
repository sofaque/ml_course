# Model Deployment

This project demonstrates the deployment of a machine learning model using Docker, Airflow, and Flask for predictions.

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone git@github.com:sofaque/ml_course.git

2. **Navigate to the project directory**
   ```bash
   cd ml_course_test/5_model_deployment

3. **Prepare the necessary directories**
   ```bash
   mkdir ./logs ./plugins

4. **Set environment variables**
   ```bash
   echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" >> .env
   echo "DATA_PATH=$(pwd)/data" >> .env
   echo "DOCKER_GID=$(getent group docker | cut -d: -f3)" >> .env

5. **Build and start Docker containers**
   ```bash
   docker-compose build
   docker-compose up -d

6. **Open a new terminal and run the following command to execute tests:**
   Open your browser and go to:
   ```bash
   docker exec -it 5_model_deployment-api-1 pytest /tests

8. **Verify Flask API**
   
   In your browser, go to:
   http://localhost:5000/predict_form
   Enter your comment in the text field and press "Submit". 
   Check if the comment was correctly classified.

9. **Verify Airflow DAG Execution**

    Check the next scheduled execution for the batch_prediction DAG:
    Open a new terminal and run:
    ```bash
    docker exec -it 5_model_deployment-api-1 airflow dags next-execution batch_prediction
    ```
    Trigger the batch_prediction DAG manually:
    ```bash
    docker exec -it 5_model_deployment-airflow-webserver-1 airflow dags trigger batch_prediction
    ```

    Check the list of DAGs:
    ```bash
    docker exec -it 5_model_deployment-airflow-webserver-1 airflow dags list
    ```

    List tasks for batch_prediction:
    ```bash
    docker exec -it 5_model_deployment-airflow-webserver-1 airflow tasks list batch_prediction
    ```

    After running the DAG, verify that the output.csv file has been generated in the /data folder within the container:
    ```bash
    docker exec -it 5_model_deployment-api-1 ls /data
    ```


    To monitor the DAG execution in UI, open the Airflow web UI in your browser:
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
        
10. **Shutdown**

    When finished, stop and remove volumes:
    ```bash
    docker-compose down --volumes
