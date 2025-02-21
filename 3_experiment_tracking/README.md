# Instructions
## 1 Clone the repository
- If the container is already built, run:
```bash
git clone https://github.com/sofaque/ml_course.git
```
## 2. Navigate to the project directory:
```bash
cd ml_course_test/3_experiment_tracking
```
## 3 Start the Docker containers:
```bash
docker-compose build
docker-compose up
```
## 4 Access the Jupyter Notebook:
 - Open your browser and go to http://localhost:8888/notebooks/regr.ipynb

 - Run all cells in the notebook to execute the experiment.
## 5 Check the MLFlow UI:
 - After running the notebook, open http://localhost:5000 in your browser.
 - 
 - Ensure that the models and artifacts have been successfully logged.
## 6 Exit the container:

> Ctrl+C

```bash
docker-compose down
```
