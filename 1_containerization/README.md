1. Clone repository to folder on your computer
> git clone https://github.com/sofaque/ml_course.git
2. Navigate to folder with first task
> ../ml_course/1_containerization
3. (Optional) Set environment variables if neccesary. Otherwise default will be used
> export USER_ID=$(id -u)
> export GROUP_ID=$(id -g)
4. Ensure Docker is running, then start the container:
> docker-compose up
5. Open your browser and go to:
> http://localhost:8888
6. To stop the container, press Ctrl+C in the terminal where the container is running, then enter:
> docker-compose down






