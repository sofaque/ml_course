# Use the base Python image
FROM python:3.9-slim

# Define build arguments for UID and GID
ARG USER_ID
ARG GROUP_ID

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install dvc to use container for 2nd task, comment if not needed
RUN pip install dvc[all]

# installing lightgbm or 3rd task, comment if not needed
RUN apt-get update && apt-get install -y libgomp1

# Create a group and user with the specified UID and GID
RUN addgroup --gid $GROUP_ID user || true && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

# turning off password/token protection for notebook
RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py

# Copy all project files into the container
WORKDIR /app

# Set access rights
RUN chown -R user:user /app

# Switch to the created user
USER user

# Specify the command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
