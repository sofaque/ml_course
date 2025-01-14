FROM apache/airflow:latest

WORKDIR /opt/airflow

COPY requirements.txt /requirements.txt
COPY ./src /opt/airflow/src
COPY ./setup.py /opt/airflow/setup.py

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /requirements.txt \
    && pip install --no-cache-dir -e /opt/airflow

RUN mkdir -p /opt/airflow/data