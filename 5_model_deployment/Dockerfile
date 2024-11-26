FROM apache/airflow:latest

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /requirements.txt \
    && pip install --no-cache-dir -e /opt/airflow

COPY ./src /opt/airflow/src

RUN mkdir -p /opt/airflow/data

WORKDIR /opt/airflow