# Используем официальный образ Airflow
FROM apache/airflow:2.7.1

# Копируем файл зависимостей
COPY requirements.txt .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Создаем необходимые директории
RUN mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/plugins

# Убедись, что нужные права доступа выставлены для Airflow пользователя
USER airflow