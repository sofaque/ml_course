import pandas as pd
import os
from src.model_loader import load_model

# Загрузка модели
classifier = load_model()

def batch_predict(input_file, output_file):
    try:
        # Проверяем, существует ли входной файл
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file '{input_file}' does not exist.")
        
        # Читаем данные
        df = pd.read_csv(input_file)

        # Проверяем наличие колонки `comment`
        if 'comment' not in df.columns:
            raise ValueError("Input CSV must contain a 'comment' column.")
        
        # Применяем модель для предсказания
        df['prediction'] = df['comment'].apply(lambda x: classifier(x)[0]['label'])
        df['score'] = df['comment'].apply(lambda x: classifier(x)[0]['score'])

        # Сохраняем результат
        df.to_csv(output_file, index=False)
        print(f"Batch predictions saved to '{output_file}'.")
    except Exception as e:
        print(f"Error during batch prediction: {e}")

if __name__ == "__main__":
    input_path = "/opt/airflow/data/input.csv"
    output_path = "/opt/airflow/data/output.csv"
    batch_predict(input_path, output_path)