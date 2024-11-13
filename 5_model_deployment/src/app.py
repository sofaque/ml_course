from flask import Flask, request, jsonify
from transformers import pipeline

model_path = "citizenlab/distilbert-base-multilingual-cased-toxicity"

app = Flask(__name__)
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    comment = data.get("comment", "")
    if not comment:
        return jsonify({"error": "Comment text is required"}), 400
    result = classifier(comment)
    return jsonify(result[0])

# Главная функция для запуска
def main():
    app.run(host="0.0.0.0", port=5000)