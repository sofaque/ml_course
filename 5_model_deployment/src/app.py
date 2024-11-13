from flask import Flask, request, jsonify
from transformers import pipeline
from flasgger import Swagger, swag_from

model_path = "citizenlab/distilbert-base-multilingual-cased-toxicity"

app = Flask(__name__)
swagger = Swagger(app)

classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)

@app.route("/predict", methods=["POST"])
@swag_from({
    'tags': ['Prediction'],
    'parameters': [
        {
            "name": "comment",
            "in": "body",
            "required": True,
            "schema": {
                "type": "object",
                "properties": {
                    "comment": {
                        "type": "string",
                        "example": "This is a test comment."
                    }
                }
            }
        }
    ],
    'responses': {
        200: {
            "description": "Prediction result",
            "schema": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "score": {"type": "number"}
                }
            }
        }
    }
})

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

if __name__ == "__main__":
    main()