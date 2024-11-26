from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
from src.load_model import load_model

app = Flask(__name__)
swagger = Swagger(app)

classifier = load_model()

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