from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
from toxicity_detection.load_model import load_model

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

@app.route('/')
def home():
    return "It works! For prediction navigate to /prediction_form"

@app.route('/predict_form', methods=['GET', 'POST'])
def predict_form():
    result_html = "<p>Type your comment for prediction</p>"  # Initialize with a default message
    if request.method == 'POST':
        comment = request.form.get('comment', "")
        if comment:
            # Use the model for prediction
            result = classifier(comment)
            label = result[0]['label']
            score = result[0]['score']
            result_html = f"<p>Predicted label: {label}, with score: {score}</p>"

    # Form with possible result
    return f'''
        {result_html}
        <form method="post">
            <label for="comment">Comment:</label>
            <input type="text" id="comment" name="comment">
            <button type="submit">Submit</button>
        </form>
    '''

def main():
    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    main()
