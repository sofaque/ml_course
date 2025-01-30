from flask import Flask, request, jsonify 
from flasgger import Swagger, swag_from
from toxicity_detection.load_model import load_model

# Initialize Flask app
app = Flask(__name__)

# Enable API documentation with Swagger
swagger = Swagger(app)

# Load the toxicity classification model
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
    """
    API endpoint for toxicity prediction.
    Accepts a JSON payload with a "comment" field and returns the predicted label and confidence score.
    """
    data = request.get_json(force=True)
    comment = data.get("comment", "")

    # Validate input
    if not comment:
        return jsonify({"error": "Comment text is required"}), 400

    # Perform prediction using the loaded model
    result = classifier(comment)

    # Return prediction result as JSON
    return jsonify(result[0])

@app.route('/')
def home():
    """ Root endpoint that provides a basic message. """
    return "It works! For prediction navigate to /predict_form"

@app.route('/predict_form', methods=['GET', 'POST'])
def predict_form():
    """
    Simple web form for making predictions.
    Allows users to input a comment and receive a prediction result.
    """
    result_html = "<p>Type your comment for prediction</p>"  # Default message

    if request.method == 'POST':
        comment = request.form.get('comment', "")

        if comment:
            # Perform prediction
            result = classifier(comment)
            label = result[0]['label']
            score = result[0]['score']

            # Display the result
            result_html = f"<p>Predicted label: {label}, with score: {score}</p>"

    # Render HTML form with the result
    return f'''
        {result_html}
        <form method="post">
            <label for="comment">Comment:</label>
            <input type="text" id="comment" name="comment">
            <button type="submit">Submit</button>
        </form>
    '''

def main():
    """ Starts the Flask application on port 5000, accessible from any network. """
    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    # Run the Flask app when the script is executed
    main()
