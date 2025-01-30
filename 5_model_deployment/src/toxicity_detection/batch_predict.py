import pandas as pd
import os
from toxicity_detection.load_model import load_model
from datetime import datetime

# Load the toxicity detection model
classifier = load_model()

def batch_predict(input_file, output_file):
    """
    Performs batch prediction on a CSV file containing comments.

    Args:
        input_file (str): Path to the input CSV file containing a 'comment' column.
        output_file (str): Path to save the predictions in CSV format.

    The function:
    - Checks if the input file exists.
    - Reads the file and ensures it has a 'comment' column.
    - Applies the model to generate predictions.
    - Adds prediction results, confidence scores, and timestamps.
    - Saves the updated data to an output CSV file.
    """
    try:
        # Ensure the input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file '{input_file}' does not exist.")

        # Read the CSV file
        df = pd.read_csv(input_file)

        # Validate that the 'comment' column is present
        if 'comment' not in df.columns:
            raise ValueError("Input CSV must contain a 'comment' column.")

        # Apply the classifier to each comment
        predictions = df['comment'].apply(classifier)

        # Extract prediction labels and confidence scores
        df['prediction'] = predictions.apply(lambda x: x[0]['label'])
        df['score'] = predictions.apply(lambda x: x[0]['score'])

        # Add a timestamp for when the prediction was made
        df['prediction_time'] = datetime.now().isoformat()

        # Save the results to a new CSV file
        df.to_csv(output_file, index=False)
        print(f"Batch predictions saved to '{output_file}'.")
    
    except Exception as e:
        # Handle errors and print an error message
        print(f"Error during batch prediction: {e}")

if __name__ == "__main__":
    # Define input and output file paths (expected to be in /data/)
    input_path = "/data/input.csv"
    output_path = "/data/output.csv"

    # Run batch prediction
    batch_predict(input_path, output_path)
