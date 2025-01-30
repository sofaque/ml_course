from transformers import pipeline

def load_model(model_path="citizenlab/distilbert-base-multilingual-cased-toxicity"):
    """
    Loads a pre-trained text classification model for toxicity detection.

    Args:
        model_path (str): Path or name of the pre-trained model.
                          Defaults to a multilingual toxicity detection model.

    Returns:
        transformers.pipeline: A text classification pipeline ready for inference.
    """
    print("Loading model and tokenizer...")

    # Initialize and return a text classification pipeline with the specified model and tokenizer
    return pipeline("text-classification", model=model_path, tokenizer=model_path)
