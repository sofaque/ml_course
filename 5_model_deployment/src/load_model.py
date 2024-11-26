from transformers import pipeline

def load_model(model_path="citizenlab/distilbert-base-multilingual-cased-toxicity"):
    print("Loading model and tokenizer...")
    return pipeline("text-classification", model=model_path, tokenizer=model_path)