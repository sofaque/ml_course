import os
import tarfile
import json
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

# Identifying device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

# standart model path in SageMaker
MODEL_DIR = "/opt/ml/model"

def load_model():
    """
    Loads and initializes the model and tokenizer.
    If a model archive is present, it is unpacked.
    """
    global tokenizer, model

    # Extracting model
    model_tar = os.path.join(MODEL_DIR, "model.tar.gz")
    if os.path.exists(model_tar):
        with tarfile.open(model_tar, "r:gz") as tar:
            tar.extractall(MODEL_DIR)

    # Load the tokenizer and model from the MODEL_DIR directory
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    model.to(device)

# Loading the model when the container starts
load_model()

@app.route("/ping", methods=["GET"])
def ping():
    """
    Endpoint for checking container health.
    SageMaker uses this to check the server's readiness.
    """
    return "Healthy\n", 200

@app.route("/invocations", methods=["POST"])
def invocations():
    """
    The main endpoint for predictions.
    Processes incoming requests in JSON or raw text format.
    """
    # ОProcessing input
    if request.content_type == "application/json":
        try:
            data = request.get_json(force=True)
        except Exception as e:
            return jsonify({"error": f"JSON parsing error - {e}"}), 400
        input_text = data.get("text", "")
    else:
        input_text = request.data.decode("utf-8")

    # If input text is empty, using bos_token
    if not input_text:
        input_text = tokenizer.bos_token

    # Tokenizing input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    # If pad_token_id is not set, using eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    # generating output
    try:
        generated_output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_length=100,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            repetition_penalty=1.5,
            #top_p=0.92,
            #temperature=0.85,
            #do_sample=True,
            #top_k=125,
        )
    except Exception as e:
        return jsonify({"error": f"Generation error: {e}"}), 500

    generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    return json.dumps({"Generated text": generated_text}, ensure_ascii=False) + "\n", 200, {'Content-Type': 'application/json'}

if __name__ == "__main__":
    # local launch for testing
    app.run(host="0.0.0.0", port=8080)
