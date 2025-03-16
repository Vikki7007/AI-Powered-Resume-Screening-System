from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
from pdf2image import convert_from_path
import os
from transformers import BertTokenizer, BertModel
import torch
from pyngrok import ngrok

app = Flask(__name__)
CORS(app)
NGROK_AUTH_TOKEN = "your-ngrok-auth-token"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
public_url = ngrok.connect(5000)
print("Public URL:", public_url)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

@app.route('/process', methods=['POST'])
def process_resume():
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['resume']
    file_path = "uploaded_resume.pdf"
    file.save(file_path)
    images = convert_from_path(file_path)
    text = pytesseract.image_to_string(images[0])
    tokens = tokenizer(text, return_tensors="pt")
    embeddings = model(**tokens).last_hidden_state.mean(dim=1).detach().numpy().tolist()

    return jsonify({"processed_text": text, "embeddings": embeddings})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
