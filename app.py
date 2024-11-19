from flask import Flask, request, jsonify
import torch
from fraud_detection import FraudDetectionNN  # Import the model class
from transformers import BertTokenizer  # Import the tokenizer

app = Flask(__name__)

# Load the model
def load_model():
    model = FraudDetectionNN()  # Create an instance of the model
    model_path = "fraud_detection_model.pth"  # Path to your saved model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load weights
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Define the function to get BERT embeddings
def get_bert_embeddings(text):
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize the input text
    inputs = tokenizer(
        text, 
        padding='max_length', 
        truncation=True, 
        max_length=256, 
        return_tensors="pt"
    )
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    return input_ids, attention_mask

@app.route('/predict', methods=['POST'])
def predict_fraud():
    data = request.json
    text = data.get('text')  # Get the live transcription from the frontend
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Get the BERT embeddings for the input text
    input_ids, attention_mask = get_bert_embeddings(text)
    
    # Move tensors to CPU (or GPU if available)
    input_ids = input_ids.to(torch.device('cpu'))
    attention_mask = attention_mask.to(torch.device('cpu'))
    
    # Make the prediction using the model
    with torch.no_grad():  # Disable gradients during inference
        output = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))  # Add batch dimension
        prediction = torch.argmax(output, dim=1).item()  # Get predicted class
    
    # Return the prediction
    return jsonify({"prediction": 'fraud' if prediction == 1 else 'normal'})

if __name__ == '__main__':
    app.run(debug=True)
