import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# Define the model architecture
class FraudDetectionNN(nn.Module):
    def __init__(self, input_size=768):
        super(FraudDetectionNN, self).__init__()
        # Load local BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')  
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        x = torch.relu(self.fc1(cls_embedding))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# Load the tokenizer from the local directory
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FraudDetectionNN(input_size=768)  # Same input size used during training
model.load_state_dict(torch.load("fraud_detection_model.pth", map_location=torch.device('cpu')))
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Preprocessing function to tokenize input text
def preprocess_text(text):
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    return inputs['input_ids'].flatten(), inputs['attention_mask'].flatten()

# Prediction function to classify the input as 'fraud' or 'normal'
def predict_fraud(input_text):
    input_ids, attention_mask = preprocess_text(input_text)
    # Move input to the same device as the model (GPU or CPU)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Get model predictions
    with torch.no_grad():
        output = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))  # Add batch dimension
        _, prediction = torch.max(output, dim=1)  # Get predicted class (0 or 1)
    
    if prediction.item() == 1:
        return "Fraud"
    else:
        return "Normal"

# Test function to allow user input and prediction
def test_model():
    print("Fraud Detection Test (Type 'exit' to quit)\n")
    while True:
        input_text = input("Enter conversation text: ")
        # Exit condition
        if input_text.lower() == 'exit':
            print("Exiting test mode...")
            break
        # Get prediction from the model
        prediction = predict_fraud(input_text)
        # Display the result
        print(f"Prediction: {prediction}\n")

# Start testing the model
if __name__ == '__main__':
    test_model()
