import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# Step 1: Load the dataset without headers (header=None)
file_path = 'fraud_calls_data.csv'
df = pd.read_csv(file_path, header=None)

# Step 2: The first column contains the classifications ("fraud" or "normal")
# and the second column contains the data (text).
X = df.iloc[:, 1].values  # Text data
y = df.iloc[:, 0].values  # Labels (fraud/normal)

# Step 3: Convert labels to numeric format (fraud=1, normal=0)
y = np.where(y == 'fraud', 1, 0)

# Step 4: Print the first few rows to verify
print(df.head())

# Define Dataset class
class FraudDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Define the Fraud Detection Model
class FraudDetectionNN(nn.Module):
    def __init__(self, input_size=768):
        super(FraudDetectionNN, self).__init__()
        # Initialize the pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Define the fully connected layers
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        # Pass the input through BERT
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # Get the CLS token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        # Pass through fully connected layers with dropout
        x = torch.relu(self.fc1(cls_embedding))
        x = self.dropout(x)  # Add dropout after first layer
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  # Add dropout after second layer
        return self.fc3(x)

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = FraudDetectionNN()

# Define optimizer and loss function with a higher learning rate
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Set up DataLoader with new parameters: max_len=256 and batch_size=8
train_data = FraudDataset(X, y, tokenizer, max_len=256)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Initialize the mixed-precision scaler
scaler = GradScaler()

# Training loop
model.train()
for epoch in range(10):  # Training for 10 epochs
    epoch_loss = 0.0  # Track epoch-level loss for diagnostics
    for batch in train_loader:
        optimizer.zero_grad()
        # Move batch to device (GPU or CPU)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Mixed precision forward pass
        with autocast():  # Enables mixed precision
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
    
    # Print loss at the end of each epoch
    print(f"Epoch [{epoch + 1}/10], Loss: {epoch_loss / len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'fraud_detection_model.pth')
