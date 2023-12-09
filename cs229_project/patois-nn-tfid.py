import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Define constants
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess Jamaican Patois NLI dataset
train_file_path = 'datasets/patoisnli/jampatoisnli-train.csv'
test_file_path = 'datasets/patoisnli/jampatoisnli-test.csv'
val_file_path = 'datasets/patoisnli/jampatoisnli-val.csv'

train_df = pd.read_csv(train_file_path)
val_df = pd.read_csv(val_file_path)
test_df = pd.read_csv(test_file_path)

# Tokenize and encode text using TF-IDF
def tfidf_vectorize(dataframe):
    vectorizer = TfidfVectorizer(max_features=MAX_SEQ_LENGTH)
    X = vectorizer.fit_transform(dataframe['premise'] + ' ' + dataframe['hypothesis'])
    return torch.tensor(X.toarray(), dtype=torch.float32)

train_tfidf = tfidf_vectorize(train_df)
val_tfidf = tfidf_vectorize(val_df)
test_tfidf = tfidf_vectorize(test_df)

label_mapping = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

# Define Jamaican Patois NLI Dataset class
class JamaicanPatoisNLIDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    # Modify __getitem__ method in JamaicanPatoisNLIDataset
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'label': torch.tensor(label_mapping[self.labels[idx]], dtype=torch.long),
        }

# Instantiate DataLoader for training, validation, and test sets
train_dataset = JamaicanPatoisNLIDataset(train_tfidf, train_df['label'])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = JamaicanPatoisNLIDataset(val_tfidf, val_df['label'])
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = JamaicanPatoisNLIDataset(test_tfidf, test_df['label'])
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define MLP model
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model
input_size = MAX_SEQ_LENGTH
hidden_size = 256
output_size = 3  # Multi-class classification
mlp_model = MLPModel(input_size, hidden_size, output_size).to(DEVICE)

# Define optimizer and loss function
optimizer = optim.Adam(mlp_model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCHS):
    mlp_model.train()
    total_loss = 0.0

    for batch in train_loader:
        features, labels = batch['features'].to(DEVICE), batch['label'].to(DEVICE)

        optimizer.zero_grad()
        outputs = mlp_model(features)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Average Loss: {avg_loss}')

# Validation loop
mlp_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in val_loader:
        features, labels = batch['features'].to(DEVICE), batch['label'].to(DEVICE)
        outputs = mlp_model(features)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Evaluate the model on the test set
mlp_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        features, labels = batch['features'].to(DEVICE), batch['label'].to(DEVICE)
        outputs = mlp_model(features)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')
