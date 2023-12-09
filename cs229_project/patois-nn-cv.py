import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Define constants
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 1e-2
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
# tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# Load and preprocess Jamaican Patois NLI dataset
file_path = 'datasets/patoisnli/jampatoisnli-train.csv'
test_file_path = 'datasets/patoisnli/jampatoisnli-test.csv'

df = pd.read_csv(file_path)
test_df = pd.read_csv(test_file_path)

label_mapping = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

# Tokenize and encode text using BERT tokenizer
def tokenize_and_encode(row):
    premise = str(row['premise'])
    hypothesis = str(row['hypothesis'])
    inputs = tokenizer(
        text=premise,
        text_pair=hypothesis,
        padding='max_length',
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors='pt'
    )
    return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), torch.tensor(label_mapping[row['label']], dtype=torch.long)

df['input_ids'], df['attention_mask'], df['label'] = zip(*df.apply(tokenize_and_encode, axis=1))
test_df['input_ids'], test_df['attention_mask'], test_df['label'] = zip(*test_df.apply(tokenize_and_encode, axis=1))

# Define Jamaican Patois NLI Dataset class
class JamaicanPatoisNLIDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data.iloc[idx]['input_ids'],
            'attention_mask': self.data.iloc[idx]['attention_mask'],
            'label': self.data.iloc[idx]['label']
        }

# Instantiate DataLoader for training and validation sets
train_dataset = JamaicanPatoisNLIDataset(df)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = JamaicanPatoisNLIDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define MLP model
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x.float())  # Cast input to torch.float
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model
input_size = MAX_SEQ_LENGTH  # Adjust input size based on BERT tokenized sequence length
hidden_size = 256
output_size = 3 # Multi-class classification
mlp_model = MLPModel(input_size, hidden_size, output_size).to(DEVICE)

# Define optimizer and loss function
optimizer = optim.Adam(mlp_model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# Define the number of folds for cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Lists to store accuracy values for plotting
train_accuracies = []
val_accuracies = []

# Train and evaluate models for each fold
for fold, (train_index, val_index) in enumerate(kf.split(df)):
    print(f"Fold {fold + 1}/{num_folds}")

    # Create DataLoader for training set
    train_fold_dataset = JamaicanPatoisNLIDataset(df.iloc[train_index])
    train_fold_loader = DataLoader(train_fold_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create DataLoader for validation set
    val_fold_dataset = JamaicanPatoisNLIDataset(df.iloc[val_index])
    val_fold_loader = DataLoader(val_fold_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Instantiate the model for each fold
    mlp_model = MLPModel(input_size, hidden_size, output_size).to(DEVICE)

    # Define optimizer and loss function
    optimizer = optim.Adam(mlp_model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(EPOCHS):
        mlp_model.train()
        total_loss = 0.0

        for batch in train_fold_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['label'].to(DEVICE)

            optimizer.zero_grad()
            outputs = mlp_model(input_ids)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_fold_loader)
        print(f'Epoch {epoch + 1}/{EPOCHS}, Average Loss: {avg_loss}')

    # Validation loop
    mlp_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_fold_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['label'].to(DEVICE)
            outputs = mlp_model(input_ids)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')

    # Save the accuracy values for plotting
    val_accuracies.append(accuracy)

# Plotting
plt.figure(figsize=(6, 4))
plt.plot(range(1, num_folds + 1), val_accuracies, marker='o', linestyle='-', color='b')
plt.title('Validation Accuracy Across Folds')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.savefig('cross_validation_acc_plot_nn.png')
plt.show()
