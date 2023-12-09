import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define constants
# MAX_SEQ_LENGTH = 128
# MAX_SEQ_LENGTH = 256
MAX_SEQ_LENGTH = 512
# MAX_SEQ_LENGTH = 1026
# BATCH_SIZE = 32
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
# EPOCHS = 10
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pretrained_name = "bert-base-uncased"
# pretrained_name = "bert-base-multilingual-uncased"
pretrained_name = "bert-base-multilingual-cased"
# pretrained_name = "bert-base-chinese"
# pretrained_name = "MayaGalvez/bert-base-multilingual-cased-finetuned-nli"
# pretrained_name = "dbmdz/bert-base-french-europeana-cased"
# pretrained_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
# pretrained_name = 'distilbert-base-uncased'

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(pretrained_name)
# tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

# Load and preprocess Jamaican Patois NLI dataset
train_file_path = 'datasets/patoisnli/jampatoisnli-train.csv'
test_file_path = 'datasets/patoisnli/jampatoisnli-test.csv'
val_file_path = 'datasets/patoisnli/jampatoisnli-val.csv'

train_df = pd.read_csv(train_file_path)
val_df = pd.read_csv(val_file_path)
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

train_df['input_ids'], train_df['attention_mask'], train_df['label'] = zip(*train_df.apply(tokenize_and_encode, axis=1))
val_df['input_ids'], val_df['attention_mask'], val_df['label'] = zip(*val_df.apply(tokenize_and_encode, axis=1))
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
            'label': self.data.iloc[idx]['label'],
            'premise': self.data.iloc[idx]['premise'],
            'hypothesis': self.data.iloc[idx]['hypothesis']
        }

# Instantiate DataLoader for training, validation, and test sets
train_dataset = JamaicanPatoisNLIDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = JamaicanPatoisNLIDataset(val_df)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

# Training loop
for epoch in range(EPOCHS):
    mlp_model.train()
    total_loss = 0.0

    for batch in train_loader:
        input_ids, attention_mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['label'].to(DEVICE)

        optimizer.zero_grad()
        outputs = mlp_model(input_ids)
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
        input_ids, attention_mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['label'].to(DEVICE)
        outputs = mlp_model(input_ids)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Function to train the model
def train_model(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Average Loss: {avg_loss}')
    return avg_loss

# Function to evaluate the model
def evaluate_model(model, data_loader, device, graph = False):
    model.eval()
    all_labels = []
    all_predictions = []
    all_premises = []
    all_hypotheses = []
    
    num_entail, num_correct_entail = 0, 0 
    num_contradict, num_correct_contradict = 0, 0
    num_neutral, num_correct_neutral = 0, 0 

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
            outputs = model(input_ids)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_premises.extend(batch['premise'])
            all_hypotheses.extend(batch['hypothesis'])

    accuracy = sum([1 if label == pred else 0 for label, pred in zip(all_labels, all_predictions)]) / len(all_labels)
    for label, pred in zip(all_labels, all_predictions):
        if label == 0:
            num_entail += 1
            if label == pred:
                num_correct_entail += 1
        elif label == 1:
            num_contradict += 1
            if label == pred:
                num_correct_contradict += 1
        else:
            num_neutral += 1
            if label == pred:
                num_correct_neutral += 1
    precision = precision_score(all_labels, all_predictions, average='weighted')
    acc_entail = num_correct_entail / num_entail
    acc_contradict = num_correct_contradict / num_contradict
    acc_neutral = num_correct_neutral / num_neutral
    print(f'Entail Accuracy {acc_entail}, Contradict Accuracy {acc_contradict}, Neutral Accuracy {acc_neutral}')

    if graph:
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Entail', 'Contradict', 'Neutral'], yticklabels=['Entail', 'Contradict', 'Neutral'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    return accuracy, precision, all_labels, all_predictions, all_premises, all_hypotheses

# Define regularization strengths to test
reg_strengths = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
# reg_strengths = [1e-2]

# Lists to store accuracy values for plotting
train_accuracies = []
val_accuracies = []
test_accuracies = []
test_precisions = []
losses = []

# Train and evaluate models with different regularization strengths
for reg_strength in reg_strengths:
    # Instantiate the model with L2 regularization
    mlp_model = MLPModel(input_size, hidden_size, output_size).to(DEVICE)

    # Define optimizer with L2 regularization
    optimizer = optim.Adam(mlp_model.parameters(), lr=LEARNING_RATE, weight_decay=reg_strength)

    # Lists to store accuracy values during training
    epoch_train_accuracies = []
    epoch_val_accuracies = []
    epoch_losses = []

    # Training loop
    for epoch in range(EPOCHS):
        avg_loss = train_model(mlp_model, train_loader, optimizer, loss_fn, DEVICE)
        
        # Calculate training and validation accuracies
        train_acc, train_prec, train_labels, train_predictions, train_premises, train_hypotheses = evaluate_model(mlp_model, train_loader, DEVICE)
        val_acc, val_prec, val_labels, val_predictions, val_premises, val_hypotheses = evaluate_model(mlp_model, val_loader, DEVICE)

        epoch_train_accuracies.append(train_acc)
        epoch_val_accuracies.append(val_acc)
        epoch_losses.append(avg_loss)

        print(f'Epoch {epoch + 1}/{EPOCHS}, Average Loss: {avg_loss}, Train Accuracy: {train_acc}, Validation Accuracy: {val_acc}')

    # Save the accuracy values for plotting
    train_accuracies.append(epoch_train_accuracies)
    val_accuracies.append(epoch_val_accuracies)
    losses.append(epoch_losses)
    
    print(f"Evaluating the model on the test data at Epoch: {epoch}")
    test_acc, test_prec, test_labels, test_predictions, test_premises, test_hypotheses = evaluate_model(mlp_model, test_loader, DEVICE, graph = False)
    test_accuracies.append(test_acc)
    test_precisions.append(test_prec)
    
    print("\nTop 10 Test Examples:")
    for i in range(min(10, len(test_labels))):
        print(f'\nExample {i + 1}:')
        print(f'Premise: {test_premises[i]}')
        print(f'Hypothesis: {test_hypotheses[i]}')
        print(f'True Label: {test_labels[i]}, Predicted Label: {test_predictions[i]}')


print(pretrained_name)
for i, reg_strength in enumerate(reg_strengths):
    print(f'Test Accuracy {test_accuracies[i]}, Precision {test_precisions[i]}, Regularization {reg_strength}')

# Plotting
plt.figure(figsize=(12, 6))

# Plotting training and validation accuracies
plt.subplot(1, 2, 1)
for i, reg_strength in enumerate(reg_strengths):
    plt.plot(range(1, EPOCHS + 1), train_accuracies[i], label=f'Train (Reg Strength: {reg_strength})')
    plt.plot(range(1, EPOCHS + 1), val_accuracies[i], label=f'Validation (Reg Strength: {reg_strength})', linestyle='--')

plt.title('Training and Validation Accuracy Over Time')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting training loss
plt.subplot(1, 2, 2)
for i, reg_strength in enumerate(reg_strengths):
    plt.plot(range(1, EPOCHS + 1), losses[i], label=f'Train Loss (Reg Strength: {reg_strength})')

plt.title('Training Loss Over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_validation_loss_acc_plot_nn.png')
plt.show()


