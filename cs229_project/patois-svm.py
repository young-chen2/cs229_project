import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.svm import SVC
from sklearn.metrics import precision_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np

# Constants
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_name = "bert-base-multilingual-cased"

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(pretrained_name)

# Load and preprocess Jamaican Patois NLI dataset
train_file_path = 'datasets/patoisnli/jampatoisnli-train.csv'
val_file_path = 'datasets/patoisnli/jampatoisnli-val.csv'
test_file_path = 'datasets/patoisnli/jampatoisnli-test.csv'

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

# Define SVM model
class SVMModel(nn.Module):
    def __init__(self, input_size):
        super(SVMModel, self).__init__()
        self.svm = SVC(kernel='linear')

    def forward(self, x):
        pass  # No forward pass for SVM

# Instantiate the SVM model
svm_model = SVMModel(input_size=MAX_SEQ_LENGTH).to(DEVICE)

# Training loop for SVM model
for epoch in range(EPOCHS):
    svm_model.train()
    features = []
    labels = []

    for batch in train_loader:
        input_ids, attention_mask, batch_labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['label'].to(DEVICE)

        # Flatten and concatenate input_ids and attention_mask
        batch_features = torch.cat([input_ids, attention_mask], dim=1)
        features.append(batch_features.cpu().numpy())
        labels.append(batch_labels.cpu().numpy())

    # Flatten the features and labels
    features = np.vstack(features)
    labels = np.hstack(labels)

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Train SVM
    svm_model.svm.fit(features, labels)

# Validation loop for SVM model
svm_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['label'].to(DEVICE)

        # Flatten and concatenate input_ids and attention_mask
        batch_features = torch.cat([input_ids, attention_mask], dim=1)
        batch_features = scaler.transform(batch_features.cpu().numpy())  # Standardize

        outputs = torch.tensor(svm_model.svm.predict(batch_features), dtype=torch.long).to(DEVICE)

        total += labels.size(0)
        correct += (outputs == labels).sum().item()

accuracy = correct / total
print(f'SVM Validation Accuracy: {accuracy * 100:.2f}%')

# Function to train the SVM model
def train_svm_model(model, train_loader, device):
    model.train()
    features = []
    labels = []

    for batch in train_loader:
        input_ids, attention_mask, batch_labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)

        # Flatten and concatenate input_ids and attention_mask
        batch_features = torch.cat([input_ids, attention_mask], dim=1)
        features.append(batch_features.cpu().numpy())
        labels.append(batch_labels.cpu().numpy())

    # Flatten the features and labels
    features = np.vstack(features)
    labels = np.hstack(labels)

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Train SVM
    model.svm.fit(features, labels)

# Function to evaluate the SVM model
def evaluate_svm_model(model, data_loader, device, graph=False):
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

            # Flatten and concatenate input_ids and attention_mask
            batch_features = torch.cat([input_ids, attention_mask], dim=1)
            batch_features = scaler.transform(batch_features.cpu().numpy())  # Standardize

            outputs = torch.tensor(model.svm.predict(batch_features), dtype=torch.long).to(device)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())
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

# Train and evaluate SVM model
train_svm_model(svm_model, train_loader, DEVICE)

# Validation loop for SVM model
svm_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['label'].to(DEVICE)

        # Flatten and concatenate input_ids and attention_mask
        batch_features = torch.cat([input_ids, attention_mask], dim=1)
        batch_features = scaler.transform(batch_features.cpu().numpy())  # Standardize

        outputs = torch.tensor(svm_model.svm.predict(batch_features), dtype=torch.long).to(DEVICE)

        total += labels.size(0)
        correct += (outputs == labels).sum().item()

accuracy = correct / total
print(f'SVM Validation Accuracy: {accuracy * 100:.2f}%')

# Evaluate SVM model on the test set
print("Evaluating the SVM model on the test data:")
test_acc, test_prec, test_labels, test_predictions, test_premises, test_hypotheses = evaluate_svm_model(svm_model, test_loader, DEVICE, graph=True)

print(f"SVM Test Accuracy: {test_acc}, Precision: {test_prec}")
