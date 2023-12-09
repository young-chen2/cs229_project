import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm

# Define constants
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.to(DEVICE)

# Jamaican Patois NLI Dataset class
class JamaicanPatoisNLIDataset(Dataset):
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, sep='\t')
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        premise = str(self.df.loc[idx, 'premise'])
        hypothesis = str(self.df.loc[idx, 'hypothesis'])
        label = int(self.df.loc[idx, 'label'])

        inputs = self.tokenizer(
            text=premise,
            text_pair=hypothesis,
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'token_type_ids': inputs['token_type_ids'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Instantiate DataLoader for training and validation sets
train_dataset = JamaicanPatoisNLIDataset(file_path='datasets/patoisnli/jampatoisnli-train.csv')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = JamaicanPatoisNLIDataset(file_path='datasets/patoisnli/jampatoisnli-test.csv')
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}'):
        inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Average Loss: {avg_loss}')

# Save the finetuned model
model.save_pretrained("uncased_L-12_H-768_A-12/finetune_bert_model")
tokenizer.save_pretrained("uncased_L-12_H-768_A-12/finetune_bert_tokenizer")
