import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
import time

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
print("Loading data...")
train_df = pd.read_csv('train_clean.csv')
test_df = pd.read_csv('test_clean.csv')

# Use a subset to keep training fast
train_df = train_df.sample(5000, random_state=42)
test_df = test_df.sample(1000, random_state=42)

# Dataset class
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Load tokenizer and model
print("Loading DistilBERT model...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', 
    num_labels=2
)
model.to(device)

# Create datasets and loaders
train_dataset = ReviewDataset(train_df['text'].values, 
                               train_df['label'].values, tokenizer)
test_dataset = ReviewDataset(test_df['text'].values, 
                              test_df['label'].values, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
EPOCHS = 3
print(f"\nStarting training for {EPOCHS} epochs...")
print(f"Training batches per epoch: {len(train_loader)}\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_num, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, 
                       attention_mask=attention_mask, 
                       labels=labels)

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Print progress every 50 batches
        if (batch_num + 1) % 50 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_num+1}/{len(train_loader)} "
                  f"| Loss: {total_loss/(batch_num+1):.4f}")

    avg_loss = total_loss / len(train_loader)
    elapsed = time.time() - start_time
    print(f"\nEpoch {epoch+1} complete | Avg Loss: {avg_loss:.4f} "
          f"| Time: {elapsed:.0f}s\n")

# Save the model
print("Saving model...")
model.save_pretrained('sentiment_model')
tokenizer.save_pretrained('sentiment_model')
print("Model saved to 'sentiment_model' folder!")
print("\nPhase 3 complete!")