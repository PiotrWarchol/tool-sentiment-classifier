import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer

# Load cleaned data
print("Loading cleaned data...")
train_df = pd.read_csv('train_clean.csv')
test_df = pd.read_csv('test_clean.csv')

print(f"Train size: {len(train_df)}")
print(f"Test size:  {len(test_df)}")

# Load DistilBERT tokenizer
print("\nLoading DistilBERT tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Test the tokenizer on one review
sample = train_df['text'][0]
print(f"\nSample review:\n{sample[:100]}...")

tokens = tokenizer(sample, truncation=True, padding='max_length', 
                   max_length=128, return_tensors='pt')
print(f"\nToken IDs shape: {tokens['input_ids'].shape}")
print(f"Attention mask shape: {tokens['attention_mask'].shape}")

# Create PyTorch Dataset class
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

# Create datasets
print("\nCreating datasets...")
train_dataset = ReviewDataset(
    train_df['text'].values,
    train_df['label'].values,
    tokenizer
)

test_dataset = ReviewDataset(
    test_df['text'].values,
    test_df['label'].values,
    tokenizer
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Test batches:  {len(test_loader)}")

# Test one batch
print("\nTesting one batch...")
batch = next(iter(train_loader))
print(f"Input IDs shape:      {batch['input_ids'].shape}")
print(f"Attention mask shape: {batch['attention_mask'].shape}")
print(f"Labels shape:         {batch['label'].shape}")

print("\nPhase 2 complete!")