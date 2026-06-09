import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load saved model
print("Loading saved model...")
tokenizer = DistilBertTokenizer.from_pretrained('sentiment_model')
model = DistilBertForSequenceClassification.from_pretrained('sentiment_model')
model.to(device)
model.eval()

# Load test data
test_df = pd.read_csv('test_clean.csv').sample(1000, random_state=42)

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

# Create test loader
test_dataset = ReviewDataset(test_df['text'].values,
                              test_df['label'].values, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Run predictions
print("Running predictions...")
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print(f"\nResults:")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"F1 Score: {f1:.4f}")
print(f"\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im)

classes = ['Negative', 'Positive']
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black',
                fontsize=14)

ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Sentiment Classifier - Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("\nConfusion matrix saved as confusion_matrix.png!")

# Test on a custom review
print("\n--- Testing on custom reviews ---")
def predict(text):
    inputs = tokenizer(text, return_tensors='pt',
                      truncation=True, padding=True,
                      max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits).item()
    return "POSITIVE" if pred == 1 else "NEGATIVE"

test_reviews = [
    "This drill is amazing, works perfectly every time!",
    "Terrible product, broke after one use. Total waste of money.",
    "Decent quality for the price, would recommend to a friend.",
    "Very disappointed, the battery died within an hour."
]

for review in test_reviews:
    print(f"\nReview: {review[:60]}...")
    print(f"Prediction: {predict(review)}")

print("\nPhase 4 complete!")