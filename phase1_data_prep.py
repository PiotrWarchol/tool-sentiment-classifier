import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading dataset...")

# Load a sample of 50,000 rows - full dataset is too large
df = pd.read_csv("train.csv", header=None, 
                  names=["label", "title", "text"],
                  nrows=50000)

# Labels are 1 and 2 - convert to 0 and 1
df['label'] = df['label'].apply(lambda x: 1 if x == 2 else 0)

# Keep only text and label
df = df[['text', 'label']].dropna()

# Remove very short or very long reviews
df = df[df['text'].str.split().str.len().between(5, 256)]
df = df.reset_index(drop=True)

print(f"Total samples: {len(df)}")
print("\nClass distribution:")
print(df['label'].value_counts())
print("\nSample reviews:")
print(df[['text', 'label']].head())

# Train/test split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"\nTraining samples: {len(train_df)}")
print(f"Test samples:     {len(test_df)}")

# Save cleaned versions
train_df.to_csv('train_clean.csv', index=False)
test_df.to_csv('test_clean.csv', index=False)
print("\nSaved train_clean.csv and test_clean.csv!")
print("\nPhase 1 complete!")