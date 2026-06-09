# Tool Review Sentiment Classifier

A PyTorch deep learning model that classifies product reviews as positive 
or negative using a fine-tuned DistilBERT transformer.

## Results
- **Accuracy:** 89.5%
- **F1 Score:** 0.89
- **Test samples:** 1,000 reviews

## Tech Stack
- Python
- PyTorch
- HuggingFace Transformers (DistilBERT)
- Pandas
- Scikit-learn
- Matplotlib

## Project Structure
- `phase1_data_prep.py` — Data loading and preprocessing
- `phase2_tokenizer.py` — DistilBERT tokenization and PyTorch Dataset class
- `phase3_training.py` — Model fine-tuning and training loop
- `phase4_evaluation.py` — Evaluation with accuracy, F1, and confusion matrix

## How It Works
1. Loads 50,000 Amazon product reviews
2. Tokenizes text using DistilBERT tokenizer with max length 128
3. Fine-tunes DistilBERT for binary sentiment classification
4. Evaluates on 1,000 held-out test reviews

## Model Performance
The model correctly classifies 9 out of 10 reviews with balanced 
precision and recall across both positive and negative classes.

## Business Application
This same approach could be applied to power tool customer feedback, 
warranty claims, or field technician reports to automatically surface 
product quality issues at scale.