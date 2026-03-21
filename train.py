"""
train.py
Fine-tunes DistilBERT on the loan approval dataset.
Usage: python train.py
Saves model to models/bert_loan_model/
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME   = 'distilbert-base-uncased'
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
WARMUP_RATIO = 0.1
SAVE_DIR     = 'models/bert_loan_model'
DATA_PATH    = 'data/loan_data.csv'
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🔧 Device: {DEVICE}")
print(f"🔧 Model: {MODEL_NAME}")

# ── Dataset ─────────────────────────────────────────────────────────────────
class LoanDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ── Helpers ──────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            ids   = batch['input_ids'].to(DEVICE)
            mask  = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            total_loss += out.loss.item()
            probs = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
            preds = out.logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    avg_loss = total_loss / len(loader)
    acc  = accuracy_score(all_labels, all_preds)
    auc  = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, auc, all_preds, all_labels, all_probs


def save_training_plots(history, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.title.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'o-', color='#58a6ff', label='Train')
    axes[0].plot(epochs, history['val_loss'],   'o-', color='#f85149', label='Val')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend(facecolor='#21262d', labelcolor='white')

    axes[1].plot(epochs, history['val_acc'], 'o-', color='#3fb950')
    axes[1].set_title('Validation Accuracy'); axes[1].set_xlabel('Epoch')
    axes[1].set_ylim(0, 1)

    axes[2].plot(epochs, history['val_auc'], 'o-', color='#d2a8ff')
    axes[2].set_title('ROC-AUC'); axes[2].set_xlabel('Epoch')
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 Saved training_curves.png")


def save_confusion_matrix(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'],
                ax=ax, cbar=False,
                annot_kws={'color': 'white', 'fontsize': 14})
    ax.set_xlabel('Predicted', color='#8b949e')
    ax.set_ylabel('Actual', color='#8b949e')
    ax.set_title('Confusion Matrix', color='#e6edf3')
    ax.tick_params(colors='#8b949e')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 Saved confusion_matrix.png")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"   {len(df)} records | Approval rate: {df['approved'].mean()*100:.1f}%")

    texts  = df['description'].tolist()
    labels = df['approved'].tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    train_ds = LoanDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_ds   = LoanDataset(X_val,   y_val,   tokenizer, MAX_LEN)
    test_ds  = LoanDataset(X_test,  y_test,  tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    print("\n🤖 Loading DistilBERT...")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(DEVICE)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    best_auc = 0

    # Training loop
    print(f"\n🚀 Training for {EPOCHS} epochs on {DEVICE}...\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            ids    = batch['input_ids'].to(DEVICE)
            mask   = batch['attention_mask'].to(DEVICE)
            labels_b = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=labels_b)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += out.loss.item()
            pbar.set_postfix({'loss': f'{out.loss.item():.4f}'})

        avg_train = train_loss / len(train_loader)
        val_loss, val_acc, val_auc, _, _, _ = evaluate(model, val_loader)

        history['train_loss'].append(avg_train)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        print(f"\n📈 Epoch {epoch}: train_loss={avg_train:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"   ✅ Best model saved (AUC={best_auc:.4f})")

    # Final test evaluation
    print("\n🧪 Final test evaluation...")
    _, test_acc, test_auc, test_preds, test_labels, _ = evaluate(model, test_loader)
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test ROC-AUC:  {test_auc:.4f}")
    print("\n" + classification_report(test_labels, test_preds, target_names=['Rejected', 'Approved']))

    # Save plots
    save_training_plots(history, SAVE_DIR)
    save_confusion_matrix(test_labels, test_preds, SAVE_DIR)

    # Save metadata
    meta = {
        'model_name': MODEL_NAME,
        'max_len': MAX_LEN,
        'epochs': EPOCHS,
        'best_val_auc': round(best_auc, 4),
        'test_accuracy': round(test_acc, 4),
        'test_auc': round(test_auc, 4),
        'labels': ['Rejected', 'Approved'],
    }
    with open(os.path.join(SAVE_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Training complete! Model saved to {SAVE_DIR}/")
    print(f"   Best Val AUC : {best_auc:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test AUC     : {test_auc:.4f}")


if __name__ == '__main__':
    main()
