# 🤖 LoanBERT — BERT Fine-Tuned Loan Approval System

A full-stack NLP application that fine-tunes **DistilBERT** to make loan approval decisions from natural language descriptions of applicant profiles.

## 📁 Project Structure

```
loan_bert/
├── app.py              # Flask web application
├── train.py            # DistilBERT fine-tuning script
├── predictor.py        # Inference engine (BERT + rule-based fallback)
├── generate_data.py    # Synthetic dataset generator
├── requirements.txt
├── data/
│   └── loan_data.csv   # Generated after running generate_data.py
├── models/
│   └── bert_loan_model/  # Saved after training
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer files
│       ├── meta.json
│       ├── training_curves.png
│       └── confusion_matrix.png
├── templates/
│   ├── base.html
│   ├── apply.html          # Loan application form with live NLP preview
│   ├── result.html         # Decision page with BERT confidence score
│   ├── dashboard.html      # All applications overview
│   └── model_info.html     # Architecture & performance metrics
└── static/
    ├── css/main.css
    └── js/main.js
```

---

## 🚀 Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate dataset
```bash
python generate_data.py
```
Creates 5,000 synthetic loan records with realistic approval logic.

### Step 3 — Fine-tune BERT
```bash
python train.py
```
- Downloads DistilBERT (~250MB on first run)
- Trains for 3 epochs (~10-30 min depending on GPU/CPU)
- Saves model + plots to `models/bert_loan_model/`

### Step 4 — Run the web app
```bash
python app.py
```
Visit **http://localhost:5000**

> 💡 The app works immediately without training (uses rule-based scoring as fallback). Train for full BERT power.

---

## 🧠 How BERT is Used

### The NLP Trick
Instead of feeding raw numbers to BERT, we convert structured data into **natural language**:

```
"Applicant is a 34-year-old employed individual requesting a $45,000 loan 
for home purchase over 60 months with property as collateral. Annual income 
is $82,000 with a credit score of 720. Debt-to-income ratio is 0.32. 
The applicant has no existing loans."
```

BERT reads this text and learns to associate language patterns with approval decisions.

### Model Architecture
```
Input Text → DistilBERT Tokenizer → DistilBERT Encoder (6 layers)
→ [CLS] token → Dropout → Linear(768→2) → Softmax → [P(Reject), P(Approve)]
```

### Scoring Formula
- **60%** Skill/feature score (rule-based explainability layer)
- **40%** BERT semantic confidence

---

## 📊 Expected Performance

After 3 epochs on 5,000 records:
| Metric | Expected |
|--------|----------|
| Test Accuracy | ~87–92% |
| ROC-AUC | ~0.92–0.96 |
| Training Time (CPU) | ~20–40 min |
| Training Time (GPU) | ~3–8 min |

---

## 🌐 Routes

| Route | Description |
|-------|-------------|
| `/apply` | Loan application form with live BERT input preview |
| `/result/<id>` | Decision page: confidence ring, factors, BERT text |
| `/dashboard` | All applications with stats |
| `/model-info` | Architecture diagram + training metrics |
| `/api/predict` | POST JSON → prediction |
| `/api/stats` | GET pipeline statistics |

---

## 🔧 Extending

- **Better model**: Swap DistilBERT for `bert-base-uncased` for ~2% accuracy gain
- **More features**: Add education level, zip code, employment years
- **SHAP explanations**: Use `shap` library for token-level importance
- **Real data**: Replace synthetic data with Lending Club dataset from Kaggle
- **Bias auditing**: Add fairness metrics (demographic parity, equalized odds)
