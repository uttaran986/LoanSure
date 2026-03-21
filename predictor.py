"""
predictor.py
Loads fine-tuned BERT and runs inference.
Falls back to rule-based scoring if model not yet trained.
"""

import os
import json
import torch
import numpy as np

MODEL_DIR = 'models/bert_loan_model'
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_tokenizer = None
_model     = None
_meta      = None


def _load_model():
    global _tokenizer, _model, _meta
    if _model is not None:
        return True
    if not os.path.exists(os.path.join(MODEL_DIR, 'config.json')):
        return False
    try:
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        _tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
        _model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        _model     = _model.to(DEVICE)
        _model.eval()
        meta_path = os.path.join(MODEL_DIR, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                _meta = json.load(f)
        return True
    except Exception as e:
        print(f"Model load error: {e}")
        return False


def _rule_based_score(data: dict) -> dict:
    """Fallback rule-based scoring when BERT model not trained yet."""
    score = 0
    reasons = []
    flags   = []

    cs = int(data.get('credit_score', 650))
    if cs >= 750:   score += 30; reasons.append(f"Excellent credit score ({cs})")
    elif cs >= 700: score += 20; reasons.append(f"Good credit score ({cs})")
    elif cs >= 650: score += 10; reasons.append(f"Fair credit score ({cs})")
    elif cs >= 600: score += 0;  flags.append(f"Below-average credit score ({cs})")
    else:           score -= 20; flags.append(f"Poor credit score ({cs}) — high risk")

    income = int(data.get('annual_income', 50000))
    if income >= 100000:  score += 25; reasons.append(f"High income (${income:,})")
    elif income >= 60000: score += 15; reasons.append(f"Solid income (${income:,})")
    elif income >= 35000: score += 5
    else:                 score -= 10; flags.append(f"Low income (${income:,})")

    dti = float(data.get('debt_to_income', 0.4))
    if dti <= 0.3:   score += 20; reasons.append(f"Low debt-to-income ratio ({dti:.2f})")
    elif dti <= 0.5: score += 10
    elif dti <= 0.8: score -= 5;  flags.append(f"Elevated DTI ratio ({dti:.2f})")
    else:            score -= 20; flags.append(f"High DTI ratio ({dti:.2f}) — over-leveraged")

    emp = data.get('employment_status', 'Employed')
    if emp == 'Employed':           score += 15; reasons.append("Stable employment")
    elif emp in ('Self-Employed', 'Business Owner'): score += 8; reasons.append(f"{emp} — variable income")
    elif emp == 'Retired':          score += 5
    elif emp == 'Student':          score -= 5;  flags.append("Student — limited income")
    else:                           score -= 15; flags.append("Unemployed — high risk")

    collateral = data.get('collateral', 'None')
    if collateral != 'None':
        score += 10; reasons.append(f"{collateral} provided as collateral")

    existing = int(data.get('existing_loans', 0))
    if existing == 0:   score += 5; reasons.append("No existing loans")
    elif existing <= 2: pass
    else:               score -= 10; flags.append(f"{existing} existing loans — debt burden")

    loan_amount = int(data.get('loan_amount', 50000))
    ratio = loan_amount / max(income, 1)
    if ratio > 5: flags.append(f"Loan amount is {ratio:.1f}x annual income")

    confidence = min(99, max(1, score + 50))
    approved   = score >= 20

    return {
        'approved':   approved,
        'confidence': round(confidence, 1),
        'score':      score,
        'reasons':    reasons,
        'flags':      flags,
        'model_used': 'Rule-Based (BERT not trained yet)',
    }


def build_description(data: dict) -> str:
    """Convert form data dict into natural language for BERT."""
    coll = data.get('collateral', 'None')
    coll_text = f" with {coll.lower()} as collateral" if coll != 'None' else " with no collateral offered"
    existing = int(data.get('existing_loans', 0))
    existing_text = (f"The applicant has {existing} existing loan(s)."
                     if existing > 0 else "The applicant has no existing loans.")
    loan_amount = int(data.get('loan_amount', 0))
    income      = int(data.get('annual_income', 0))
    cs          = int(data.get('credit_score', 650))
    dti         = float(data.get('debt_to_income', 0.4))
    age         = int(data.get('age', 30))
    emp         = data.get('employment_status', 'employed').lower()
    purpose     = data.get('loan_purpose', 'personal').lower()
    term        = int(data.get('loan_term_months', 60))

    desc = (
        f"Applicant is a {age}-year-old {emp} individual requesting a "
        f"${loan_amount:,} loan for {purpose} over {term} months"
        f"{coll_text}. "
        f"Annual income is ${income:,} with a credit score of {cs}. "
        f"Debt-to-income ratio is {dti:.2f}. "
        f"{existing_text}"
    )
    return desc


def predict(data: dict) -> dict:
    """
    Main prediction function.
    Returns dict with: approved, confidence, reasons, flags, model_used, description
    """
    description = build_description(data)

    if not _load_model():
        result = _rule_based_score(data)
        result['description'] = description
        return result

    # BERT inference
    enc = _tokenizer(
        description,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt',
    )
    input_ids = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = _model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits
        probs   = torch.softmax(logits, dim=1)[0].cpu().numpy()

    approved   = bool(probs[1] > 0.5)
    confidence = round(float(probs[1] * 100 if approved else probs[0] * 100), 1)

    # Generate human-readable reasons
    reasons, flags = _generate_explanations(data)

    return {
        'approved':    approved,
        'confidence':  confidence,
        'prob_approve': round(float(probs[1] * 100), 1),
        'prob_reject':  round(float(probs[0] * 100), 1),
        'reasons':     reasons,
        'flags':       flags,
        'description': description,
        'model_used':  f"DistilBERT (AUC={_meta['test_auc'] if _meta else 'N/A'})",
    }


def _generate_explanations(data: dict):
    """Generate human-readable positive and negative factors."""
    reasons = []
    flags   = []

    cs = int(data.get('credit_score', 650))
    if cs >= 750:   reasons.append(f"Excellent credit score ({cs})")
    elif cs >= 700: reasons.append(f"Good credit score ({cs})")
    elif cs >= 650: reasons.append(f"Fair credit score ({cs})")
    elif cs < 600:  flags.append(f"Poor credit score ({cs})")

    income = int(data.get('annual_income', 50000))
    if income >= 80000:   reasons.append(f"Strong annual income (${income:,})")
    elif income >= 40000: reasons.append(f"Adequate income (${income:,})")
    else:                 flags.append(f"Low annual income (${income:,})")

    dti = float(data.get('debt_to_income', 0.4))
    if dti <= 0.35:  reasons.append(f"Healthy debt-to-income ratio ({dti:.2f})")
    elif dti > 0.6:  flags.append(f"High debt-to-income ratio ({dti:.2f})")

    emp = data.get('employment_status', '')
    if emp == 'Employed':    reasons.append("Stable employment status")
    elif emp == 'Unemployed': flags.append("Currently unemployed")

    coll = data.get('collateral', 'None')
    if coll != 'None': reasons.append(f"{coll} provided as collateral")

    existing = int(data.get('existing_loans', 0))
    if existing == 0:  reasons.append("No existing loan obligations")
    elif existing >= 3: flags.append(f"{existing} existing active loans")

    loan_amount = int(data.get('loan_amount', 0))
    if income > 0 and loan_amount / income > 4:
        flags.append(f"Loan amount is {loan_amount/income:.1f}× annual income")

    return reasons, flags
