"""
predictor.py
LoanBERT 2.0 Underwriting & Risk Intelligence Engine.
Provides:
1. Multi-Task Loan Decisioning (Approval, Risk Tier A+ to D, APR %, Max Line, PD %).
2. Token-Level Explainability (XAI) with Word Saliency & Sentiment Heatmap.
3. Counterfactual "What-If" Simulation Engine.
4. Comprehensive Quantitative Factor Decomposition.
"""

import os
import re
import json
import joblib
import numpy as np

MODEL_DIR = 'models/bert_loan_model'
_pipeline = None
_token_saliency = None
_meta = None


def _load_engine():
    global _pipeline, _token_saliency, _meta
    if _pipeline is not None:
        return True
    pipeline_path = os.path.join(MODEL_DIR, 'model_pipeline.joblib')
    if os.path.exists(pipeline_path):
        try:
            _pipeline = joblib.load(pipeline_path)
            meta_path = os.path.join(MODEL_DIR, 'meta.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    _meta = json.load(f)
            saliency_path = os.path.join(MODEL_DIR, 'token_saliency.json')
            if os.path.exists(saliency_path):
                with open(saliency_path, 'r', encoding='utf-8') as f:
                    _token_saliency = json.load(f)
            return True
        except Exception as e:
            print(f"[-] Pipeline load warning: {e}")
    return False


def build_description(data: dict) -> str:
    """Generate high-fidelity, contextual natural language credit dossier."""
    age         = int(data.get('age', 32))
    emp         = str(data.get('employment_status', 'Employed'))
    emp_years   = float(data.get('employment_years', 4.5))
    income      = int(data.get('annual_income', 65000))
    cs          = int(data.get('credit_score', 700))
    loan_amount = int(data.get('loan_amount', 45000))
    term        = int(data.get('loan_term_months', 60))
    purpose     = str(data.get('loan_purpose', 'Personal'))
    dti         = float(data.get('debt_to_income', 0.35))
    existing    = int(data.get('existing_loans', 0))
    collateral  = str(data.get('collateral', 'None'))
    revolving   = float(data.get('revolving_utilization', 0.25))
    delinq      = int(data.get('delinquencies', 0))
    notes       = str(data.get('underwriter_notes', '')).strip()

    coll_text = f"secured by {collateral.lower()}" if collateral != 'None' else "with no asset collateral offered"
    existing_text = f"{existing} active loan liability(ies)" if existing > 0 else "zero existing debt obligations"
    delinq_text = f"with {delinq} past delinquency flag(s)" if delinq > 0 else "with zero recorded delinquencies"
    notes_text = f" Underwriting assessment: {notes}" if notes else ""

    desc = (
        f"Applicant is a {age}-year-old {emp.lower()} professional ({emp_years:.1f} years experience) "
        f"requesting a ${loan_amount:,} credit facility for {purpose.lower()} over {term} months {coll_text}. "
        f"Declared annual income is ${income:,} against a credit score of {cs} and DTI ratio of {dti:.2f}. "
        f"Credit file shows {existing_text}, {revolving*100:.0f}% revolving utilization, and {delinq_text}.{notes_text}"
    )
    return desc


def _extract_token_saliency(description: str, approved: bool) -> list:
    """
    Extract token-level word saliency for inline HTML interactive heatmaps.
    Returns list of dicts: {'word': str, 'weight': float, 'impact': 'positive'|'negative'|'neutral'}
    """
    tokens = re.findall(r"[\w'$%.,-]+|\s+|[^\w\s]", description)
    saliency_data = []

    pos_keywords = {
        'clean': 1.8, 'low': 1.6, 'zero': 2.0, 'prime': 2.2, 'stable': 1.5,
        'property': 1.4, 'investments': 1.4, 'real estate': 1.5, 'portfolio': 1.4,
        'strong': 1.8, 'solid': 1.6, 'employed': 1.2, 'business owner': 1.1,
        'comfortable': 1.3, 'disciplined': 1.6, 'exemplary': 2.0, 'liquid': 1.4,
        '800': 2.5, '750': 2.0, '700': 1.2, 'income': 0.8, 'reserve': 1.2
    }
    neg_keywords = {
        'unemployed': -2.8, 'poor': -2.4, 'delinquency': -2.5, 'delinquencies': -2.8,
        'bankruptcy': -3.5, 'elevated': -1.8, 'high': -1.5, 'over-leveraged': -2.2,
        'student': -1.2, 'burden': -1.6, 'shocks': -1.5, 'none': -0.4,
        '550': -2.2, '500': -2.5, '450': -2.8, 'liability': -1.1, 'aggressive': -1.4
    }

    # Merge with trained vocabulary if present
    trained_weights = _token_saliency or {}

    for token in tokens:
        clean = token.lower().strip(",.$%'-")
        weight = 0.0
        
        if clean in trained_weights:
            weight = trained_weights[clean]
        elif clean in pos_keywords:
            weight = pos_keywords[clean]
        elif clean in neg_keywords:
            weight = neg_keywords[clean]
        elif clean.isdigit():
            val = int(clean)
            if 700 <= val <= 850: weight = 1.6
            elif 300 <= val < 600: weight = -1.9
            elif val > 80000: weight = 1.3
        
        if weight > 0.15:
            impact = 'positive'
        elif weight < -0.15:
            impact = 'negative'
        else:
            impact = 'neutral'

        saliency_data.append({
            'text': token,
            'weight': round(float(weight), 2),
            'impact': impact
        })

    return saliency_data


def _calculate_score_and_factors(data: dict):
    """Deep multi-factor underwriting score calculation & quantitative factor breakdown."""
    reasons = []
    flags = []
    factor_breakdown = []

    cs = int(data.get('credit_score', 680))
    if cs >= 800:
        pts = 38
        reasons.append(f"Exceptional FICO credit score ({cs}) — Tier 1 Credit Profile")
    elif cs >= 740:
        pts = 28
        reasons.append(f"Very Good credit score ({cs}) — Low Historical Risk")
    elif cs >= 680:
        pts = 16
        reasons.append(f"Good credit score ({cs}) — Standard Prime Qualification")
    elif cs >= 620:
        pts = 4
        reasons.append(f"Fair credit score ({cs})")
    elif cs >= 580:
        pts = -15
        flags.append(f"Below-average credit score ({cs}) — Subprime Risk Bracket")
    else:
        pts = -35
        flags.append(f"Poor credit score ({cs}) — High Probability of Default")
    factor_breakdown.append({'factor': 'Credit Score (FICO)', 'points': pts, 'type': 'positive' if pts >= 0 else 'negative'})

    income = int(data.get('annual_income', 60000))
    if income >= 120000:
        pts = 22
        reasons.append(f"High annual gross income (${income:,})")
    elif income >= 75000:
        pts = 14
        reasons.append(f"Solid annual earnings (${income:,})")
    elif income >= 45000:
        pts = 6
        reasons.append(f"Adequate income level (${income:,})")
    elif income >= 25000:
        pts = -6
        flags.append(f"Modest annual income (${income:,})")
    else:
        pts = -18
        flags.append(f"Low income threshold (${income:,})")
    factor_breakdown.append({'factor': 'Annual Income', 'points': pts, 'type': 'positive' if pts >= 0 else 'negative'})

    dti = float(data.get('debt_to_income', 0.35))
    if dti <= 0.25:
        pts = 22
        reasons.append(f"Excellent debt-to-income ratio ({dti:.2f}) — Strong Debt Coverage")
    elif dti <= 0.40:
        pts = 12
        reasons.append(f"Healthy debt-to-income ratio ({dti:.2f})")
    elif dti <= 0.55:
        pts = -2
        flags.append(f"Elevated DTI ratio ({dti:.2f})")
    elif dti <= 0.70:
        pts = -16
        flags.append(f"High debt-to-income leverage ({dti:.2f}) — Tight Cashflow")
    else:
        pts = -30
        flags.append(f"Severe debt-to-income overload ({dti:.2f})")
    factor_breakdown.append({'factor': 'Debt-To-Income (DTI)', 'points': pts, 'type': 'positive' if pts >= 0 else 'negative'})

    emp = str(data.get('employment_status', 'Employed'))
    emp_yr = float(data.get('employment_years', 3))
    if emp == 'Employed':
        pts = 15 + min(int(emp_yr * 0.8), 8)
        reasons.append(f"Stable primary employment ({emp_yr:.1f} years in position)")
    elif emp in ('Self-Employed', 'Business Owner'):
        pts = 10 + min(int(emp_yr * 0.6), 6)
        reasons.append(f"{emp} with {emp_yr:.1f} years track record")
    elif emp == 'Retired':
        pts = 8
        reasons.append("Retirement pension / fixed verified income")
    elif emp == 'Freelancer':
        pts = 4
        flags.append("Freelance / variable income stream")
    elif emp == 'Student':
        pts = -8
        flags.append("Student status — limited current full-time cash flow")
    else:
        pts = -25
        flags.append("Currently unemployed — severe repayment hazard")
    factor_breakdown.append({'factor': 'Employment & Tenure', 'points': pts, 'type': 'positive' if pts >= 0 else 'negative'})

    collateral = str(data.get('collateral', 'None'))
    if collateral != 'None':
        pts = 14
        reasons.append(f"{collateral} provided as asset security")
    else:
        pts = 0
    factor_breakdown.append({'factor': 'Collateral Security', 'points': pts, 'type': 'positive' if pts > 0 else 'neutral'})

    existing = int(data.get('existing_loans', 0))
    if existing == 0:
        pts = 8
        reasons.append("No existing loan liabilities on record")
    elif existing <= 2:
        pts = 2
    elif existing <= 4:
        pts = -8
        flags.append(f"{existing} active loan obligations")
    else:
        pts = -18
        flags.append(f"{existing} active credit lines — multiple obligation stress")
    factor_breakdown.append({'factor': 'Existing Debt Count', 'points': pts, 'type': 'positive' if pts >= 0 else 'negative'})

    delinq = int(data.get('delinquencies', 0))
    if delinq > 0:
        pts = -delinq * 12
        flags.append(f"{delinq} past payment delinquency record(s) on file")
        factor_breakdown.append({'factor': 'Delinquencies', 'points': pts, 'type': 'negative'})

    revolving = float(data.get('revolving_utilization', 0.25))
    if revolving > 0.70:
        pts = -12
        flags.append(f"High credit card utilization ({revolving*100:.0f}%)")
        factor_breakdown.append({'factor': 'Credit Utilization', 'points': pts, 'type': 'negative'})
    elif revolving <= 0.30:
        pts = 6
        reasons.append(f"Low credit card utilization ({revolving*100:.0f}%)")
        factor_breakdown.append({'factor': 'Credit Utilization', 'points': pts, 'type': 'positive'})

    # Composite Health Index
    total_pts = sum(f['points'] for f in factor_breakdown)
    health_index = min(99.0, max(1.0, float(total_pts + 40)))

    return health_index, reasons, flags, factor_breakdown


def predict(data: dict) -> dict:
    """
    Main Multi-Task Prediction API.
    Returns:
    - approved (bool)
    - decision_status (str: Approved | Conditional Approval | Denied)
    - confidence (float %)
    - prob_approve, prob_reject (float %)
    - risk_tier (str)
    - recommended_apr (float %)
    - max_approved_amount (int $)
    - default_prob (float %)
    - credit_health_index (float)
    - reasons (list[str])
    - flags (list[str])
    - factor_breakdown (list[dict])
    - token_saliency (list[dict])
    - description (str)
    - model_used (str)
    """
    _load_engine()
    description = build_description(data)
    health_index, reasons, flags, factor_breakdown = _calculate_score_and_factors(data)

    if _pipeline is not None:
        try:
            vec = _pipeline['vectorizer'].transform([description])
            probs = _pipeline['calibrated_clf'].predict_proba(vec)[0]
            prob_approve = round(float(probs[1] * 100), 1)
            prob_reject  = round(float(probs[0] * 100), 1)
            
            # Predict APR and max line
            pred_apr = float(_pipeline['apr_regressor'].predict(vec)[0])
            recommended_apr = round(max(5.25, min(28.0, pred_apr)), 2)
            
            pred_line = float(_pipeline['line_regressor'].predict(vec)[0])
            max_approved_amount = int(max(2500, min(750000, pred_line)))
            
            model_used = "LoanBERT FinBERT/Ensemble v2.0 (Calibrated XAI Engine)"
        except Exception as e:
            print(f"[-] Inference exception: {e}")
            prob_approve = round(float(health_index), 1)
            prob_reject  = round(100.0 - prob_approve, 1)
            recommended_apr = round(5.25 + (100 - health_index) * 0.20, 2)
            max_approved_amount = int(int(data.get('annual_income', 50000)) * 0.45 * (health_index / 60.0))
            model_used = "Rule-Based Multi-Task Underwriting Engine"
    else:
        prob_approve = round(float(health_index), 1)
        prob_reject  = round(100.0 - prob_approve, 1)
        recommended_apr = round(5.25 + (100 - health_index) * 0.20, 2)
        max_approved_amount = int(int(data.get('annual_income', 50000)) * 0.45 * (health_index / 60.0))
        model_used = "Rule-Based Multi-Task Underwriting Engine"

    # Multi-task decisions
    approved = bool(prob_approve >= 48.0)
    if prob_approve >= 65.0:
        decision_status = 'Approved'
    elif prob_approve >= 48.0:
        decision_status = 'Conditional Approval'
    else:
        decision_status = 'Denied'

    # Risk Tier assignment
    if health_index >= 80:
        risk_tier = 'Tier A+ (Exceptional Prime)'
    elif health_index >= 65:
        risk_tier = 'Tier A (Prime)'
    elif health_index >= 48:
        risk_tier = 'Tier B (Near-Prime)'
    elif health_index >= 32:
        risk_tier = 'Tier C (Subprime)'
    else:
        risk_tier = 'Tier D (High Risk)'

    confidence = round(prob_approve if approved else prob_reject, 1)
    default_prob = round(1.0 / (1.0 + np.exp((health_index - 45) / 10.0)) * 100, 1)
    
    # Token Saliency for interactive text highlight
    token_saliency = _extract_token_saliency(description, approved)

    return {
        'approved':            approved,
        'decision_status':     decision_status,
        'confidence':          confidence,
        'prob_approve':        prob_approve,
        'prob_reject':         prob_reject,
        'risk_tier':           risk_tier,
        'recommended_apr':     recommended_apr,
        'max_approved_amount': max_approved_amount,
        'default_prob':        default_prob,
        'credit_health_index': round(health_index, 1),
        'reasons':             reasons,
        'flags':               flags,
        'factor_breakdown':    factor_breakdown,
        'token_saliency':      token_saliency,
        'description':         description,
        'model_used':          model_used,
    }


def counterfactual_simulation(base_data: dict, deltas: dict) -> dict:
    """
    Simulate impact of user counterfactual adjustments (credit score, income, loan amount, DTI).
    Returns before vs after comparison with specific delta explanations.
    """
    original_pred = predict(base_data)
    
    modified_data = dict(base_data)
    if 'credit_score' in deltas:
        modified_data['credit_score'] = int(deltas['credit_score'])
    if 'annual_income' in deltas:
        modified_data['annual_income'] = int(deltas['annual_income'])
    if 'loan_amount' in deltas:
        modified_data['loan_amount'] = int(deltas['loan_amount'])
    if 'debt_to_income' in deltas:
        modified_data['debt_to_income'] = float(deltas['debt_to_income'])
    if 'collateral' in deltas:
        modified_data['collateral'] = str(deltas['collateral'])

    new_pred = predict(modified_data)

    delta_prob = round(new_pred['prob_approve'] - original_pred['prob_approve'], 1)
    delta_apr  = round(new_pred['recommended_apr'] - original_pred['recommended_apr'], 2)
    delta_line = new_pred['max_approved_amount'] - original_pred['max_approved_amount']

    return {
        'original': original_pred,
        'simulated': new_pred,
        'delta_prob': delta_prob,
        'delta_apr': delta_apr,
        'delta_max_line': delta_line,
        'status_changed': original_pred['decision_status'] != new_pred['decision_status']
    }

