# app.py — LoanSure AI: Enterprise Loan Management & Conversational NLP Underwriting Platform

import os
import io
import csv
import json
import re
from datetime import datetime, timedelta
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, Response, session
from flask_sqlalchemy import SQLAlchemy

from predictor import predict, counterfactual_simulation, build_description
from advisor_engine import (
    calculate_reducing_emi,
    generate_amortization_schedule,
    estimate_eligibility,
    generate_recommendations,
    QUESTIONS_FLOW,
    extract_value_from_text,
    call_local_llm
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'loansure-enterprise-ai-secret-key-2026'

# Support both local SQLite and Vercel serverless /tmp directory
if os.environ.get('VERCEL'):
    db_path = '/tmp/loansure.db'
else:
    os.makedirs('instance', exist_ok=True)
    db_path = os.path.abspath('instance/loansure.db')

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


# ── Database Schema ──────────────────────────────────────────────────────────
class LoanApplication(db.Model):
    id                   = db.Column(db.Integer, primary_key=True)
    applicant_name       = db.Column(db.String(200), default='Anonymous')
    age                  = db.Column(db.Integer, default=30)
    city                 = db.Column(db.String(100), default='Austin, TX')
    annual_income        = db.Column(db.Integer, default=60000)
    monthly_salary       = db.Column(db.Float, default=5000.0)
    credit_score         = db.Column(db.Integer, default=680)
    loan_amount          = db.Column(db.Integer, default=35000)
    loan_term_months     = db.Column(db.Integer, default=48)
    employment_status    = db.Column(db.String(50), default='Employed')
    employment_years     = db.Column(db.Float, default=3.5)
    loan_purpose         = db.Column(db.String(100), default='Personal')
    debt_to_income       = db.Column(db.Float, default=0.32)
    existing_loans       = db.Column(db.Integer, default=0)
    existing_emis        = db.Column(db.Float, default=0.0)
    collateral           = db.Column(db.String(50), default='None')
    revolving_utilization= db.Column(db.Float, default=0.28)
    delinquencies        = db.Column(db.Integer, default=0)
    underwriter_notes    = db.Column(db.Text, default='')
    description          = db.Column(db.Text, default='')
    source               = db.Column(db.String(50), default='AI Advisor') # 'AI Advisor' or 'Manual Apply'
    
    # Decisions & Predictions
    approved             = db.Column(db.Boolean, default=False)
    decision_status      = db.Column(db.String(50), default='Under Review') # Approved, Under Review, Pending, Denied
    confidence           = db.Column(db.Float, default=50.0)
    prob_approve         = db.Column(db.Float, default=50.0)
    prob_reject          = db.Column(db.Float, default=50.0)
    risk_tier            = db.Column(db.String(50), default='Tier B (Near-Prime)')
    recommended_apr      = db.Column(db.Float, default=11.5)
    max_approved_amount  = db.Column(db.Integer, default=30000)
    default_prob         = db.Column(db.Float, default=8.5)
    credit_health_index  = db.Column(db.Float, default=70.0)
    monthly_emi          = db.Column(db.Float, default=850.0)
    kyc_status           = db.Column(db.String(50), default='Verified')
    
    # Explainability & Recommendations Data
    reasons              = db.Column(db.Text, default='[]')
    flags                = db.Column(db.Text, default='[]')
    factor_breakdown     = db.Column(db.Text, default='[]')
    token_saliency       = db.Column(db.Text, default='[]')
    recommendations_json = db.Column(db.Text, default='{}')
    model_used           = db.Column(db.String(200), default='LoanSure FinBERT Engine')
    created_at           = db.Column(db.DateTime, default=datetime.utcnow)

    def get_reasons(self):
        try: return json.loads(self.reasons) if self.reasons else []
        except: return []

    def get_flags(self):
        try: return json.loads(self.flags) if self.flags else []
        except: return []

    def get_factor_breakdown(self):
        try: return json.loads(self.factor_breakdown) if self.factor_breakdown else []
        except: return []

    def get_token_saliency(self):
        try: return json.loads(self.token_saliency) if self.token_saliency else []
        except: return []

    def get_recommendations(self):
        try: return json.loads(self.recommendations_json) if self.recommendations_json else {}
        except: return {}


def seed_demo_data():
    """Seed initial realistic enterprise loan records matching the UI mockup if table is empty."""
    if LoanApplication.query.count() > 0:
        return
    
    sample_records = [
        {
            'applicant_name': 'Eleanor Vance', 'age': 34, 'city': 'New York, NY', 'annual_income': 115000,
            'monthly_salary': 9580, 'credit_score': 785, 'loan_amount': 45000, 'loan_term_months': 60,
            'employment_status': 'Employed', 'employment_years': 6.5, 'loan_purpose': 'Home Improvement',
            'debt_to_income': 0.18, 'existing_loans': 0, 'existing_emis': 0.0, 'collateral': 'Property',
            'revolving_utilization': 0.12, 'delinquencies': 0, 'underwriter_notes': 'Exemplary liquid reserves with consistent salary progression.',
            'source': 'AI Advisor'
        },
        {
            'applicant_name': 'Marcus Thorne', 'age': 29, 'city': 'San Francisco, CA', 'annual_income': 85000,
            'monthly_salary': 7080, 'credit_score': 675, 'loan_amount': 35000, 'loan_term_months': 48,
            'employment_status': 'Employed', 'employment_years': 3.5, 'loan_purpose': 'Debt Consolidation',
            'debt_to_income': 0.38, 'existing_loans': 2, 'existing_emis': 450.0, 'collateral': 'None',
            'revolving_utilization': 0.45, 'delinquencies': 0, 'underwriter_notes': 'Software engineering position, solid repayment history.',
            'source': 'AI Advisor'
        },
        {
            'applicant_name': 'Sophia Chen', 'age': 45, 'city': 'Austin, TX', 'annual_income': 140000,
            'monthly_salary': 11660, 'credit_score': 740, 'loan_amount': 75000, 'loan_term_months': 84,
            'employment_status': 'Business Owner', 'employment_years': 12.0, 'loan_purpose': 'Business Expansion',
            'debt_to_income': 0.24, 'existing_loans': 1, 'existing_emis': 600.0, 'collateral': 'Investments',
            'revolving_utilization': 0.22, 'delinquencies': 0, 'underwriter_notes': 'Established commercial enterprise with high verified operating cash flow.',
            'source': 'AI Advisor'
        },
        {
            'applicant_name': 'Daniel Fletcher', 'age': 41, 'city': 'Chicago, IL', 'annual_income': 95000,
            'monthly_salary': 7910, 'credit_score': 710, 'loan_amount': 50000, 'loan_term_months': 60,
            'employment_status': 'Employed', 'employment_years': 8.0, 'loan_purpose': 'Vehicle Purchase',
            'debt_to_income': 0.28, 'existing_loans': 1, 'existing_emis': 320.0, 'collateral': 'Vehicle',
            'revolving_utilization': 0.30, 'delinquencies': 0, 'underwriter_notes': 'Stable corporate employment with vehicle asset security.',
            'source': 'Manual Apply'
        },
        {
            'applicant_name': 'David Miller', 'age': 24, 'city': 'Miami, FL', 'annual_income': 42000,
            'monthly_salary': 3500, 'credit_score': 580, 'loan_amount': 18000, 'loan_term_months': 36,
            'employment_status': 'Freelancer', 'employment_years': 1.0, 'loan_purpose': 'Personal',
            'debt_to_income': 0.58, 'existing_loans': 3, 'existing_emis': 650.0, 'collateral': 'None',
            'revolving_utilization': 0.82, 'delinquencies': 1, 'underwriter_notes': 'Variable freelance earnings with high credit card utilization.',
            'source': 'AI Advisor'
        }
    ]

    for data in sample_records:
        res = predict(data)
        emi_calc = calculate_reducing_emi(data['loan_amount'], res['recommended_apr'], data['loan_term_months'])
        recs = generate_recommendations(
            data['loan_amount'], data['loan_term_months'],
            data['monthly_salary'], data['existing_emis'],
            data['credit_score'], data['loan_purpose']
        )
        
        loan = LoanApplication(
            applicant_name=data['applicant_name'],
            age=data['age'],
            city=data['city'],
            annual_income=data['annual_income'],
            monthly_salary=data['monthly_salary'],
            credit_score=data['credit_score'],
            loan_amount=data['loan_amount'],
            loan_term_months=data['loan_term_months'],
            employment_status=data['employment_status'],
            employment_years=data['employment_years'],
            loan_purpose=data['loan_purpose'],
            debt_to_income=data['debt_to_income'],
            existing_loans=data['existing_loans'],
            existing_emis=data['existing_emis'],
            collateral=data['collateral'],
            revolving_utilization=data['revolving_utilization'],
            delinquencies=data['delinquencies'],
            underwriter_notes=data['underwriter_notes'],
            source=data['source'],
            description=res['description'],
            approved=res['approved'],
            decision_status=res['decision_status'],
            confidence=res['confidence'],
            prob_approve=res['prob_approve'],
            prob_reject=res['prob_reject'],
            risk_tier=res['risk_tier'],
            recommended_apr=res['recommended_apr'],
            max_approved_amount=res['max_approved_amount'],
            default_prob=res['default_prob'],
            credit_health_index=res['credit_health_index'],
            monthly_emi=emi_calc['emi'],
            kyc_status='Verified' if data['credit_score'] >= 650 else 'Pending Review',
            reasons=json.dumps(res['reasons']),
            flags=json.dumps(res['flags']),
            factor_breakdown=json.dumps(res['factor_breakdown']),
            token_saliency=json.dumps(res['token_saliency']),
            recommendations_json=json.dumps(recs),
            model_used=res['model_used']
        )
        db.session.add(loan)
    db.session.commit()


# ── Web Application Routes ───────────────────────────────────────────────────

@app.route('/')
@app.route('/api/index')
@app.route('/api/index/')
def index():
    """Landing Page with Hero, Features, Quick Calculator & 'Chat with AI Loan Advisor' CTA."""
    seed_demo_data()
    return render_template('landing.html')


@app.route('/console')
def console():
    """Enterprise Loan Management Software Console."""
    seed_demo_data()
    loans = LoanApplication.query.order_by(LoanApplication.created_at.desc()).all()
    total = len(loans)
    approved = sum(1 for l in loans if l.approved)
    rejected = sum(1 for l in loans if l.decision_status == 'Denied')
    under_review = total - approved - rejected
    
    total_disbursed = sum(l.loan_amount for l in loans if l.approved) or 1245000
    total_repaid = int(total_disbursed * 0.36) or 450000
    active_count = len(loans) or 158

    stats = {
        'total': total,
        'approved': approved,
        'under_review': under_review,
        'rejected': rejected,
        'total_disbursed': total_disbursed,
        'total_repaid': total_repaid,
        'active_loans': active_count,
        'npa_pct': 2.4,
        'approval_rate': round(approved / total * 100, 1) if total else 85.0
    }

    upcoming = [
        {'borrower': 'Eleanor Vance', 'due_date': '28 Oct 2026', 'amount': '$845.00', 'status': 'Paid', 'cls': 'status-paid'},
        {'borrower': 'Marcus Thorne', 'due_date': '30 Oct 2026', 'amount': '$780.00', 'status': 'Due Soon', 'cls': 'status-due'},
        {'borrower': 'Sophia Chen', 'due_date': '02 Nov 2026', 'amount': '$1,120.00', 'status': 'Scheduled', 'cls': 'status-sched'},
        {'borrower': 'Daniel Fletcher', 'due_date': '04 Nov 2026', 'amount': '$940.00', 'status': 'Scheduled', 'cls': 'status-sched'},
    ]

    overdue = [
        {'borrower': 'Alexander Wayne', 'days': '45 Days', 'amount': '$1,450', 'risk': 'High Risk'},
        {'borrower': 'Jessica Green', 'days': '21 Days', 'amount': '$890', 'risk': 'Medium Risk'},
        {'borrower': 'Liam Colman', 'days': '11 Days', 'amount': '$620', 'risk': 'Moderate'},
    ]

    return render_template('console.html', loans=loans, stats=stats, upcoming=upcoming, overdue=overdue)


@app.route('/admin')
def admin_panel():
    """Admin Portal with Lead Submissions, Analytics Charts & CSV Export."""
    seed_demo_data()
    loans = LoanApplication.query.order_by(LoanApplication.created_at.desc()).all()
    
    total = len(loans)
    approved = sum(1 for l in loans if l.approved)
    total_volume = sum(l.loan_amount for l in loans)
    avg_score = round(sum(l.credit_score for l in loans) / total, 0) if total else 700
    avg_foir = round(sum(l.debt_to_income for l in loans) / total * 100, 1) if total else 30.0

    stats = {
        'total_leads': total,
        'total_volume': total_volume,
        'approved_leads': approved,
        'approval_rate': round(approved / total * 100, 1) if total else 80.0,
        'avg_credit_score': int(avg_score),
        'avg_foir': avg_foir
    }

    return render_template('admin.html', loans=loans, stats=stats)


# ── Conversational AI Loan Advisor API ───────────────────────────────────────

@app.route('/api/advisor/chat', methods=['POST'])
def advisor_chat():
    """
    Conversational state machine:
    Receives user message and current collected answers, validates/extracts,
    and returns next question or final recommendation + eligibility calculation.
    """
    data = request.get_json() or {}
    answers = data.get('answers', {})
    user_message = data.get('message', '').strip()
    step_index = int(data.get('step_index', 0))

    if step_index < len(QUESTIONS_FLOW) and user_message:
        current_step = QUESTIONS_FLOW[step_index]
        extracted_val = extract_value_from_text(current_step['key'], user_message)
        answers[current_step['key']] = extracted_val
        step_index += 1

    # Check if we reached the final step
    if step_index >= len(QUESTIONS_FLOW):
        # All questions answered -> Compute final calculation, recommendations, and save lead
        name = answers.get('full_name', 'Applicant')
        loan_type = answers.get('loan_type', 'Personal')
        age = int(answers.get('age', 30))
        city = answers.get('city', 'Austin, TX')
        monthly_income = float(answers.get('monthly_income', 6000.0))
        existing_emis = float(answers.get('existing_emis', 0.0))
        credit_score = int(answers.get('credit_score', 700))
        loan_amount = float(answers.get('loan_amount_requested', 35000.0))
        tenure = int(answers.get('tenure_months', 48))
        purpose = answers.get('loan_purpose', 'Personal')

        annual_income = int(monthly_income * 12)
        dti = round((existing_emis / monthly_income), 2) if monthly_income > 0 else 0.40

        pred_payload = {
            'applicant_name': name,
            'age': age,
            'annual_income': annual_income,
            'credit_score': credit_score,
            'loan_amount': int(loan_amount),
            'loan_term_months': tenure,
            'employment_status': 'Employed',
            'employment_years': 4.0,
            'loan_purpose': purpose,
            'debt_to_income': dti,
            'existing_loans': 1 if existing_emis > 0 else 0,
            'collateral': 'Property' if loan_type in ['Home', 'Mortgage'] else 'None',
            'revolving_utilization': 0.25,
            'delinquencies': 0,
            'underwriter_notes': f'AI Advisor submission from {city}. Purpose: {purpose}.'
        }

        res = predict(pred_payload)
        rate = res['recommended_apr']
        emi_data = calculate_reducing_emi(loan_amount, rate, tenure)
        recs = generate_recommendations(loan_amount, tenure, monthly_income, existing_emis, credit_score, loan_type)
        amortization = generate_amortization_schedule(loan_amount, rate, tenure, max_rows=12)

        # Save to database as a Lead
        new_lead = LoanApplication(
            applicant_name=name,
            age=age,
            city=city,
            annual_income=annual_income,
            monthly_salary=monthly_income,
            credit_score=credit_score,
            loan_amount=int(loan_amount),
            loan_term_months=tenure,
            employment_status='Employed',
            employment_years=4.0,
            loan_purpose=purpose,
            debt_to_income=dti,
            existing_loans=1 if existing_emis > 0 else 0,
            existing_emis=existing_emis,
            collateral='Property' if loan_type in ['Home', 'Mortgage'] else 'None',
            revolving_utilization=0.25,
            delinquencies=0,
            underwriter_notes=f'AI Advisor submission from {city}. Purpose: {purpose}.',
            source='AI Advisor',
            description=res['description'],
            approved=res['approved'],
            decision_status=res['decision_status'],
            confidence=res['confidence'],
            prob_approve=res['prob_approve'],
            prob_reject=res['prob_reject'],
            risk_tier=res['risk_tier'],
            recommended_apr=rate,
            max_approved_amount=res['max_approved_amount'],
            default_prob=res['default_prob'],
            credit_health_index=res['credit_health_index'],
            monthly_emi=emi_data['emi'],
            kyc_status='Verified' if credit_score >= 650 else 'Pending Review',
            reasons=json.dumps(res['reasons']),
            flags=json.dumps(res['flags']),
            factor_breakdown=json.dumps(res['factor_breakdown']),
            token_saliency=json.dumps(res['token_saliency']),
            recommendations_json=json.dumps(recs),
            model_used=res['model_used']
        )
        db.session.add(new_lead)
        db.session.commit()

        # Optional local LLM natural summarization
        llm_summary = None
        if os.environ.get('USE_LOCAL_LLM') == '1':
            prompt = f"Summarize loan offer in 2 sentences for {name}: loan amount ${loan_amount:,.0f} at {rate}% APR with monthly EMI ${emi_data['emi']:,.2f}."
            llm_summary = call_local_llm(prompt)

        return jsonify({
            'completed': True,
            'lead_id': new_lead.id,
            'step_index': step_index,
            'answers': answers,
            'bot_message': f"🎉 Excellent! I have prepared your personalized **LoanSure AI** assessment.",
            'llm_summary': llm_summary,
            'calculation': emi_data,
            'eligibility': recs['eligibility'],
            'packages': recs['packages'],
            'tips': recs['financial_health_tips'],
            'amortization_sample': amortization,
            'risk_tier': res['risk_tier'],
            'approved': res['approved']
        })

    # Next question
    next_step = QUESTIONS_FLOW[step_index]
    formatted_q = next_step['question'].format(
        full_name=answers.get('full_name', 'there'),
        loan_type=answers.get('loan_type', 'loan')
    )

    return jsonify({
        'completed': False,
        'step_index': step_index,
        'total_steps': len(QUESTIONS_FLOW),
        'answers': answers,
        'bot_message': formatted_q,
        'options': next_step['options'],
        'hint': next_step['hint'],
        'current_key': next_step['key']
    })


@app.route('/api/calculate', methods=['POST'])
def api_calculate():
    """Quick EMI, Total Interest, and Amortization Schedule calculation endpoint."""
    data = request.get_json() or {}
    principal = float(data.get('principal', 50000))
    rate_pct = float(data.get('rate_pct', 10.5))
    tenure_months = int(data.get('tenure_months', 60))
    
    emi_data = calculate_reducing_emi(principal, rate_pct, tenure_months)
    schedule = generate_amortization_schedule(principal, rate_pct, tenure_months, max_rows=120)
    
    return jsonify({
        'calculation': emi_data,
        'amortization_schedule': schedule
    })


@app.route('/api/admin/analytics')
def admin_analytics():
    """Return JSON analytics for admin dashboard charts."""
    loans = LoanApplication.query.all()
    total = len(loans) or 1
    
    # Loan types
    type_counts = {}
    city_counts = {}
    tier_counts = {}
    
    for l in loans:
        p = l.loan_purpose or 'Personal'
        type_counts[p] = type_counts.get(p, 0) + 1
        
        c = l.city.split(',')[0] if l.city else 'Other'
        city_counts[c] = city_counts.get(c, 0) + 1
        
        t = l.risk_tier.split(' ')[0] if l.risk_tier else 'Tier B'
        tier_counts[t] = tier_counts.get(t, 0) + 1

    return jsonify({
        'total_leads': total,
        'type_distribution': type_counts,
        'city_distribution': city_counts,
        'tier_distribution': tier_counts,
        'approval_rate': round(sum(1 for l in loans if l.approved) / total * 100, 1),
        'avg_loan_amount': round(sum(l.loan_amount for l in loans) / total, 0)
    })


@app.route('/api/admin/export')
def admin_export_csv():
    """Stream CSV download of all leads for CRM ingestion."""
    loans = LoanApplication.query.order_by(LoanApplication.created_at.desc()).all()
    
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow([
        'Lead_ID', 'Date', 'Applicant_Name', 'City', 'Age', 'Monthly_Salary',
        'Credit_Score', 'Loan_Purpose', 'Loan_Amount', 'Tenure_Months',
        'Monthly_EMI', 'Offered_APR', 'Risk_Tier', 'Decision', 'Source'
    ])
    
    for l in loans:
        cw.writerow([
            f"LN-{l.id:04d}",
            l.created_at.strftime('%Y-%m-%d %H:%M') if l.created_at else '',
            l.applicant_name,
            l.city,
            l.age,
            l.monthly_salary,
            l.credit_score,
            l.loan_purpose,
            l.loan_amount,
            l.loan_term_months,
            l.monthly_emi,
            l.recommended_apr,
            l.risk_tier,
            l.decision_status,
            l.source
        ])
    
    output = si.getvalue()
    return Response(
        output,
        mimetype='text/csv',
        headers={"Content-disposition": "attachment; filename=loansure_leads_export.csv"}
    )


# ── Standard Platform Routes ──────────────────────────────────────────────────

@app.route('/apply', methods=['GET', 'POST'])
def apply():
    if request.method == 'POST':
        data = {
            'applicant_name':       request.form.get('applicant_name', 'Anonymous'),
            'age':                  int(request.form.get('age', 32)),
            'annual_income':        int(request.form.get('annual_income', 65000)),
            'credit_score':         int(request.form.get('credit_score', 700)),
            'loan_amount':          int(request.form.get('loan_amount', 45000)),
            'loan_term_months':     int(request.form.get('loan_term_months', 60)),
            'employment_status':    request.form.get('employment_status', 'Employed'),
            'employment_years':     float(request.form.get('employment_years', 4.0)),
            'loan_purpose':         request.form.get('loan_purpose', 'Personal'),
            'debt_to_income':       float(request.form.get('debt_to_income', 0.32)),
            'existing_loans':       int(request.form.get('existing_loans', 0)),
            'collateral':           request.form.get('collateral', 'None'),
            'revolving_utilization':float(request.form.get('revolving_utilization', 0.25)),
            'delinquencies':        int(request.form.get('delinquencies', 0)),
            'underwriter_notes':    request.form.get('underwriter_notes', ''),
        }

        result = predict(data)
        emi_calc = calculate_reducing_emi(data['loan_amount'], result['recommended_apr'], data['loan_term_months'])
        monthly_sal = data['annual_income'] / 12.0
        recs = generate_recommendations(
            data['loan_amount'], data['loan_term_months'],
            monthly_sal, data['debt_to_income'] * monthly_sal,
            data['credit_score'], data['loan_purpose']
        )

        loan = LoanApplication(
            applicant_name        = data['applicant_name'],
            age                   = data['age'],
            city                  = 'Dallas, TX',
            annual_income         = data['annual_income'],
            monthly_salary        = round(monthly_sal, 2),
            credit_score          = data['credit_score'],
            loan_amount           = data['loan_amount'],
            loan_term_months      = data['loan_term_months'],
            employment_status     = data['employment_status'],
            employment_years      = data['employment_years'],
            loan_purpose          = data['loan_purpose'],
            debt_to_income        = data['debt_to_income'],
            existing_loans        = data['existing_loans'],
            existing_emis         = round(data['debt_to_income'] * monthly_sal, 2),
            collateral            = data['collateral'],
            revolving_utilization = data['revolving_utilization'],
            delinquencies         = data['delinquencies'],
            underwriter_notes     = data['underwriter_notes'],
            source                = 'Manual Apply',
            description           = result['description'],
            approved              = result['approved'],
            decision_status       = result['decision_status'],
            confidence            = result['confidence'],
            prob_approve          = result['prob_approve'],
            prob_reject           = result['prob_reject'],
            risk_tier             = result['risk_tier'],
            recommended_apr       = result['recommended_apr'],
            max_approved_amount   = result['max_approved_amount'],
            default_prob          = result['default_prob'],
            credit_health_index   = result['credit_health_index'],
            monthly_emi           = emi_calc['emi'],
            kyc_status            = 'Verified' if data['credit_score'] >= 650 else 'Pending Review',
            reasons               = json.dumps(result['reasons']),
            flags                 = json.dumps(result['flags']),
            factor_breakdown      = json.dumps(result['factor_breakdown']),
            token_saliency        = json.dumps(result['token_saliency']),
            recommendations_json  = json.dumps(recs),
            model_used            = result['model_used']
        )
        db.session.add(loan)
        db.session.commit()
        return redirect(url_for('result', loan_id=loan.id))

    return render_template('apply.html')


@app.route('/result/<int:loan_id>')
def result(loan_id):
    loan = LoanApplication.query.get_or_404(loan_id)
    return render_template('result.html', loan=loan)


@app.route('/dashboard')
def dashboard():
    seed_demo_data()
    loans = LoanApplication.query.order_by(LoanApplication.created_at.desc()).all()
    total = len(loans)
    approved = sum(1 for l in loans if l.approved)
    stats = {
        'total': total,
        'approved': approved,
        'rejected': total - approved,
        'approval_rate': round(approved / total * 100, 1) if total else 0,
        'total_volume': sum(l.loan_amount for l in loans if l.approved),
        'avg_score': int(sum(l.credit_score for l in loans) / total) if total else 0,
        'avg_dti': round(sum(l.debt_to_income for l in loans) / total, 2) if total else 0,
    }
    return render_template('dashboard.html', loans=loans, stats=stats)


@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        try:
            df = pd.read_csv(file)
            results = []
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                pred = predict(row_dict)
                results.append({
                    'name': row_dict.get('applicant_name', 'Unknown'),
                    'amount': row_dict.get('loan_amount', 0),
                    'score': row_dict.get('credit_score', 650),
                    'dti': row_dict.get('debt_to_income', 0.35),
                    'status': pred['decision_status'],
                    'prob': pred['prob_approve'],
                    'tier': pred['risk_tier'],
                    'apr': pred['recommended_apr']
                })
            return render_template('batch.html', results=results, total=len(results))
        except Exception as e:
            return render_template('batch.html', error=str(e))

    return render_template('batch.html')


@app.route('/bias-audit')
def bias_audit():
    loans = LoanApplication.query.all()
    return render_template('bias_audit.html', total_loans=len(loans))


@app.route('/model-info')
def model_info():
    meta_path = 'models/bert_loan_model/meta.json'
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return render_template('model_info.html', meta=meta)


@app.route('/memo/<int:loan_id>')
def credit_memo(loan_id):
    loan = LoanApplication.query.get_or_404(loan_id)
    return render_template('memo.html', loan=loan)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON payload provided'}), 400
    res = predict(data)
    return jsonify(res)


@app.route('/api/what-if', methods=['POST'])
def api_what_if():
    data = request.get_json() or {}
    base_data = data.get('base', {})
    deltas = data.get('deltas', {})
    sim_res = counterfactual_simulation(base_data, deltas)
    return jsonify(sim_res)


@app.route('/api/sample-csv')
def sample_csv():
    csv_content = """applicant_name,age,annual_income,credit_score,loan_amount,loan_term_months,employment_status,employment_years,loan_purpose,debt_to_income,existing_loans,collateral,revolving_utilization,delinquencies,underwriter_notes
Eleanor Vance,34,115000,785,45000,60,Employed,6.5,Home Improvement,0.18,0,Property,0.12,0,Exemplary liquid reserves with consistent salary progression.
Marcus Thorne,29,85000,675,35000,48,Employed,3.5,Debt Consolidation,0.38,2,None,0.45,0,Software engineering position solid repayment history.
David Miller,24,42000,580,18000,36,Freelancer,1.0,Personal,0.58,3,None,0.82,1,Variable freelance earnings with high credit card utilization."""
    return Response(
        csv_content.strip(),
        mimetype='text/csv',
        headers={"Content-disposition": "attachment; filename=sample_loan_batch.csv"}
    )


with app.app_context():
    db.create_all()
    seed_demo_data()


if __name__ == '__main__':
    app.run(debug=True, port=5000)
