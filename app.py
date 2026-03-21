"""
app.py — Flask application for BERT Loan Approval System
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bert-loan-secret-2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///loans.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


# ── Model ────────────────────────────────────────────────────────────────────
class LoanApplication(db.Model):
    id                = db.Column(db.Integer, primary_key=True)
    applicant_name    = db.Column(db.String(200))
    age               = db.Column(db.Integer)
    annual_income     = db.Column(db.Integer)
    credit_score      = db.Column(db.Integer)
    loan_amount       = db.Column(db.Integer)
    loan_term_months  = db.Column(db.Integer)
    employment_status = db.Column(db.String(50))
    loan_purpose      = db.Column(db.String(100))
    debt_to_income    = db.Column(db.Float)
    existing_loans    = db.Column(db.Integer)
    collateral        = db.Column(db.String(50))
    description       = db.Column(db.Text)
    approved          = db.Column(db.Boolean)
    confidence        = db.Column(db.Float)
    prob_approve      = db.Column(db.Float)
    prob_reject       = db.Column(db.Float)
    reasons           = db.Column(db.Text)   # JSON
    flags             = db.Column(db.Text)   # JSON
    model_used        = db.Column(db.String(200))
    created_at        = db.Column(db.DateTime, default=datetime.utcnow)

    def get_reasons(self):
        try: return json.loads(self.reasons) if self.reasons else []
        except: return []

    def get_flags(self):
        try: return json.loads(self.flags) if self.flags else []
        except: return []


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return redirect(url_for('apply'))


@app.route('/apply', methods=['GET', 'POST'])
def apply():
    if request.method == 'POST':
        from predictor import predict

        data = {
            'age':               request.form.get('age', 30),
            'annual_income':     request.form.get('annual_income', 50000),
            'credit_score':      request.form.get('credit_score', 650),
            'loan_amount':       request.form.get('loan_amount', 50000),
            'loan_term_months':  request.form.get('loan_term_months', 60),
            'employment_status': request.form.get('employment_status', 'Employed'),
            'loan_purpose':      request.form.get('loan_purpose', 'Personal'),
            'debt_to_income':    request.form.get('debt_to_income', 0.4),
            'existing_loans':    request.form.get('existing_loans', 0),
            'collateral':        request.form.get('collateral', 'None'),
        }

        result = predict(data)

        # Save to DB
        loan = LoanApplication(
            applicant_name    = request.form.get('applicant_name', 'Anonymous'),
            age               = int(data['age']),
            annual_income     = int(data['annual_income']),
            credit_score      = int(data['credit_score']),
            loan_amount       = int(data['loan_amount']),
            loan_term_months  = int(data['loan_term_months']),
            employment_status = data['employment_status'],
            loan_purpose      = data['loan_purpose'],
            debt_to_income    = float(data['debt_to_income']),
            existing_loans    = int(data['existing_loans']),
            collateral        = data['collateral'],
            description       = result.get('description', ''),
            approved          = result['approved'],
            confidence        = result['confidence'],
            prob_approve      = result.get('prob_approve', result['confidence'] if result['approved'] else 100 - result['confidence']),
            prob_reject       = result.get('prob_reject',  100 - result['confidence'] if result['approved'] else result['confidence']),
            reasons           = json.dumps(result.get('reasons', [])),
            flags             = json.dumps(result.get('flags', [])),
            model_used        = result.get('model_used', 'Unknown'),
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
    loans = LoanApplication.query.order_by(LoanApplication.created_at.desc()).all()
    total     = len(loans)
    approved  = sum(1 for l in loans if l.approved)
    rejected  = total - approved
    avg_score = round(sum(l.credit_score for l in loans) / total, 0) if total else 0
    avg_conf  = round(sum(l.confidence for l in loans) / total, 1) if total else 0
    stats = {
        'total': total, 'approved': approved, 'rejected': rejected,
        'approval_rate': round(approved / total * 100, 1) if total else 0,
        'avg_credit_score': avg_score, 'avg_confidence': avg_conf,
    }
    return render_template('dashboard.html', loans=loans, stats=stats)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    from predictor import predict
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    result = predict(data)
    return jsonify(result)


@app.route('/api/stats')
def api_stats():
    loans = LoanApplication.query.all()
    total = len(loans)
    if total == 0:
        return jsonify({'total': 0})
    approved = sum(1 for l in loans if l.approved)
    purposes = {}
    for l in loans:
        purposes[l.loan_purpose] = purposes.get(l.loan_purpose, 0) + 1
    return jsonify({
        'total': total,
        'approved': approved,
        'rejected': total - approved,
        'approval_rate': round(approved / total * 100, 1),
        'by_purpose': purposes,
    })


@app.route('/model-info')
def model_info():
    import os, json
    meta_path = 'models/bert_loan_model/meta.json'
    meta = None
    has_model = os.path.exists(meta_path)
    if has_model:
        with open(meta_path) as f:
            meta = json.load(f)
    plots = {
        'training_curves':  os.path.exists('models/bert_loan_model/training_curves.png'),
        'confusion_matrix': os.path.exists('models/bert_loan_model/confusion_matrix.png'),
    }
    return render_template('model_info.html', meta=meta, has_model=has_model, plots=plots)


@app.route('/model-plot/<name>')
def model_plot(name):
    from flask import send_file
    allowed = ['training_curves', 'confusion_matrix']
    if name not in allowed:
        return 'Not found', 404
    path = f'models/bert_loan_model/{name}.png'
    if not os.path.exists(path):
        return 'Plot not found', 404
    return send_file(path, mimetype='image/png')


# ── Init ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('models/bert_loan_model', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)
