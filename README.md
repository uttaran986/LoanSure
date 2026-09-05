# 🏦 LoanSure — Enterprise Loan Management & NLP Underwriting Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![Vercel](https://img.shields.io/badge/Deployed-Vercel-black.svg)](https://vercel.com)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **LoanSure** is a production-grade **Loan Management Software & AI Underwriting Platform**. It unifies automated NLP loan decisioning, EMI collection schedules, real-time KYC stage-gate workflows, borrower risk speedometer analytics, fair lending bias audits, and instant credit memo exports.

---

## 🌟 Key Modules & Features

### 1. 💼 Enterprise Loan Management Console (`/console`)
- **Executive KPIs**: Real-time tracking of Total Disbursed ($1.25M+), Total Repaid, Active Portfolios (158 loans), and NPA / Default Rate (< 2.5%).
- **Interactive Loan Registry**: Searchable & filterable applications table with live credit scores, risk tiers, monthly EMI calculations, and automated approval decisions.
- **Payment Approvals & Settlement Ledger**: Incoming EMI transactions, instant reconciliation, and pre-auth settlement queues.
- **KYC & Stage-Gate Progression**: Multi-stage loan pipeline (*Ingestion → FinBERT Scoring → Biometric KYC & Anti-Fraud → Disbursement*).
- **Borrower Risk Speedometer**: Semicircle dynamic gauge (Prime 720+, Moderate 640-719, Subprime <640) with real-time profile metrics.
- **EMI Calendar & Overdue NPA Alerts**: Monthly repayment collection schedule with 1-click overdue reminders.

### 2. 🤖 Multi-Task AI Underwriting Engine (`/apply`)
- **Simultaneous Predictions**:
  - Binary Approval Decision (`Approved` vs `Denied`)
  - Continuous Calibrated Confidence (%)
  - Risk Tier Classification (`Tier A+ Prime` through `Tier D Subprime`)
  - Risk-Based Dynamic APR (`5.5%` to `24.0%`)
  - Maximum Approved Credit Line
  - Probability of Default (PD %)
- **Live Counterfactual "What-If" Simulator**: Sliders to test credit score increases, income jumps, or debt reductions to view instant approval probability deltas.

### 3. 🔍 Explainable AI (XAI) & Word Saliency
- Token-level attention & keyword attribution highlighting positive credit signals (e.g. *promotions, liquid reserves, low DTI*) vs adverse risk flags.
- Comprehensive 7-factor quantitative breakdown (Credit, DTI, Employment, Collateral, Utilization, Delinquencies, Loan-to-Income).

### 4. ⚖️ Fair Lending & ECOA Compliance Audit (`/bias-audit`)
- Real-time demographic parity and disparate impact ratio evaluations ensuring compliance with the **Equal Credit Opportunity Act (ECOA)** and **FCRA**.

### 5. ⚡ Batch CSV Underwriting (`/batch`)
- Drag-and-drop CSV batch processor to underwrite thousands of applications in seconds with downloadable decision results.

### 6. 📄 Underwriting Credit Memo Export (`/memo/<id>`)
- Institutional committee memorandum with printable layout, audit logs, model governance details, and e-signature blocks.

---

## 🚀 Deployment on Vercel

LoanSure includes full Vercel Serverless Function configuration (`vercel.json` and `api/index.py`):

### Option A: 1-Click via Vercel CLI
```bash
npm install -g vercel
vercel
```

### Option B: GitHub Integration
1. Push this repository to GitHub (`main` branch).
2. Go to [Vercel Dashboard](https://vercel.com/dashboard) and click **"Add New Project"**.
3. Select your `loan_bert` / `LoanSure` repository.
4. Click **Deploy**. Vercel will automatically detect `vercel.json` and build the Python serverless runtime.

---

## 💻 Local Development Setup

```bash
# 1. Clone repository
git clone https://github.com/uttaran986/loan_bert.git
cd loan_bert

# 2. Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run test suite
python test_suite.py

# 5. Launch application
python app.py
```
Open **`http://localhost:5000`** in your browser.

---

## 🛡️ Architecture & Technology Stack

- **Backend**: Python 3.10+, Flask 3.0, Flask-SQLAlchemy, SQLite (with Serverless `/tmp` ephemeral support on Vercel).
- **ML / NLP Engine**: Scikit-Learn, Joblib, BERT-aligned token saliency scoring, multi-task decision trees & linear calibrators.
- **Frontend**: Responsive modern CSS3 design system, Lucide Icons, Chart.js.

---

## 📄 License
MIT License &copy; 2026 LoanSure Enterprise Financial Technologies.
