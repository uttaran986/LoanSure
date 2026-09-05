# 🤖 LoanSure AI — Intelligent Conversational Loan Advisor & Enterprise Underwriting

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![Local LLM](https://img.shields.io/badge/AI-Ollama%20%7C%20Llama%203-purple.svg)](https://ollama.ai/)
[![Vercel](https://img.shields.io/badge/Deployed-Vercel-black.svg)](https://vercel.com)
[![License](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)

> **LoanSure AI** is a production-ready **Conversational AI Loan Advisor, Financial Underwriting Engine, and Analytics Platform**. It combines step-by-step natural language borrower dialogue, dynamic FOIR eligibility calculations, reducing-balance EMI estimations, personalized loan recommendations, and an executive administration dashboard.

---

## 🌟 Core Capabilities & Architecture

### 1. 💬 Conversational AI Loan Advisor (`/`)
- **Step-by-Step Interactive Chat**: Guides borrowers one-by-one through:
  1. Loan Type (*Personal, Home, Auto, Business, Education*)
  2. Full Legal Name & Age
  3. City & State of Residence
  4. Monthly In-Hand Salary / Income
  5. Active Monthly EMIs
  6. Estimated Credit Score Tier
  7. Desired Loan Amount & Preferred Tenure
  8. Primary Purpose of Loan
- **Local LLM Integration**: Connects seamlessly to local **Ollama** (`llama3`, `qwen2.5`, `mistral`) with an intelligent, zero-latency local NLP rule fallback for 100% offline uptime.
- **Instant Pre-Approval Offer Cards**: Delivers 3 customized loan packages (*Optimal Balanced, Lower Monthly Burden, Fast Payoff Plan*) + credit health tips.

### 2. 🧮 Financial Calculation & Amortization Engine
- **Exact Reducing Balance EMI**:
  $$\text{EMI} = P \times r \times \frac{(1 + r)^n}{(1 + r)^n - 1}$$
- **FOIR / DTI Maximum Sizing**: Evaluates disposable income and credit risk tiers to calculate safe credit limits.
- **Full Repayment Schedule**: Month-by-month amortization table breaking down opening balance, principal paid, interest paid, and closing balance.

### 3. 📊 Admin Portal & Analytics (`/admin`)
- **Real-Time Submissions Ledger**: Searchable & filterable table of all leads ingested via AI Advisor or Manual Underwriting.
- **Interactive Visualizations**:
  - Loan Category Distribution (Doughnut Chart)
  - Top Metropolitan Areas (Bar Chart)
  - Risk Tier & Pipeline Volume Tracking
- **1-Click CSV Export**: Stream downloadable CSV data formatted for direct CRM ingestion.

### 4. 🎨 Modern Dark / Light Mode Experience
- Smooth CSS variables and theme toggle with `localStorage` state persistence.
- High-converting landing page with interactive sliders, loan product cards, trust badges, and mobile-responsive layout.

---

## 🚀 Quickstart & Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/uttaran986/LoanSure.git
cd LoanSure

# 2. Set up virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated test suite
python test_suite.py

# 5. Launch the application
python app.py
```
Open **`http://localhost:5000`** in your browser.

---

## 🤖 (Optional) Connecting Local LLM with Ollama

LoanSure AI works 100% out of the box with its built-in NLP engine. To optionally enable local Ollama LLM generation:
```bash
# 1. Start Ollama with Llama 3
ollama run llama3

# 2. Set environment variable in your terminal
set USE_LOCAL_LLM=1
python app.py
```

---

## ☁️ Deploying to Vercel

1. Push to your GitHub repository (`main` branch).
2. Go to [Vercel Dashboard](https://vercel.com/dashboard) and click **"Add New Project"**.
3. Select your **`LoanSure`** repository.
4. Click **Deploy**. Vercel will automatically build the Python runtime!

---

## 📄 License
MIT License &copy; 2026 LoanSure AI Financial Technologies.
