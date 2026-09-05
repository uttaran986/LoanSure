# advisor_engine.py — Conversational AI Loan Advisor, Financial Calculations & Recommendations

import os
import json
import re
import urllib.request
import urllib.error

# ── 1. Financial Calculation Engine ──────────────────────────────────────────

def calculate_reducing_emi(principal: float, annual_rate_pct: float, tenure_months: int):
    """Calculate accurate reducing balance EMI, total interest, and total payable."""
    if principal <= 0 or tenure_months <= 0:
        return {'emi': 0.0, 'total_interest': 0.0, 'total_payable': 0.0, 'monthly_rate_pct': 0.0}
    
    monthly_rate = (annual_rate_pct / 100.0) / 12.0
    if monthly_rate <= 0:
        emi = principal / tenure_months
        return {'emi': round(emi, 2), 'total_interest': 0.0, 'total_payable': round(principal, 2), 'monthly_rate_pct': 0.0}
    
    factor = (1.0 + monthly_rate) ** tenure_months
    emi = (principal * monthly_rate * factor) / (factor - 1.0)
    total_payable = emi * tenure_months
    total_interest = total_payable - principal
    
    return {
        'emi': round(emi, 2),
        'total_interest': round(total_interest, 2),
        'total_payable': round(total_payable, 2),
        'monthly_rate_pct': round(monthly_rate * 100, 4)
    }


def generate_amortization_schedule(principal: float, annual_rate_pct: float, tenure_months: int, max_rows: int = 60):
    """Generate full month-by-month repayment schedule (amortization table)."""
    if principal <= 0 or tenure_months <= 0:
        return []
    
    monthly_rate = (annual_rate_pct / 100.0) / 12.0
    calc = calculate_reducing_emi(principal, annual_rate_pct, tenure_months)
    emi = calc['emi']
    
    schedule = []
    balance = principal
    
    for month in range(1, tenure_months + 1):
        interest = balance * monthly_rate if monthly_rate > 0 else 0.0
        principal_paid = emi - interest
        if principal_paid > balance or month == tenure_months:
            principal_paid = balance
            emi_actual = principal_paid + interest
        else:
            emi_actual = emi
        
        closing_balance = max(0.0, balance - principal_paid)
        
        schedule.append({
            'month': month,
            'opening_balance': round(balance, 2),
            'emi': round(emi_actual, 2),
            'principal': round(principal_paid, 2),
            'interest': round(interest, 2),
            'closing_balance': round(closing_balance, 2)
        })
        
        balance = closing_balance
        if balance <= 0:
            break
            
    return schedule[:max_rows]


def estimate_eligibility(monthly_salary: float, existing_emis: float, credit_score: int, loan_type: str = 'Personal'):
    """Estimate maximum eligible loan amount and FOIR risk assessment."""
    if credit_score >= 750:
        max_foir = 0.55
        base_rate = 9.5 if loan_type in ['Home', 'Mortgage'] else (10.5 if loan_type in ['Auto', 'Vehicle'] else 11.5)
        risk_grade = 'Prime (Excellent)'
    elif credit_score >= 680:
        max_foir = 0.45
        base_rate = 11.0 if loan_type in ['Home', 'Mortgage'] else (12.5 if loan_type in ['Auto', 'Vehicle'] else 13.5)
        risk_grade = 'Near-Prime (Good)'
    elif credit_score >= 600:
        max_foir = 0.35
        base_rate = 14.0 if loan_type in ['Home', 'Mortgage'] else (15.5 if loan_type in ['Auto', 'Vehicle'] else 16.5)
        risk_grade = 'Moderate Risk'
    else:
        max_foir = 0.25
        base_rate = 18.0
        risk_grade = 'Subprime (High Risk)'

    max_total_emi = monthly_salary * max_foir
    available_emi = max(0.0, max_total_emi - existing_emis)
    current_dti = round((existing_emis / monthly_salary * 100) if monthly_salary > 0 else 100.0, 1)

    default_tenure_months = 240 if loan_type in ['Home', 'Mortgage'] else (60 if loan_type in ['Auto', 'Vehicle', 'Business'] else 48)
    r = (base_rate / 100.0) / 12.0
    
    if r > 0 and available_emi > 0:
        factor = (1.0 + r) ** default_tenure_months
        max_loan_amount = (available_emi * (factor - 1.0)) / (r * factor)
    else:
        max_loan_amount = available_emi * default_tenure_months

    return {
        'max_eligible_amount': round(max_loan_amount, 0),
        'available_monthly_emi': round(available_emi, 2),
        'base_rate_pct': base_rate,
        'risk_grade': risk_grade,
        'current_dti_pct': current_dti,
        'max_allowed_foir_pct': round(max_foir * 100, 1),
        'is_eligible': available_emi > 50 and credit_score >= 580
    }


def generate_recommendations(requested_amount: float, requested_tenure: int, monthly_salary: float, existing_emis: float, credit_score: int, loan_type: str = 'Personal'):
    """Generate 3 tailored loan package recommendations (Optimal, Low Monthly, Fast Payoff)."""
    elig = estimate_eligibility(monthly_salary, existing_emis, credit_score, loan_type)
    rate = elig['base_rate_pct']
    
    opt_tenure = requested_tenure if requested_tenure > 0 else (60 if loan_type in ['Home', 'Business'] else 36)
    opt_calc = calculate_reducing_emi(requested_amount, rate, opt_tenure)
    
    low_tenure = min(84 if loan_type != 'Home' else 360, int(opt_tenure * 1.5))
    low_calc = calculate_reducing_emi(requested_amount, rate + 0.25, low_tenure)
    
    fast_tenure = max(12, int(opt_tenure * 0.65))
    fast_calc = calculate_reducing_emi(requested_amount, max(7.5, rate - 0.5), fast_tenure)
    
    tips = []
    if elig['current_dti_pct'] > 40:
        tips.append('Paying off smaller active credit card balances could boost your maximum loan amount by up to 25%.')
    if credit_score < 720:
        tips.append('Maintaining on-time payments for 3-6 months can lower your offered APR by 1.5% to 3.0%.')
    else:
        tips.append('Your prime credit score qualifies you for instant pre-approved processing and zero documentation fees.')

    return {
        'eligibility': elig,
        'packages': [
            {
                'name': 'Optimal Balanced Choice',
                'tag': 'Recommended',
                'amount': requested_amount,
                'tenure_months': opt_tenure,
                'rate_pct': rate,
                'monthly_emi': opt_calc['emi'],
                'total_interest': opt_calc['total_interest'],
                'total_payable': opt_calc['total_payable'],
                'highlight': 'Best balance of monthly cash flow and minimal overall interest.'
            },
            {
                'name': 'Lower Monthly Burden',
                'tag': 'Budget Friendly',
                'amount': requested_amount,
                'tenure_months': low_tenure,
                'rate_pct': round(rate + 0.25, 2),
                'monthly_emi': low_calc['emi'],
                'total_interest': low_calc['total_interest'],
                'total_payable': low_calc['total_payable'],
                'highlight': f'Reduces your monthly commitment to only ${low_calc["emi"]:,.2f}/mo.'
            },
            {
                'name': 'Fast Payoff Plan',
                'tag': 'Max Savings',
                'amount': requested_amount,
                'tenure_months': fast_tenure,
                'rate_pct': round(max(7.5, rate - 0.5), 2),
                'monthly_emi': fast_calc['emi'],
                'total_interest': fast_calc['total_interest'],
                'total_payable': fast_calc['total_payable'],
                'highlight': f'Saves ${round(opt_calc["total_interest"] - fast_calc["total_interest"], 2):,.2f} in lifetime interest.'
            }
        ],
        'financial_health_tips': tips
    }


# ── 2. Conversational AI Step Machine & LLM Integration ───────────────────────

QUESTIONS_FLOW = [
    {
        'key': 'loan_type',
        'question': 'Hello! I am your **LoanSure AI Advisor**. Which type of loan are you looking for today?',
        'options': ['Personal Loan', 'Home Loan', 'Auto / Vehicle', 'Business Loan', 'Education Loan'],
        'hint': 'Select an option or type your loan category.'
    },
    {
        'key': 'full_name',
        'question': 'Great! What is your full legal name?',
        'options': [],
        'hint': 'e.g. Eleanor Vance'
    },
    {
        'key': 'age',
        'question': 'Nice to meet you, {full_name}! What is your age?',
        'options': ['21 - 30', '31 - 45', '46 - 60', '60+'],
        'hint': 'e.g. 32'
    },
    {
        'key': 'city',
        'question': 'Which city and state do you currently reside in?',
        'options': ['New York, NY', 'San Francisco, CA', 'Austin, TX', 'Chicago, IL', 'Seattle, WA'],
        'hint': 'e.g. Austin, TX'
    },
    {
        'key': 'monthly_income',
        'question': 'What is your average monthly in-hand income or salary ($)?',
        'options': ['$3,500/mo', '$6,500/mo', '$10,000/mo', '$15,000+/mo'],
        'hint': 'e.g. 6500'
    },
    {
        'key': 'existing_emis',
        'question': 'How much do you currently pay each month towards active EMIs or loans ($)?',
        'options': ['$0 (No active loans)', '$250/mo', '$500/mo', '$1,200/mo'],
        'hint': 'e.g. 0'
    },
    {
        'key': 'credit_score',
        'question': 'What is your estimated credit score range?',
        'options': ['Excellent (750+)', 'Good (700 - 749)', 'Fair (640 - 699)', 'Needs Work (< 640)'],
        'hint': 'e.g. 740'
    },
    {
        'key': 'loan_amount_requested',
        'question': 'How much loan amount are you looking to borrow ($)?',
        'options': ['$15,000', '$35,000', '$50,000', '$100,000'],
        'hint': 'e.g. 45000'
    },
    {
        'key': 'tenure_months',
        'question': 'What is your preferred repayment tenure?',
        'options': ['12 Months (1 Yr)', '24 Months (2 Yrs)', '36 Months (3 Yrs)', '60 Months (5 Yrs)', '84 Months (7 Yrs)'],
        'hint': 'e.g. 36'
    },
    {
        'key': 'loan_purpose',
        'question': 'Lastly, what is the primary purpose of this loan?',
        'options': ['Debt Consolidation', 'Home Renovation', 'Business Expansion', 'Medical / Emergency', 'Major Purchase'],
        'hint': 'e.g. Debt Consolidation'
    }
]


def extract_value_from_text(step_key: str, text: str):
    """Extract numerical or categorical values cleanly from natural language strings."""
    clean = text.strip()
    if step_key == 'age':
        m = re.search(r'\b(\d{2})\b', clean)
        return int(m.group(1)) if m else 30
    
    if step_key in ['monthly_income', 'existing_emis', 'loan_amount_requested']:
        nums = re.findall(r'[\d,]+', clean.replace('$', '').replace('/mo', ''))
        if nums:
            raw = nums[0].replace(',', '')
            try: return float(raw)
            except: pass
        return 6500.0 if step_key == 'monthly_income' else (0.0 if step_key == 'existing_emis' else 35000.0)
    
    if step_key == 'credit_score':
        m = re.search(r'\b(3\d{2}|[4-8]\d{2})\b', clean)
        if m: return int(m.group(1))
        if 'excellent' in clean.lower() or '750' in clean: return 780
        if 'good' in clean.lower() or '700' in clean: return 720
        if 'fair' in clean.lower() or '640' in clean: return 670
        return 620
    
    if step_key == 'tenure_months':
        m = re.search(r'\b(\d+)\s*(?:months?|mos?|years?|yrs?)\b', clean, re.I)
        if m:
            val = int(m.group(1))
            if 'year' in clean.lower() or 'yr' in clean.lower(): val *= 12
            return val
        nums = re.findall(r'\b\d+\b', clean)
        if nums:
            val = int(nums[0])
            return val * 12 if val <= 10 else val
        return 36

    return clean


def call_local_llm(prompt: str, model: str = 'llama3', timeout_sec: float = 3.0):
    """Attempt to call local Ollama server if running on http://localhost:11434."""
    url = 'http://localhost:11434/api/generate'
    payload = json.dumps({'model': model, 'prompt': prompt, 'stream': False}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            return res_data.get('response', '')
    except Exception:
        return None
