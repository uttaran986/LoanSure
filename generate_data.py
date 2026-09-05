"""
generate_data.py
Generates a realistic, comprehensive synthetic loan dataset with multi-task targets:
- Approval Status (Approved, Conditional, Rejected)
- Credit Risk Tier (Tier A+ Exceptional, Tier A Prime, Tier B Near-Prime, Tier C Subprime, Tier D High Risk)
- Recommended APR (%)
- Default Probability (PD %)
- Multi-template natural language descriptions & underwriter notes.

Usage: python generate_data.py
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)
N = 10000

def generate_loan_dataset(n=N):
    # Core Demographics & Employment
    ages = np.random.randint(20, 72, n)
    employment_statuses = np.random.choice(
        ['Employed', 'Self-Employed', 'Business Owner', 'Unemployed', 'Retired', 'Student', 'Freelancer'],
        n, p=[0.48, 0.18, 0.12, 0.05, 0.07, 0.04, 0.06]
    )
    
    # Employment tenure in years (correlated with age)
    emp_years = np.clip((ages - 21) * np.random.uniform(0.1, 0.85, n), 0, 40).round(1)
    emp_years[np.isin(employment_statuses, ['Unemployed', 'Student'])] = 0

    # Income distributions with realistic log-normal spread
    incomes = np.random.lognormal(mean=11.15, sigma=0.55, size=n).astype(int)
    # Adjust income by employment status
    income_multipliers = {
        'Employed': 1.0, 'Self-Employed': 1.1, 'Business Owner': 1.45,
        'Unemployed': 0.25, 'Retired': 0.75, 'Student': 0.35, 'Freelancer': 0.95
    }
    for status, mult in income_multipliers.items():
        incomes[employment_statuses == status] = (incomes[employment_statuses == status] * mult).astype(int)
    incomes = np.clip(incomes, 15000, 450000)

    # Credit score (FICO style 300-850)
    credit_scores = np.random.normal(685, 78, n).astype(int)
    credit_scores = np.clip(credit_scores, 320, 850)

    # Loan characteristics
    loan_amounts = np.random.lognormal(mean=10.6, sigma=0.75, size=n).astype(int)
    loan_amounts = np.clip(loan_amounts, 2500, 500000)

    loan_terms = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240, 360], n,
                                  p=[0.05, 0.08, 0.25, 0.12, 0.28, 0.07, 0.05, 0.04, 0.03, 0.03])

    purposes = np.random.choice(
        ['Home Purchase', 'Debt Consolidation', 'Business Expansion',
         'Education', 'Medical', 'Vehicle Purchase', 'Home Improvement', 
         'Personal', 'Green Energy & Solar', 'Commercial Equipment'],
        n, p=[0.20, 0.22, 0.11, 0.08, 0.06, 0.12, 0.09, 0.05, 0.04, 0.03]
    )

    existing_loans = np.random.choice([0, 1, 2, 3, 4, 5, 6], n, p=[0.30, 0.32, 0.20, 0.10, 0.05, 0.02, 0.01])
    
    # Financial leverage: Debt-to-income and revolving credit utilization
    monthly_debt = (loan_amounts * 0.006 + existing_loans * 450 + np.random.uniform(100, 1200, n))
    monthly_income = np.maximum(incomes / 12, 1000)
    debt_to_income = np.clip(monthly_debt / monthly_income, 0.02, 2.5).round(2)
    
    revolving_util = np.clip(np.random.beta(2, 5, n) * (1.2 - (credit_scores - 300) / 700), 0.02, 0.98).round(2)
    delinquencies = np.random.choice([0, 1, 2, 3, 4], n, p=[0.75, 0.14, 0.06, 0.03, 0.02])
    bankruptcies = np.random.choice([0, 1], n, p=[0.94, 0.06])

    collaterals = np.random.choice(
        ['None', 'Property / Real Estate', 'Vehicle', 'Investment Portfolio', 'Commercial Equipment', 'Cash Reserve'],
        n, p=[0.32, 0.30, 0.14, 0.12, 0.07, 0.05]
    )

    # ── Multi-Factor Underwriting Scoring Engine ──────────────────────────
    score = np.zeros(n, dtype=float)

    # 1. Credit Score Contribution (-40 to +40)
    score += np.where(credit_scores >= 800, 38,
             np.where(credit_scores >= 750, 28,
             np.where(credit_scores >= 700, 18,
             np.where(credit_scores >= 650, 8,
             np.where(credit_scores >= 600, -10,
             np.where(credit_scores >= 550, -25, -45))))))

    # 2. Income & DTI Contribution (-30 to +30)
    score += np.where(incomes >= 120000, 20,
             np.where(incomes >= 75000, 12,
             np.where(incomes >= 45000, 5,
             np.where(incomes >= 25000, -5, -18))))

    score += np.where(debt_to_income <= 0.25, 22,
             np.where(debt_to_income <= 0.40, 14,
             np.where(debt_to_income <= 0.55, 2,
             np.where(debt_to_income <= 0.75, -15, -30))))

    # 3. Employment & Stability (-20 to +20)
    score += np.where(employment_statuses == 'Employed', 15,
             np.where(np.isin(employment_statuses, ['Self-Employed', 'Business Owner']), 10,
             np.where(employment_statuses == 'Freelancer', 4,
             np.where(employment_statuses == 'Retired', 6,
             np.where(employment_statuses == 'Student', -8, -25)))))
    score += np.clip(emp_years * 0.8, 0, 10)

    # 4. Collateral & Existing Debt Obligations (-15 to +15)
    score += np.where(collaterals != 'None', 14, 0)
    score += np.where(existing_loans == 0, 8,
             np.where(existing_loans <= 2, 2,
             np.where(existing_loans <= 4, -8, -18)))

    # 5. Risk penalties (Delinquencies & Bankruptcies)
    score -= delinquencies * 12
    score -= bankruptcies * 35
    score -= np.where(revolving_util > 0.70, 12, np.where(revolving_util > 0.45, 4, 0))

    # Add realistic stochastic financial volatility
    score += np.random.normal(0, 4.5, n)

    # Normalize score scale to 0-100
    credit_health_index = np.clip(score + 40, 1, 99).round(1)

    # Multi-task targets:
    # 1. Approval Decision
    approved = (credit_health_index >= 48.0).astype(int)
    
    # 2. Risk Tiers
    risk_tiers = np.where(credit_health_index >= 80, 'Tier A+ (Prime)',
                 np.where(credit_health_index >= 65, 'Tier A (Prime)',
                 np.where(credit_health_index >= 48, 'Tier B (Near-Prime)',
                 np.where(credit_health_index >= 32, 'Tier C (Subprime)', 'Tier D (High Risk)'))))

    # 3. Recommended APR (Base Fed Rate 5.25% + Risk Spread)
    base_apr = 5.25
    spread = np.clip((100 - credit_health_index) * 0.22, 1.0, 21.0)
    recommended_apr = (base_apr + spread).round(2)

    # 4. Default Probability (PD %)
    default_prob = np.clip(1.0 / (1.0 + np.exp((credit_health_index - 45) / 10.0)), 0.02, 0.94).round(3)

    # 5. Max Approved Credit Line
    max_approved_amount = np.clip(incomes * 0.45 * (credit_health_index / 60.0), 3000, 600000).astype(int)

    # ── Diverse Natural Language Generation (4 Templates) ─────────────────
    descriptions = []
    underwriter_notes_list = []

    positive_notes = [
        "Borrower demonstrates exemplary cash reserves and disciplined credit utilization.",
        "Solid balance sheet with multi-year steady employment and low debt obligations.",
        "Clean repayment history with strong collateral backing.",
        "Verified disposable cash flow comfortably covers debt service coverage ratio.",
        "Strong liquidity profile and prime credit score history."
    ]
    negative_notes = [
        "Elevated debt-to-income ratio presents elevated default risk under rate shocks.",
        "Limited asset collateral and history of short employment duration.",
        "High revolving credit utilization with recent delinquency inquiries on file.",
        "Aggressive loan request relative to stated annual gross income.",
        "Unfavorable leverage metrics and multiple active loan liabilities."
    ]

    for i in range(n):
        age = ages[i]
        emp = employment_statuses[i]
        emp_yr = emp_years[i]
        purp = purposes[i]
        amt = loan_amounts[i]
        term = loan_terms[i]
        inc = incomes[i]
        cs = credit_scores[i]
        dti = debt_to_income[i]
        coll = collaterals[i]
        ex = existing_loans[i]
        util = revolving_util[i]
        delinq = delinquencies[i]
        bk = bankruptcies[i]
        tier = risk_tiers[i]

        # Select underwriter note based on score
        if credit_health_index[i] >= 65:
            note = np.random.choice(positive_notes)
        elif credit_health_index[i] >= 48:
            note = "Moderate risk profile; requires verification of secondary liquid assets."
        else:
            note = np.random.choice(negative_notes)
        underwriter_notes_list.append(note)

        coll_str = f"secured by {coll.lower()}" if coll != 'None' else "without collateral backing"
        ex_str = f"{ex} existing loan liability(ies)" if ex > 0 else "no existing debt obligations"
        delinq_str = f"with {delinq} past delinquency flag(s)" if delinq > 0 else "with zero delinquencies"
        bk_str = " Prior bankruptcy recorded." if bk == 1 else ""

        # Rotate through 4 distinct NLP linguistic styles
        template_id = i % 4
        if template_id == 0:
            # Underwriting Dossier Style
            desc = (
                f"Applicant is a {age}-year-old {emp.lower()} professional ({emp_yr} years in field) "
                f"requesting a ${amt:,} credit facility for {purp.lower()} over {term} months {coll_str}. "
                f"Declared annual income is ${inc:,} against a credit score of {cs} and DTI ratio of {dti:.2f}. "
                f"Credit file shows {ex_str}, {util*100:.0f}% revolving utilization, and {delinq_str}.{bk_str} "
                f"Underwriting assessment: {note}"
            )
        elif template_id == 1:
            # Executive Summary Style
            desc = (
                f"Credit application summary: ${amt:,} {purp.lower()} loan sought by a {age}-year-old {emp.lower()} individual. "
                f"Repayment horizon is {term} months with {coll.lower()} collateral. "
                f"Applicant commands an annual income of ${inc:,} with a FICO score of {cs} and debt ratio of {dti:.2f}. "
                f"Borrower carries {ex_str} and revolving utilization of {util*100:.0f}%. "
                f"Credit notes: {note}"
            )
        elif template_id == 2:
            # Financial Narrative Style
            desc = (
                f"Loan profile: The applicant is {age} years of age, working as {emp.lower()} with {emp_yr} years of experience. "
                f"Seeking ${amt:,} financing for {purp.lower()} across {term} monthly installments {coll_str}. "
                f"Earning ${inc:,} annually with an established credit score of {cs} and monthly DTI of {dti:.2f}. "
                f"The borrower currently maintains {ex_str}. Risk note: {note}"
            )
        else:
            # Comprehensive Underwriting Statement
            desc = (
                f"Borrower review: {age}-year-old {emp.lower()} requesting a ${amt:,} {term}-month loan for {purp.lower()}. "
                f"Financial standing: ${inc:,} annual gross income, credit rating of {cs}, and leverage ratio of {dti:.2f}. "
                f"Collateral posture: {coll}. Obligation structure: {ex_str}, {delinq_str}, {util*100:.0f}% revolving usage.{bk_str} "
                f"Analyst review: {note}"
            )

        descriptions.append(desc)

    df = pd.DataFrame({
        'description': descriptions,
        'age': ages,
        'employment_status': employment_statuses,
        'employment_years': emp_years,
        'income': incomes,
        'credit_score': credit_scores,
        'loan_amount': loan_amounts,
        'loan_term_months': loan_terms,
        'loan_purpose': purposes,
        'debt_to_income': debt_to_income,
        'existing_loans': existing_loans,
        'collateral': collaterals,
        'revolving_utilization': revolving_util,
        'delinquencies': delinquencies,
        'bankruptcies': bankruptcies,
        'underwriter_notes': underwriter_notes_list,
        'credit_health_index': credit_health_index,
        'approved': approved,
        'risk_tier': risk_tiers,
        'recommended_apr': recommended_apr,
        'default_prob': default_prob,
        'max_approved_amount': max_approved_amount,
    })

    return df


import sys
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    print("[*] Generating 10,000 synthetic multi-task loan records...")
    df = generate_loan_dataset(N)
    df.to_csv('data/loan_data.csv', index=False)
    print(f"[+] Generated {len(df):,} loan records")
    print(f"    Approval rate: {df['approved'].mean()*100:.1f}%")
    print(f"    Risk Tier Breakdown:")
    for tier, cnt in df['risk_tier'].value_counts().items():
        print(f"      - {tier}: {cnt:,} ({cnt/len(df)*100:.1f}%)")
    print(f"    Average Recommended APR: {df['recommended_apr'].mean():.2f}%")
    print(f"    Saved to data/loan_data.csv")
    print(f"\nSample NLP Description:\n{df['description'].iloc[0]}")


