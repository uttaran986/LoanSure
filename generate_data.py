"""
generate_data.py
Generates a synthetic loan dataset and saves to data/loan_data.csv
Run once before training: python generate_data.py
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 5000

def generate_loan_dataset(n=N):
    ages = np.random.randint(21, 65, n)
    incomes = np.random.lognormal(mean=11.0, sigma=0.5, size=n).astype(int)
    incomes = np.clip(incomes, 20000, 300000)

    credit_scores = np.random.normal(680, 80, n).astype(int)
    credit_scores = np.clip(credit_scores, 300, 850)

    loan_amounts = np.random.lognormal(mean=10.5, sigma=0.7, size=n).astype(int)
    loan_amounts = np.clip(loan_amounts, 5000, 500000)

    loan_terms = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240, 360], n)

    employments = np.random.choice(
        ['Employed', 'Self-Employed', 'Business Owner', 'Unemployed', 'Retired', 'Student'],
        n, p=[0.50, 0.20, 0.12, 0.06, 0.07, 0.05]
    )

    purposes = np.random.choice(
        ['Home Purchase', 'Debt Consolidation', 'Business Expansion',
         'Education', 'Medical', 'Vehicle Purchase', 'Home Improvement', 'Personal'],
        n, p=[0.20, 0.22, 0.12, 0.10, 0.06, 0.12, 0.10, 0.08]
    )

    existing_loans = np.random.randint(0, 5, n)
    debt_to_income = (loan_amounts * 0.005 + existing_loans * 500) / (incomes / 12)
    debt_to_income = np.clip(debt_to_income, 0.01, 2.5).round(2)

    collateral = np.random.choice(['None', 'Property', 'Vehicle', 'Investments', 'Equipment'], n,
                                   p=[0.35, 0.30, 0.15, 0.12, 0.08])

    # ── Approval logic (rule-based, realistic) ─────────────────────────────
    score = np.zeros(n)
    score += np.where(credit_scores >= 750, 3, np.where(credit_scores >= 700, 2,
              np.where(credit_scores >= 650, 1, np.where(credit_scores >= 600, 0, -2))))
    score += np.where(incomes >= 100000, 2, np.where(incomes >= 60000, 1,
              np.where(incomes >= 35000, 0, -1)))
    score += np.where(debt_to_income <= 0.3, 2, np.where(debt_to_income <= 0.5, 1,
              np.where(debt_to_income <= 0.8, 0, -2)))
    score += np.where(employments == 'Employed', 2,
              np.where(employments.isin(['Self-Employed', 'Business Owner']), 1,
              np.where(employments == 'Retired', 0,
              np.where(employments == 'Student', -1, -2)))
              if False else np.where(
                  np.isin(employments, ['Self-Employed', 'Business Owner']), 1,
                  np.where(np.isin(employments, ['Retired']), 0,
                  np.where(np.isin(employments, ['Student']), -1, -2))))
    score += np.where(collateral != 'None', 1, 0)
    score += np.where(existing_loans == 0, 1, np.where(existing_loans <= 2, 0, -1))

    noise = np.random.normal(0, 0.5, n)
    score += noise

    approved = (score >= 2).astype(int)

    # ── Natural language description (the NLP input) ───────────────────────
    descriptions = []
    for i in range(n):
        emp = employments[i]
        purp = purposes[i]
        coll = collateral[i]
        coll_text = f" with {coll.lower()} as collateral" if coll != 'None' else " with no collateral offered"
        existing_text = (f"The applicant has {existing_loans[i]} existing loan(s)." 
                         if existing_loans[i] > 0 else "The applicant has no existing loans.")
        
        desc = (
            f"Applicant is a {ages[i]}-year-old {emp.lower()} individual requesting a "
            f"${loan_amounts[i]:,} loan for {purp.lower()} over {loan_terms[i]} months"
            f"{coll_text}. "
            f"Annual income is ${incomes[i]:,} with a credit score of {credit_scores[i]}. "
            f"Debt-to-income ratio is {debt_to_income[i]:.2f}. "
            f"{existing_text}"
        )
        descriptions.append(desc)

    df = pd.DataFrame({
        'description': descriptions,
        'age': ages,
        'income': incomes,
        'credit_score': credit_scores,
        'loan_amount': loan_amounts,
        'loan_term_months': loan_terms,
        'employment_status': employments,
        'loan_purpose': purposes,
        'debt_to_income': debt_to_income,
        'existing_loans': existing_loans,
        'collateral': collateral,
        'approved': approved,
    })

    return df


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    df = generate_loan_dataset()
    df.to_csv('data/loan_data.csv', index=False)
    print(f"✅ Generated {len(df)} loan records")
    print(f"   Approval rate: {df['approved'].mean()*100:.1f}%")
    print(f"   Saved to data/loan_data.csv")
    print(f"\nSample description:")
    print(df['description'].iloc[0])
