# test_suite.py
import os
import json
from app import app, db, LoanApplication
from predictor import predict, counterfactual_simulation, build_description

def test_all():
    print("\n[*] Starting LoanSure Comprehensive Test Suite...")

    # 1. Test Prediction Engine
    test_applicant = {
        'applicant_name': 'Eleanor Vance',
        'age': 34,
        'annual_income': 115000,
        'credit_score': 785,
        'loan_amount': 45000,
        'loan_term_months': 60,
        'employment_status': 'Employed',
        'employment_years': 6.5,
        'loan_purpose': 'Home Improvement',
        'debt_to_income': 0.18,
        'existing_loans': 0,
        'collateral': 'Property / Real Estate',
        'revolving_utilization': 0.12,
        'delinquencies': 0,
        'underwriter_notes': 'Strong liquid reserves with excellent credit history.'
    }

    result = predict(test_applicant)
    assert result['approved'] == True, "Prime applicant should be approved"
    assert result['decision_status'] == 'Approved'
    assert 'Tier A' in result['risk_tier']
    assert 5.0 <= result['recommended_apr'] <= 25.0
    assert result['max_approved_amount'] > 10000
    assert len(result['token_saliency']) > 0
    print("  [+] Predict API: Passed (Status: Approved, Tier:", result['risk_tier'], ", APR:", result['recommended_apr'], "%)")

    # 2. Test Subprime Applicant
    subprime_applicant = {
        'applicant_name': 'David Miller',
        'age': 24,
        'annual_income': 28000,
        'credit_score': 540,
        'loan_amount': 25000,
        'loan_term_months': 36,
        'employment_status': 'Unemployed',
        'employment_years': 0,
        'loan_purpose': 'Personal',
        'debt_to_income': 0.85,
        'existing_loans': 4,
        'collateral': 'None',
        'revolving_utilization': 0.90,
        'delinquencies': 2,
        'underwriter_notes': 'Multiple delinquent records on file.'
    }
    sub_res = predict(subprime_applicant)
    assert sub_res['approved'] == False
    assert sub_res['decision_status'] == 'Denied'
    assert 'Tier' in sub_res['risk_tier']
    print("  [+] Subprime Reject Test: Passed (Status: Denied, Tier:", sub_res['risk_tier'], ")")

    # 3. Test Counterfactual "What-If" Simulation
    cf_res = counterfactual_simulation(subprime_applicant, {'credit_score': 740, 'annual_income': 65000, 'debt_to_income': 0.30})
    assert cf_res['delta_prob'] > 0, "Improving credit score and income must increase approval probability"
    print(f"  [+] Counterfactual Simulation: Passed (Probability Delta: +{cf_res['delta_prob']}%, APR Delta: {cf_res['delta_apr']}%)")

    # 4. Test Flask Routes & Database
    with app.app_context():
        db.create_all()
        client = app.test_client()

        # Console GET
        r = client.get('/console')
        assert r.status_code == 200
        print("  [+] Route GET /console (Loan Management Software UI): Passed")

        # Apply POST
        r = client.post('/apply', data={
            'applicant_name': 'Sophia Chen',
            'age': '45',
            'annual_income': '140000',
            'credit_score': '810',
            'loan_amount': '75000',
            'loan_term_months': '84',
            'employment_status': 'Business Owner',
            'employment_years': '12',
            'loan_purpose': 'Business Expansion',
            'debt_to_income': '0.22',
            'existing_loans': '1',
            'collateral': 'Investment Portfolio',
            'revolving_utilization': '0.15',
            'delinquencies': '0',
            'underwriter_notes': 'High commercial cashflow.'
        }, follow_redirects=True)
        assert r.status_code == 200
        print("  [+] Route POST /apply -> /result/<id>: Passed")

        # Dashboard GET
        r = client.get('/dashboard')
        assert r.status_code == 200
        print("  [+] Route GET /dashboard: Passed")

        # Batch GET & Sample CSV download
        r = client.get('/batch')
        assert r.status_code == 200
        print("  [+] Route GET /batch: Passed")

        # Bias Audit GET
        r = client.get('/bias-audit')
        assert r.status_code == 200
        print("  [+] Route GET /bias-audit: Passed")

        # Model Info GET
        r = client.get('/model-info')
        assert r.status_code == 200
        print("  [+] Route GET /model-info: Passed")

        # API Predict POST
        r = client.post('/api/predict', json=test_applicant)
        assert r.status_code == 200
        data = r.get_json()
        assert data['approved'] == True
        print("  [+] Route POST /api/predict: Passed")

        # API What-If POST
        r = client.post('/api/what-if', json={'base': test_applicant, 'deltas': {'credit_score': 820}})
        assert r.status_code == 200
        print("  [+] Route POST /api/what-if: Passed")

        # API Sample CSV GET
        r = client.get('/api/sample-csv')
        assert r.status_code == 200
        assert 'text/csv' in r.content_type
        print("  [+] Route GET /api/sample-csv: Passed")

        # Memo View GET
        first_loan = LoanApplication.query.first()
        if first_loan:
            r = client.get(f'/memo/{first_loan.id}')
            assert r.status_code == 200
            print(f"  [+] Route GET /memo/{first_loan.id}: Passed")

    print("\n[+] ALL LoanSure Tests Passed with Zero Errors!\n")

if __name__ == '__main__':
    test_all()
