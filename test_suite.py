# test_suite.py — LoanSure AI Comprehensive Automated Test Suite
import os
import json
from app import app, db, LoanApplication
from predictor import predict, counterfactual_simulation, build_description
from advisor_engine import (
    calculate_reducing_emi,
    generate_amortization_schedule,
    estimate_eligibility,
    generate_recommendations,
    extract_value_from_text
)

def test_all():
    print("\n[*] Starting LoanSure AI Comprehensive Verification Suite...")

    # 1. Test Reducing Balance EMI & Amortization
    calc = calculate_reducing_emi(50000, 10.5, 60)
    assert calc['emi'] > 1000 and calc['emi'] < 1200, f"Unexpected EMI: {calc['emi']}"
    assert calc['total_payable'] > 50000
    assert calc['total_interest'] > 0
    print(f"  [+] Financial Calculator: Passed (P=$50k, 10.5%, 60m -> EMI=${calc['emi']}, Total Interest=${calc['total_interest']})")

    # 2. Test Amortization Schedule
    sched = generate_amortization_schedule(50000, 10.5, 60, max_rows=12)
    assert len(sched) == 12
    assert sched[0]['opening_balance'] == 50000.0
    assert sched[0]['principal'] + sched[0]['interest'] == sched[0]['emi']
    print(f"  [+] Amortization Engine: Passed (Generated {len(sched)} month rows with accurate Principal/Interest split)")

    # 3. Test Eligibility & FOIR Engine
    elig = estimate_eligibility(monthly_salary=8000, existing_emis=1000, credit_score=760, loan_type='Personal')
    assert elig['is_eligible'] == True
    assert elig['max_eligible_amount'] > 50000
    assert elig['max_allowed_foir_pct'] == 55.0
    print(f"  [+] FOIR Eligibility Sizing: Passed (Max Eligible: ${elig['max_eligible_amount']:,.0f}, Grade: {elig['risk_grade']})")

    # 4. Test 3-Tier Recommendation Engine
    recs = generate_recommendations(45000, 48, 8000, 1000, 760, 'Personal')
    assert len(recs['packages']) == 3
    assert recs['packages'][0]['tag'] == 'Recommended'
    assert len(recs['financial_health_tips']) > 0
    print(f"  [+] Recommendation Engine: Passed (3 packages generated: Optimal, Low Monthly, Fast Payoff)")

    # 5. Test Natural Language Value Extraction
    assert extract_value_from_text('age', 'I am 34 years old') == 34
    assert extract_value_from_text('monthly_income', 'Around $7,500/mo after taxes') == 7500.0
    assert extract_value_from_text('credit_score', 'My score is 745') == 745
    assert extract_value_from_text('tenure_months', '3 years') == 36
    print("  [+] Natural Language Entity Extractor: Passed")

    # 6. Test Flask Routes & Database Context
    with app.app_context():
        db.create_all()
        client = app.test_client()

        # Landing Page GET
        r = client.get('/')
        assert r.status_code == 200
        assert b'LoanSure AI' in r.data
        print("  [+] Route GET / (Landing Page with AI Advisor CTA): Passed")

        # Admin Portal GET
        r = client.get('/admin')
        assert r.status_code == 200
        print("  [+] Route GET /admin (Admin & Analytics Portal): Passed")

        # Admin Analytics JSON GET
        r = client.get('/api/admin/analytics')
        assert r.status_code == 200
        data = r.get_json()
        assert 'total_leads' in data
        assert 'type_distribution' in data
        print("  [+] Route GET /api/admin/analytics: Passed")

        # Admin Export CSV GET
        r = client.get('/api/admin/export')
        assert r.status_code == 200
        assert 'text/csv' in r.content_type
        print("  [+] Route GET /api/admin/export (CSV Stream): Passed")

        # Conversational AI Step 1 POST
        r = client.post('/api/advisor/chat', json={
            'step_index': 0,
            'answers': {},
            'message': ''
        })
        assert r.status_code == 200
        res1 = r.get_json()
        assert res1['completed'] == False
        assert len(res1['options']) > 0
        print("  [+] Route POST /api/advisor/chat (Step 1 Greeting): Passed")

        # Conversational AI Full Final Flow POST
        full_answers = {
            'loan_type': 'Personal Loan',
            'full_name': 'Eleanor Vance',
            'age': 34,
            'city': 'Austin, TX',
            'monthly_income': 8500.0,
            'existing_emis': 250.0,
            'credit_score': 760,
            'loan_amount_requested': 45000.0,
            'tenure_months': 48,
            'loan_purpose': 'Debt Consolidation'
        }
        r = client.post('/api/advisor/chat', json={
            'step_index': 9,
            'answers': full_answers,
            'message': 'Debt Consolidation'
        })
        assert r.status_code == 200
        final_res = r.get_json()
        assert final_res['completed'] == True
        assert 'lead_id' in final_res
        assert 'packages' in final_res
        print(f"  [+] Route POST /api/advisor/chat (Final Step -> Lead #{final_res['lead_id']} created): Passed")

        # Quick Calculate API POST
        r = client.post('/api/calculate', json={'principal': 60000, 'rate_pct': 9.5, 'tenure_months': 48})
        assert r.status_code == 200
        c_res = r.get_json()
        assert 'calculation' in c_res
        assert len(c_res['amortization_schedule']) > 0
        print("  [+] Route POST /api/calculate (EMI & Amortization): Passed")

        # Console View GET
        r = client.get('/console')
        assert r.status_code == 200
        print("  [+] Route GET /console (Enterprise Console): Passed")

    print("\n[+] ALL 12 LoanSure AI Tests Passed with 100% Success!\n")

if __name__ == '__main__':
    test_all()
