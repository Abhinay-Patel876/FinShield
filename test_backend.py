import unittest
import json
import io
import pandas as pd
from app import (
    app, calculate_altman_z_score, calculate_ohlson_o_score,
    ml_predictor, generate_early_warning_signals
)

class TestFinShieldBackend(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

        self.healthy_data = {
            'company_name': 'Solvent Tech Corp',
            'total_assets': 10000000,
            'current_assets': 4500000,
            'current_liabilities': 1800000,
            'total_liabilities': 3500000,
            'retained_earnings': 2800000,
            'ebit': 1400000,
            'sales': 12000000,
            'net_income': 950000,
            'operating_cash_flow': 1600000,
            'market_value_equity': 15000000,
            'net_income_prior': 850000
        }

        self.distressed_data = {
            'company_name': 'Distressed Mfg Ltd',
            'total_assets': 5000000,
            'current_assets': 1100000,
            'current_liabilities': 2500000,
            'total_liabilities': 5500000, # liabilities > assets (OENEG=1)
            'retained_earnings': -800000,
            'ebit': -300000,
            'sales': 3000000,
            'net_income': -500000,
            'operating_cash_flow': -200000,
            'market_value_equity': 100000,
            'net_income_prior': -400000 # two consecutive loss years (INTWO=1)
        }

    # 1. Statistical Altman Z-Score Tests
    def test_altman_z_score_healthy(self):
        res = calculate_altman_z_score(self.healthy_data)
        self.assertGreaterEqual(res['z_score'], 2.99)
        self.assertEqual(res['zone'], 'Safe Zone')
        self.assertGreater(res['working_capital'], 0)
        self.assertIn('z_double_prime', res)

    def test_altman_z_score_distressed(self):
        res = calculate_altman_z_score(self.distressed_data)
        self.assertLess(res['z_score'], 1.81)
        self.assertEqual(res['zone'], 'Distress Zone')
        self.assertLess(res['working_capital'], 0)

    # 2. Statistical Ohlson O-Score Tests
    def test_ohlson_o_score_healthy(self):
        res = calculate_ohlson_o_score(self.healthy_data)
        self.assertLess(res['probability_percent'], 25.0)
        self.assertEqual(res['intwo'], 0)
        self.assertEqual(res['oeneg'], 0)

    def test_ohlson_o_score_distressed(self):
        res = calculate_ohlson_o_score(self.distressed_data)
        self.assertGreater(res['probability_percent'], 50.0)
        self.assertEqual(res['risk_category'], 'High Risk')
        self.assertEqual(res['intwo'], 1)
        self.assertEqual(res['oeneg'], 1)

    # 3. Machine Learning Logic Tests
    def test_ml_predictor(self):
        healthy_pred = ml_predictor.predict_risk(self.healthy_data)
        self.assertLess(healthy_pred['bankruptcy_probability_percent'], 30.0)
        self.assertEqual(healthy_pred['risk_tier'], 'Low Risk')
        self.assertIn('pillar_scores', healthy_pred)
        self.assertGreaterEqual(healthy_pred['pillar_scores']['liquidity'], 50)

        distressed_pred = ml_predictor.predict_risk(self.distressed_data)
        self.assertGreater(distressed_pred['bankruptcy_probability_percent'], 60.0)
        self.assertEqual(distressed_pred['risk_tier'], 'High Risk')
        self.assertIn('top_factors', distressed_pred)

    # 4. Early Warning Signals Tests
    def test_early_warning_advisory(self):
        altman = calculate_altman_z_score(self.distressed_data)
        ohlson = calculate_ohlson_o_score(self.distressed_data)
        ml_res = ml_predictor.predict_risk(self.distressed_data)
        advisory = generate_early_warning_signals(self.distressed_data, altman, ohlson, ml_res)

        self.assertEqual(advisory['severity'], 'HIGH')
        self.assertGreater(len(advisory['alerts']), 0)
        self.assertGreater(len(advisory['recommendations']), 0)
        self.assertGreater(advisory['consensus_risk_score'], 60.0)

    # 5. Flask Endpoints & Integration Tests
    def test_pages_render_200(self):
        routes = ['/', '/home', '/manual-entry', '/upload', '/dashboard', '/prediction-result', '/report', '/help']
        for route in routes:
            resp = self.client.get(route)
            self.assertEqual(resp.status_code, 200, f"Route {route} failed with status {resp.status_code}")

    def test_api_calculate_stats(self):
        resp = self.client.post('/api/calculate-stats',
                                data=json.dumps(self.healthy_data),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('altman', data)
        self.assertIn('ohlson', data)
        self.assertIn('ml', data)

    def test_predict_form_redirect(self):
        resp = self.client.post('/predict', data=self.healthy_data)
        self.assertEqual(resp.status_code, 302) # Redirect to prediction_result_page

    def test_api_upload_csv(self):
        csv_content = (
            "company_name,total_assets,current_assets,current_liabilities,total_liabilities,retained_earnings,ebit,sales,net_income,operating_cash_flow,market_value_equity,net_income_prior\n"
            "UploadedTestCorp,8000000,3500000,1500000,2800000,2000000,1100000,9500000,750000,1200000,11000000,700000\n"
        )
        data = {
            'file': (io.BytesIO(csv_content.encode('utf-8')), 'test_financials.csv')
        }
        resp = self.client.post('/api/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)
        json_data = resp.get_json()
        self.assertEqual(json_data['status'], 'success')
        self.assertEqual(json_data['data']['company_name'], 'UploadedTestCorp')
        self.assertEqual(json_data['data']['altman']['zone'], 'Safe Zone')

    def test_export_pdf_and_excel(self):
        # Set up a session first
        self.client.post('/predict', data=self.healthy_data)

        # PDF export
        resp_pdf = self.client.get('/download-report/pdf')
        self.assertEqual(resp_pdf.status_code, 200)
        self.assertEqual(resp_pdf.mimetype, 'application/pdf')

        # Excel export
        resp_excel = self.client.get('/download-report/excel')
        self.assertEqual(resp_excel.status_code, 200)
        self.assertEqual(resp_excel.mimetype, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    def test_sample_template_download(self):
        resp = self.client.get('/sample-template')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, 'text/csv')

if __name__ == '__main__':
    unittest.main()
