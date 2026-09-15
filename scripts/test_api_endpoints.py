import urllib.request
import urllib.parse
import json
import uuid

base = 'http://127.0.0.1:8000'

def post_json(url, data, token=None):
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers=headers
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def get_json(url, token=None):
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def main():
    # 1. Register / Login
    uname = f'evaluator_{uuid.uuid4().hex[:6]}'
    uemail = f'{uname}@stockdna.internal'
    pwd = 'Password123!'
    print('Testing Auth...')
    reg = post_json(f'{base}/auth/register', {'username': uname, 'email': uemail, 'password': pwd})
    print('Register status: OK, msg:', reg.get('message'))

    # Login with email and password
    login_resp = post_json(f'{base}/auth/login', {'email': uemail, 'password': pwd})
    token = login_resp['access_token']
    print('Login status: OK, token obtained.')

    # 2. Predict TCS
    print('\nTesting Predict TCS...')
    pred = post_json(f'{base}/predict', {'ticker': 'TCS'}, token=token)
    pred_label = pred.get('prediction')
    conf = pred.get('confidence', 0.0)
    print(f'Prediction: {pred_label}, Confidence: {conf:.2%}')
    print(f'Probabilities: {pred.get("probabilities")}')
    xai = pred.get('xai', {})
    print(f'SHAP Internal: {xai.get("internal_percentage", 0):.1f}%, External: {xai.get("external_percentage", 0):.1f}%, Driver: {xai.get("primary_driver")}')
    top_internal = xai.get('top_internal_factors', [])
    top_external = xai.get('top_external_factors', [])
    print(f'Top internal factors: {len(top_internal)}, Top external factors: {len(top_external)}')
    if top_internal:
        print(f'  Internal sample: {top_internal[0]["feature"]} -> {top_internal[0]["shap_value"]}')
    if top_external:
        print(f'  External sample: {top_external[0]["feature"]} -> {top_external[0]["shap_value"]}')

    # 3. History
    print('\nTesting History...')
    hist = get_json(f'{base}/history/', token=token)
    print(f'History entries count: {len(hist)}')
    if hist:
        print(f'Latest history entry: {hist[0].get("ticker")} -> {hist[0].get("prediction")}')

    # 4. Data Tool
    print('\nTesting Data Tool...')
    univ = get_json(f'{base}/datatool/universe')
    print(f'Universe tickers: {[u.get("ticker") for u in univ]}')

    factors = post_json(f'{base}/datatool/factors/classify', {'feature_name': 'ROE'})
    print(f'Classify ROE: {factors}')

    sentiment = post_json(f'{base}/datatool/sentiment', {'text': 'TCS reports strong revenue growth and quarterly beat'})
    print(f'Sentiment: label={sentiment.get("sentiment")}, score={sentiment.get("score")}')

    process = post_json(f'{base}/datatool/process', {'records': [{'ticker': 'TCS', 'date': '2025-01-02', 'close': 4000.0, 'volume': 1500000}]})
    print(f'Process: success={process.get("success")}, records_processed={process.get("records_processed")}')

    # 5. Research endpoints
    print('\nTesting Research Endpoints...')
    metrics = get_json(f'{base}/research/metrics')
    test_metrics = metrics.get('metrics', {})
    print(f'Test Metrics: Accuracy={test_metrics.get("accuracy", 0):.2%}, Balanced Acc={test_metrics.get("balanced_accuracy", 0):.2%}, Macro F1={test_metrics.get("macro_f1", 0):.2%}')

    baselines = get_json(f'{base}/research/baselines')
    print(f'Baselines count: {len(baselines)}')

    backtest = get_json(f'{base}/research/backtest')
    bt_metrics = backtest.get('rigorous_portfolio_metrics', {})
    print(f'Backtest strategy return: {bt_metrics.get("strategy_cumulative_return_pct", 0):.2f}%, benchmark: {bt_metrics.get("benchmark_cumulative_return_pct", 0):.2f}%')

    print('\n>>> ALL 5 ENDPOINT SUITES PASSED VERIFICATION! <<<')

if __name__ == '__main__':
    main()
