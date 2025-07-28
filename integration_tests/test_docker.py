import json

import requests
from deepdiff import DeepDiff

with open('event.json', 'rb') as f:
    event = json.load(f)

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
actual_response = requests.post(url, json=event, timeout=180).json()
print(actual_response)
expected_response = {
    'predictions': [
        {
            'prediction': 0,
            'model_version': '13',
            'trans_num': '0b242abb623afc578575680df30655b9',
        }
    ]
}

diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(f'diff={diff}')

assert 'type_changes' not in diff
assert 'values_changed' not in diff
