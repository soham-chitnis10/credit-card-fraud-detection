import requests
from deepdiff import DeepDiff

trans_event = {
    "trans_date_trans_time": "2019-01-01 00:00:18",
    "cc_num": 2703186189652095,
    "merchant": "fraud_Rippin, Kub and Mann",
    "category": "misc_net",
    "amt": 4.97,
    "first": "Jennifer",
    "last": "Banks",
    "gender": "F",
    "street": "561 Perry Cove",
    "city": "Moravian Falls",
    "state": "NC",
    "zip": 28654,
    "lat": 36.0788,
    "long": -81.1781,
    "city_pop": 3495,
    "job": "Psychologist, counselling",
    "dob": "1988-03-09",
    "trans_num": "0b242abb623afc578575680df30655b9",
    "unix_time": 1325376018,
    "merch_lat": 36.011293,
    "merch_long": -82.048315,
}

response = requests.post("http://localhost:8000/predict", json=trans_event, timeout=180)

actual_response = response.json()

expected_response = {
    "trans_num": "0b242abb623afc578575680df30655b9",
    "prediction": 0,
    "model_version": "15",
}

diff = DeepDiff(actual_response, expected_response, ignore_order=True)

if diff:
    print("Differences found:")
    print(diff)
else:
    print("No differences found.")
