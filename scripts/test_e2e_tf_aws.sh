#!/bin/bash
export KINESIS_STREAM_INPUT="stg-trans-event-credit-card-fraud-detection"
export KINESIS_STREAM_OUTPUT="stg-trans-event-prediction-credit-card-fraud-detection"

SHARD_ID=$(aws kinesis put-record  \
        --stream-name ${KINESIS_STREAM_INPUT}   \
        --partition-key 1  --cli-binary-format raw-in-base64-out  \
        --data '{
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
    }'  \
        --query 'ShardId' \
        --output text
    )
echo "Shard ID: $SHARD_ID"
SHARD_ITERATOR=$(aws kinesis get-shard-iterator --shard-id ${SHARD_ID} --shard-iterator-type TRIM_HORIZON --stream-name ${KINESIS_STREAM_OUTPUT} --query 'ShardIterator' --output text)

aws kinesis get-records --shard-iterator $SHARD_ITERATOR
