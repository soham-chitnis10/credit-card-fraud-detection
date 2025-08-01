#!/bin/bash

set -a
source .env

python eval.py --data_path $1 --output_path $2 --num_workers 8 --batch_size 5000 --url $3
python monitoring.py --reference_data_path $4 --current_data_path $2
