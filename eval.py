import os
import argparse
import concurrent.futures

import pandas as pd
import requests
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Get predictions from the model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/fraudTest.csv",
        help="Path to the test data CSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/predictions.csv",
        help="Path to save the predictions CSV file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=5000, help="Batch size for predictions"
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL of the prediction service",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for concurrent requests",
    )
    return parser.parse_args()


def main(args):
    df = pd.read_csv(args.data_path)
    test_df = df.drop(columns=["is_fraud", "Unnamed: 0"])
    print(len(test_df))

    def predict(df, output_path):
        predict_df = pd.DataFrame(columns=["trans_num", "prediction"])
        for row in tqdm(df.iterrows(), total=len(df)):
            transaction_event = row[1].to_dict()
            response = requests.post(
                args.url, json=transaction_event, timeout=180
            ).json()
            predict_df.loc[len(predict_df)] = {
                "trans_num": response["trans_num"],
                "prediction": response['prediction'],
            }
        predict_df.to_csv(output_path, index=False)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers)

    futures = []
    files = []
    for i in range(0, len(test_df), args.batch_size):
        batch = test_df.iloc[i : i + args.batch_size]
        files.append(args.output_path.replace(".csv", f"_{i//args.batch_size}.csv"))
        futures.append(
            pool.submit(
                predict,
                batch,
                args.output_path.replace(".csv", f"_{i//args.batch_size}.csv"),
            )
        )

    pool.shutdown(wait=True)

    dfs = [pd.read_csv(f) for f in files]
    predictions = pd.concat(dfs, ignore_index=True)
    df['is_fraud_prediction'] = predictions['prediction']

    df.to_csv(args.output_path, index=False)

    for f in files:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
