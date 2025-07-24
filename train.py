import argparse
import os
import pickle
import time

import mlflow
import numpy as np
import sklearn.metrics as metrics
import torch
from dotenv import load_dotenv
from prefect import flow, task

import utils

if load_dotenv():
    print("Loaded environment variables from .env file")
else:
    print("No .env file found, using default environment variables")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model for credit card fraud detection"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Size of the hidden layer in the model",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="Perform grid search for hyperparameter tuning",
    )
    # parser.add_argument(
    #     "--register_model",
    #     action="store_true",
    #     help="Register the best model in MLflow Model Registry",
    # )
    parser.add_argument(
        "--quick_debug",
        action="store_true",
        default=False,
        help="Run a quick debug with reduced dataset and epochs",
    )
    return parser.parse_args()


@task(name="load_preprocess_data", log_prints=True)
def load_preprocess_data():
    df_train = utils.load_data("data/fraudTrain.csv")
    df_test = utils.load_data("data/fraudTest.csv")
    df_train = utils.preprocess_data(df_train)
    df_test = utils.preprocess_data(df_test)
    scaler, X_train_scaled = utils.get_scaler_and_features(
        df_train.drop(columns=["is_fraud"])
    )
    y_train = df_train["is_fraud"].values
    X_test_scaled = scaler.transform(df_test.drop(columns=["is_fraud"]))
    y_test = df_test["is_fraud"].values
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_value = loss_fn(outputs, targets.long())
        loss_value.backward()
        optimizer.step()
        total_loss += loss_value.item()
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device):
    model.eval()
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            all_targets.append(targets)
            all_preds.append(preds)
    targets = torch.cat(all_targets)
    predicted = torch.cat(all_preds)
    accuracy = metrics.accuracy_score(
        targets.detach().cpu().numpy(), predicted.detach().cpu().numpy()
    )
    f1 = metrics.f1_score(
        targets.detach().cpu().numpy(),
        predicted.detach().cpu().numpy(),
        average="binary",
    )
    recall = metrics.recall_score(
        targets.detach().cpu().numpy(),
        predicted.detach().cpu().numpy(),
        average="binary",
    )
    precision = metrics.precision_score(
        targets.detach().cpu().numpy(),
        predicted.detach().cpu().numpy(),
        average="binary",
    )
    return accuracy, f1, recall, precision


@task(name="run_experiment", log_prints=True)
def run_experiment(
    train_loader,
    test_loader,
    device,
    hidden_size,
    epochs,
    feature_size,
    scaler,
    learning_rate=1e-3,
):
    with mlflow.start_run():
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_artifact("scaler.pkl")
        model = utils.get_model(input_size=feature_size, hidden_size=hidden_size).to(
            device
        )
        optimizer = utils.get_optimizer(model, lr=learning_rate)
        loss = utils.get_loss_function()
        best_recall = 0.0
        best_f1 = 0.0
        best_precision = 0.0
        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, loss, device)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")
            accuracy, f1, recall, precision = evaluate(model, test_loader, device)
            print(f"Epoch {epoch+1}, Accuracy: {accuracy:.4f}")
            print(
                f"Epoch {epoch+1}, F1 Score: {f1:.4f} Recall: {recall:.4f} Precision: {precision:.4f}"
            )
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)  # type: ignore
            mlflow.log_metric("f1_score", f1, step=epoch)  # type: ignore
            mlflow.log_metric("recall", recall, step=epoch)  # type: ignore
            mlflow.log_metric("precision", precision, step=epoch)  # type: ignore
            if recall >= best_recall:
                best_recall = recall
                best_f1 = f1
                best_precision = precision
                mlflow.pytorch.log_model(  # type: ignore
                    model,
                    name="model",
                )
                print(f"Saved best model with Recall score: {best_recall:.2f}")
        print(f"Best model Recall: {best_recall:.2f}")
        mlflow.log_metric("best_recall", best_recall)  # type: ignore
        mlflow.log_metric("best_f1", best_f1)  # type: ignore
        mlflow.log_metric("best_precision", best_precision)  # type: ignore
    return best_recall, best_f1, best_precision


@task(name="grid_search", log_prints=True)
def grid_search(
    train_loader, test_loader, device, hidden_size, epochs, feature_size, scaler
):
    learning_rates = torch.arange(0.0001, 0.001 + 1e-9, 0.0001)
    best_model_recall = 0.0
    best_model_f1 = 0.0
    best_model_precision = 0.0
    best_parameters = {}
    hidden_sizes = [256, 512]
    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            print(f"Training with learning rate: {lr:.4f}")
            model_recall, model_f1, model_precision = run_experiment(
                train_loader,
                test_loader,
                device,
                hidden_size,
                epochs,
                feature_size,
                scaler=scaler,
                learning_rate=lr.item(),
            )
            if model_recall > best_model_recall:
                best_model_recall = model_recall
                best_model_f1 = model_f1
                best_model_precision = model_precision
                best_parameters["lr"] = lr.item()
                best_parameters["hidden_size"] = hidden_size

    print(f"Best model Recall: {best_model_recall:.4f}")
    print(f"Best model F1 Score: {best_model_f1:.4f}")
    print(f"Best model Precision: {best_model_precision:.4f}")
    print(f"Best learning rate: {best_parameters['lr']:.4f}")
    print(f"Best hidden size: {best_parameters['hidden_size']}")
    return best_model_recall


# @task(name="register_model", log_prints=True)
# def register_model():
#     """
#     Register the best model in MLflow Model Registry.
#     """
#     client = mlflow.MlflowClient(TRACKING_URI)
#     experiments = client.search_experiments()
#     if not experiments:
#         print("No experiments found.")
#         return
#     best_experiment = experiments[0]
#     runs = client.search_runs(
#         experiment_ids=[best_experiment.experiment_id],
#         order_by=[
#             "metrics.best_recall desc",
#         ],
#         filter_string="metrics.best_f1 >= 0.7"
#     )
#     if not runs:
#         print("No runs found.")
#         return
#     best_run = runs[0]
#     print(f"Best run: {best_run.info.run_id}")
#     mlflow.register_model(
#         model_uri=f"runs:/{best_run.info.run_id}/model",
#         name="CreditCardFraudDetector-MLP",
#     )


@flow(name="main_flow", log_prints=True)
def main(args):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(f"credit-card-fraud-detection-{int(time.time())}")
    utils.seed_everything(42)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_preprocess_data()
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    train_loader = utils.get_dataloader(
        X_train_scaled, y_train, batch_size=args.batch_size, train=True
    )
    test_loader = utils.get_dataloader(
        X_test_scaled, y_test, batch_size=args.batch_size, train=False
    )

    device = utils.get_device(args.use_cpu)
    print(f"Using device: {device}")
    feature_size = X_train_scaled.shape[1]
    if args.grid_search and not args.quick_debug:
        print("Ignoring provided model parameters")
        print("Performing grid search for hyperparameter tuning")
        best_f1 = grid_search(
            train_loader,
            test_loader,
            device,
            args.hidden_size,
            args.epochs,
            feature_size,
            scaler=scaler,
        )
    else:
        if args.quick_debug:
            args.epochs = 3
        f1 = run_experiment(
            train_loader,
            test_loader,
            device,
            args.hidden_size,
            args.epochs,
            feature_size,
            scaler=scaler,
            learning_rate=args.learning_rate,
        )
    # if args.register_model:
    #     register_model()


if __name__ == "__main__":
    args = parse_args()
    main(args)
