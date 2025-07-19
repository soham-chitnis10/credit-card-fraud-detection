import argparse
import time

import mlflow
import numpy as np
import sklearn.metrics as metrics
import torch
from tqdm import tqdm, trange

import utils


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
    return parser.parse_args()


def load_preprocess_data():
    df_train = utils.load_data("data/fraudTrain.csv")
    df_test = utils.load_data("data/fraudTest.csv")
    df_train = utils.preprocess_data(df_train)
    df_test = utils.preprocess_data(df_test)
    X_train_scaled = utils.normalize_data(df_train.drop(columns=["is_fraud"]))
    y_train = df_train["is_fraud"].values
    X_test_scaled = utils.normalize_data(df_test.drop(columns=["is_fraud"]))
    y_test = df_test["is_fraud"].values
    return X_train_scaled, X_test_scaled, y_train, y_test


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


def run_experiment(
    train_loader,
    test_loader,
    device,
    hidden_size,
    epochs,
    feature_size,
    learning_rate=1e-3,
):
    with mlflow.start_run():
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", train_loader.batch_size)
        model = utils.get_model(input_size=feature_size, hidden_size=hidden_size).to(
            device
        )
        optimizer = utils.get_optimizer(model, lr=learning_rate)
        loss = utils.get_loss_function()
        best_recall = 0.0

        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, loss, device)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")
            accuracy, f1, recall, precision = evaluate(model, test_loader, device)
            print(f"Epoch {epoch+1}, Accuracy: {accuracy:.2f}%")
            print(
                f"Epoch {epoch+1}, F1 Score: {f1:.2f} Recall: {recall:.2f} Precision: {precision:.2f}"
            )
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)  # type: ignore
            mlflow.log_metric("f1_score", f1, step=epoch)  # type: ignore
            mlflow.log_metric("recall", recall, step=epoch)  # type: ignore
            mlflow.log_metric("precision", precision, step=epoch)  # type: ignore
            if recall >= best_recall:
                best_recall = recall
                mlflow.pytorch.log_model(model, name="model")  # type: ignore
                torch.save(model.state_dict(), "best_model.pth")
                print(f"Saved best model with Recall score: {best_recall:.2f}")
        print(f"Best model Recall: {best_recall:.2f}")
        mlflow.log_metric("best_recall", best_recall)  # type: ignore
    return best_recall


def grid_search(train_loader, test_loader, device, hidden_size, epochs, feature_size):
    learning_rates = torch.arange(0.0001, 0.001 + 1e-9, 0.0001)
    best_model_recall = 0.0
    best_lr = 0
    for lr in learning_rates:
        print(f"Training with learning rate: {lr:.4f}")
        model_recall = run_experiment(
            train_loader,
            test_loader,
            device,
            hidden_size,
            epochs,
            feature_size,
            learning_rate=lr.item(),
        )
        if model_recall > best_model_recall:
            best_model_recall = model_recall
            best_lr = lr

    print(f"Best model Recall: {best_model_recall:.2f}")
    print(f"Best learning rate: {best_lr:.4f}")
    return best_model_recall


def register_model():
    """
    Register the best model in MLflow Model Registry.
    """
    client = mlflow.tracking.MlflowClient("http://localhost:5000")  # type: ignore
    experiments = client.search_experiments()
    if not experiments:
        print("No experiments found.")
        return
    best_experiment = experiments[0]
    runs = client.search_runs(
        experiment_ids=[best_experiment.experiment_id],
        order_by=["metrics.best_recall desc"],
    )
    if not runs:
        print("No runs found.")
        return
    best_run = runs[0]
    print(f"Best run: {best_run.info.run_id}")
    try:
        client.get_registered_model("CreditCardFraudDetector-MLP")
        print("Model already registered.")
    except:
        client.create_registered_model("CreditCardFraudDetector-MLP")
    client.create_model_version(
        name="CreditCardFraudDetector-MLP",
        source=best_run.info.artifact_uri,  # type: ignore
        run_id=best_run.info.run_id,
    )


def main(args):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(f"credit-card-fraud-detection-{int(time.time())}")
    utils.seed_everything(42)
    X_train_scaled, X_test_scaled, y_train, y_test = load_preprocess_data()

    train_loader = utils.get_dataloader(
        X_train_scaled, y_train, batch_size=args.batch_size, train=True
    )
    test_loader = utils.get_dataloader(
        X_test_scaled, y_test, batch_size=args.batch_size, train=False
    )

    device = utils.get_device(args.use_cpu)
    print(f"Using device: {device}")
    feature_size = X_train_scaled.shape[1]
    grid_search(
        train_loader, test_loader, device, args.hidden_size, args.epochs, feature_size
    )
    register_model()


if __name__ == "__main__":
    args = parse_args()
    main(args)
