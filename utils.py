import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import CreditCardFraudDetector

features = [
    "amt",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "trans_time",
    "month",
    "category_food_dining",
    "category_gas_transport",
    "category_grocery_net",
    "category_grocery_pos",
    "category_health_fitness",
    "category_home",
    "category_kids_pets",
    "category_misc_net",
    "category_misc_pos",
    "category_personal_care",
    "category_shopping_net",
    "category_shopping_pos",
    "category_travel",
]


def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(file_path)


def preprocess_data(df):
    """
    Preprocess the DataFrame by dropping unnecessary columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"], format="%Y-%m-%d %H:%M:%S"
    )
    df["trans_time"] = df["trans_date_trans_time"].apply(
        lambda x: x.hour * 3600 + x.minute * 60 + x.second
    )
    df["month"] = df["trans_date_trans_time"].apply(lambda x: x.month)

    columns_to_drop = [
        "Unnamed: 0",
        "trans_date_trans_time",
        "unix_time",
        "street",
        "city",
        "state",
        "first",
        "last",
        "dob",
        "zip",
        "city_pop",
        "merchant",
        "cc_num",
        "gender",
        "trans_num",
        "job",
    ]
    df.drop(columns=columns_to_drop, inplace=True)

    encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded_array = encoder.fit_transform(df[["category"]])
    encoded_df = pd.DataFrame(
        encoded_array, columns=encoder.get_feature_names_out(["category"])
    )
    df = pd.concat([df.drop(columns=["category"]), encoded_df], axis=1)

    return df


def get_scaler_and_features(df):
    """
    Normalize the DataFrame by scaling numerical features.

    Parameters:
    df (pd.DataFrame): The DataFrame to normalize.

    Returns:
    tuple: A tuple containing the scaler and the normalized features.
    """

    scaler = StandardScaler()
    features = scaler.fit_transform(df)

    return scaler, features


def seed_everything(seed=42):
    """
    Set the random seed for reproducibility.

    Parameters:
    seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_dataloader(X, y, batch_size=128, train=True):
    """
    Create a DataLoader from the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert into a DataLoader.
    batch_size (int): The size of each batch.
    shuffle (bool): Whether to shuffle the data.

    Returns:
    torch.utils.data.DataLoader: The DataLoader for the DataFrame.
    """
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def get_device(use_cpu=False):
    """
    Get the device to use for PyTorch operations.

    Returns:
    torch.device: The device (CPU or GPU) to use.
    """
    if use_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    else:
        return torch.device("cuda")


def get_optimizer(model, lr=0.001):
    """
    Get the optimizer for the model.

    Parameters:
    model (torch.nn.Module): The model to optimize.
    lr (float): The learning rate.

    Returns:
    torch.optim.Optimizer: The optimizer for the model.
    """
    return torch.optim.Adam(model.parameters(), lr=lr)


def get_model(input_size, hidden_size=256):
    """
    Get the CreditCardFraudDetector model.

    Parameters:
    input_size (int): The number of input features.
    hidden_size (int): The size of the hidden layer.

    Returns:
    CreditCardFraudDetector: The model instance.
    """

    return CreditCardFraudDetector(input_size=input_size, hidden_size=hidden_size)


def get_loss_function():
    """
    Get the loss function for training.

    Returns:
    torch.nn.Module: The loss function.
    """
    return torch.nn.CrossEntropyLoss()
