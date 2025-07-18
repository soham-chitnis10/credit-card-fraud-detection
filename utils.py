import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import os
import random

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
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S')
    df['trans_time'] = df['trans_date_trans_time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    df['month'] = df['trans_date_trans_time'].apply(lambda x: x.month)

    columns_to_drop = ['Unnamed: 0','trans_date_trans_time', "unix_time", 'street', 'city', 'state', 'first', 'last',
                       'dob', 'zip', 'city_pop', 'merchant', 'cc_num', 'gender', 'trans_num', 'job']
    df.drop(columns=columns_to_drop, inplace=True)

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_array = encoder.fit_transform(df[['category']])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['category']))
    df = pd.concat([df.drop(columns=['category']), encoded_df], axis=1)


    return df


def normalize_data(df):
    """
    Normalize the DataFrame by scaling numerical features.

    Parameters:
    df (pd.DataFrame): The DataFrame to normalize.

    Returns:
    pd.DataFrame: The normalized DataFrame.
    """
    scaler = StandardScaler()
    features = scaler.fit_transform(df)

    return features

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
    os.environ['PYTHONHASHSEED'] = str(seed)
