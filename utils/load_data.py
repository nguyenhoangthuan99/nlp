import pandas as pd
import random
from sklearn.model_selection import train_test_split


def load_data(data_path, train_name, test_name, val_name=None):
    train_path = data_path + '/' + train_name
    test_path = data_path + '/' + test_name
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if val_name is None:
        train_df, val_df = train_test_split(train_df, test_rate=0.1)
    else:
        val_path = data_path + '/' + val_name
        val_df = pd.read_csv(val_path)

    return train_df, val_df, test_df


def load_lm_data(data_path, train_name, test_name, val_name=None, val_ratio=0.1):
    train_path = data_path + '/' + train_name
    test_path = data_path + '/' + test_name

    with open(train_path, 'r') as pf:
        train = pf.readlines()
        train = [t.replace('\n', '') for t in train]

    with open(test_path, 'r') as pf:
        test = pf.readlines()
        test = [t.replace('\n', '') for t in test]

    if val_name is None:
        temp = train.copy()
        random.shuffle(temp)
        n = len(temp)
        n_train = int((1-val_ratio) * n)
        train = temp[:n_train]
        val = temp[n_train:-1]

    else:
        val_path = data_path + '/' + val_name
        with open(val_path, 'r') as pf:
            val = pf.readlines()
            val = [t.replace('\n', '') for t in val]

    return train, val, test