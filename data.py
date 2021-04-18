import requests
from config import api_key
import json
import pandas as pd
import numpy as np


def get_real_time_data():
    base_url = 'https://www.alphavantage.co'
    symbol = 'BTc'
    url = f'{base_url}/query?function=CRYPTO_INTRADAY&symbol={symbol}&market=CAD&interval=15min&apikey={api_key}'

    res = requests.get(url)
    data = json.loads(res.content)
    print(data)
    with open('data.json', 'w')as f:
        json.dump(data, f)

    with open('data.json', 'r') as f:
        data = json.load(f)


def get_disjoint_sets():
    dataset = ''
    df = pd.read_csv(dataset)
    df = df.dropna()

    sets = []
    current_set = []
    prev_row = -1
    count = 0
    for row in df.iterrows():
        count += 1
        index = row[0]
        if index == prev_row + 1:
            current_set.append(row)
        else:
            sets.append(current_set)
            current_set = [row]

        prev_row = index

        if count > 10:
            break

    if current_set:
        sets.append(current_set)

    set_lengths = [len(x) for x in sets]
    d = {}
    for x in sets:
        d[len(x)] = x

    set_lengths.sort(reverse=True)
    return [d[set_lengths[i]] for i in range(20)]


def to_train(contiguous_set):
    """
    partition data for training
    :return:
    """

    training_data = []
    training_labels = []
    for i in range(len(contiguous_set) - 3000):
        train = [contiguous_set[i + j] for j in range(1500, 0, 5)]
        labels = [contiguous_set[i + j] for j in range(3000, 1500, 5)]

        for j in range(1500):
            _, _, _, high, low, close, volume, *rest = train[j]
            train.append(
                [high, low, close, volume]
            )
            _, _, _, _, _, _, _, _, price = labels[j]
            labels.append(price)

    return training_data, training_labels


def process_data():
    """
    :return:
    """

    training_data, labels = [], []

    disjoint_sets = get_disjoint_sets()

    for disjoint_set in disjoint_sets:
        train, label = to_train(disjoint_set)
        training_data = training_data + train
        labels = labels + label

    return np.array(training_data), np.array(labels)
