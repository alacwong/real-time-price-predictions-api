import requests
from config import api_key
import json
import numpy as np
import csv
import pickle
import time


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
    prev_timestamp = 0
    disjoint_sets = []
    current_set = []
    with open('dataset.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys = [row[key] for key in row]
            if 'NaN' in keys:
                continue
            else:
                data = {
                    key: float(row[key]) for key in row
                }
                # print(data['Timestamp'], prev_timestamp + 60, data['Timestamp'] - (prev_timestamp + 60))
                if data['Timestamp'] == prev_timestamp + 60:
                    current_set.append(data)
                else:
                    disjoint_sets.append(current_set)
                    current_set = [data]
                prev_timestamp = data['Timestamp']

        if current_set:
            disjoint_sets.append(current_set)

    set_lengths = [len(sets) for sets in disjoint_sets]
    set_map = {}
    for sets in disjoint_sets:
        set_map[len(sets)] = sets

    set_lengths.sort(reverse=True)
    with open('disjoint_sets.json', 'w') as f:
        json.dump({
            'disjoint_sets': [set_map[set_lengths[i]] for i in range(20)],
        }, f)


def to_train(contiguous_set):
    """
    partition data for training
    :return:
    """

    training_data = []
    training_labels = []
    for i in range(len(contiguous_set) - 3000):
        train = [contiguous_set[i + j] for j in range(0, 1500, 15)]
        labels = [contiguous_set[i + j] for j in range(1500, 3000, 15)]
        for j in range(100):
            training_data.append(
                [train[j]['High'], train[j]['Low'], train[j]['Close'], train[j]['Volume_(BTC)']]
            )
            price = labels[j]['Weighted_Price']
            training_labels.append(price)

    return training_data, training_labels


def process_data():
    """
    :return:
    """

    training_data, labels = [], []

    with open('disjoint_sets.json', 'r') as f:
        disjoint_sets = json.load(f)['disjoint_sets']

    for disjoint_set in disjoint_sets:
        train, label = to_train(disjoint_set)
        training_data = training_data + train
        labels = labels + label

    data = {'training': np.array(training_data), 'labels': np.array(labels)}

    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    start = time.time()
    process_data()
    print(f'Generated dataset in {time.time() - start} s')
