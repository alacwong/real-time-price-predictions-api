import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import pickle

path = 'nn_model'
load = True
train = False
epochs = 10000
lr = 0.00003    # 0.00003, 0.0001 log loss

sliding_dim = 100
features_dim = 4
output_dim = 1
batch_size = 32

device = torch.device('cpu')

class Model(torch.nn.Sequential):
    def __init__(self):
        super().__init__()
        self.add_module('fc0', torch.nn.Linear(sliding_dim * features_dim, 2000))
        self.add_module('fc1', torch.nn.Linear(2000, 1000))
        self.add_module('fc2', torch.nn.Linear(1000, 1000))
        self.add_module('fc3', torch.nn.Linear(1000, sliding_dim * output_dim))

    def forward(self, input: Tensor) -> Tensor:
        input = input.to(device)
        means = []
        stds = []

        for i in range(input.shape[0]):
            mean = torch.mean(input[i], 0)
            input[i] -= mean
            std = torch.std(input[i], 0)
            input[i] /= std

            means += [mean[2]]
            stds += [std[2]]

        input = torch.reshape(input, (-1, sliding_dim * features_dim)).to(device)
        means = Tensor(means).reshape((input.shape[0], -1)).to(device)
        stds = Tensor(stds).reshape((input.shape[0], -1)).to(device)

        # print('output_network before', output_network)
        # output_network = output_network * stds + means
        # print('output_network after', output_network)
        # exit()

        return super().forward(input) * stds + means


if __name__ == '__main__':
    # ----------DATA PROCESSING---------------

    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)

    X_data = data['training']
    y_data = data['labels']

    # X_data = torch.reshape(Tensor(X_data), (-1, 100, 4))
    X_data = torch.reshape(Tensor(X_data), (-1, sliding_dim, features_dim))
    print(X_data.shape)
    # X_data = X_data[0:4000]
    print(X_data.shape)
    y_data = torch.reshape(Tensor(y_data), (-1, sliding_dim * output_dim))
    # y_data = y_data[0:4000]
    n_sets = X_data.shape[0]
    print("num data:", n_sets)
    print(X_data.shape)

    # print(data)
    # exit()
    # X_data = [[[0, 0], [0, 0]], [[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]]  # (n_sets, sliding_dim, features_dim)
    # X_data = torch.FloatTensor(X_data)
    # print(X_data.shape[1])
    # y_data = [[[0], [0]], [[1], [1]], [[2], [2]], [[3], [3]]]                # dim: n_batches x sliding_dim x output_dim
    # y_data = torch.FloatTensor(y_data)

    assert X_data.shape[0] == n_sets
    assert X_data.shape[0] == y_data.shape[0]
    assert X_data.shape[1] == sliding_dim
    assert X_data.shape[2] == features_dim
    assert y_data.shape[1] == sliding_dim * output_dim
    # assert y_data.shape[2] == output_dim

    x_test = X_data
    y_test = y_data

    train_data = TensorDataset(Tensor(x_test), Tensor(y_test))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # -----------ML---------------
    model = Model().to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=lr)

    if load:
        print("Loading model from " + path)
        model.load_state_dict(torch.load(path))
        optim.load_state_dict(torch.load(path + "_optim"))

    criterion = torch.nn.MSELoss()

    count = 0
    sum_loss = 0
    sum_loss_raw = 0
    sum_loss_spread = 0

    if train:
        for e in range(epochs):
            for i, (x_cpu, y_true_cpu) in enumerate(train_loader):
                x = x_cpu.to(device)
                y_true = y_true_cpu.to(device)
                # print("test t")
                # assert x.shape[1] == sliding_dim * features_dim
                # print(x.shape)
                # print(y_true.shape)

                y = model(x)

                assert x.shape[0] == y.shape[0]
                assert y.shape[1] == y_true.shape[1]

                y = torch.log(y)
                y_true = torch.log(y_true)
                loss_raw = criterion(y, y_true)
                loss_spread = torch.sum((y.std(dim=0) - y_true.std(dim=0)) * (y.std(dim=0) - y_true.std(dim=0)))
                loss = loss_raw + loss_spread
                sum_loss_raw += loss_raw
                sum_loss_spread += loss_spread
                sum_loss += loss

                # print(e, y)
                if count % 100 == 0:
                    print(e, i, sum_loss / 100, sum_loss_raw / 100, sum_loss_spread / 100)
                    sum_loss = 0
                    sum_loss_raw = 0
                    sum_loss_spread = 0
                    torch.save(model.state_dict(), path)
                    torch.save(optim.state_dict(), path + "_optim")

                optim.zero_grad()
                loss.backward()
                optim.step()

                count += 1

                if count > 10000:
                    break

            if count > 10000:
                break

        torch.save(model.state_dict(), path)
        torch.save(optim.state_dict(), path + "_optim")


    #-----------PLOT-------------
    example_i = 20000
    example_x = X_data[example_i].unsqueeze(dim=0)
    # print(example_x)
    show_x = example_x[:, :, 2].tolist()[0]
    print(show_x, type(show_x))
    # exit()
    assert example_x.shape == (1, sliding_dim, features_dim)
    example_y = y_data[example_i].tolist()
    example_y_pred = model(example_x).tolist()[0]

    # print(example_y_pred)
    # print(example_y + example_y_pred)

    print('show x', show_x)
    print('show y', example_y_pred)
    plt.plot([15 * i for i in range(0, sliding_dim)], show_x)
    plt.plot([15 * sliding_dim + 15 * i for i in range(0, sliding_dim)], example_y_pred, label='prediction')
    plt.plot([15 * sliding_dim + 15 * i for i in range(0, sliding_dim)], example_y, label='real')
    plt.legend()
    plt.show()
