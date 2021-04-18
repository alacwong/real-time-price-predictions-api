import torch
from train_nn import path
from train_nn import Model
from train_nn import device
from torch import Tensor

model = Model().to(device)

print("Loading model from " + path)
model.load_state_dict(torch.load(path))

def evaluate(x):
    return model(x).tolist()[0]