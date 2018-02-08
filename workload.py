import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import time, datetime


def create_net(n_features, n_hidden, n_layers=50, n_output=1):
    layers = [nn.Linear(n_features, n_hidden),
              nn.Relu()]
    for i in range(n_layers):
        layers.append(nn.linear(n_hidden, n_hidden))
        layers.append(nn.Relu)

    layers.append(nn.Linear(n_hidden, n_output))

    return nn.Sequential(layers)


def generate_data(batch_size, n_features):
    x = Variable(torch.randn(batch_size, n_features)).gpu()
    return x


def main(n_features=784, n_hidden=500, n_layers=50, n_output=1,
         batch_size=1000, epochs=100, seed=0):
    ls_size = 60000

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = create_net(n_features, n_hidden, n_layers, n_output)

    x = generate_data(batch_size, n_features)

    for epoch in range(epochs):
        start = time.time()
        for iteration in range(int(ls_size/batch_size)):
            model(x)
        end = time.time() - start
        print("Epoch duration:", datetime.timedelta(seconds=end - start))

