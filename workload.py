import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import time, datetime
import sys


def create_net(n_features, n_hidden, n_layers=50, n_output=1):
    layers = [nn.Linear(n_features, n_hidden),
              nn.ReLU()]
    for i in range(n_layers):
        layers.append(nn.Linear(n_hidden, n_hidden))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(n_hidden, n_output))

    return nn.Sequential(*layers)


def generate_data(batch_size, n_features):
    x = Variable(torch.randn(batch_size, n_features)).cuda()
    return x


def print_duration(durations):
    durations = np.array(durations)
    print()
    print("Duration [s]")
    print("Min", " " * 25, "Mean/std", " " * 25, "Max", sep="")
    mask = "{{:^9}}{0:15}{{:^8}}{0:15}{{:^8}}".format(" ")
    print(mask.format("{:.4f}".format(durations.min()),
                      "{:.4f} +/- {:.4f}".format(durations.mean(), durations.std()),
                      "{:.4f}".format(durations.max())))
    print()
    print(repr(durations))


def main(n_features=784, n_hidden=5000, n_layers=50, n_output=1,
         batch_size=1000, epochs=1000, seed=0):
    durations = []
    try:

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        model = create_net(n_features, n_hidden, n_layers, n_output)
        model.cuda()

        x = generate_data(batch_size, n_features)

        for epoch in range(epochs):
            start = time.time()
            for iteration in range(100):
                model(x)
            duration = time.time() - start
            durations.append(duration)
            if epoch % 10 == 0:
                print("Epoch {} duration:".format(epoch),
                      datetime.timedelta(seconds=duration))
                sys.stdout.flush()

    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)
    finally:
        print_duration(durations)
        print()


if __name__ == '__main__':
    main()
