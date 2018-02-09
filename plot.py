%matplotlib

from matplotlib import pyplot as plt
from data import gpu0, gpu1, gpu2, gpu3

def plot_times():
    plt.plot(gpu0, color="b", label="GPU 0")
    plt.plot(gpu1, color="r", label="GPU 1")
    plt.plot(gpu2, color="g", label="GPU 2")
    plt.plot(gpu3, color="violet", label="GPU 3")
    plt.xlabel("Epoch")
    plt.ylabel("Duration [s]")
    plt.title("Epoch duration per GPU")
    plt.legend()


def box_times():
    plt.boxplot([gpu0, gpu1, gpu2, gpu3],
                labels=["GPU 0", "GPU 1", "GPU 2", "GPU 3"])
    plt.ylabel("Epoch duration [s]")
    plt.title("Epoch duration per GPU")

def plot_usage(df):
    for i,c in (0, "b"), (1, "r"), (2, "g"), (3, "violet"):
        plt.plot(df[i::4][' utilization.gpu [%]'].as_matrix(),
                 color=c, label="GPU {}".format(i))
    plt.xlabel("Time [s]")
    plt.ylabel("GPU utilization [%]")
    plt.title("Utilization per GPU")
    plt.legend()

def plot_clock(df):
    for i,c in (0, "b"), (1, "r"), (2, "g"), (3, "violet"):
        plt.plot(df[i::4][' clocks.current.sm [MHz]'].as_matrix(),
                 color=c, label="GPU {}".format(i))
    plt.xlabel("Time [s]")
    plt.ylabel("Clock speed [MHz]")
    plt.title("Clock speed per GPU")
    plt.legend()

def plot_temperature(df):
    for i,c in (0, "b"), (1, "r"), (2, "g"), (3, "violet"):
        plt.plot(df[i::4][' temperature.gpu'].as_matrix(),
                 color=c, label="GPU {}".format(i))
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature per GPU")
    plt.legend()


def plot_fanspeed(df):
    for i,c in (0, "b"), (1, "r"), (2, "g"), (3, "violet"):
        plt.plot(df[i::4][' fan.speed [%]'].as_matrix(),
                 color=c, label="GPU {}".format(i))
    plt.xlabel("Time [s]")
    plt.ylabel("Fan speed [%]")
    plt.title("Fan speed per GPU")
    plt.legend()



