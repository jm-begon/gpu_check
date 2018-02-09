import pandas as pd

def utilization(df):



if __name__ == '__main__':
    df = pd.read_csv("gpu_test.log")

    gpu0_df = df[::4]
    gpu1_df = df[1::4]
    gpu2_df = df[2::4]
    gpu3_df = df[3::4]
