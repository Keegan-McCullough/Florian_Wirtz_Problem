import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

if __name__ == "__main__":
    df = load_data("Stats.csv")
    print(df.head())