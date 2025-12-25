import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def calculate_statistics(df):
    goalsPerGame = (df['Goals'] / df['Appearances']).round(2)
    assistsPerGame = (df['Assists'] / df['Appearances']).round(2)
    df['GoalsXGame'] = goalsPerGame
    df['AssistsXGame'] = assistsPerGame
    return df

if __name__ == "__main__":
    df = load_data("Stats.csv")
    df = calculate_statistics(df)
    print(df.head())