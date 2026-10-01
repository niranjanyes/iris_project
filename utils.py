import pandas as pd
from config import DATA_PATH, TARGET_COLUMN

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df, df[TARGET_COLUMN]

def print_shape(df):
    print(f"Dataset shape: {df.shape}")
    print(f"Classes: {df[TARGET_COLUMN].unique()}")   