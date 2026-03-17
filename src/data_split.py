import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_data():

    #Path to raw data
    data_path = Path('/workspaces/telecom-customer-churn-ml-project/data/raw/telecom_customer_churn.csv')

    #Load data
    df = pd.read_csv(data_path)

    print("Data Loaded")
    print(f"Shape: {df.shape}")

    #Splitting dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"Train Shape: {train_df.shape}")
    print(f"Test Shape: {test_df.shape}")

    #Saving datasets
    train_path = Path('/workspaces/telecom-customer-churn-ml-project/data/raw/train.csv')
    test_path = Path('/workspaces/telecom-customer-churn-ml-project/data/raw/test.csv')

    train_df.to_csv(train_path, index = False)
    test_df.to_csv(test_path, index = False)

    print("Data Split Successful")

if __name__ == "__main__":
    split_data()