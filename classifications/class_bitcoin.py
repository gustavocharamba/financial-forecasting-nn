import pandas as pd
from financial_forecasting_nn.data.load_datasets import load_financial_dataset

# Load the DataFrame
df = load_financial_dataset()

# Remove rows with missing data in the Bitcoin closing price column
df = df[df['BTC_Close'].notna()]

def btc_classification(time=3, ref=0.1):

    # Calculate the future percentage change
    var_perc = (df['BTC_Close'].shift(-time) - df['BTC_Close']) / df['BTC_Close']

    # Function to classify the trend
    def classify_trend(x):
        if pd.isna(x):
            return None
        if x > ref:
            return 2   # Uptrend
        elif x < -ref:
            return 0   # Downtrend
        else:
            return 1   # Stable

    # Apply the classification function
    df['BTC_Trend'] = var_perc.apply(classify_trend)

    return df

df = btc_classification()

print(df["BTC_Trend"])
