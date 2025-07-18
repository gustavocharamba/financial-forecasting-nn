import pandas as pd
from financial_forecasting_nn.data.load_datasets import load_financial_dataset

# Load the DataFrame
df = load_financial_dataset()

# Remove rows with missing data in the SP500 closing price column
df = df[df['SP500_Close'].notna()]

def sp500_classification(time=3, ref=0.03):

    # Calculate the future percentage change
    var_perc = (df['SP500_Close'].shift(-time) - df['SP500_Close']) / df['SP500_Close']

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
    df['SP500_Trend'] = var_perc.apply(classify_trend)

    return df
