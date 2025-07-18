import pandas as pd
from financial_forecasting_nn.data.load_datasets import load_financial_dataset

def classify_market_trend():
    # Load the DataFrame
    df = load_financial_dataset()

    def classify_market_trend(df, market_col, time, ref, trend_col=None):
        if trend_col is None:
            trend_col = market_col.split('_')[0] + '_Trend'

        # Remove rows with NaN in the market column
        df_filtered = df[df[market_col].notna()].copy()

        # Calculate future percentage change
        var_perc = (df_filtered[market_col].shift(-time) - df_filtered[market_col]) / df_filtered[market_col]

        def classify_trend(x):
            if pd.isna(x):
                return None
            if x > ref:
                return 2  # Uptrend
            elif x < -ref:
                return 0  # Downtrend
            else:
                return 1  # Stable

        df_filtered[trend_col] = var_perc.apply(classify_trend)

        # Join the new column back to the original df (using join to preserve all dates)
        df = df.join(df_filtered[trend_col], how='left')

        return df

    # Parameters for each market
    market_params = [
        ('SP500_Close', 3, 0.03),
        ('NASDAQ_Close', 3, 0.03),
        ('BTC_Close', 3, 0.1)
    ]

    # Apply classification for all markets
    for col, time, ref in market_params:
        df = classify_market_trend(df, market_col=col, time=time, ref=ref)

    return df
