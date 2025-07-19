import pandas as pd
from financial_forecasting_nn.data.load_datasets import load_financial_dataset

def classify_market_trend():
    df = load_financial_dataset()

    def _classify(df, market_col, time: int, ref: float, trend_col=None):
        if trend_col is None:
            trend_col = market_col.split('_')[0] + '_Trend'
        df_filtered = df[df[market_col].notna()].copy()
        var_perc = df_filtered[market_col].pct_change(periods=time).shift(-time)

        def mapper(x):
            if x > ref: return 2  # Uptrend
            if x < -ref: return 0  # Downtrend
            return 1  # Stable

        df_filtered[trend_col] = var_perc.apply(lambda x: mapper(x) if pd.notna(x) else None)
        return df.join(df_filtered[trend_col], how='left')

    params = [
        ('SP500_Close', 3, 0.03),
        ('NASDAQ_Close',3, 0.03),
        ('BTCUSD_Close',3,0.10),
    ]
    for col, t, r in params:
        df = _classify(df, market_col=col, time=t, ref=r)
    return df
