import pandas as pd
from sklearn.preprocessing import StandardScaler
from financial_forecasting_nn.classifications.class_all_markets import classify_market_trend


def load_and_preprocess():
    # 1. Load data and apply trend classification (returns DataFrame with labels)
    df = classify_market_trend()

    # 2. Ensure datetime index and sorted
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # 3. Create full monthly datetime index from 1960-01 to last date in data
    date_range = pd.date_range(start='1960-01-01', end=df.index.max(), freq='M')

    # 4. Reindex DataFrame to complete date range
    df_aligned = df.reindex(date_range)

    # 5. Interpolate missing values, then forward/backward fill edges
    df_interpolated = df_aligned.interpolate(method='linear', limit_direction='both')
    df_interpolated = df_interpolated.fillna(method='ffill').fillna(method='bfill')

    # 6. Normalize numeric features (exclude trend label columns)
    trend_cols = ['SP500_Trend', 'NASDAQ_Trend', 'BTC_Trend']
    feature_cols = [col for col in df_interpolated.columns if col not in trend_cols]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_interpolated[feature_cols])

    df_scaled = pd.DataFrame(features_scaled, index=df_interpolated.index, columns=feature_cols)

    # 7. Add back the trend label columns without scaling
    df_scaled[trend_cols] = df_interpolated[trend_cols]

    # 8. Separate features and labels
    X = df_scaled[feature_cols]
    y = df_scaled[trend_cols]

    # 9. Split train/test by date to avoid data leakage (example: train until end 2018)
    split_date = '2018-12-31'
    X_train = X[X.index <= split_date]
    y_train = y[y.index <= split_date]
    X_test = X[X.index > split_date]
    y_test = y[y.index > split_date]

    return X_train, X_test, y_train, y_test, scaler

# Usage example:
# X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
