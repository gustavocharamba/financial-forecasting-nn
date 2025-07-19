import pandas as pd
from sklearn.preprocessing import StandardScaler
from data.classify_market_trend import classify_market_trend

def load_and_preprocess(start_date: str = '1960-01-01', split_date: str = '2018-12-31'):
    df = classify_market_trend()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # índice mensal completo
    full_idx = pd.date_range(start=start_date, end=df.index.max(), freq='M')
    df = df.reindex(full_idx)

    # interpola e preenche bordas
    df = df.interpolate(method='linear', limit_direction='both')
    df = df.fillna(method='ffill').fillna(method='bfill')

    # feature engineering contínua
    feature_cols = [c for c in df.columns if not c.endswith('_Trend')]
    # Exemplo: adicionar pct change de CPI e yield spread
    df['CPI_pct_3m'] = df['CPI_Close'].pct_change(periods=3)
    df['YieldSpread_10_2'] = df['Yield10_Close'] - df['Yield5_Close']
    feature_cols += ['CPI_pct_3m', 'YieldSpread_10_2']

    # normalização
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df[feature_cols]), index=df.index, columns=feature_cols)

    # labels multi-market
    y = df[['SP500_Trend', 'NASDAQ_Trend', 'BTC_Trend']].astype(int)

    # split temporal
    train_idx = X.index <= split_date
    X_train, X_test = X.loc[train_idx], X.loc[~train_idx]
    y_train, y_test = y.loc[train_idx], y.loc[~train_idx]

    return X_train, X_test, y_train, y_test, scaler