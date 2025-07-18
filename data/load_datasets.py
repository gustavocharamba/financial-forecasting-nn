import os
import pandas as pd

def load_financial_dataset():
    base_path = os.path.dirname(__file__)  # pasta onde está este arquivo

    # CPI (Consumer Price Index)
    cpi = pd.read_csv(os.path.join(base_path, "CPI.csv"), parse_dates=["Date"], index_col="Date")
    cpiCore = pd.read_csv(os.path.join(base_path, "CPICore.csv"), parse_dates=["Date"], index_col="Date")

    # PPI (Producer Price Index)
    ppi = pd.read_csv(os.path.join(base_path, "PPI.csv"), parse_dates=["Date"], index_col="Date")

    # IPI ( Industrial Production Index)
    ipi = pd.read_csv(os.path.join(base_path, "IPI.csv"), parse_dates=["Date"], index_col="Date")

    # Payroll and Unemployment
    payroll = pd.read_csv(os.path.join(base_path, "Payroll.csv"), parse_dates=["Date"], index_col="Date")
    unemploy = pd.read_csv(os.path.join(base_path, "Unemployment.csv"), parse_dates=["Date"], index_col="Date")

    # Yields
    yield5 = pd.read_csv(os.path.join(base_path, "Yield5.csv"), parse_dates=["Date"], index_col="Date")
    yield10 = pd.read_csv(os.path.join(base_path, "Yield10.csv"), parse_dates=["Date"], index_col="Date")
    yield30 = pd.read_csv(os.path.join(base_path, "Yield30.csv"), parse_dates=["Date"], index_col="Date")

    # Interest Rates
    rates = pd.read_csv(os.path.join(base_path, "FEDRates.csv"), parse_dates=["Date"], index_col="Date")

    # Monetary Aggregates (M1 - Narrow Money) (M2 - Broad Money)
    m1 = pd.read_csv(os.path.join(base_path, "M1.csv"), parse_dates=["Date"], index_col="Date")
    m2 = pd.read_csv(os.path.join(base_path, "M2.csv"), parse_dates=["Date"], index_col="Date")

    # SP500, Nasdaq, Bitcoin (Monthly)
    sp500 = pd.read_csv(os.path.join(base_path, "SP500.csv"), parse_dates=["Date"], index_col="Date").rename(columns=lambda x: f"SP500_{x}")
    nasdaq = pd.read_csv(os.path.join(base_path, "NASDAQ.csv"), parse_dates=["Date"], index_col="Date").rename(columns=lambda x: f"NASDAQ_{x}")
    bitcoin = pd.read_csv(os.path.join(base_path, "BTCUSD.csv"), parse_dates=["Date"], index_col="Date").rename(columns=lambda x: f"BTC_{x}")

    df = pd.concat([cpi, cpiCore, ppi, ipi, payroll, unemploy, yield5, yield10, yield30, rates, m1, m2, sp500, nasdaq, bitcoin], axis=1, join="outer")

    return df