import pandas as pd

def load_financial_dataset():
    # CPI (Consumer Price Index)
    cpi = pd.read_csv("data/CPI.csv", parse_dates=["Date"], index_col="Date")
    cpiCore = pd.read_csv("data/CPICore.csv", parse_dates=["Date"] ,index_col="Date")

    # PPI (Producer Price Index)
    ppi = pd.read_csv("data/PPI.csv", parse_dates=["Date"] ,index_col="Date")

    # IPI ( Industrial Production Index)
    ipi = pd.read_csv("data/IPI.csv", parse_dates=["Date"] ,index_col="Date")

    # Payrool and Unemployment
    payroll = pd.read_csv("data/Payroll.csv", parse_dates=["Date"] ,index_col="Date")
    unemploy = pd.read_csv("data/Unemployment.csv", parse_dates=["Date"] ,index_col="Date")

    # Yields
    yield5 = pd.read_csv("data/Yield5.csv", parse_dates=["Date"] ,index_col="Date")
    yield10 = pd.read_csv("data/Yield10.csv", parse_dates=["Date"] ,index_col="Date")
    yield30 = pd.read_csv("data/Yield30.csv", parse_dates=["Date"] ,index_col="Date")

    # Interest Rates
    rates = pd.read_csv("data/FEDRates.csv", parse_dates=["Date"] ,index_col="Date")

    # Monetary Aggregates (M1 - Narrow Money) (M2 - Broad Money)
    m1 = pd.read_csv("data/M1.csv", parse_dates=["Date"] ,index_col="Date")
    m2 = pd.read_csv("data/M2.csv", parse_dates=["Date"] ,index_col="Date")

    # SP500, Nasdaq, Bitcoin (Monthly)
    sp500 = pd.read_csv("data/SP500.csv", parse_dates=["Date"] ,index_col="Date")
    nasdaq = pd.read_csv("data/NASDAQ.csv", parse_dates=["Date"] ,index_col="Date")
    bitcoin = pd.read_csv("data/BTCUSD.csv", parse_dates=["Date"] ,index_col="Date")

    df = pd.concat([cpi, cpiCore, ppi, ipi, payroll, unemploy, yield5, yield10, yield30, rates, m1, m2, sp500, nasdaq, bitcoin], axis = 1, join = "outer")

    return df
