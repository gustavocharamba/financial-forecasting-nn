import pandas as pd
from financial_forecasting_nn.data.load_datasets import load_financial_dataset

# Carrega o DataFrame
df = load_financial_dataset()

# Remove linhas com dados ausentes na coluna de fechamento do SP500
df = df[df['SP500_Close'].notna()]

def sp500_classification(time=3, ref=0.03):

    # Calcula a variação percentual futura
    var_perc = (df['SP500_Close'].shift(-time) - df['SP500_Close']) / df['SP500_Close']

    # Função para classificar a tendência
    def classify_trend(x):
        if pd.isna(x):
            return None
        if x > ref:
            return 2   # Alta
        elif x < -ref:
            return 0  # Baixa
        else:
            return 1   # Estável

    # Aplica a função de classificação
    df['SP500_Trend'] = var_perc.apply(classify_trend)

    return df
