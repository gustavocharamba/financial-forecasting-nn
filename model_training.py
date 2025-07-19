from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocessing import load_and_preprocess
from model import build_mlp

def run_pipeline(target: str):
    # load e preprocess
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

    # seleciona labels do target (SP500, NASDAQ ou BTC)
    ytr = y_train[f'{target}_Trend']
    yte = y_test[f'{target}_Trend']

    # treino/teste split adicional (se quiser) ou validar direto X_train
    Xtr, Xval, ytr_sub, yval = train_test_split(
        X_train, ytr, test_size=0.1, stratify=ytr, random_state=42)

    model = build_mlp(input_dim=X_train.shape[1])
    callbacks = [
        EarlyStopping(patience=7, restore_best_weights=True),
        ModelCheckpoint(f'best_{target}.h5', save_best_only=True)
    ]

    # treinamento
    model.fit(
        Xtr, ytr_sub,
        validation_data=(Xval, yval),
        epochs=60,
        batch_size=32,
        callbacks=callbacks
    )

    # avaliação
    yhat = model.predict(X_test).argmax(axis=1)
    print(f"\n=== Relatório para {target} ===")
    print(classification_report(yte, yhat))
    print("Matriz de confusão:\n", confusion_matrix(yte, yhat))

if __name__ == '__main__':
    for t in ['SP500', 'NASDAQ', 'BTC']:
        run_pipeline(t)