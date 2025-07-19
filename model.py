from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_mlp(input_dim: int,
              layers: list[int] = [128, 64, 32],
              dropouts: list[float] = [0.4, 0.3, 0.2],
              lr: float = 1e-3) -> Sequential:
    model = Sequential()
    # primeira camada
    model.add(Dense(layers[0], activation='relu', input_dim=input_dim))
    model.add(Dropout(dropouts[0]))
    # camadas ocultas
    for units, d in zip(layers[1:], dropouts[1:]):
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(d))
    # saída 3 classes
    model.add(Dense(3, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model