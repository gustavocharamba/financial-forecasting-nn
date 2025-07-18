import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from preprocessing import load_and_preprocess
from model import create_model

# 1. Load processed data
X_train, X_test, y_train_df, y_test_df, scaler = load_and_preprocess()

# 2. Choose one target to predict (e.g., SP500 trend)
target_col = 'SP500_Trend'

y_train = y_train_df[target_col].values
y_test = y_test_df[target_col].values

# 3. Create the model
model = create_model(input_shape=X_train.shape[1], num_classes=3)

# 4. Define callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
]

# 5. Train the model
model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 6. Evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {accuracy:.4f}")

# 7. Predict and report
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
