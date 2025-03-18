from typing import Sequence
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam

# Load data
leftpunch_df = pd.read_csv("h_left_punch.txt")
rightpunch_df = pd.read_csv("h_right_punch.txt")
neutral_df = pd.read_csv("h_neutral.txt")


# Check for NaN values in the raw data
print("NaN values in left punch data:", leftpunch_df.isna().sum().sum())
print("NaN values in right punch data:", rightpunch_df.isna().sum().sum())
print("NaN values in neutral data:", neutral_df.isna().sum().sum())


leftpunch_df = leftpunch_df.dropna()
rightpunch_df = rightpunch_df.dropna()
neutral_df = neutral_df.dropna()


X = []
y = []
no_of_timesteps = 20

# Process left punch da

# Process left punch data
datasets = leftpunch_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(0)

# Process right punch data
datasets = rightpunch_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(1)  

# Convert to numpy arrays
X, y = np.array(X), np.array(y)
print("Shape of X:", X.shape, "Shape of y:", y.shape)

# Check for NaN values in X and y
print("Number of NaN values in X:", np.isnan(X).sum())
print("Number of NaN values in y:", np.isnan(y).sum())

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Check for NaN values after normalization
print("Number of NaN values in X after normalization:", np.isnan(X).sum())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Check for NaN values in training and testing data
print("Number of NaN values in X_train:", np.isnan(X_train).sum())
print("Number of NaN values in X_test:", np.isnan(X_test).sum())
print("Number of NaN values in y_train:", np.isnan(y_train).sum())
print("Number of NaN values in y_test:", np.isnan(y_test).sum())

# Build model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=2, activation="softmax"))  # Assuming 2 classes

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), metrics=["accuracy"], loss="sparse_categorical_crossentropy")

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

# Check for NaN values in loss and val_loss during training
if np.isnan(history.history['loss']).any():
    print("NaN values found in training loss!")
if np.isnan(history.history['val_loss']).any():
    print("NaN values found in validation loss!")

# Save model
model.save("lstm-hand-punch.h5")