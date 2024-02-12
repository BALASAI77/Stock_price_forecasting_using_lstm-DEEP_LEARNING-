import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

def load_stock_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df_close = df['close']
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(np.array(df_close).reshape(-1, 1))
    return df_scaled, scaler

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data)-time_steps-1):
        X.append(data[i:(i+time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

def build_model(time_steps, features):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, features)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
    return train_rmse, test_rmse

def plot_results(df, train_predict, test_predict):
    plt.plot(df)
    plt.plot(train_predict)
    plt.plot(test_predict)
    plt.legend(['Original data', 'Training predictions', 'Testing predictions'])
    plt.show()

def predict_future(model, data, scaler, num_prediction, time_steps):
    prediction_list = data[-time_steps:]
    for _ in range(num_prediction):
        x_input = prediction_list[-time_steps:].reshape((1, time_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        prediction_list = np.append(prediction_list, yhat)
    prediction_list = prediction_list[time_steps-1:]
    predicted_values = scaler.inverse_transform(prediction_list.reshape(-1, 1))
    return predicted_values

# Load data
file_path = 'AAPL.csv'  # Update with your file path
df = load_stock_data(file_path)
df_scaled, scaler = preprocess_data(df)

# Prepare data
time_steps = 100
X, y = create_sequences(df_scaled, time_steps)

# Split data
train_size = int(len(df_scaled) * 0.70)
test_size = len(df_scaled) - train_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build and train model
model = build_model(time_steps, 1)
model = train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=60)

# Evaluate model
train_rmse, test_rmse = evaluate_model(model, X_train, y_train, X_test, y_test, scaler)
print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

# Plot results
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
plot_results(df_scaled, train_predict, test_predict)

# Predict future values
num_prediction = 20
future_predictions = predict_future(model, df_scaled, scaler, num_prediction, time_steps)
print(f"Future Predictions: {future_predictions}")

