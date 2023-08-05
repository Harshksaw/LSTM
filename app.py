import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st
# from statsmodels.tsa.seasonal import seasonal_decompose, plot_components


today = date.today()
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=2000)
d2.strftime("%Y-%m-%d")
start_date = d2

data = yf.download('MSFT', start=start_date, end=end_date, progress=False)

# Importing Data and Plotting
user_input = st.text_input('Enter Stock Ticker', 'MSFT')

df = yf.download(user_input, start=start_date, end=end_date, progress=False)

st.subheader(f"Data from {start_date} - {end_date}")

st.write(df.describe())

#visualizations

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

# st.subheader('Closing Price')
# st.subheader('Closing Price vs Time chart with 100MA*')'

# ma100 = df.Close.rolling(100).mean

# fig = plt.figure(figsize = (12,6))

# plt.plot(ma100)

# plt.plot(df.Close)
# st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA* & 200 MA') 

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

fig = plt.figure(figsize = (12,6))

plt.plot (ma100 , 'g')
plt.plot (ma200,'y')

plt.plot (df.Close  , 'b')
st.pyplot(fig)

#training ans testing data

# Function to get the data
def get_data():
    today = date.today()
    end_date = today.strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=2000)).strftime("%Y-%m-%d")
    data = yf.download('MSFT', start=start_date, end=end_date, progress=False)
    return data

# Function to preprocess the data
def preprocess_data(data):
    df = data.copy()
    df.reset_index(inplace=True)
    df["Date"] = df['Date'].dt.date
    df.set_index('Date', inplace=True)
    return df

# Function to get the seasonal decomposition
def get_seasonal_decomposition(data):
    results = seasonal_decompose(data['Close'], model='multiplicative', period=12)
    return results

# Function to plot the data and decomposition
def plot_data_decomposition(data, decomposition):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(data['Close'])
    ax[0].set_title('Closing Price of Stock')
    decomposition.plot(ax=ax[1])
    st.pyplot(fig)

# Function to create LSTM model and make predictions
def predict_next_days(data, time_steps=100, num_days=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(np.array(data['Close']).reshape(-1, 1))

    X_test = df[-time_steps:].reshape(1, -1)

    model = load_model('yourmodel.keras')

    predictions = []
    for _ in range(num_days):
        X_test = X_test.reshape(1, time_steps, 1)
        yhat = model.predict(X_test, verbose=0)
        X_test = np.concatenate((X_test[:, 1:, :], yhat), axis=1)
        predictions.append(scaler.inverse_transform(yhat)[0][0])

    return predictions

def main():
    # Getting data and preprocessing
    data = get_data()
    df = preprocess_data(data)

    # Show data summary
    # st.subheader(f"Data from {df.index[0]} - {df.index[-1]}")
    # st.write(df.describe())

    # Plotting closing price and seasonal decomposition
    plot_data_decomposition(df, get_seasonal_decomposition(df))

    # Predict future prices
    # st.subheader("Predicted Closing Prices for Next 30 Days")
    # predictions = predict_next_days(df)
    # st.write(pd.DataFrame(predictions, columns=['Predicted Closing Price']))

    # Plotting the predictions
    # Function to plot the data and decomposition
def plot_data_decomposition(data, decomposition):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(data['Close'])
    ax[0].set_title('Closing Price of Stock')
    decomposition.trend.plot(ax=ax[1])
    ax[1].set_title('Trend')
    decomposition.seasonal.plot(ax=ax[1])
    ax[1].set_title('Seasonal')
    decomposition.resid.plot(ax=ax[1])
    ax[1].set_title('Residual')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
