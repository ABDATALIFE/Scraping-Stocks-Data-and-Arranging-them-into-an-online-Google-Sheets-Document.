
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf
import os
import datetime as dt
from datetime import date,datetime
import glob
from sklearn.metrics import mean_squared_error, r2_score
import os


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

xlsx_files = glob.glob('stocks/*.xlsx')
columns_to_extract = ['Open','High', 'Low', 'Close', 'Adj Close']

dfs = []


for xlsx_file in xlsx_files:
    print(xlsx_file)
    # Load the XLSX file into a dictionary of dataframes
    xl_dict = pd.read_excel(xlsx_file, sheet_name=None, parse_dates=True)
    # Loop through each worksheet in the dictionary and append it to the list of dataframes
    for sheet_name, df in xl_dict.items():
        if sheet_name != 'Profile':
            df['sheet_name'] = sheet_name
            dfs.append(df)

df = pd.DataFrame(columns=['Stock', 'Error'])

for stock_data_ in dfs:

    stock=stock_data_['sheet_name'][0]

    stock_data=stock_data_[columns_to_extract]
    # Fill missing values with the mean
    stock_data.fillna(stock_data.mean(), inplace=True)

    # Extract the target variable and scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data.values.reshape(-1, 1))

    test_data = scaled_data
    time_step = 60
    X_test, y_test = create_dataset(test_data, time_step)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model_filename = "models/"+stock+".h5"

    try:
        model = load_model(model_filename)
        # Predict the stock prices on the testing data
        y_pred = model.predict(X_test)

        y_pred = scaler.inverse_transform(y_pred)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        # Plot the predicted and actual stock prices
        diff = y_test - y_pred
        percent_diff = np.divide(diff, y_test, out=np.zeros_like(diff), where=y_test != 0)
        mape = np.mean(np.abs(percent_diff)) * 100
        print("{}:Mean Absolute Percentage Error (MAPE): {:.2f}%".format(stock,mape))

        stock_return = (y_pred[-1] - y_pred[0]) / y_pred[0] * 100
        growth_rate = (y_pred[-1] / y_pred[0]) ** (1 / len(y_pred)) - 1

        print("Stock Return: {:.2f}%".format(stock_return[0]))
        print("Growth Rate: {:.2f}%".format(growth_rate[0] * 100))

        data = {'Stock': stock, 'Error': float(mape)}
        df = df.append(data, ignore_index=True)


    except Exception:
        pass

    # plt.plot(y_test[:, 3], label='Actual Values')  # Use second column for Close values
    # plt.plot(y_pred, color='red', label='Predicted Values')
    # plt.title('Prediction vs Actual')
    # plt.xlabel('Time Period')
    # plt.ylabel('Close Value')
    # plt.legend()
    # plt.show()

writer = pd.ExcelWriter('error_stocks.xlsx', engine='xlsxwriter')

# Write the dataframe to the Excel file
df.to_excel(writer, sheet_name='Stocks', index=False)


writer.save()