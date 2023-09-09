#%%
import google.auth
from google.oauth2 import service_account
from googleapiclient.discovery import build
import csv
import time
from googleapiclient.errors import HttpError
#%%
import pandas as pd
import numpy as np
#%%
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
#%%
import matplotlib.pyplot as plt
#%%
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

from keras.models import load_model
#%%
import tensorflow as tf
#%%
import os
import datetime as dt
from datetime import date,datetime
import datetime as dt
import glob


#%%
import gspread

#%%
# Replace with the path to your JSON key file

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def sort_dict_by_value_desc(d):
    sorted_dict = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    return sorted_dict

json_key_path = 'mlso-395416-0f858298bf47.json'
#%%


# Replace with the shared file URL
file_urls= ['https://docs.google.com/spreadsheets/d/1t5OiXmflNA__3J6uHqoqtcc14BEEDcDYRRKaEgHL3uo/edit#gid=1354523557']
#%%
# Extract the file ID from the URL
file_ids=[]
for file_url in file_urls:

    file_id = file_url.split('/')[-2]
    file_ids.append(file_id)
#%%
# Authenticate using the JSON key file
creds = service_account.Credentials.from_service_account_file(json_key_path, scopes=['https://www.googleapis.com/auth/drive'])

# Create a Drive API client
service = build('sheets', 'v4', credentials=creds)






#%%
# Export the Google Sheets file as a CSV file
i=1
for file_id in file_ids:
    #time.sleep(1000)
    sheet_metadata = service.spreadsheets().get(spreadsheetId=file_id).execute()
    sheets = sheet_metadata.get('sheets', [])
    for sheet in sheets:
        #time.sleep(1000)
        sheet_name = sheet['properties']['title']
        range_name = f"{sheet_name}!A1:ZZZ"
        result = service.spreadsheets().values().get(spreadsheetId=file_id, range=range_name).execute()
        data = result.get('values', [])
        if data:
            # Save the data to a CSV file with the sheet name
            with open(f'NewStocks/{sheet_name}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data)

#%%
csv_files  = glob.glob('NewStocks/*.csv')
stocks = [file.split('\\')[-1].split('.')[0] for file in csv_files]
stocks_rank={}


#%%xx
for stock_file in stocks:

    stock = pd.read_csv(f'NewStocks/{stock_file}.csv', names = range(16))
    stock = stock.iloc[:,:6]
    stock.columns = stock.iloc[0]
    stock_data = stock[1:]
    stock_data = stock_data.reset_index(drop=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['Date'] = stock_data['Date'].dt.date
    #column_names = ['Date','Open', 'High', 'Low', 'Close', 'Adj Close']
    ##stock = pd.read_csv(f'stocks_updated/{stock_file}.csv', parse_dates=['Date'],names=['Date','Open', 'High', 'Low', 'Close', 'Adj Close'])
    #stock_data = stock[['Date','Open', 'High', 'Low', 'Close', 'Adj Close']]

    # Fill missing values with the mean
    #stock_data.fillna(stock_data.mean(), inplace=True)

    # Extract the target variable and scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close']].values.reshape(-1, 1))

    size=int(len(scaled_data) * 0.1)
    test_data = scaled_data[0:size, :]
    time_step = 60
    X_test, y_test = create_dataset(test_data, time_step)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model_filename = "models/"+stock_file+".h5"


    model = load_model(model_filename)
    # Predict the stock prices on the testing data
    y_pred = model.predict(X_test)

    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    # Plot the predicted and actual stock prices
    diff = y_test - y_pred
    percent_diff = np.divide(diff, y_test, out=np.zeros_like(diff), where=y_test != 0)
    mape = np.mean(np.abs(percent_diff)) * 100

    stock_return = (y_pred[-1] - y_pred[0]) / y_pred[0] * 100
    growth_rate = (y_pred[-1] / y_pred[0]) ** (1 / len(y_pred)) - 1

    daily_growth_rate = []
    stock_return_next_day = []
    date_list=[]
    last_date = stock_data.iloc[0, 0]
    for i in range(len(y_pred)):
        next_date = last_date + pd.DateOffset(days=1)
        next_date = datetime.strptime(str(next_date), '%Y-%m-%d %H:%M:%S')
        next_date = next_date.date()
        date_list.append(next_date)
        daily_growth_rate.append((y_pred[i] / y_pred[i - 1]) - 1)
        stock_return_next_day.append(y_pred[i] - y_pred[i-1])
        # print("Day {}: Date: {}, Daily Growth Rate: {:.2f}%, Daily Stock Return: {:.2f}".format(i + 1, next_date,
        #                                                                                          daily_growth_rate[i][0] * 100,stock_return_next_day[i][0]))

        last_date = next_date

    # Get the index of today's date in the date list

    today_date = dt.date.today()
    tomorrow_date = today_date + dt.timedelta(days=1)

    # Get the index of today's date in the date list
    today_index = date_list.index(tomorrow_date)

    # Get the daily growth rate and daily stock return for today's date
    daily_growth_rate_today = daily_growth_rate[today_index][0]
    stock_return_today = stock_return_next_day[today_index][0]
    stocks_rank[stock_file]=daily_growth_rate_today*stock_return_today
    # Print the daily growth rate and daily stock return for today's date
    print("Today's Date: {}, Daily Growth Rate: {:.2f}%, Daily Stock Return: ${:.2f}".format(today_date,
                                                                                             daily_growth_rate_today * 100,
                                                                                             stock_return_today))

#%%
sorted_stocks_rank = sort_dict_by_value_desc(stocks_rank)

for stock_name, rank in sorted_stocks_rank.items():
    print(f'{stock_name}: {rank}')

df = pd.DataFrame(list(sorted_stocks_rank.items()), columns=['Stock', 'Rank'])

# Save the sorted data to an Excel file
tomorrow_date_str = tomorrow_date.strftime('%Y-%m-%d')

excel_file_name = f'sorted_stocks_rank{tomorrow_date_str}.xlsx'
df.to_excel(excel_file_name, index=False)

print(f"The sorted data has been saved to the Excel file '{excel_file_name}'")


################################################################ Google Sheet ###############################################
#%%

file_id='1HrvcR6GnIA_g0ZidF5zcpxSNQqHa8OxUb1NEdWfwTfU'

new_sheet_name = tomorrow_date_str

# Authenticate with Google using service account credentials
creds = service_account.Credentials.from_service_account_file(json_key_path, scopes=['https://www.googleapis.com/auth/drive'])

# Create a Sheets API client
sheets_service = build('sheets', 'v4', credentials=creds)

# Create a request to add a new sheet to the Google Sheets file
request = {
    'addSheet': {
        'properties': {
            'title': new_sheet_name
        }
    }
}

# Execute the request to add the new sheet to the Google Sheets file
try:
    response = sheets_service.spreadsheets().batchUpdate(spreadsheetId=file_id, body={'requests': [request]}).execute()
    print(f"Sheet '{new_sheet_name}' added to Google Sheets file with ID '{file_id}'")
except HttpError as e:
    print(f"An error occurred: {e}")










