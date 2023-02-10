import pandas as pd
import time
import numpy as np
import datetime

# encoding the timestamp data cyclically. See Medium Article.
def process_data(source):

    df = pd.read_csv(source)
    timestamps = df["Date"].tolist()
    # timestamps_day = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d').day) for t in timestamps])
    # timestamps_month = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d').month) for t in timestamps])
    # timestamps_day = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').day) for t in timestamps])
    # timestamps_month = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').month) for t in timestamps])
    try:
        timestamps_day = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d').day) for t in timestamps])
        timestamps_month = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d').month) for t in timestamps])
    except:
        timestamp = np.array([datetime.datetime.strptime(t, '%d-%m-%Y').strftime('%Y-%m-%d') for t in timestamps])
        timestamps_day = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d').day) for t in timestamp])
        timestamps_month = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d').month) for t in timestamp])

    hours_in_day = 24
    days_in_month = 30
    month_in_year = 12

    # df['sin_hour'] = np.sin(2*np.pi*timestamps_hour/hours_in_day)
    # df['cos_hour'] = np.cos(2*np.pi*timestamps_hour/hours_in_day)
    df['sin_day'] = np.sin(2*np.pi*timestamps_day/days_in_month)
    df['cos_day'] = np.cos(2*np.pi*timestamps_day/days_in_month)
    df['sin_month'] = np.sin(2*np.pi*timestamps_month/month_in_year)
    df['cos_month'] = np.cos(2*np.pi*timestamps_month/month_in_year)

    return df

# train_dataset = process_data('E:/my_ML programs/nikita/practice 4/monthly_train.csv')
# print("train dataset prepared")
test_dataset = process_data('E:/my_ML programs/nikita/practice 4/weekly_test.csv')
print("test dataset prepared")
# train_dataset.to_csv(r'E:/my_ML programs/nikita/practice/stock indices/NIFTY 50_dataset.csv', index=False)
test_dataset.to_csv(r'weekly_test_dataset.csv', index=False)

# train_dataset.to_csv(r'nifty_train_dataset.csv', index=False)
# test_dataset.to_csv(r'nifty_test_dataset.csv', index=False)