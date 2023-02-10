import torch
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import dump


def next_window(df,start,training_length,forecast_length):

    _input = torch.tensor(df[["Close","Open","High","Low","Volume", "sin_day", "cos_day", "sin_month", "cos_month"]][start : start + training_length].values)
    target = torch.tensor(df[["Close","Open","High","Low","Volume", "sin_day", "cos_day", "sin_month", "cos_month"]][start + training_length-1 : start + training_length + forecast_length-1].values) 
    comparing_target = torch.tensor(df[["Close","Open","High","Low","Volume", "sin_day", "cos_day", "sin_month", "cos_month"]][start + training_length : start + training_length + forecast_length].values) 
    
    scaler = MinMaxScaler()

    scaler.fit(_input[:,0].unsqueeze(-1))
    _input[:,0] = torch.tensor(scaler.transform(_input[:,0].unsqueeze(-1)).squeeze(-1))  #[:,0] picks up all 0th elements of the nested lists,humidity here
    target[:,0] = torch.tensor(scaler.transform(target[:,0].unsqueeze(-1)).squeeze(-1))
    comparing_target[:,0] = torch.tensor(scaler.transform(target[:,0].unsqueeze(-1)).squeeze(-1))
    dump(scaler, 'scalar_item_close.joblib')

        
    scaler.fit(_input[:,1].unsqueeze(-1))
    _input[:,1] = torch.tensor(scaler.transform(_input[:,1].unsqueeze(-1)).squeeze(-1))  #[:,0] picks up all 0th elements of the nested lists,humidity here
    target[:,1] = torch.tensor(scaler.transform(target[:,1].unsqueeze(-1)).squeeze(-1))
    comparing_target[:,1] = torch.tensor(scaler.transform(target[:,1].unsqueeze(-1)).squeeze(-1))
    dump(scaler, 'scalar_item_open.joblib')

    scaler.fit(_input[:,2].unsqueeze(-1))
    _input[:,2] = torch.tensor(scaler.transform(_input[:,2].unsqueeze(-1)).squeeze(-1))  #[:,0] picks up all 0th elements of the nested lists,humidity here
    target[:,2] = torch.tensor(scaler.transform(target[:,2].unsqueeze(-1)).squeeze(-1))
    comparing_target[:,2] = torch.tensor(scaler.transform(target[:,2].unsqueeze(-1)).squeeze(-1))
    dump(scaler, 'scalar_item_high.joblib')

    scaler.fit(_input[:,3].unsqueeze(-1))
    _input[:,3] = torch.tensor(scaler.transform(_input[:,3].unsqueeze(-1)).squeeze(-1))  #[:,0] picks up all 0th elements of the nested lists,humidity here
    target[:,3] = torch.tensor(scaler.transform(target[:,3].unsqueeze(-1)).squeeze(-1))
    comparing_target[:,3] = torch.tensor(scaler.transform(target[:,3].unsqueeze(-1)).squeeze(-1))
    dump(scaler, 'scalar_item_low.joblib')

    scaler.fit(_input[:,4].unsqueeze(-1))
    _input[:,4] = torch.tensor(scaler.transform(_input[:,4].unsqueeze(-1)).squeeze(-1))  
    target[:,4] = torch.tensor(scaler.transform(target[:,4].unsqueeze(-1)).squeeze(-1))
    comparing_target[:,4] = torch.tensor(scaler.transform(target[:,4].unsqueeze(-1)).squeeze(-1))
    #save the scalar to be used later when inverse translating the data for plotting.
    dump(scaler, 'scalar_item_volume.joblib')
    return 1,1, _input, target, comparing_target, 1  #1 is index_in, index_, placeholder for sensor_number