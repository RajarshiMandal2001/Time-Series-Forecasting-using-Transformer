from cProfile import label
import torch,matplotlib
import matplotlib.pyplot as plt
from model import Transformer
from joblib import load
from joblib import dump
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import datetime
from imp import reload
from math import isnan
from tqdm import tqdm


def get_pad_mask(self, x, pad_idx):
    """ x: (batch_size, seq_len)
    """
    x = (x != pad_idx).unsqueeze(-2)  # (batch_size, 1, seq_len)

    # x: (batch_size, 1, seq_len) 
    return x

def get_subsequent_mask(x):
    sz = len(x)
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    print("mask:",mask,"shape:",mask.shape)
    return mask

def demo_future_prediction(src_file,device):
    device = torch.device(device)

    df1 = pd.read_csv(src_file)
    _input = torch.tensor(df1[["Close","Open","High","Low","Volume","sin_day", "cos_day", "sin_month", "cos_month"]].values)
    close = df1["Close"].tolist()
    print("input  shape:",_input.shape)
    df2 = pd.read_csv("E:/my_ML programs/nikita/practice 4/daily_model_close/test_target.csv")
    true = df2["Close"].tolist()

    scaler = MinMaxScaler()
    scaler.fit(_input[:,0].unsqueeze(-1))
    _input[:,0] = torch.tensor(scaler.transform(_input[:,0].unsqueeze(-1)).squeeze(-1)) 
    dump(scaler, 'scalar_item_forecast_close.joblib')
    scaler = MinMaxScaler()
    scaler.fit(_input[:,1].unsqueeze(-1))
    _input[:,1] = torch.tensor(scaler.transform(_input[:,1].unsqueeze(-1)).squeeze(-1))  
    dump(scaler, 'scalar_item_forecast_open.joblib')
    scaler = MinMaxScaler()
    scaler.fit(_input[:,2].unsqueeze(-1))
    _input[:,2] = torch.tensor(scaler.transform(_input[:,2].unsqueeze(-1)).squeeze(-1))  
    dump(scaler, 'scalar_item_forecast_high.joblib')
    scaler = MinMaxScaler()
    scaler.fit(_input[:,3].unsqueeze(-1))
    _input[:,3] = torch.tensor(scaler.transform(_input[:,3].unsqueeze(-1)).squeeze(-1)) 
    dump(scaler, 'scalar_item_forecast_low.joblib')
    scaler = MinMaxScaler()
    scaler.fit(_input[:,4].unsqueeze(-1))
    _input[:,4] = torch.tensor(scaler.transform(_input[:,4].unsqueeze(-1)).squeeze(-1)) 
    dump(scaler, 'scalar_item_forecast_volume.joblib')

    _input = _input.view(len(_input),1,-1)
    # _input = _input.permute(1,0,2).double().to(device)
    
    model_close = Transformer().double().to(device)
    model_close.load_state_dict(torch.load("E:/my_ML programs/nikita/practice 4/daily_model_close/best_train_455.pth"))
#*******************************************************************************************************************************
    with torch.no_grad():
        print("shape of input to model:",_input.shape)
        enc_output = model_close.encoder_for_test(_input,device)  # (1, source_seq_len, d_model)
    
    target_start = [_input[-1:,:,:].tolist()]   #the last element of input
    target_start = torch.tensor(target_start)
    target_start = target_start.squeeze(0)
    print("initiial target:",target_start)
    model_close.eval()
    for i in range(1,_input.shape[0]+1):   #30
        print("i=",i)
        # target = torch.tensor(target_start).unsqueeze(0).to(device)  # (1, target_seq_len)
        # target = torch.tensor(target_start).view(1,1,_input.shape[2]).to(device)  # (1, target_seq_len)
        target = torch.tensor(target_start).view(-1,1,_input.shape[2]).to(device) 
        print("(target)",torch.tensor(target),"shape:",target.shape)
        # s=(1,_input.shape[1]-1,_input.shape[2])   #(1,29,9)
        # y = torch.tensor(np.ones(s))
        # z=torch.cat((target,y),dim=1)
        # target_mask = get_pad_mask(target, PAD_IDX) & get_subsequent_mask(target)
        target_mask = get_subsequent_mask(target)
        # decode the sequence
        with torch.no_grad():
            # target = target.float().to(device)
            # output = model_close.decoder_for_test(target, target_mask, enc_output, device, _input) 
            output = model_close.decoder_for_test(target[:i,:,:], target_mask, enc_output, device) 
            print("output=",output,"shape:",output.shape)
        # target_id = output.argmax(dim=-1)[:, -1].item()
        # target_start.append(target_id)
        new_output_generated = output[-1,:,:]
        new_output_generated = new_output_generated.view(1,1,_input.shape[2])
        target_start = torch.cat((target_start,new_output_generated))
        print("target_start=",target_start)
    print("final output from decoder=",target_start)
    print("target close:",true)
    scaler = load('scalar_item_forecast_close.joblib')
    prediction_price = scaler.inverse_transform(output[:,:,0].detach().cpu().numpy())
    print("predicted close:",prediction_price)
    prediction_price = torch.tensor(prediction_price).view(len(prediction_price))
    total = close + true
    plt.plot(total, color="green",label="true results",linewidth=2)
    plt.plot(close,color="blue",label="input",linewidth=3)
    total = close + prediction_price.tolist()
    plt.plot(total,color="red",label="forecast",linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.savefig("save_prediction_forecast/"+"forecast.png")
    plt.legend()
    plt.title("Forecast from given input")
    plt.close('all')


#*********************call the prediction method with input dataset************************************
demo_future_prediction("E:/my_ML programs/nikita/practice 4/daily_model_close/test_input.csv","cpu")


 


