import argparse
from train_with_sampling import *
from torch.utils.data import DataLoader
import torch.nn as nn
from helper import *
from my_dataloader import *
import sys

def main(
    epoch: int = 1000,
    k: int = 6,
    training_length = 30,    #48   one month is required to predict new prices
    forecast_window = 30,   #24
    test_length = 100,
    test_forecast_window = 10,
    train_csv = "E:/my_ML programs/nikita/practice 4/monthly_train.csv",
    test_csv = "E:/my_ML programs/nikita/practice 4/temp.csv",   #no need
    path_to_save_model = "save_model/",
    path_to_save_loss = "save_loss/", 
    path_to_save_predictions = "save_predictions/", 

    path_to_save_model_open = "save_model_open/",
    path_to_save_model_high = "save_model_high/",
    path_to_save_model_low = "save_model_low/",
    path_to_save_model_volume = "save_model_volume/",
    path_to_save_predictions_open = "save_predictions_open/",
    path_to_save_predictions_high = "save_predictions_high/",
    path_to_save_predictions_low = "save_predictions_low/",
    path_to_save_predictions_volume = "save_predictions_volume/",    
    path_to_save_loss_open = "save_loss_open/",
    path_to_save_loss_high = "save_loss_high/",  
    path_to_save_loss_low = "save_loss_low/", 
    path_to_save_loss_volume = "save_loss_volume/", 

    device = "cpu"
):

    clean_directory()  #in helpers.py

    # train_dataset = niftyDataset(csv_name = train_csv, root_dir = "", training_length = training_length, forecast_window = forecast_window)
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # test_dataset = SensorDataset(csv_name = test_csv, root_dir = "", training_length = test_length, forecast_window = test_forecast_window)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # with open("output.txt","a") as sys.stdout:
    #     print("Data has been preprocessed and now sending those to transformer() in train_with_sampling.py")

    best_model = transformer(train_csv, training_length, forecast_window, epoch, k, path_to_save_model, path_to_save_loss, path_to_save_predictions, device)
    #inference(path_to_save_predictions, forecast_window, test_dataloader, device, path_to_save_model, best_model)


    # best_model_for_open = transformer_ohlv(train_csv, training_length, forecast_window, epoch, k, path_to_save_model_open, path_to_save_loss_open, path_to_save_predictions_open, device,"Open")
    #inference_ohlv(path_to_save_predictions_open, forecast_window, test_dataloader, device, path_to_save_model_open, best_model_for_open,"Open")
    
    # best_model_for_high = transformer_ohlv(train_csv, training_length, forecast_window, epoch, k, path_to_save_model_high, path_to_save_loss_high, path_to_save_predictions_high, device,"High")
    #inference_ohlv(path_to_save_predictions_high, forecast_window, test_dataloader, device, path_to_save_model_high, best_model_for_high,"High")

    # best_model_for_volume = transformer_ohlv(train_csv, training_length, forecast_window, epoch, k, path_to_save_model_volume, path_to_save_loss_volume, path_to_save_predictions_volume, device,"Volume")
    #inference_ohlv(path_to_save_predictions_volume, forecast_window, test_dataloader, device, path_to_save_model_volume, best_model_for_volume,"Volume")
    
    # best_model_for_low = transformer_ohlv(train_csv, training_length, forecast_window, epoch, k, path_to_save_model_low, path_to_save_loss_low,path_to_save_predictions_low, device,"Low")
    #inference_ohlv(path_to_save_predictions_low, forecast_window, test_dataloader, device, path_to_save_model_low, best_model_for_low,"Low")
    
    #forecast 30 steps
    #predict_future(best_model,best_model_for_open,best_model_for_high,best_model_for_low,best_model_for_volume,500,"save_prediction_forecast/",device,path_to_save_model,path_to_save_model_open,path_to_save_model_high,path_to_save_model_low,path_to_save_model_volume)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frequency", type=int, default=100)
    parser.add_argument("--path_to_save_model",type=str,default="save_model/")
    parser.add_argument("--path_to_save_loss",type=str,default="save_loss/")
    parser.add_argument("--path_to_save_predictions",type=str,default="save_predictions/")

    parser.add_argument("--path_to_save_model_open",type=str,default="save_model_open/")
    parser.add_argument("--path_to_save_loss_open",type=str,default="save_loss_open/")
    parser.add_argument("--path_to_save_predictions_open",type=str,default="save_predictions_open/")

    parser.add_argument("--path_to_save_model_high",type=str,default="save_model_high/")
    parser.add_argument("--path_to_save_loss_high",type=str,default="save_loss_high/")
    parser.add_argument("--path_to_save_predictions_high",type=str,default="save_predictions_high/")

    parser.add_argument("--path_to_save_model_low",type=str,default="save_model_low/")
    parser.add_argument("--path_to_save_loss_low",type=str,default="save_loss_low/")
    parser.add_argument("--path_to_save_predictions_low",type=str,default="save_predictions_low/")

    parser.add_argument("--path_to_save_model_volume",type=str,default="save_model_volume/")
    parser.add_argument("--path_to_save_loss_volume",type=str,default="save_loss_volume/")
    parser.add_argument("--path_to_save_predictions_volume",type=str,default="save_predictions_volume/")

    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(
        epoch=args.epoch,
        k = args.k,
        path_to_save_model=args.path_to_save_model,
        path_to_save_loss=args.path_to_save_loss,
        path_to_save_predictions=args.path_to_save_predictions,

        path_to_save_model_open=args.path_to_save_model_open,
        path_to_save_loss_open=args.path_to_save_loss_open,
        path_to_save_predictions_open=args.path_to_save_predictions_open,

        path_to_save_model_high=args.path_to_save_model_high,
        path_to_save_loss_high=args.path_to_save_loss_high,
        path_to_save_predictions_high=args.path_to_save_predictions_high,

        path_to_save_model_low=args.path_to_save_model_low,
        path_to_save_loss_low=args.path_to_save_loss_low,
        path_to_save_predictions_low=args.path_to_save_predictions_low,

        path_to_save_model_volume=args.path_to_save_model_volume,
        path_to_save_loss_volume=args.path_to_save_loss_volume,
        path_to_save_predictions_volume=args.path_to_save_predictions_volume,

        device=args.device,
    )


#ohlc last training loss : 0.08941861793252309
#inference Loss On Unseen Dataset: 0.00117315166822706

#on not /10 value
#  Training loss: 0.08673433652225876
# inference Loss On Unseen Dataset: 0.0074599722437802135

#ohlcv
# Training loss: 0.11362474944298641
# inference Loss On Unseen Dataset: 0.006627966120116341

# import torch
# a = torch.tensor([ [[1,2,3]], [[3,4,5]], [[6,7,8]] ]) 
# for i in range(len(a)):
#     c = a[i][0][0]
#     d=c.item()
#     o = a[i][0][1]
#     a[i][0][0] = o
#     a[i][0][1] = d
# print(a)