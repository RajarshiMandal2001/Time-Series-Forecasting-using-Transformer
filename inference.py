from model import Transformer
from plot import *
import torch
import logging
import sys
from joblib import load

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def inference(path_to_save_predictions, forecast_window, dataloader, device, path_to_save_model, best_model):
    # with open("output.txt","a") as sys.stdout:
    #     print("inside inference() in inference.py")
    device = torch.device(device)
    
    model = Transformer().double().to(device)
    model.load_state_dict(torch.load(path_to_save_model+best_model))   #save the model
    criterion = torch.nn.MSELoss()

    val_loss = 0
    dataloader_counter = 0
    with torch.no_grad():

        model.eval()   #testing mode on
        for plot in range(25):

            for index_in, index_tar, _input, target, sensor_number in dataloader:

                dataloader_counter = dataloader_counter + 1
                # starting from 1 so that src matches with target, but has same length as when training
                src = _input.permute(1,0,2).double().to(device)[1:, :, :] # 47, 1, 7: t1 -- t47
                target = target.permute(1,0,2).double().to(device) # t48 - t59

                # with open("output.txt","a") as sys.stdout:
                #     print("src (last 3 values) with shape:",src.shape)
                #     print(src[-3:])
                #     print("target (last 3 values) with shape:",target.shape)
                #     print(target[-3:])

                next_input_for_model = src
                all_predictions = []

                for i in range(forecast_window - 1):
                    
                    prediction = model(next_input_for_model, device, dataloader_counter) # 47,1,1: t2' - t48'

                    # with open("output.txt","a") as sys.stdout:
                    #     print("input to transformer (last 2 values) with shape:",next_input_for_model.shape)
                    #     print(next_input_for_model[-2:])
                    #     print("transformer output (last 2 values) with shape:",prediction.shape)

                    if all_predictions == []:
                        all_predictions = prediction # 47,1,1: t2' - t48'
                    else:
                        all_predictions = torch.cat((all_predictions, prediction[-1 ,:,:].unsqueeze(0))) # 47+,1,1: t2' - t48', t49', t50'

                    pos_encoding_old_vals = src[i+1: , :, 1:] # 46, 1, 6, pop positional encoding first value: t2 -- t47
                    pos_encoding_new_val = target[i+1, :, 1:].unsqueeze(1) # 1, 1, 6, append positional encoding of last predicted value: t48
                    pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val)) # 47, 1, 6 positional encodings matched with prediction: t2 -- t48
                    
                    next_input_for_model = torch.cat((src[i+1:, :, 0].unsqueeze(-1), prediction[-1 ,:,:].unsqueeze(0))) #t2 -- t47, t48'
                    next_input_for_model = torch.cat((next_input_for_model, pos_encodings), dim = 2) # making input dimention = 47, 1, 7  for next round
                    
                    # with open("output.txt","a") as sys.stdout:
                    #     print("positional_encoding_old_vals (last 2 values) with shape:",pos_encoding_old_vals.shape)
                    #     print(pos_encoding_old_vals[-2:])
                    #     print("positional_encoding_new_val (last 2 values) with shape:",pos_encoding_new_val.shape)
                    #     print(pos_encoding_new_val[-2:])
                    #     print("pos_encoding (last 2 values) with shape:",pos_encodings.shape)
                    #     print(pos_encodings[-2:])
                    #     print("all_predictions (last 2 values) with shape:",all_predictions.shape)
                    #     print(all_predictions[-2:])
                    #     print("..............................................................")
                        
                true = torch.cat((src[1:,:,0],target[:-1,:,0]))
                
                # with open("output.txt","a") as sys.stdout:
                #     print("src[1:,:,0], target[:-1,:,0] are concatinated and sent to loss function")
                #     print("src[1:,:,0] :")
                #     print(src[1:,:,0])
                #     print("target[:-1,:,0]")
                #     print(target[:-1,:,0])
                
                loss = criterion(true, all_predictions[:,:,0])
                
                # with open("output.txt","a") as sys.stdout:
                #     print(f"(first 3) MSE loss between {true} and {all_predictions[:,:,0]} = {loss}")

                val_loss += loss
            
            val_loss = val_loss/10

            target_price = target[:,:,0].cpu()
            src_price = src[:,:,0].cpu()
            prediction_price = all_predictions[:,:,0].detach().cpu().numpy()

            # scaler = load('scalar_item_close.joblib')
            # src_price = scaler.inverse_transform(src[:,:,0].cpu())
            # target_price = scaler.inverse_transform(target[:,:,0].cpu())
            # prediction_price = scaler.inverse_transform(all_predictions[:,:,0].detach().cpu().numpy())

            plot_prediction(plot, path_to_save_predictions, src_price, target_price, prediction_price, sensor_number, index_in, index_tar,"Close")

        logger.info(f"Loss On Unseen Dataset: {val_loss.item()}")

#********************************************************************************************************************************************


def inference_ohlv(path_to_save_predictions, forecast_window, dataloader, device, path_to_save_model, best_model, feature):

    if feature == "Open":
        device = torch.device(device)
        
        model = Transformer().double().to(device)
        model.load_state_dict(torch.load(path_to_save_model + best_model))   
        criterion = torch.nn.MSELoss()

        val_loss = 0
        dataloader_counter = 0
        with torch.no_grad():

            model.eval()   #testing mode on
            for plot in range(25):

                for index_in, index_tar, _input, target, sensor_number in dataloader:

                    dataloader_counter = dataloader_counter + 1
                    # starting from 1 so that src matches with target, but has same length as when training
                    src = _input.permute(1,0,2).double().to(device)[1:, :, :] # 47, 1, 7: t1 -- t47
                    target = target.permute(1,0,2).double().to(device) # t48 - t59
                    #.............................................
                    for i in range(len(src)):
                        c = src[i][0][0]
                        d = c.item()
                        o = src[i][0][1]
                        src[i][0][0] = o
                        src[i][0][1] = d
                    for i in range(len(target)):
                        c = target[i][0][0]
                        d = c.item()
                        o = target[i][0][1]
                        target[i][0][0] = o
                        target[i][0][1] = d
                    #..............................................

                    next_input_for_model = src
                    all_predictions = []

                    for i in range(forecast_window - 1):
                        
                        prediction = model(next_input_for_model, device, dataloader_counter) 

                        if all_predictions == []:
                            all_predictions = prediction # 47,1,1: t2' - t48'
                        else:
                            all_predictions = torch.cat((all_predictions, prediction[-1 ,:,:].unsqueeze(0))) # 47+,1,1: t2' - t48', t49', t50'

                        pos_encoding_old_vals = src[i+1: , :, 1:] # 46, 1, 6, pop positional encoding first value: t2 -- t47
                        pos_encoding_new_val = target[i+1, :, 1:].unsqueeze(1) # 1, 1, 6, append positional encoding of last predicted value: t48
                        pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val)) # 47, 1, 6 positional encodings matched with prediction: t2 -- t48
                        
                        next_input_for_model = torch.cat((src[i+1:, :, 0].unsqueeze(-1), prediction[-1 ,:,:].unsqueeze(0))) #t2 -- t47, t48'
                        next_input_for_model = torch.cat((next_input_for_model, pos_encodings), dim = 2) # making input dimention = 47, 1, 7  for next round
                        
                            
                    true = torch.cat((src[1:,:,0],target[:-1,:,0]))   # why 0? since open may be comminng at 0th position
                    

                    loss = criterion(true, all_predictions[:,:,0])

                    val_loss += loss
                
                val_loss = val_loss/10
                target_open = target[:,:,0].cpu()
                src_open = src[:,:,0].cpu()
                prediction_open = all_predictions[:,:,0].detach().cpu().numpy()
                # scaler = load('scalar_item_open.joblib')
                # src_open = scaler.inverse_transform(src[:,:,0].cpu())
                # target_open = scaler.inverse_transform(target[:,:,0].cpu())
                # prediction_open = scaler.inverse_transform(all_predictions[:,:,0].detach().cpu().numpy())

                plot_prediction(plot, path_to_save_predictions, src_open, target_open, prediction_open, sensor_number, index_in, index_tar,"Open")

            logger.info(f"Loss On Unseen Dataset: {val_loss.item()}")

    elif feature == "High":

        device = torch.device(device)
        
        model = Transformer().double().to(device)
        model.load_state_dict(torch.load(path_to_save_model + best_model))   
        criterion = torch.nn.MSELoss()

        val_loss = 0
        dataloader_counter = 0
        with torch.no_grad():

            model.eval()   #testing mode on
            for plot in range(25):

                for index_in, index_tar, _input, target, sensor_number in dataloader:

                    dataloader_counter = dataloader_counter + 1
                    # starting from 1 so that src matches with target, but has same length as when training
                    src = _input.permute(1,0,2).double().to(device)[1:, :, :] # 47, 1, 7: t1 -- t47
                    target = target.permute(1,0,2).double().to(device) # t48 - t59
                    #.............................................
                    for i in range(len(src)):
                        c = src[i][0][0]
                        d = c.item()
                        o = src[i][0][2]
                        src[i][0][0] = o
                        src[i][0][2] = d
                    for i in range(len(target)):
                        c = target[i][0][0]
                        d = c.item()
                        o = target[i][0][2]
                        target[i][0][0] = o
                        target[i][0][2] = d
                    #..............................................

                    next_input_for_model = src
                    all_predictions = []

                    for i in range(forecast_window - 1):
                        
                        prediction = model(next_input_for_model, device, dataloader_counter) 

                        if all_predictions == []:
                            all_predictions = prediction # 47,1,1: t2' - t48'
                        else:
                            all_predictions = torch.cat((all_predictions, prediction[-1 ,:,:].unsqueeze(0))) # 47+,1,1: t2' - t48', t49', t50'

                        pos_encoding_old_vals = src[i+1: , :, 1:] # 46, 1, 6, pop positional encoding first value: t2 -- t47
                        pos_encoding_new_val = target[i+1, :, 1:].unsqueeze(1) # 1, 1, 6, append positional encoding of last predicted value: t48
                        pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val)) # 47, 1, 6 positional encodings matched with prediction: t2 -- t48
                        
                        next_input_for_model = torch.cat((src[i+1:, :, 0].unsqueeze(-1), prediction[-1 ,:,:].unsqueeze(0))) #t2 -- t47, t48'
                        next_input_for_model = torch.cat((next_input_for_model, pos_encodings), dim = 2) # making input dimention = 47, 1, 7  for next round
                        
                            
                    true = torch.cat((src[1:,:,0],target[:-1,:,0]))   # why 0? since open may be comminng at 0th position
                    

                    loss = criterion(true, all_predictions[:,:,0])

                    val_loss += loss
                
                val_loss = val_loss/10
                target_high = target[:,:,0].cpu()
                src_high = src[:,:,0].cpu()
                prediction_high = all_predictions[:,:,0].detach().cpu().numpy()

                # scaler = load('scalar_item_high.joblib')
                # src_high = scaler.inverse_transform(src[:,:,0].cpu())
                # target_high = scaler.inverse_transform(target[:,:,0].cpu())
                # prediction_high = scaler.inverse_transform(all_predictions[:,:,0].detach().cpu().numpy())

                plot_prediction(plot, path_to_save_predictions, src_high, target_high, prediction_high, sensor_number, index_in, index_tar,"High")

            logger.info(f"Loss On Unseen Dataset: {val_loss.item()}")

    elif feature == "Low":

        device = torch.device(device)
        
        model = Transformer().double().to(device)
        model.load_state_dict(torch.load(path_to_save_model + best_model))   
        criterion = torch.nn.MSELoss()

        val_loss = 0
        dataloader_counter = 0
        with torch.no_grad():

            model.eval()   #testing mode on
            for plot in range(25):

                for index_in, index_tar, _input, target, sensor_number in dataloader:

                    dataloader_counter = dataloader_counter + 1
                    # starting from 1 so that src matches with target, but has same length as when training
                    src = _input.permute(1,0,2).double().to(device)[1:, :, :] # 47, 1, 7: t1 -- t47
                    target = target.permute(1,0,2).double().to(device) # t48 - t59
                    #.............................................
                    for i in range(len(src)):
                        c = src[i][0][0]
                        d = c.item()
                        o = src[i][0][3]
                        src[i][0][0] = o
                        src[i][0][3] = d
                    for i in range(len(target)):
                        c = target[i][0][0]
                        d = c.item()
                        o = target[i][0][3]
                        target[i][0][0] = o
                        target[i][0][3] = d
                    #..............................................

                    next_input_for_model = src
                    all_predictions = []

                    for i in range(forecast_window - 1):
                        
                        prediction = model(next_input_for_model, device, dataloader_counter) 

                        if all_predictions == []:
                            all_predictions = prediction # 47,1,1: t2' - t48'
                        else:
                            all_predictions = torch.cat((all_predictions, prediction[-1 ,:,:].unsqueeze(0))) # 47+,1,1: t2' - t48', t49', t50'

                        pos_encoding_old_vals = src[i+1: , :, 1:] # 46, 1, 6, pop positional encoding first value: t2 -- t47
                        pos_encoding_new_val = target[i+1, :, 1:].unsqueeze(1) # 1, 1, 6, append positional encoding of last predicted value: t48
                        pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val)) # 47, 1, 6 positional encodings matched with prediction: t2 -- t48
                        
                        next_input_for_model = torch.cat((src[i+1:, :, 0].unsqueeze(-1), prediction[-1 ,:,:].unsqueeze(0))) #t2 -- t47, t48'
                        next_input_for_model = torch.cat((next_input_for_model, pos_encodings), dim = 2) # making input dimention = 47, 1, 7  for next round
                        
                            
                    true = torch.cat((src[1:,:,0],target[:-1,:,0]))   # why 0? since open may be comminng at 0th position
                    

                    loss = criterion(true, all_predictions[:,:,0])

                    val_loss += loss
                
                val_loss = val_loss/10
                target_low = target[:,:,0].cpu()
                src_low = src[:,:,0].cpu()
                prediction_low = all_predictions[:,:,0].detach().cpu().numpy()
                # scaler = load('scalar_item_low.joblib')
                # src_low = scaler.inverse_transform(src[:,:,0].cpu())
                # target_low = scaler.inverse_transform(target[:,:,0].cpu())
                # prediction_low = scaler.inverse_transform(all_predictions[:,:,0].detach().cpu().numpy())

                plot_prediction(plot, path_to_save_predictions, src_low, target_low, prediction_low, sensor_number, index_in, index_tar,"Low")

            logger.info(f"Loss On Unseen Dataset: {val_loss.item()}")

    elif feature == "Volume":

        device = torch.device(device)
        
        model = Transformer().double().to(device)
        model.load_state_dict(torch.load(path_to_save_model + best_model))   
        criterion = torch.nn.MSELoss()

        val_loss = 0
        dataloader_counter = 0
        with torch.no_grad():

            model.eval()   #testing mode on
            for plot in range(25):

                for index_in, index_tar, _input, target, sensor_number in dataloader:

                    dataloader_counter = dataloader_counter + 1
                    # starting from 1 so that src matches with target, but has same length as when training
                    src = _input.permute(1,0,2).double().to(device)[1:, :, :] # 47, 1, 7: t1 -- t47
                    target = target.permute(1,0,2).double().to(device) # t48 - t59
                    #.............................................
                    for i in range(len(src)):
                        c = src[i][0][0]
                        d = c.item()
                        o = src[i][0][4]
                        src[i][0][0] = o
                        src[i][0][4] = d
                    for i in range(len(target)):
                        c = target[i][0][0]
                        d = c.item()
                        o = target[i][0][4]
                        target[i][0][0] = o
                        target[i][0][4] = d
                    #..............................................

                    next_input_for_model = src
                    all_predictions = []

                    for i in range(forecast_window - 1):
                        
                        prediction = model(next_input_for_model, device, dataloader_counter) 

                        if all_predictions == []:
                            all_predictions = prediction # 47,1,1: t2' - t48'
                        else:
                            all_predictions = torch.cat((all_predictions, prediction[-1 ,:,:].unsqueeze(0))) # 47+,1,1: t2' - t48', t49', t50'

                        pos_encoding_old_vals = src[i+1: , :, 1:] # 46, 1, 6, pop positional encoding first value: t2 -- t47
                        pos_encoding_new_val = target[i+1, :, 1:].unsqueeze(1) # 1, 1, 6, append positional encoding of last predicted value: t48
                        pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val)) # 47, 1, 6 positional encodings matched with prediction: t2 -- t48
                        
                        next_input_for_model = torch.cat((src[i+1:, :, 0].unsqueeze(-1), prediction[-1 ,:,:].unsqueeze(0))) #t2 -- t47, t48'
                        next_input_for_model = torch.cat((next_input_for_model, pos_encodings), dim = 2) # making input dimention = 47, 1, 7  for next round
                        
                            
                    true = torch.cat((src[1:,:,0],target[:-1,:,0]))   # why 0? since open may be comminng at 0th position
                    

                    loss = criterion(true, all_predictions[:,:,0])

                    val_loss += loss
                
                val_loss = val_loss/10
                # target_volume = target[:,:,0].cpu()
                # src_volume = src[:,:,0].cpu()
                # prediction_volume = all_predictions[:,:,0].detach().cpu().numpy()
                scaler = load('scalar_item_volume.joblib')
                src_volume = scaler.inverse_transform(src[:,:,0].cpu())
                target_volume = scaler.inverse_transform(target[:,:,0].cpu())
                prediction_volume = scaler.inverse_transform(all_predictions[:,:,0].detach().cpu().numpy())

                plot_prediction(plot, path_to_save_predictions, src_volume, target_volume, prediction_volume, sensor_number, index_in, index_tar,"Volume")

            logger.info(f"Loss On Unseen Dataset: {val_loss.item()}")