from model import Transformer
import torch
import torch.nn as nn
import logging
from joblib import load
import pandas as pd
import math, random
from tqdm import tqdm
from my_dataloader import next_window
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)


def flip_from_probability(p):
    return True if random.random() < p else False

def transformer(dataset, training_length ,forecast_lenfth, EPOCH, k, path_to_save_model, path_to_save_loss, path_to_save_predictions, device):

    device = torch.device(device)
    df = pd.read_csv(dataset)


    # model = Transformer().double().to(device)
    # optimizer = torch.optim.Adam(model.parameters())
    model = Transformer().double().to(device)
    # model.load_state_dict(torch.load("E:/my_ML programs/nikita/practice 4/monthly_model_close/best_train_705.pth"))
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')
    dataloader_counter = 0

    for epoch in range(EPOCH):
        train_loss = 0

        ## TRAIN -- TEACHER FORCING
        model.train()
        # for index_in, index_tar, _input, target, sensor_number in dataloader:
        for l in tqdm(range(len(df)-(training_length+forecast_lenfth))):
            index_in, index_tar, _input, target, comparing_target, sensor_number = next_window(df,l,training_length,forecast_lenfth)
            _input = _input.view(len(_input),1,-1)
            target = target.view(len(target),1,-1)
            comparing_target = target.view(len(comparing_target),1,-1)
            # print("(before permutation) input to model:",_input)
            # print("(before permutation) its target:",target)
            optimizer.zero_grad()

            prediction = model(_input, target ,device) # torch.Size([1xw, 1, 1])  #model returns a single value first then two then three and so on
            # print("predictiopn shape:",prediction.shape)
            # print("prediction:",prediction)
                #if i < 24: # One day, enough data to make inferences about cycles
            #prob_true_val = True
                # else:
                #     ## coin flip
                #     v = k/(k+math.exp(epoch/k)) # probability of heads/tails depends on the epoch, evolves with time.
                #     prob_true_val = flip_from_probability(v) # starts with over 95 % probability of true val for each flip in epoch 0.
                    ## if using true value as new value

            #if prob_true_val: # Using true value as next value, i.e teacher forcing
            #sampled_src = torch.cat((sampled_src.detach(), sampled_src[i+1 , :, :].unsqueeze(0).detach()))  #add the next data in src
                    
                # else:   # using prediction as new value
                #     positional_encodings_new_val = src[i+1,:,1:].unsqueeze(0)   #maybe need to change something here after adding ohl 
                #     predicted_humidity = torch.cat((prediction[-1,:,:].unsqueeze(0), positional_encodings_new_val), dim=2)
                #     sampled_src = torch.cat((sampled_src.detach(), predicted_humidity.detach()))
                    
            loss = criterion(comparing_target, prediction)  #prediction is a list of lists with singleton elements
            # print(f"MSELoss at l :{l} = {loss}")
            # print("loss at l=",l,"=",loss,"from",with_target_sampled_src[:,-1,0].unsqueeze(-1),"and ",prediction[:,-1,:])
            # with open("output.txt","a") as sys.stdout:
            #     print(f"At epoch {epoch} and dataload_counter {dataloader_counter}:")
            #     print("target[:-1,:,0] shape=",target[:-1,:,0].shape,"with prediction shape:",prediction.shape) #target[:-1,:,0] returns 0th element of each list in target
            #     print(f"(last 3) MSE loss between {target[:-1,:,0].unsqueeze(-1)} and {prediction} = {loss}")
                # print("..............................................................")
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
        print("train loss after epoch ",epoch,"=",train_loss)

        #following is for saving the best model
        if train_loss < min_train_loss:
            torch.save(model.state_dict(), path_to_save_model + f"best_train_{epoch}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{epoch}.pth")
            min_train_loss = train_loss
            best_model = f"best_train_{epoch}.pth"


        if epoch % 10 == 0: # Plot 1-Step Predictions

            logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")


        # train_loss /= len(dataset)
        scaler = load('scalar_item_close.joblib')
        # print("comparing_target[:,:,0].unsqueeze(-1):",comparing_target[:,:,0]," prediction[:,:,:0]",prediction[:,:,0])
        a = scaler.inverse_transform(comparing_target[:,:,0])
        b = scaler.inverse_transform(prediction[:,:,0].detach().cpu().numpy())
        print("comparing_target[:,:,0]:",a)
        print("prediction[:,:,0]:",b)
        print("accuracy at epoch",epoch,"=",(b/a)*100,"%","loss=",train_loss)
    print("final accuracy =",np.sum(((b/a)*100))/comparing_target.shape[0],"%","loss=",train_loss)

    return best_model

#**********************************************************************************************************************************************
