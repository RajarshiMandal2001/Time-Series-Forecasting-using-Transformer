import torch.nn as nn
import torch, math
import time
import sys

class Transformer(nn.Module):

    def __init__(self, feature_size=9, num_layers=3, dropout=0):
        super(Transformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=9, dropout=dropout)  
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)    
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size,nhead=9,dropout=0)    
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer,num_layers=num_layers)
        # self.decoder = nn.Linear(feature_size,1)
        # self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.transformer_decoder.bias.data.zero_()
        self.transformer_decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        # print("mask:",mask,"shape:",mask.shape)
        return mask

    def forward(self, src, target, device):  
        # print("len(src)",len(src),"with shape:",src.shape)
        # print("len(src)",len(target) ,"with shape:",target.shape)
        encoder_mask = self._generate_square_subsequent_mask(len(src)).to(device)
        decoder_mask = self._generate_square_subsequent_mask(len(target)).to(device)
        encoder_output = self.transformer_encoder(src, encoder_mask)
        # print("encoder_output",encoder_output)
        decoder_output = self.transformer_decoder(target, encoder_output, decoder_mask)
        # output = self.decoder(trans_output)
        return decoder_output

    def encoder_for_test(self,src, device):
        encoder_mask = self._generate_square_subsequent_mask(len(src)).to(device)
        encoder_output = self.transformer_encoder(src, encoder_mask)
        return encoder_output

    def decoder_for_test(self,target,target_mask,encoder_output,device):
        t = target.to(torch.float64)
        decoder_output = self.transformer_decoder(t, encoder_output, target_mask)
        # print("decoder_output:",decoder_output)
        return decoder_output

    
    # def decoder_for_test(self, target, target_mask, encoder_output, device, src):
    #     print("len(src):",len(src))
    #     print("len(target):",len(target))
    #     encoder_mask = self._generate_square_subsequent_mask(len(src)).to(device)
    #     # decoder_mask = self._generate_square_subsequent_mask(len(target)).to(device)
    #     print("raw input for decoder_for_test:",src,"shape:",src.shape)
    #     print("target:",target,"shape=",target.shape)
    #     print("encoder output:",encoder_output)
    #     # print("encoder mask:",encoder_mask)
    #     # print("decoder mask:",decoder_mask)
    #     t = target.to(torch.float64)
    #     print("target type:",t.dtype)
    #     decoder_output = self.transformer_decoder(t, encoder_output, encoder_mask, target_mask)
    #     return decoder_output
        

