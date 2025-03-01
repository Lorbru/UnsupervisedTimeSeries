import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch


class CausalCut(torch.nn.Module):
    
    def __init__(self, cut_size):
        super(CausalCut, self).__init__()
        self.cut_size = cut_size

    def forward(self, x):
        return x[:, :, :-self.cut_size]


class CausalConvolutionBlock(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation):

        super(CausalConvolutionBlock, self).__init__()

        self.input_channels = in_channels
        self.output_channels = out_channels
        padding = (kernel_size - 1) * dilation

        cc1 = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        ccut1 = CausalCut(padding)
        relu1 = torch.nn.LeakyReLU()

        cc2 = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        ccut2 = CausalCut(padding)
        relu2 = torch.nn.LeakyReLU()

        self.causal_block = torch.nn.Sequential(
            cc1, 
            ccut1, 
            relu1, 
            cc2, 
            ccut2,
            relu2
        )

        self.is_equalsample = (in_channels == out_channels)
        self.equalsampler = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) 


    def forward(self, x):
        y = self.causal_block(x)
        if self.is_equalsample : return y + x
        return y + self.equalsampler(x)



class CausalBlockSeries(torch.nn.Module):
   
    def __init__(self, in_channels, hidden_channels, out_channels, depth, kernel_size):
        super(CausalBlockSeries, self).__init__()

        layers = nn.ModuleList() # List of causal convolution blocks

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else hidden_channels
            layers.append(
                CausalConvolutionBlock(
                    in_channels_block, 
                    hidden_channels, 
                    kernel_size=kernel_size,  
                    dilation=2**i
                )
            )


        layers.append(
            CausalConvolutionBlock(
                hidden_channels, 
                out_channels, 
                kernel_size,
                2**depth
            )
        )

        self.dilated_conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.dilated_conv(x)


class Encoder(torch.nn.Module):
 
    def __init__(self, in_channels, hidden_CNN_channels, 
                 hidden_out_CNN_channels, out_channels, CNN_depth, 
                 kernel_size):
        super(Encoder, self).__init__()
        self.causal_cnn = CausalBlockSeries(
            in_channels, hidden_CNN_channels, 
            hidden_out_CNN_channels, CNN_depth, kernel_size
        )
        self.pool = torch.nn.AdaptiveMaxPool1d(1)
        self.linear = torch.nn.Linear(hidden_out_CNN_channels, out_channels)
        

    def forward(self, x):
        y = self.causal_cnn(x)
        y = self.pool(y)
        y = y.squeeze(2)
        return self.linear(y)
    
    def transform(self, x):
        self.eval()
        res = []
        with torch.no_grad():
            res = [self.forward(x[i].unsqueeze(0)) for i in range(x.size(0))]
            return torch.stack(res)