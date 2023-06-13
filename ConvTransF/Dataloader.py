#!/usr/bin/env python
# coding: utf-8


import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_excel('data_new.xlsx', sheet_name='1')
data_d = np.array([data.AQI, data.PM10, data.O3, data.SO2, data.PM2_5, data.NO2, data.CO, data.V13305, data.V10004_700,
                   data.V11291_700, data.V12001_700, data.V13003_700]).T
(data_d.astype('int64') * 10) // 10
df = pd.DataFrame(data_d.astype('int64'))
df = df.replace(-9223372036854775808, -10)
data_0 = np.array(df[0][1:])
data_1 = np.array(df[1][1:])


class time_series_decoder_paper(Dataset):
    """synthetic time series dataset from section 5.1"""
    
    def __init__(self,t0=96,N=4500,transform=None):
        """
        Args:
            t0: previous t0 data points to predict from
            N: number of data points
            transform: any transformations to be applied to time series
        """
        self.t0 = t0
        self.N = N
        self.transform = None
        # time points
        self.x = torch.cat(torch.chunk(torch.tensor(data_0).unsqueeze(0), 152, dim=1), 0)

        # sinuisoidal signal
        self.fx = torch.cat(torch.chunk(torch.tensor(data_1).unsqueeze(0), 152, dim=1), 0)

        
        # add noise
        self.fx = self.fx + torch.randn(self.fx.shape)
        
        self.masks = self._generate_square_subsequent_mask(t0)
                
        
        # print out shapes to confirm desired output
        print("x: {}*{}".format(*list(self.x.shape)),
              "fx: {}*{}".format(*list(self.fx.shape)))        
        
    def __len__(self):
        return len(self.fx)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        sample = (self.x[idx,:],
                  self.fx[idx,:],
                  self.masks)
        
        if self.transform:
            sample=self.transform(sample)
            
        return sample
    
    def _generate_square_subsequent_mask(self,t0):
        mask = torch.zeros(t0+5,t0+5)
        for i in range(0,t0):
            mask[i,t0:] = 1 
        for i in range(t0,t0+5):
            mask[i,i+1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))#.masked_fill(mask == 1, float(0.0))
        return mask