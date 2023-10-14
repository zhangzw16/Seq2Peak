import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''
    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)          # B x 1 x E
        x = torch.cat([x, stats], dim=1) # B x 2 x E
        x = x.view(batch_size, -1) # B x 2E
        y = self.backbone(x)       # B x O

        return y

class Projectorh(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''
    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, feature_num, kernel_size=3):
        super(Projectorh, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='reflect', bias=False)
        
        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        layers_fc = [nn.Linear(feature_num, 1), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)
        self.feature_concentration = nn.Sequential(*layers_fc)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        f_size = x.shape[-1]
        x = self.series_conv(x.reshape(batch_size,-1,24*f_size)).reshape(batch_size,-1,24,f_size)        # B x 1 x E
        x = torch.cat([x, stats], dim=1) # B x 2 x E * feature
        
        x = x.view(batch_size, -1,f_size) # B x 2E
        x = self.feature_concentration(x).squeeze()
        y = self.backbone(x)      # B x O

        return y

class ShiftingModule(nn.Module):
    '''
    MLP to learn the shifting of mean and standard variation
    '''
    def __init__(self, enc_in, pred_len, seq_len, hidden_dims, hidden_layers, output_dim=2, kernel_size=3,):
        super(ShiftingModule, self).__init__()
        self.pred_len = pred_len
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        #self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)
        layers = [nn.Linear((2+seq_len//pred_len*2)*enc_in, hidden_dims[0], bias=False)]
        #for i in range(hidden_layers-1):
        #    layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats1,stats2):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        
        hist_mean = torch.zeros([batch_size,x.shape[1]//self.pred_len,24]).cuda()
        hist_std = torch.zeros([batch_size,x.shape[1]//self.pred_len,24]).cuda()
        for i in range(0,x.shape[1],self.pred_len):
            hist_mean[:,i//self.pred_len,:] = torch.flip(x,dims=[1])[:,i:i+self.pred_len,:].mean(1, keepdim=False).detach()
            hist_std[:,i//self.pred_len,:] = torch.sqrt(torch.var(torch.flip(x,dims=[1])[:,i:i+self.pred_len,:], dim=1, keepdim=False, unbiased=False) + 1e-5).detach()
        hist_stat = torch.cat([hist_mean, hist_std], dim=1)        # B x 1 x E
        x = torch.cat([hist_stat, stats1, stats2], dim=1) # B x 3 x E
        
        #x = self.series_conv(x)       # B x 1 x E
        #x = torch.cat([stats1, stats2], dim=1) # B x 2 x E
        x = x.view(batch_size, -1) # B x 2E
        y = self.backbone(x)       # B x 2 

        return y

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hour_day = configs.hour_day
        self.with_shift = configs.with_shift

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            
        self.tau_learner   = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=configs.seq_len)

        if self.hour_day=='h':
            if configs.with_shift == 'ws':
                self.shift_learner = ShiftingModule(enc_in=24,pred_len = self.pred_len//24,seq_len=self.seq_len//24, hidden_dims=[48,48], hidden_layers=1,output_dim=24*2)
            self.tau_learner_hour   = Projectorh(enc_in=24, seq_len=self.seq_len//24, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=24, feature_num=configs.enc_in)
            self.delta_learner_hour = Projectorh(enc_in=24, seq_len=self.seq_len//24, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=24,feature_num=configs.enc_in)    
            
        if configs.busy_decoder:
            kernel_size = 24
            if configs.busy_decoder_modal == 'm':
                self.busy_decoder = torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            else:
                self.busy_decoder = torch.nn.Linear(self.pred_len+self.label_len,(self.pred_len+self.label_len)//24)


    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x_raw = x.clone().detach()
        
        if self.hour_day == 'h':
            x_hour_day = x.reshape(x.shape[0],-1,24,x.shape[-1])
            mean_hour = x_hour_day.mean(1, keepdim=True).detach() # B x 1 x 24
            std_hour = torch.sqrt(torch.var(x_hour_day, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x 24
            x_hour_day = (x_hour_day-mean_hour)/std_hour
            x = x_hour_day.reshape(x.shape[0],-1,x.shape[-1])
            if self.with_shift == 'ws':
                shift_hour = self.shift_learner(x.reshape(x.shape[0],-1,24).clone().detach(),mean_hour,std_hour).unsqueeze(1)
        elif self.hour_day == 'd':
            x_hour_day = x.reshape(x.shape[0],-1,24)
            mean_day = x_hour_day.mean(2, keepdim=True).detach() # B x 1 x day
            std_day = torch.sqrt(torch.var(x_hour_day, dim=2, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x day
            x_hour_day = (x_hour_day-mean_day)/std_day
            x = x_hour_day.reshape(x.shape[0],-1,1)
        else:
            mean = x.mean(1, keepdim=True).detach() # B x 1 x E
            x = x - mean
            std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x E
            if self.with_shift == 'ws':
                shift_n = self.shift_learner(x_raw,mean,std).unsqueeze(1)
            x = x / std 
       
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = x.permute(0,2,1)
        
        # De-normalization
        '''
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        '''
        if self.hour_day=='h':
            if self.with_shift != 'ws':
                std_hour=std_hour.squeeze().repeat(1,x.shape[1]//24,1)
                mean_hour=mean_hour.squeeze().repeat(1,x.shape[1]//24,1)
            else:
                #shift_hour = self.shift_learner(dec_out.reshape(x.shape[0],-1,24),mean_hour,std_hour).unsqueeze(1)
                std_hour = shift_hour[:,:,:24].exp().permute(0,2,1).repeat(1,x.shape[1]//24,1)
                mean_hour = shift_hour[:,:,24:].permute(0,2,1).repeat(1,x.shape[1]//24,1)
            x = x * std_hour + mean_hour
        elif self.hour_day=='d':
            std_day=std_day.repeat(1,1,24).reshape(std_day.shape[0],1,-1)
            mean_day=mean_day.repeat(1,1,24).reshape(std_day.shape[0],1,-1)
            x = x * x + mean_day
        else:
            if self.with_shift == 'ws':
                std = shift_n[:,:,0].exp().unsqueeze(-1)
                mean = shift_n[:,:,1].unsqueeze(-1)
            x = x * std + mean
        
        if self.busy_decoder:
            busy_out = self.busy_decoder(x.permute(0,2,1)).permute(0,2,1)
            return x[:, -self.pred_len:, :], busy_out[:,-self.pred_len//24:]
        
        return x # to [Batch, Output length, Channel]
