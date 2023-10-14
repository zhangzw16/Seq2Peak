import torch
import torch.nn as nn
from ns_layers.AutoCorrelation import DSAutoCorrelation, AutoCorrelationLayer
from ns_layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from layers.Embed import DataEmbedding_wo_pos


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
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.hour_day = configs.hour_day
        self.with_shift = configs.with_shift
        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        DSAutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        DSAutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        DSAutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        
        self.tau_learner   = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=configs.seq_len)

        if self.hour_day=='h':
            if configs.with_shift == 'ws':
                self.shift_learner = ShiftingModule(enc_in=24,pred_len = self.pred_len//24,seq_len=self.seq_len//24, hidden_dims=[48,48], hidden_layers=1,output_dim=24*2)
            self.tau_learner_hour   = Projectorh(enc_in=24, seq_len=self.seq_len//24, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=24, feature_num=configs.enc_in)
            self.delta_learner_hour = Projectorh(enc_in=24, seq_len=self.seq_len//24, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=24,feature_num=configs.enc_in)
        #else:
        #    if configs.with_shift == 'ws':
        #        self.shift_learner = ShiftingModule(enc_in=self.seq_len//24,pred_len = self.pred_len, seq_len=24, hidden_dims=[48,48], hidden_layers=1,output_dim=2)
        #    self.tau_learner_day   = Projectorh(enc_in=self.seq_len//24, seq_len=24, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=self.seq_len//24,feature_num=configs.enc_in)
        #    self.delta_learner_day = Projectorh(enc_in=self.seq_len//24, seq_len=24, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=self.seq_len//24,feature_num=configs.enc_in)


        if configs.busy_decoder:
            kernel_size = 24
            if configs.busy_decoder_modal == 'm':
                self.busy_decoder = torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            else:
                self.busy_decoder = torch.nn.Linear(self.pred_len+self.label_len,(self.pred_len+self.label_len)//24)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x_raw = x_enc.clone().detach()

        # Normalization

        if self.hour_day == 'h':
            x_enc_hour_day = x_enc.reshape(x_enc.shape[0],-1,24,x_enc.shape[-1])
            mean_enc_hour = x_enc_hour_day.mean(1, keepdim=True).detach() # B x 1 x 24
            #mean_enc_day = x_enc_hour_day.mean(2, keepdim=True).detach() # B x 1 x day
            std_enc_hour = torch.sqrt(torch.var(x_enc_hour_day, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x 24
            #std_enc_day = torch.sqrt(torch.var(x_enc_hour_day, dim=2, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x day
            x_enc_hour_day = (x_enc_hour_day-mean_enc_hour)/std_enc_hour
            x_enc = x_enc_hour_day.reshape(x_enc.shape[0],-1,x_enc.shape[-1])
            tau_hour = self.tau_learner_hour(x_enc_hour_day,std_enc_hour).exp()
            delta_hour = self.delta_learner_hour(x_enc_hour_day,mean_enc_hour)
            if self.with_shift == 'ws':
                shift_hour = self.shift_learner(x_enc.reshape(x_enc.shape[0],-1,24).clone().detach(),mean_enc_hour,std_enc_hour).unsqueeze(1)
            tau = None
            delta = None
            tau_day = None
            delta_day = None
        elif self.hour_day == 'd':
            x_enc_hour_day = x_enc.reshape(x_enc.shape[0],-1,24)
            mean_enc_day = x_enc_hour_day.mean(2, keepdim=True).detach() # B x 1 x day
            std_enc_day = torch.sqrt(torch.var(x_enc_hour_day, dim=2, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x day
            x_enc_hour_day = (x_enc_hour_day-mean_enc_day)/std_enc_day
            x_enc = x_enc_hour_day.reshape(x_enc.shape[0],-1,1)
            tau_day = self.tau_learner_day(x_enc_hour_day.permute(0,2,1),std_enc_day.permute(0,2,1)).exp()
            delta_day = self.delta_learner_day(x_enc_hour_day.permute(0,2,1),mean_enc_day.permute(0,2,1))
            tau = None
            delta = None
            tau_hour = None
            delta_hour = None
        else:
            tau_hour = None
            delta_hour = None
            tau_day = None
            delta_day = None
            mean_enc = x_enc.mean(1, keepdim=True).detach() # B x 1 x E
            x_enc = x_enc - mean_enc
            std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x E
            if self.with_shift == 'ws':
                shift_n = self.shift_learner(x_raw,mean_enc,std_enc).unsqueeze(1)
            x_enc = x_enc / std_enc
            tau = self.tau_learner(x_raw, std_enc).exp()     # B x S x E, B x 1 x E -> B x 1, positive scalar    
            delta = self.delta_learner(x_raw, mean_enc) 
        x_dec_new = torch.cat([x_enc[:, -self.label_len: , :], torch.zeros_like(x_dec.squeeze()[:, -self.pred_len:, :])], dim=1).to(x_enc.device).clone()
        # Model Inference


        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec_new.shape[0], self.pred_len, x_dec_new.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta, tau_hour=tau_hour,delta_hour=delta_hour,tau_day=tau_day,delta_day=delta_day)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init, tau=tau, delta=delta, tau_hour=tau_hour,delta_hour=delta_hour,tau_day=tau_day,delta_day=delta_day)
        # final
        dec_out = trend_part + seasonal_part

        # De-normalization
        '''
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        '''
        if self.hour_day=='h':
            if self.with_shift != 'ws':
                std_enc_hour=std_enc_hour.squeeze().repeat(1,dec_out.shape[1]//24,1)
                mean_enc_hour=mean_enc_hour.squeeze().repeat(1,dec_out.shape[1]//24,1)
            else:
                #shift_hour = self.shift_learner(dec_out.reshape(x_enc.shape[0],-1,24),mean_enc_hour,std_enc_hour).unsqueeze(1)
                std_enc_hour = shift_hour[:,:,:24].exp().permute(0,2,1).repeat(1,dec_out.shape[1]//24,1)
                mean_enc_hour = shift_hour[:,:,24:].permute(0,2,1).repeat(1,dec_out.shape[1]//24,1)
            dec_out = dec_out * std_enc_hour + mean_enc_hour
        elif self.hour_day=='d':
            std_enc_day=std_enc_day.repeat(1,1,24).reshape(std_enc_day.shape[0],1,-1)
            mean_enc_day=mean_enc_day.repeat(1,1,24).reshape(std_enc_day.shape[0],1,-1)
            dec_out = dec_out * std_enc_day + mean_enc_day
        else:
            if self.with_shift == 'ws':
                std_enc = shift_n[:,:,0].exp().unsqueeze(-1)
                mean_enc = shift_n[:,:,1].unsqueeze(-1)
            dec_out = dec_out * std_enc + mean_enc
        if self.busy_decoder:
            busy_out = self.busy_decoder(dec_out.permute(0,2,1)).permute(0,2,1)
            return dec_out[:, -self.pred_len:, :], busy_out[:,-self.pred_len//24:]
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
