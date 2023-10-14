from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Transformer, Informer, Autoformer, DLinear, peak_FiLM, peak_Transformer, peak_Informer, peak_Autoformer, peak_DLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'peak_Transformer': peak_Transformer,
            'peak_Informer': peak_Informer,
            'peak_Autoformer': peak_Autoformer,
            'peak_FiLM': peak_FiLM,
            'peak_DLinear': peak_DLinear,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                #batch_x = batch_x.reshape([batch_y.shape[0],-1,24,batch_y.shape[-1]]).max(-2)[0]
                #batch_y = batch_y.reshape([batch_y.shape[0],-1,24,batch_y.shape[-1]]).max(-2)[0]
                #batch_x = batch_x.reshape([batch_x.shape[0],-1,24])
                #batch_y = batch_y.reshape([batch_y.shape[0],-1,24])
                batch_x_mark = batch_x_mark.float().to(self.device)
                #batch_x_mark = batch_x_mark[:,::24,:]
                batch_y_mark = batch_y_mark.float().to(self.device)
                #batch_y_mark = batch_y_mark[:,::24,:]
                if self.args.busy_decoder:
                    batch_y_busy = batch_y.reshape([batch_y.shape[0],-1,24,batch_y.shape[-1]]).max(-2)[0]
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                            outputs, busy_outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs, busy_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                busy_outputs = busy_outputs[:, -self.args.pred_len//24:].to(self.device)
                batch_y_busy = batch_y_busy[:, -self.args.pred_len//24:].to(self.device) 
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                pred_busy = busy_outputs.detach().cpu()
                true_busy = batch_y_busy.detach().cpu()
                loss = criterion(pred, true)*(1-self.args.busy_ratio)+criterion(pred_busy, true_busy)*self.args.busy_ratio

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)#.unsqueeze(2)
                #batch_x = batch_x.reshape([batch_y.shape[0],-1,24,batch_y.shape[-1]]).max(-2)[0]
                batch_y = batch_y.float().to(self.device)#.unsqueeze(2)
                #batch_y = batch_y.reshape([batch_y.shape[0],-1,24,batch_y.shape[-1]]).max(-2)[0]
                #batch_y = batch_y.reshape([batch_y.shape[0],-1,24])
                if self.args.busy_decoder:
                    batch_y_busy = batch_y.reshape([batch_y.shape[0],-1,24,batch_y.shape[-1]]).max(-2)[0]
                batch_x_mark = batch_x_mark.float().to(self.device)
                #batch_x_mark = batch_x_mark[:,::24,:]
                batch_y_mark = batch_y_mark.float().to(self.device)
                #batch_y_mark = batch_y_mark[:,::24,:]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len//24:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len//24:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model:
                            outputs, busy_outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs, busy_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    busy_outputs = busy_outputs[:, -self.args.pred_len//24:].to(self.device)
                    batch_y_busy = batch_y_busy[:, -self.args.pred_len//24:].to(self.device)
                    loss = criterion(outputs, batch_y)*(1-self.args.busy_ratio)+criterion(busy_outputs, batch_y_busy)*self.args.busy_ratio
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            #temp = early_stopping.counter
            early_stopping(vali_loss, self.model, path)
            #if early_stopping.counter != temp and self.args.busy_ratio<=0.8:
            #    self.args.busy_ratio += 0.2
            #    early_stopping.counter = temp = 0

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        preds_busy = []
        trues_busy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                #batch_y = batch_y.reshape([batch_y.shape[0],-1,24,batch_y.shape[-1]]).max(-2)[0]
                #batch_x = batch_x.reshape([batch_y.shape[0],-1,24,batch_y.shape[-1]]).max(-2)[0]
                #batch_x = batch_x.reshape([batch_x.shape[0],-1,24])
                #batch_y = batch_y.reshape([batch_y.shape[0],-1,24])
                if self.args.busy_decoder:
                    batch_y_busy = batch_y.reshape([batch_y.shape[0],-1,24,batch_y.shape[-1]]).max(-2)[0]
                batch_x_mark = batch_x_mark.float().to(self.device)
                #batch_x_mark = batch_x_mark[:,::24,:]
                batch_y_mark = batch_y_mark.float().to(self.device)
                #batch_y_mark = batch_y_mark[:,::24,:]
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                            outputs, busy_outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs, busy_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                busy_outputs = busy_outputs[:, -self.args.pred_len//24:].to(self.device)
                batch_y_busy = batch_y_busy[:, -self.args.pred_len//24:].to(self.device)
                outputs = outputs.detach().cpu().numpy()

                batch_y = batch_y.detach().cpu().numpy()

                batch_y_busy = batch_y_busy.detach().cpu().numpy()
                busy_outputs = busy_outputs.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                pred_busy = busy_outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true_busy = batch_y_busy  # batch_y.detach().cpu().numpy()  # .squeeze()
                
                preds.append(pred)
                trues.append(true)
                preds_busy.append(pred_busy)
                trues_busy.append(true_busy)
                if i % 10 == 0:

                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    input_busy = input.reshape([input.shape[0],-1,24,input.shape[-1]]).max(-2)
                    gt_b = np.concatenate((input_busy[0, :,-1], true_busy[0,:,-1]), axis=0)
                    pd_b = np.concatenate((input_busy[0, :,-1], pred_busy[0,:,-1]), axis=0)
                    visual(gt_b, pd_b, os.path.join(folder_path, str(i) + '.pdf'))
                    #np.save('/home/wx/busy_hour/Nonstationary_Transformers/four_schemes/s2b/' + str(i) + 'gt.npy', gt)
                    #np.save('/home/wx/busy_hour/Nonstationary_Transformers/four_schemes/s2b/' + str(i) + 'pd.npy', pd)
                    
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds_busy = np.concatenate(preds_busy, axis=0)
        trues_busy = np.concatenate(trues_busy, axis=0)
        print('test shape:', preds.shape, trues.shape)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae_b,mse_b,rmse_b,mape_b,mspe_b = metric(preds_busy,trues_busy)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, mse_b:{}, mae_b:{}'.format(mse, mae,mse_b,mae_b))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, mse_b:{}, mae_b:{}'.format(mse, mae,mse_b,mae_b))
        f.write('\n')
        f.write('\n')
        f.close()

        #np.save(folder_path + 'metrics.npy', np.array([mae, mse, mae_b,mse_b]))
        #np.save(folder_path + 'pred.npy', preds)
        #np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        preds_busy = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                #batch_x = batch_x.reshape([batch_y.shape[0],-1,24,batch_y.shape[-1]]).max(-2)[0]
                batch_y = batch_y.float()
                #batch_y = batch_y.reshape([batch_y.shape[0],-1,24,batch_y.shape[-1]]).max(-2)[0]
                batch_x_mark = batch_x_mark.float().to(self.device)
                #batch_x_mark = batch_x_mark[:,::24,:]
                batch_y_mark = batch_y_mark.float().to(self.device)
                #batch_y_mark = batch_y_mark[:,::24,:]
                if self.args.busy_decoder:
                    batch_y_busy = batch_y.max(-1)[0]
                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs,busy_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                pred_busy = busy_outputs.detach().cpu().numpy()
                preds.append(pred)
                preds_busy.append(pred_busy)
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        '''
        preds_busy = np.array(preds_busy)
        preds_busy = preds_busy.reshape(-1, preds_busy.shape[-2], preds_busy.shape[-1])
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        np.save(folder_path + 'real_prediction_busy.npy', pred_busy)
        '''
        return

class Exp_Main_busy(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_busy, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'peak_Transformer': peak_Transformer,
            'peak_Informer': peak_Informer,
            'peak_Autoformer': peak_Autoformer,
            'peak_DLinear': peak_DLinear,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                #batch_x = batch_x.reshape([batch_x.shape[0],-1,24])
                batch_y = batch_y.reshape([batch_y.shape[0],-1,24])
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs,busy_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_x = batch_x.reshape([batch_x.shape[0],-1,24])
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y.reshape([batch_y.shape[0],-1,24])
                if self.args.busy_decoder:
                    batch_y_busy = batch_y.max(-1)[0]
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs, busy_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len//24:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len//24:, f_dim:].to(self.device)
                    batch_y_busy = batch_y_busy[:, -self.args.pred_len//24:].to(self.device)
                    loss = criterion(outputs, batch_y)+criterion(busy_outputs, batch_y_busy)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        preds_busy = []
        trues_busy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x = batch_x.reshape([batch_x.shape[0],-1,24])
                batch_y = batch_y.reshape([batch_y.shape[0],-1,24])
                if self.args.busy_decoder:
                    batch_y_busy = batch_y.max(-1)[0]
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs,busy_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len//24:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len//24:, f_dim:].to(self.device)
                batch_y_busy = batch_y_busy[:, -self.args.pred_len//24:].to(self.device)
                outputs = outputs.detach().cpu().numpy()

                batch_y = batch_y.detach().cpu().numpy()

                batch_y_busy = batch_y_busy.detach().cpu().numpy()
                busy_outputs = busy_outputs.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                pred_busy = busy_outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true_busy = batch_y_busy  # batch_y.detach().cpu().numpy()  # .squeeze()
                
                preds.append(pred)
                trues.append(true)
                preds_busy.append(pred_busy[:,:,0])
                trues_busy.append(true_busy)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        preds_busy = np.array(preds_busy)
        trues_busy = np.array(trues_busy)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae_b,mse_b,rmse_b,mape_b,mspe_b = metric(preds_busy,trues_busy)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{},mse_b:{},mae_b:{}'.format(mse, mae,mse_b,mae_b))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if self.args.busy_decoder:
                    batch_y_busy = batch_y.max(-1)[0]
                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs,busy_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
