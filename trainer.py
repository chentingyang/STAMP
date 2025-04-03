

import os
import torch
from lib.logger import get_logger
import time
import copy
import numpy as np
import pandas as pd

class Trainer(object):
    def __init__(self, pred_model, pred_loss, pred_optimizer, ae_model, ae_loss, ae_optimizer, train_loader, val_loader, test_loader, args, scaler, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.pred_model = pred_model
        self.pred_loss = pred_loss
        self.pred_optimizer = pred_optimizer

        self.ae_model = ae_model
        self.ae_loss = ae_loss
        self.ae_optimizer = ae_optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)

        self.best_path = os.path.join(self.args.log_dir, 'best_model_' + self.args.data + "_" + self.args.model + '.pth')
        #self.best_path = os.path.join(self.args.log_dir, 'best_model_unsup_weights_init_' + self.args.data + "_" + self.args.model + '.pth') #用该版本读取无监督节点权重
        self.transfer_path = os.path.join(self.args.log_dir_transfer, 'best_model_' + self.args.data + "_" + self.args.model + '.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')

        ## log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug, data = args.data)
        #self.logger = get_logger(args.log_dir, name=args.model + '_unsup', debug=args.debug, data = args.data)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))

        self.ae_channels = self.args.window_size*self.args.nnodes*self.args.in_channels

    def pred_model_batch(self, batch, training=True, mas = None):
        self.pred_model.train(training)
        x, target = batch[:, :self.args.window_size - self.args.n_pred, ...], batch[:, -self.args.n_pred:, ...]
        # if self.args.is_mas:
        #     mas = mas[:, :self.args.window_size - self.args.n_pred, ...]
        output = self.pred_model(x, mas=mas)

        if self.args.real_value:
            #     outputs = self.scaler.inverse_transform(outputs)
            target = self.scaler.inverse_transform(target.reshape(-1, self.args.n_pred , self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
            target = torch.from_numpy(target).float().view(-1, self.args.n_pred, self.args.nnodes, self.args.out_channels).to(batch.device)

            x = self.scaler.inverse_transform(x.reshape(-1, (self.args.window_size - self.args.n_pred) , self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
            x = torch.from_numpy(x).float().view(-1, self.args.window_size - self.args.n_pred, self.args.nnodes, self.args.out_channels).to(batch.device)

        generate_batch = torch.cat([x, output], dim=1)
        return output, target, generate_batch

    def ae_model_batch(self, batch, training = True):
        self.ae_model.train(training)

        ## [B, T, N, C] --> [B, T * N * C]
        batch = batch.reshape(-1, self.ae_channels)

        output = self.ae_model(batch)

        if self.args.real_value:
            #     outputs = self.scaler.inverse_transform(outputs)
            batch = self.scaler.inverse_transform(batch.reshape(-1, self.args.window_size, self.args.nnodes * self.args.in_channels).detach().cpu().numpy())
            batch = torch.from_numpy(batch).view(-1, self.ae_channels).float().to(output.device)

        return output, batch

    def val_epoch(self, epoch, val_dataloader):
        self.pred_model.eval()
        self.ae_model.eval()

        total_val_pred_loss_list = []
        total_val_ae_loss_list = []
        total_val_adv_loss_list = []

        loss1_list = []
        loss2_list = []
        start_val = time.time()


        start_epoch = 0
        with torch.no_grad():
            for batch_m in val_dataloader:
                if self.args.is_mas:
                    batch,mas = batch_m
                    mas = mas[:, :self.args.window_size - self.args.n_pred, ...]
                else:
                    batch, mas = batch_m[0], None
                ## batch: [B,T,N,C]

                ## eval predModel
                output, target, generate_batch = self.pred_model_batch(batch, training=False, mas = mas)
                pred_loss = self.pred_loss(output, target)
                if not torch.isnan(pred_loss):
                    total_val_pred_loss_list.append(pred_loss.item())

                ## eval AE
                output, target = self.ae_model_batch(batch, training=False)
                ae_loss = self.ae_loss(output.reshape(-1,self.args.window_size,self.args.nnodes,self.args.out_channels), target.reshape(-1,self.args.window_size,self.args.nnodes,self.args.out_channels))
                if not torch.isnan(ae_loss):
                    total_val_ae_loss_list .append(ae_loss.item())

                if self.args.real_value:
                    generate_batch = self.scaler.transform(generate_batch.reshape(-1, self.args.window_size, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                    generate_batch = torch.from_numpy(generate_batch).float().view(-1, self.args.window_size, self.args.nnodes, self.args.out_channels).to(batch.device)

                ## eval adv
                output2, target2 = self.ae_model_batch(generate_batch, training=False)
                adv_loss = self.ae_loss(output2.reshape(-1,self.args.window_size,self.args.nnodes,self.args.out_channels), target2.reshape(-1,self.args.window_size,self.args.nnodes,self.args.out_channels))
                if not torch.isnan(ae_loss):
                    total_val_adv_loss_list.append(adv_loss.item())

                loss1 = 5/(epoch-start_epoch) *pred_loss.item() + (1- 1/(epoch - start_epoch))*adv_loss.item()
                loss2 = 3/(epoch-start_epoch) *ae_loss.item() - (1- 1/(epoch - start_epoch))*adv_loss.item()

                loss1_list.append(loss1)
                loss2_list.append(loss2)
        val_pred_loss = np.array(total_val_pred_loss_list).mean()
        val_ae_loss = np.array(total_val_ae_loss_list).mean()
        val_adv_loss = np.array(total_val_adv_loss_list).mean()
        end_val = time.time()

        loss1 = np.array(loss1_list).mean()
        loss2 = np.array(loss2_list).mean()

        self.logger.info(
            '**********Val Epoch {}: average Loos1: {:.6f}, average Loss2: {:.6f}, while average predModel Loss: {:.6f}, average AE Loss: {:.6f}, average AE Generate Loss: {:.6f}, inference time: {:.3f}s'.
            format(epoch, loss1, loss2, val_pred_loss, val_ae_loss, val_adv_loss, end_val - start_val))

        return val_pred_loss, val_ae_loss, val_adv_loss, loss1, loss2

    def train_epoch(self, epoch):
        self.pred_model.train()
        self.ae_model.train()

        loss1_list = []
        loss2_list = []
        start_time = time.time()

        start_epoch = 0
        
        for batch_m in self.train_loader:
            #data and target shape: B, T, N, F;
            # output shape: B, T, N, F
            if self.args.is_mas:
                batch, mas = batch_m
                mas = mas[:, :self.args.window_size - self.args.n_pred, ...]
            else:
                batch, mas = batch_m[0], None

            ## train predModel
            self.pred_optimizer.zero_grad()
            output, target, generate_batch = self.pred_model_batch(batch, training=True, mas=mas)
            pred_loss = self.pred_loss(output, target)
            pred_loss.backward()
            self.pred_optimizer.step()
            
            ## train AE
            self.ae_optimizer.zero_grad()
            output1, target1, = self.ae_model_batch(batch, training=True)
            ae_loss = self.ae_loss(output1.reshape(-1, self.args.window_size, self.args.nnodes, self.args.out_channels),
                                target1.reshape(-1, self.args.window_size, self.args.nnodes, self.args.out_channels))
            ae_loss.backward()
            self.ae_optimizer.step()

            ## loss1
            self.pred_optimizer.zero_grad()
            output, target, generate_batch = self.pred_model_batch(batch, training=True, mas=mas)
            pred_loss = self.pred_loss(output, target)

            if self.args.real_value:
                generate_batch = self.scaler.transform(generate_batch.reshape(-1,self.args.window_size , self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                generate_batch = torch.from_numpy(generate_batch).float().view(-1, self.args.window_size, self.args.nnodes, self.args.out_channels).to(batch.device)

            output1, batch1 = self.ae_model_batch(generate_batch, training=True)
            adv_loss = self.ae_loss(output1.reshape(-1, self.args.window_size,self.args.nnodes,self.args.out_channels), batch1.reshape(-1, self.args.window_size,self.args.nnodes,self.args.out_channels))

            if epoch > start_epoch:
                loss1 = 5/(epoch-start_epoch) *pred_loss + (1- 1/(epoch-start_epoch))*adv_loss
            else:
                loss1 = 5 / (epoch ) * pred_loss + (1 - 1 / (epoch )) * adv_loss
            loss1.backward()
            self.pred_optimizer.step()

            ## loss2
            self.ae_optimizer.zero_grad()
            output1, target1, = self.ae_model_batch(batch, training=True)
            ae_loss = self.ae_loss(output1.reshape(-1, self.args.window_size, self.args.nnodes, self.args.out_channels),
                                target1.reshape(-1, self.args.window_size, self.args.nnodes, self.args.out_channels))

            _,_, generate_batch = self.pred_model_batch(batch, training=True, mas=mas)
            if self.args.real_value:
                generate_batch = self.scaler.transform(generate_batch.reshape(-1, self.args.window_size,
                                                                              self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                generate_batch = torch.from_numpy(generate_batch).float().view(-1, self.args.window_size,self.args.nnodes, self.args.out_channels).to(batch.device)
            output2, batch2 = self.ae_model_batch(generate_batch, training=True)
            adv_loss = self.ae_loss(
                output2.reshape(-1, self.args.window_size, self.args.nnodes, self.args.out_channels),
                batch2.reshape(-1, self.args.window_size, self.args.nnodes, self.args.out_channels))

            if epoch > start_epoch:
                loss2 = 3 / (epoch - start_epoch) * ae_loss - (1 - 1 / (epoch - start_epoch)) * adv_loss
            else:
                loss2 = 3 / (epoch ) * ae_loss - (1 - 1 / (epoch )) * adv_loss
            loss2.backward()
            self.ae_optimizer.step()


            loss1_list.append(loss1.item())
            loss2_list.append(loss2.item())
            
        end_time = time.time()

        loss1 = np.array(loss1_list).mean()
        loss2 = np.array(loss2_list).mean()
        self.logger.info(
            '**********Train Epoch {}: averaged Loss1: {:.6f}, Loss2: {:.6f}, train_time: {:.3f}s'.format(epoch, loss1, loss2,
                                                                                        end_time - start_time))
        return loss1, loss2

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0

        train_history = []
        val_history = []
        start_time = time.time()
        
        for epoch in range(1, self.args.epochs + 1):
            ## train
            train_loss1, train_loss2 = self.train_epoch(epoch)
            train_result = {'train_loss1': train_loss1, 'train_loss2': train_loss2}
            train_history.append(train_result)

            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader

            ## val
            val_pred_loss, val_ae_loss, val_adv_loss, val_loss1, val_loss2 = self.val_epoch(epoch, val_dataloader)
            val_result = {"val_pred_loss":val_pred_loss, "val_ae_loss":val_ae_loss, "val_adv_loss":val_adv_loss,
                        "val_loss1":val_loss1, "val_loss2":val_loss2}
            val_history.append(val_result)
            
            train_epoch_loss = train_loss1 + train_loss2
            val_epoch_loss = val_loss1 + val_loss2
            
            # early_stop:
            # if train_epoch_loss > 1e6:
            #     self.logger.warning('Gradient explosion detected. Ending...')
            #     break

            ### early_stop
            # if val_epoch_loss < best_loss:
            #     best_loss = val_epoch_loss
            #     not_improved_count = 0
            #     best_state = True
            # else:
            #     not_improved_count += 1
            #     best_state = False
            # if val_epoch_loss - best_loss > self.args.largest_loss_diff:
            #     break
            # if self.args.early_stop:
            #     if not_improved_count == self.args.early_stop_patience:
            #         self.logger.info("Validation performance didn\'t improve for {} epochs. "
            #                         "Training stops.".format(self.args.early_stop_patience))
            #         break
            # # save the best state
            # if best_state == True:
            #     self.logger.info('*********************************Current best model saved!')
            #     best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min".format((training_time / 60)))
        # self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))


        #save the best model to file
        self.save_checkpoint()
        self.logger.info("Saving current best model to " + self.best_path)
        return train_history, val_history
    

    def save_checkpoint(self):
        state = {
            'pred_state_dict': self.pred_model.state_dict(),
            'pred_optimizer': self.pred_optimizer.state_dict(),

            'ae_state_dict': self.ae_model.state_dict(),
            'ae_optimizer': self.ae_optimizer.state_dict(),

            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)
        
    def save_checkpoint_transfer(self):
        state = {
            'pred_state_dict': self.pred_model.state_dict(),
            'pred_optimizer': self.pred_optimizer.state_dict(),

            'ae_state_dict': self.ae_model.state_dict(),
            'ae_optimizer': self.ae_optimizer.state_dict(),

            'config': self.args
        }
        torch.save(state, self.transfer_path)
        self.logger.info("Saving current best transfer model to " + self.transfer_path)
        


class Tester(object):
    def __init__(self, pred_model, ae, args, scaler, logger, path=None, alpha=.5, beta=.5, gamma = 0.):
        self.pred_model = pred_model
        self.ae = ae
        self.args = args
        # self.data_loader = data_loader
        self.scaler = scaler
        self.logger = logger
        self.path = path
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def testing(self, data_loader, map_location = None):
        if self.path != None:
            print("load model: ", self.path)
            check_point = torch.load(self.path, map_location = map_location)
            pred_state_dict = check_point['pred_state_dict']
            ae_state_dict = check_point['ae_state_dict']
            # args = check_point['config']
            self.pred_model.load_state_dict(pred_state_dict)
            self.ae.load_state_dict(ae_state_dict)
            self.pred_model.to(self.args.device)
            self.ae.to(self.args.device)

        self.pred_model.eval()
        self.ae.eval()

        scores = []
        loss1_list = []
        loss2_list = []
        gt_list = []
        pred_list = []
        origin_list = []
        construct_list = []
        generate_list = []
        generate_construct_list = []
        pred_channels = self.args.n_pred * self.args.nnodes * self.args.out_channels
        ae_channels = self.args.window_size * self.args.nnodes * self.args.out_channels
        with torch.no_grad():
            for batch_m in data_loader:
                if self.args.is_mas:
                    batch, mas = batch_m
                    mas = mas[:, :self.args.window_size - self.args.n_pred, ...]
                else:
                    batch, mas = batch_m[0], None

                x, target = batch[:, :self.args.window_size - self.args.n_pred, ...], batch[:, -self.args.n_pred:, ...]
                output = self.pred_model(x, mas=mas)

                target1 = batch.reshape(-1, ae_channels)
                output1 = self.ae(target1)

                if self.args.real_value:
                    #     outputs = self.scaler.inverse_transform(outputs)
                    target = self.scaler.inverse_transform(target.reshape(-1, self.args.n_pred,self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                    target = torch.from_numpy(target).float().view(-1, self.args.n_pred, self.args.nnodes,self.args.out_channels).to(batch.device)

                    x = self.scaler.inverse_transform(x.reshape(-1, (self.args.window_size - self.args.n_pred),self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                    x = torch.from_numpy(x).float().view(-1, self.args.window_size - self.args.n_pred, self.args.nnodes,self.args.out_channels).to(batch.device)

                    target1 = self.scaler.inverse_transform(target1.reshape(-1, self.args.window_size, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                    target1 = torch.from_numpy(target1).float().view(-1, ae_channels).to(batch.device)

                generate_batch = torch.cat([x, output], dim=1)
                generate_batch = generate_batch.reshape(-1, ae_channels)
                generate_batch_out = self.ae(generate_batch)


                loss1 = torch.mean((output.reshape(-1,pred_channels) - target.reshape(-1,pred_channels)) ** 2, axis=1)##axis=(1,2,3)
                loss2 = torch.mean((output1 - target1) ** 2, axis=1)
                loss3 = torch.mean((generate_batch - generate_batch_out) ** 2, axis=1)

                score = self.alpha * loss1 + self.beta * loss2 + self.gamma * loss3#(batch_size)
                # score = self.alpha * torch.mean((predmodel_gt - pred_out) ** 2, axis=1) + self.beta * torch.mean((x2 - x2_) ** 2, axis=1)
                scores.append(score.detach().cpu().numpy())


                loss1_list.append(loss1.detach().cpu().numpy())#(batch_size)
                loss2_list.append(loss2.detach().cpu().numpy())#(batch_size)
                

                gt_list.append(target.reshape(-1,self.args.n_pred , self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                pred_list.append(output.reshape(-1,self.args.n_pred, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())

                origin_list.append(target1.reshape(-1,self.args.window_size, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                construct_list.append(output1.reshape(-1,self.args.window_size, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())

                generate_list.append(generate_batch.reshape(-1, self.args.window_size, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                generate_construct_list.append(generate_batch_out.reshape(-1, self.args.window_size, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
        return scores, loss1_list, loss2_list, pred_list, gt_list, origin_list, construct_list, generate_list, generate_construct_list
    


        
class PredictedModelTrainer(object):
    def __init__(self, pred_model, pred_loss, pred_optimizer,  train_loader, val_loader, test_loader, args, scaler, lr_scheduler=None):
        super(PredictedModelTrainer, self).__init__()
        self.pred_model = pred_model
        self.pred_loss = pred_loss
        self.pred_optimizer = pred_optimizer


        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)

        self.best_path = os.path.join(self.args.log_dir, 'best_model_' + self.args.data +"_" +self.args.model + '.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')

        ## log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug, data = args.data)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))

    def pred_model_batch(self, batch, training=True, mas = None):
        self.pred_model.train(training)
        x, target = batch[:, :self.args.window_size - self.args.n_pred, ...], batch[:, -self.args.n_pred:, ...]
        output = self.pred_model(x, mas=mas)

        if self.args.real_value:
        #     outputs = self.scaler.inverse_transform(outputs)
            target = self.scaler.inverse_transform(target.reshape(-1,self.args.n_pred, self.args.nnodes*self.args.out_channels).detach().cpu().numpy())
            target = torch.from_numpy(target).float().view(-1,self.args.n_pred,self.args.nnodes,self.args.out_channels).to(batch.device)
        generate_batch = torch.cat([x, output], dim=1)
        return output, target, generate_batch


    def val_epoch(self, epoch, val_dataloader):
        self.pred_model.eval()

        total_val_pred_loss = 0

        loss1_list = []

        start_val = time.time()
        with torch.no_grad():
            for batch_m in val_dataloader:
                if self.args.is_mas:
                    batch, mas = batch_m
                    mas = mas[:, :self.args.window_size - self.args.n_pred, ...]
                else:
                    batch, mas = batch_m[0], None
                ## batch: [B,T,N,C]

                ## eval pred_Model
                output, target, generate_batch = self.pred_model_batch(batch, training=False, mas=mas)
                pred_loss = self.pred_loss(output, target)
                if not torch.isnan(pred_loss):
                    total_val_pred_loss += pred_loss.item()

                loss1_list.append(pred_loss.item() )

        val_pred_loss = total_val_pred_loss / len(val_dataloader)

        end_val = time.time()
        self.logger.info('**********Val Epoch {}: average predModel Loss: {:.6f}, inference time: {:.3f}s'.
                        format(epoch, val_pred_loss, end_val-start_val))

        loss1 = np.array(loss1_list).mean()

        self.logger.info( '**********Val Epoch {}: average Loos1: {:.6f}'.format(epoch, loss1))

        return val_pred_loss, loss1

    def train_epoch(self, epoch):
        self.pred_model.train()

        loss1_list = []
        start_time = time.time()

        for batch_m in self.train_loader:
            if self.args.is_mas:
                batch, mas = batch_m
                mas = mas[:, :self.args.window_size - self.args.n_pred, ...]
            else:
                batch, mas = batch_m[0], None

            #data and target shape: B, T, N, F; output shape: B, T, N, F
            self.pred_optimizer.zero_grad()
            ## train predModel
            output, target, generate_batch = self.pred_model_batch(batch, training=True, mas=mas)

            pred_loss = self.pred_loss(output, target)
            loss1 = pred_loss
            loss1.backward()
            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), self.args.max_grad_norm)

            self.pred_optimizer.step()

            loss1_list.append(loss1.item())

        end_time = time.time()
        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()

        loss1 = np.array(loss1_list).mean()

        self.logger.info('**********Train Epoch {}: averaged Loss1: {:.6f}, train_time: {:.3f}s'.format(epoch, loss1, end_time - start_time))
        return loss1

    def train(self):
        best_loss = float('inf')
        not_improved_count = 0

        history = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            ## train
            train_epoch_loss = self.train_epoch(epoch)

            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader

            ## val
            val_pred_loss, val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            result = {'train_loss': train_epoch_loss, "val_loss": val_epoch_loss}
            history.append(result)

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            ## early_stop
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min".format((training_time / 60)))

        #save the best model to file
        self.save_checkpoint()
        self.logger.info("Saving current best model to " + self.best_path)
        return history

    def save_checkpoint(self):
        state = {
            'pred_state_dict': self.pred_model.state_dict(),
            'pred_optimizer': self.pred_optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)
        



class PredTester(object):
    def __init__(self, pred_model, args, scaler, logger, path=None):
        self.pred_model = pred_model
        self.args = args
        self.scaler = scaler
        self.logger = logger
        self.path = path


    def testing(self, data_loader, map_location = None):
        if self.path != None:
            print("load model: ", self.path)
            check_point = torch.load(self.path, map_location = map_location)
            pred_state_dict = check_point['pred_state_dict']
            # args = check_point['config']
            self.pred_model.load_state_dict(pred_state_dict)

            self.pred_model.to(self.args.device)

        self.pred_model.eval()

        scores = []
        gt_list = []
        pred_list = []

        pred_channels = self.args.n_pred * self.args.nnodes * self.args.out_channels

        with torch.no_grad():
            for batch_m in data_loader:
                if self.args.is_mas:
                    batch, mas = batch_m
                    mas = mas[:, :self.args.window_size - self.args.n_pred, ...]
                else:
                    batch, mas = batch_m[0], None

                x, target = batch[:, :self.args.window_size - self.args.n_pred, ...], batch[:, -self.args.n_pred:, ...]
                output = self.pred_model(x, mas=mas)

                if self.args.real_value:
                    #     outputs = self.scaler.inverse_transform(outputs)
                    target = self.scaler.inverse_transform(target.reshape(-1, self.args.n_pred,self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                    target = torch.from_numpy(target).float().view(-1, self.args.n_pred, self.args.nnodes,self.args.out_channels).to(batch.device)

                score = torch.mean((output.reshape(-1,pred_channels) - target.reshape(-1,pred_channels)) ** 2, axis=1)##axis=(1,2,3)

                scores.append(score.detach().cpu().numpy())

                # pred_list.append(output.reshape(-1, self.args.n_pred, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())

                gt_list.append(target.reshape(-1,self.args.n_pred, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                pred_list.append(output.reshape(-1,self.args.n_pred, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())

        return scores, pred_list, gt_list


class AEModelTrainer(object):
    def __init__(self, ae_model, ae_loss, ae_optimizer,  train_loader, val_loader, test_loader, args, scaler, lr_scheduler=None):
        super(AEModelTrainer, self).__init__()
        self.ae_model = ae_model
        self.ae_loss = ae_loss
        self.ae_optimizer = ae_optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler

        self.best_path = os.path.join(self.args.log_dir, 'best_model_' + self.args.data +"_" +self.args.model + '.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')

        ## log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug, data = args.data)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))

        self.ae_channels = self.args.window_size*self.args.nnodes*self.args.in_channels

    def ae_model_batch(self, batch, training=True):
        self.ae_model.train(training)

        ## [B, T, N, C] --> [B, T * N * C]
        batch = batch.reshape(-1, self.ae_channels)

        output = self.ae_model(batch)

        if self.args.real_value:
            #     outputs = self.scaler.inverse_transform(outputs)
            batch = self.scaler.inverse_transform(batch.reshape(-1, self.args.window_size, self.args.nnodes*self.args.in_channels).detach().cpu().numpy())
            batch = torch.from_numpy(batch).view(-1, self.ae_channels).float().to(output.device)
        return output, batch

    def val_epoch(self, epoch, val_dataloader):
        self.ae_model.eval()

        ae_loss_list = []

        start_val = time.time()
        with torch.no_grad():
            for [batch] in val_dataloader:
                ## batch: [B,T,N,C]

                ## eval pred_Model
                output, batch = self.ae_model_batch(batch, training=False)
                ae_loss = self.ae_loss(output, batch)
                if not torch.isnan(ae_loss):
                    ae_loss_list.append(ae_loss.item())

                ae_loss_list.append(ae_loss.item() )

        end_val = time.time()

        ae_loss = np.array(ae_loss_list).mean()

        self.logger.info('**********Val Epoch {}: average AEModel Loss: {:.6f}, inference time: {:.3f}s'.format(epoch, ae_loss, end_val - start_val))

        return ae_loss

    def train_epoch(self, epoch):
        self.ae_model.train()

        ae_loss_list = []
        start_time = time.time()

        for [batch] in self.train_loader:
            #data and target shape: B, T, N, F; output shape: B, T, N, F
            self.ae_optimizer.zero_grad()
            ## train AE
            output, batch = self.ae_model_batch(batch, training=True)
            ae_loss = self.ae_loss(output, batch)

            ae_loss.backward()

            self.ae_optimizer.step()

            ae_loss_list.append(ae_loss.item())

        end_time = time.time()
        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()

        ae_loss = np.array(ae_loss_list).mean()

        self.logger.info(
            '**********Train Epoch {}: averaged Loss1: {:.6f}, train_time: {:.3f}s'.format(epoch, ae_loss,end_time - start_time))
        return ae_loss

    def train(self):

        best_loss = float('inf')
        not_improved_count = 0

        history = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            ## train
            train_epoch_loss = self.train_epoch(epoch)


            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader

            ## val
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            result = {'train_loss': train_epoch_loss, "val_loss": val_epoch_loss}
            history.append(result)

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            ## early_stop
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min".format((training_time / 60)))
        # self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        #save the best model to file
        self.save_checkpoint()
        self.logger.info("Saving current best model to " + self.best_path)
        return history

    def save_checkpoint(self):
        state = {
            'ae_state_dict': self.ae_model.state_dict(),
            'ae_optimizer': self.ae_optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)


class AETester(object):
    def __init__(self, ae_model, args, scaler, logger, path=None):
        self.ae_model = ae_model
        self.args = args
        self.scaler = scaler
        self.logger = logger
        self.path = path

    def testing(self, data_loader, map_location = None):
        if self.path != None:
            print("load model: ", self.path)
            check_point = torch.load(self.path, map_location = map_location)
            ae_state_dict = check_point['ae_state_dict']
            # args = check_point['config']
            self.ae_model.load_state_dict(ae_state_dict)
            self.ae_model.to(self.args.device)
        self.ae_model.eval()

        scores = []
        gt_list = []
        pred_list = []

        ae_channels = self.args.window_size * self.args.nnodes * self.args.out_channels

        with torch.no_grad():
            for [batch] in data_loader:
                ## [B, T, N, C] --> [B, T * N * C]
                batch = batch.reshape(-1, ae_channels)

                output = self.ae_model(batch)

                if self.args.real_value:
                    #     outputs = self.scaler.inverse_transform(outputs)
                    batch = self.scaler.inverse_transform(batch.reshape(-1, self.args.window_size,self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                    batch = torch.from_numpy(batch).float().view(-1, self.args.window_size, self.args.nnodes,self.args.out_channels).to(batch.device)

                score = torch.mean((output.reshape(-1, ae_channels) - batch.reshape(-1, ae_channels)) ** 2, axis=1)##axis=(1,2,3)

                scores.append(score.detach().cpu().numpy())

                # gt_list.append(batch.reshape(-1, self.args.window_size, self.args.nnodes*self.args.in_channels).detach().cpu().numpy()[:,-1,:])
                # pred_list.append(output.reshape(-1, self.args.window_size, self.args.nnodes*self.args.in_channels).detach().cpu().numpy()[:,-1,:])

                gt_list.append(batch.reshape(-1, self.args.window_size,
                                              self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                pred_list.append(output.reshape(-1, self.args.window_size,
                                                self.args.nnodes * self.args.out_channels).detach().cpu().numpy())

        return scores, pred_list, gt_list

