

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import conv2D, nodeEmbedding, graph_constructor, spatialTemporalLearningLayer, temporalLearning, Encoder, Decoder

'''
    "temp_method" is the parameter that determines which method is adopted in the spatialTemporalLearningLayer
    "temp_method" contains:
    "Conv": Conv2D layers + Graph Attention
    "Attn": Conv2D layers + Temporal-MHA (from 2 views) + Graph Attention
    "SAttn": Conv2D layers + Temporal-MHA + Spatial-MHA + Graph Attention
    where we adopt "SAttn" in STAMP
'''


class STATModel(nn.Module):
    def __init__(self, args, device, seq_len, channels_list, static_feat=None):
        super(STATModel, self).__init__()
        self.args = args
        self.nnodes = args.nnodes
        self.seq_len = seq_len
        self.layer_num = args.layer_num
        self.n_pred = args.n_pred
        self.out_channels = args.out_channels
        self.is_mas = args.is_mas
        self.temp_kernel = args.temp_kernel
        self.temp_method = args.temp_method
        

        self.idx = torch.arange(self.nnodes).to(device)
        self.gc = graph_constructor(args.nnodes, args.top_k, args.em_dim, device, alpha=args.alpha, static_feat=static_feat)

        self.input = conv2D(args.in_channels, channels_list[0][0])
        if self.is_mas:
            self.input_mas = conv2D(4, channels_list[0][0])

        self.st_layers = nn.ModuleList()
        self.t_layers = nn.ModuleList()

        for i in range(args.layer_num):
            self.st_layers.append(spatialTemporalLearningLayer(args.temp_kernel, channels_list[i], device, dropout=args.dropout, act_func = args.act_func, 
                                                            em_dim = args.em_dim, att_option = args.att_option, embed_size = args.embed_size, 
                                                            num_heads = args.num_heads, num_layers = args.num_layers, ffwd_size = args.ffwd_size, 
                                                            temp_method = args.temp_method, is_conv = args.is_conv))
            if self.is_mas and i==0:
                self.t_layers.append(temporalLearning(args,channels_list[i][0]*2, channels_list[i][-1], device, act_func="relu", dropout=args.dropout, embed_size=args.embed_size, 
                                                    num_heads=args.num_heads, num_layers=args.num_layers, ffwd_size=args.ffwd_size, temp_method=args.temp_method, 
                                                    is_conv=args.is_conv))
            else:
                self.t_layers.append(temporalLearning(args,channels_list[i][0], channels_list[i][-1], device, act_func="relu", dropout=args.dropout, embed_size=args.embed_size, 
                                                    num_heads=args.num_heads, num_layers=args.num_layers, ffwd_size=args.ffwd_size, temp_method=args.temp_method, 
                                                    is_conv=args.is_conv))

        seq_left = self.seq_len - args.layer_num * (args.temp_kernel - 1)
        
        hidden_dim = 32
        
        if self.temp_method == "Conv":    
            self.residual_conv2D = conv2D(channels_list[0][0], channels_list[-1][-1], kernel_size=(self.seq_len - seq_left + 1, 1), padding=(0, 0),
                                        stride=(1, 1), bias=True, act_func="linear")
            self.cat_conv2D = conv2D(3*channels_list[-1][-1], hidden_dim, kernel_size=(seq_left, 1), padding=(0, 0), stride=(1, 1), bias=True)
            
        elif self.temp_method == "Attn" or "SAttn":
            self.residual_conv2D = conv2D(channels_list[0][0], channels_list[-1][-1], kernel_size=(self.temp_kernel, 1), padding=(2, 0),
                                        stride=(1, 1), bias=True, act_func="linear")
            self.cat_conv2D = conv2D(3*channels_list[-1][-1], hidden_dim, kernel_size=(self.seq_len, 1), padding=(0, 0), stride=(1, 1), bias=True)


        #  cnn predictor
        self.predict = nn.Conv2d(1, self.args.n_pred * self.args.out_channels, kernel_size=(1, hidden_dim), bias=True)

    def forward(self, x, idx=None, mas = None):
        x = self.input(x)

        x_input = self.residual_conv2D(x)

        if not idx:
            adj = self.gc(self.idx)
        else:
            adj = self.gc(idx)
            
        
        z1 = z2 = x
        if self.is_mas:
            mas = self.input_mas(mas)
            z2 = torch.cat([x,mas],dim=-1)
        
        
        for i in range(self.layer_num):
            z1 = self.st_layers[i](z1, adj)
            z2 = self.t_layers[i](z2)
            

        cat_out = torch.cat([x_input,z1,z2], dim=-1)
        cat_out = self.cat_conv2D(cat_out)  ### [B, 1, N, C]

        output = self.predict(cat_out)  ### (B, T*C, N, 1)
        output = output.squeeze(-1).reshape(-1, self.args.n_pred, self.args.out_channels, self.args.nnodes)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C
        # if self.args.real_value:
        #     return output
        # else:
        #     return torch.nn.Sigmoid()(output)
        return output



class EncoderDecoder(nn.Module):
    def __init__(self, in_channels, latent_size, out_channels, is_scaled = True):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(in_channels, latent_size)
        self.decoder = Decoder(latent_size, out_channels, is_scaled)

    def forward(self, x):
        z = self.encoder(x)
        x_ = self.decoder(z)
        return x_


class Net(nn.Module):
    def __init__(self, args, latent_size, nnodes, top_k, dim, device, seq_len, temp_kernel, n_pred, in_channels, out_channels, channels_list, layer_num=2, dropout=.2, act_func = "GLU", alpha=3, static_feat=None):
        super(Net, self).__init__()

        self.n_pred = n_pred
        self.statModel = STATModel(args, device, args.window_size - args.n_pred, channels_list, static_feat=None)
        ae_in_channels = seq_len* nnodes* in_channels
        self.ae = EncoderDecoder(ae_in_channels, latent_size, ae_in_channels)

    def forward(self, x):
        '''
        :param x: [B, T, N, C]
        :return:
        '''
        B, T, N, C = x.shape
        x1 = x.reshape(-1, T*N*C)
        x1_ = self.ae(x1)

        x2 = x[:,:T-self.n_pred,...]
        statmodel_gt = x[:,-self.n_pred:,...]
        pre_out = self.statModel(x2)

        x2 = torch.cat([pre_out, statmodel_gt], dim=1)
        x2 = x2.reshape(-1, T*N*C)
        x2_ = self.ae(x2)
        return x1,x1_, statmodel_gt, pre_out, x2, x2_



