

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numbers
import math

'''
    "temp_method" is the parameter that determines which method is adopted in the spatialTemporalLearningLayer
    "temp_method" contains:
    "Conv": Conv2D layers + Graph Attention
    "Attn": Conv2D layers + Temporal-MHA (from 2 views) + Graph Attention
    "SAttn": Conv2D layers + Temporal-MHA + Spatial-MHA + Graph Attention
    where we adopt "SAttn" in STAMP
'''


class conv2D(nn.Module):
    '''
    input: [B, H, W, C_in]
    output: [B, H_out, W_out, C_out]
    '''

    def __init__(self, c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True, act_func="linear"):
        super(conv2D, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.act_func = act_func
        self.FC = torch.nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)

    def forward(self, x):
        '''
        :param x: [B, T, N, C]
        :return:
        '''
        if self.act_func == "linear":
            return self.FC(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        elif self.act_func == "relu":
            return F.relu(self.FC(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        else:
            return torch.sigmoid(self.FC(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        '''
        :param idx: self.idx = torch.arange(self.num_nodes).to(device)
        :return:
        '''
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj


class nodeEmbedding(nn.Module):
    def __init__(self, nnodes, dim):
        super(nodeEmbedding, self).__init__()
        self.nnodes = nnodes
        self.emb1 = nn.Embedding(nnodes, dim)
        self.emb2 = nn.Embedding(nnodes, dim)

    def forward(self, idx):
        emb1 = self.emb1(idx)
        emb2 = self.emb2(idx)
        return emb1, emb2




class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, dropout=.0):
        super(Spatial_Attention_layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        '''
        :param x: (batch_size, T, N, F_in)
        :return: (batch_size, T, N, N)
        '''
        batch_size, num_of_timesteps, num_of_vertices, in_channels = x.shape

        x = x.reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

        score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)

        score = self.dropout(F.softmax(score, dim=-1))  # the sum of each row is 1; (b*t, N, N)

        return score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))



class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=.0):
        super(GATLayer, self).__init__()
        # self.sym_norm_Adj_matrix = adj  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)
        self.dropout = dropout

    def forward(self, x, adj, emb1=None, emb2=None):
        '''
        spatial graph convolution operation
        :param x: (batch_size, T, N,  F_in)
        :return: (batch_size, T, N, F_out)
        '''
        x_ = self.Theta(x)

        batch_size, num_of_timesteps, num_of_vertices, out_channels = x_.shape
        x_ = x_.reshape((-1, num_of_vertices, out_channels))  # (b*t,n,f_in)

        spatial_attention = torch.matmul(x_, x_.transpose(1, 2)) / math.sqrt(
            out_channels)  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)

        zero_vec = -9e15 * torch.ones_like(spatial_attention)
        adj = adj.expand(spatial_attention.shape[0], num_of_vertices, num_of_vertices)
        attention = torch.where(adj > 0, spatial_attention, zero_vec)

        attention = F.softmax(attention, dim=-1)  ###
        attention = F.dropout(attention, self.dropout, training=self.training)

        x_ = torch.matmul(attention, x_)

        # (b*t, n, f_in)->(b*t, n, f_out)->(b,t,n,f_out)
        return F.relu(x_.reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)))


class GATLayer2(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=.0):
        super(GATLayer2, self).__init__()
        # self.sym_norm_Adj_matrix = adj  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)
        self.dropout = dropout

    def forward(self, x, adj, emb1=None, emb2=None):
        '''
        spatial graph convolution operation
        :param x: (batch_size, T, N,  F_in)
        :return: (batch_size, T, N, F_out)
        '''
        x_ = self.Theta(x)

        batch_size, num_of_timesteps, num_of_vertices, out_channels = x_.shape
        x_ = x_.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, num_of_timesteps * out_channels))  # (b,n,t*f_in)

        spatial_attention = torch.matmul(x_, x_.transpose(1, 2)) / math.sqrt(
            num_of_timesteps * out_channels)  # (b, N, t*F_in)(b, t*F_in, N)=(b, N, N)

        zero_vec = -9e15 * torch.ones_like(spatial_attention)
        adj = adj.expand(spatial_attention.shape[0], num_of_vertices, num_of_vertices)
        attention = torch.where(adj > 0, spatial_attention, zero_vec)

        attention = F.softmax(attention, dim=-1)  ###(b,n,n)
        attention = F.dropout(attention, self.dropout, training=self.training)

        x_ = torch.matmul(attention, x_)

        # (b, n, t*f_out)->(b, n, t, f_out)->(b,t,n,f_out)
        return F.relu(
            x_.reshape((batch_size, num_of_vertices, num_of_timesteps, self.out_channels)).permute(0, 2, 1, 3))


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj, emb1=None, emb2=None):
        batch_size, num_of_timesteps, num_of_vertices, in_channels = x.shape

        # x: [B, T, N, C], Wh: [B, T, N, C']--> [B * T, N, C']
        Wh = torch.mm(x, self.W).reshape((-1, num_of_vertices, self.out_features))
        e = self._prepare_attentional_mechanism_input(Wh)  ## (B * T, N, N)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape: [B*T, N, C']
        # self.a.shape (2 * C', 1)
        # Wh1&2.shape (B*T, N, 1)
        # e.shape (B*T, N, N)

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer2(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, em_dim, alpha=0.2, concat=True):
        super(GraphAttentionLayer2, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.em_dim = em_dim

        self.W = nn.Linear(in_features, out_features, bias=False)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.a2 = nn.Parameter(torch.empty(size=(2 * em_dim, 1)))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj, emb1=None, emb2=None):
        batch_size, num_of_timesteps, num_of_vertices, in_channels = x.shape

        # x: [B, T, N, C], Wh: [B, T, N, C']--> [B * T, N, C']
        Wh = self.W(x).reshape((-1, num_of_vertices, self.out_features))
        e = self._prepare_attentional_mechanism_input(Wh, emb1, emb2)  ## (B * T, N, N)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        # print("h_prime: ", h_prime.shape)

        if self.concat:
            return F.elu(h_prime.reshape((-1, num_of_timesteps, num_of_vertices, self.out_features)))
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh, emb1=None, emb2=None):
        # Wh.shape: [B*T, N, C']
        # self.a.shape (2 * C', 1)
        # Wh1&2.shape (B*T, N, 1)
        # e.shape (B*T, N, N)

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e1 = Wh1 + Wh2.permute(0, 2, 1)

        Wh1 = torch.matmul(emb1, self.a2[:self.em_dim, :])
        Wh2 = torch.matmul(emb2, self.a2[self.em_dim:, :])
        # broadcast add
        e2 = Wh1 + Wh2.T

        e2 = e2.expand(e1.shape[0], e2.shape[0], e2.shape[1])
        return self.leakyrelu(e1 + e2)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    

class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, num_layers, ffwd_size, temp_kernel, is_conv, device):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.is_conv = is_conv
        self.dropout_rate = dropout
        self.embed_size = embed_size
        self.device = device
        
        # Multi-head Self Attention
        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=self.dropout_rate)
            for _ in range(num_layers)
        ])
        
        # FFN
        if self.is_conv:
            self.position_ffwd = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(embed_size, ffwd_size, kernel_size=temp_kernel, stride=1, padding=(temp_kernel-1)//2),
                    nn.ReLU(),
                    nn.Conv1d(ffwd_size, embed_size, kernel_size=temp_kernel, stride=1, padding=(temp_kernel-1)//2)
                )
                for _ in range(num_layers)
            ])
            
        else:
            self.position_ffwd = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_size, ffwd_size),
                    nn.ReLU(),
                    nn.Linear(ffwd_size, embed_size)
                )
                for _ in range(num_layers)
            ])
        
                
    def forward(self, x, mask=None):
        attn_mask = mask 
        
        out = nn.Linear(x.shape[2], self.embed_size, device=self.device)(x) # (bs, seq_len, embed_size)

        for i in range(self.num_layers):
            
            init_out = out

            out, _ = self.self_attn[i](init_out, init_out, init_out, attn_mask=attn_mask, key_padding_mask=mask)
            
            out = nn.Dropout(self.dropout_rate)(out) + init_out
            
            out = nn.LayerNorm(self.embed_size, device=self.device)(out)
            
            # FFN
            if self.is_conv:
                ffwd_out = out.permute(0,2,1) # (bs, embed_size, seq_len)
                ffwd_out = self.position_ffwd[i](ffwd_out)
                ffwd_out = ffwd_out.permute(0,2,1) # (bs, seq_len, embed_size)
            else:
                ffwd_out = self.position_ffwd[i](out)
                
            out = nn.Dropout(self.dropout_rate)(ffwd_out) + out
            out = nn.LayerNorm(self.embed_size, device=self.device)(out)
        
        
        return out


class spatialTemporalLearningLayer(nn.Module):
    def __init__(self, temp_kernel, channels, device, dropout=.2, act_func="GLU", em_dim=4, att_option=2, embed_size=64, 
                num_heads=8, num_layers=1, ffwd_size=64, temp_method="Conv", is_conv=False):
        super(spatialTemporalLearningLayer, self).__init__()
        in_channels, graph_in_channels, out_channels = channels #[[16 8 32] [32 8 64]]
        self.act_func = act_func
        self.out_channels = out_channels
        self.temp_method = temp_method
        self.embed_size = embed_size
        self.ffwd_size = ffwd_size
        self.graph_in_channels = graph_in_channels
        self.is_conv = is_conv
        self.device = device
        
        if self.temp_method == "Conv":   
            self.residual_conv2D = conv2D(in_channels, out_channels, kernel_size=(temp_kernel, 1), padding=(0, 0),
                                        stride=(1, 1),
                                        bias=True, act_func="linear")
        elif self.temp_method == "Attn":
            self.residual_conv2D = conv2D(in_channels, out_channels, kernel_size=(temp_kernel, 1), padding=(2, 0),
                                        stride=(1, 1),
                                        bias=True, act_func="linear")
            self.temp_attn = TransformerEncoder(self.embed_size, num_heads=num_heads, dropout=dropout, 
                                                    num_layers=num_layers, ffwd_size=ffwd_size, temp_kernel=temp_kernel, is_conv=is_conv, device=self.device)
        elif self.temp_method == "SAttn":
            self.residual_conv2D = conv2D(in_channels, out_channels, kernel_size=(temp_kernel, 1), padding=(2, 0),
                                        stride=(1, 1),
                                        bias=True, act_func="linear")
            self.spat_attn = TransformerEncoder(self.embed_size, num_heads=num_heads, dropout=dropout, 
                                                    num_layers=num_layers, ffwd_size=ffwd_size, temp_kernel=temp_kernel, is_conv=is_conv, device=self.device, 
                                                )
            
        self.bn = torch.nn.BatchNorm2d(out_channels)
        

        if self.act_func in ["glu", "GLU", "gtu", "GTU"]:
            
            self.temp_conv = conv2D(in_channels, 2 * graph_in_channels, kernel_size=(temp_kernel, 1), bias=True,
                                        act_func="linear")
            
            if att_option == 1:
                self.gat = GATLayer(2 * graph_in_channels, 2 * out_channels, dropout=dropout)
            elif att_option == 2:
                self.gat = GATLayer2(2 * graph_in_channels, 2 * out_channels, dropout=dropout)
            elif att_option == 3:
                self.gat = GraphAttentionLayer(2 * graph_in_channels, 2 * out_channels, dropout=dropout, alpha=0.2,
                                                concat=True)
            else:
                self.gat = GraphAttentionLayer2(2 * graph_in_channels, 2 * out_channels, dropout=dropout, em_dim=em_dim,
                                                alpha=0.2, concat=True)
        else:
            self.temp_conv = conv2D(in_channels, graph_in_channels, kernel_size=(temp_kernel, 1), bias=True,
                                    act_func="linear")
            
            if att_option == 1:
                self.gat = GATLayer(graph_in_channels, out_channels, dropout=dropout)
            elif att_option == 2:
                self.gat = GATLayer2(graph_in_channels, out_channels, dropout=dropout)
            elif att_option == 3:
                self.gat = GraphAttentionLayer(graph_in_channels, out_channels, dropout=dropout, alpha=0.2, concat=True)
            else:
                self.gat = GraphAttentionLayer2(2 * graph_in_channels, 2 * out_channels, dropout=dropout, em_dim=em_dim,
                                                alpha=0.2, concat=True)

    def forward(self, x, adj, emb1=None, emb2=None):
        '''
        :param x: [B, T, N, C]
        :param adj: [N, N]
        :return: [B, T-K+1, N, C']
        '''
        ### residual
        x_input = self.residual_conv2D(x)
        
        if self.temp_method == "Conv":
            temporal_out = self.temp_conv(x) # (B, T-k+1, N, in_channels)
            
        elif self.temp_method == "Attn":
            seq_len = x.shape[1]
            num_nodes = x.shape[2]
            in_channels = x.shape[3]
            temporal_in = x.reshape(-1, seq_len, num_nodes * in_channels) # (B, T, N*C)
            attn_out = self.temp_attn(temporal_in) # (B, T, embed_size)
            
            if self.act_func in ["glu", "GLU", "gtu", "GTU"]:
                temporal_out = nn.Linear(self.embed_size, 2 * num_nodes * self.graph_in_channels, device=self.device)(attn_out) # (B, T, N*in_channels)
                temporal_out = temporal_out.reshape(-1, seq_len, num_nodes, 2 * self.graph_in_channels) # (B, T, N, in_channels)
            else:
                temporal_out = nn.Linear(self.embed_size, num_nodes * self.graph_in_channels, device=self.device)(attn_out)
                temporal_out = temporal_out.reshape(-1, seq_len, num_nodes, self.graph_in_channels)
            
        elif self.temp_method == "SAttn":
            seq_len = x.shape[1]
            num_nodes = x.shape[2]
            in_channels = x.shape[3]
            spatial_in = x.reshape(-1, num_nodes, in_channels) # (B*T, N, C)
            
            attn_out = self.spat_attn(spatial_in) # (B*T, N, embed_size)
            
            if self.act_func in ["glu", "GLU", "gtu", "GTU"]:
                temporal_out = nn.Linear(self.embed_size, 2 * self.graph_in_channels, device=self.device)(attn_out) # (B*T, N, in_channels)
                temporal_out = temporal_out.reshape(-1, seq_len, num_nodes, 2 * self.graph_in_channels) # (B, T, N, in_channels)
            else:
                temporal_out = nn.Linear(self.embed_size, self.graph_in_channels, device=self.device)(attn_out)
                temporal_out = temporal_out.reshape(-1, seq_len, num_nodes, self.graph_in_channels)
                
        spatial_out = self.gat(temporal_out, adj, emb1, emb2)
        
        if self.act_func in ["glu", "GLU"]:
            sp_out = (spatial_out[..., :self.out_channels] + x_input) * torch.sigmoid(
                spatial_out[..., self.out_channels:])
            return self.bn(sp_out.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        elif self.act_func in ["gtu", "GTU"]:
            sp_out = torch.tanh(spatial_out[..., :self.out_channels] + x_input) * torch.sigmoid(
                spatial_out[..., self.out_channels:])
            return self.bn(sp_out.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        elif self.act_func in ["relu", "ReLU"]:
            out = F.relu(spatial_out + x_input)
            return self.bn(out.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        elif self.act_func in ['sigmoid', "SIGMOID"]:
            out = F.sigmoid(spatial_out + x_input)
            return self.bn(out.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        elif self.act_func == "linear":
            out = spatial_out + x_input
            return self.bn(out.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        else:
            raise TypeError
        



class temporalLearning(nn.Module):
    def __init__(self, args, in_channels, out_channels, device, act_func="relu", dropout=.2, embed_size=64, 
                num_heads=8, num_layers=1, ffwd_size=64, temp_method="Conv", is_conv=False):
        super(temporalLearning, self).__init__()
        self.temp_method = temp_method
        self.out_channels = out_channels
        self.device = device
        self.embed_size = embed_size
        self.tconvs = nn.ModuleList()
        
        temp_kernels = [2, 3, 5]
        
        if self.temp_method == "Conv":
            for kt in temp_kernels:
                self.tconvs.append(conv2D(in_channels, out_channels, kernel_size=(kt, 1), act_func=act_func))
            self.out = conv2D(len(temp_kernels) * out_channels, out_channels, act_func=act_func)
            
        elif self.temp_method == "Attn":
            self.temp_attn = TransformerEncoder(self.embed_size, num_heads=num_heads, dropout=dropout, 
                                                    num_layers=num_layers, ffwd_size=ffwd_size, temp_kernel=temp_kernels[-1], is_conv=is_conv, device=self.device)
        
        elif self.temp_method == "SAttn":
            self.temp_attn = TransformerEncoder(self.embed_size, num_heads=num_heads, dropout=dropout, 
                                                    num_layers=num_layers, ffwd_size=ffwd_size, temp_kernel=temp_kernels[-1], is_conv=is_conv, device=self.device)
        

    def forward(self, x):
        '''
        :param x: [B, T, N, C]
        :return: [B, T-5+1, N, C'] (Conv)
                [B, T, N, C'] (Attn)
        '''
        if self.temp_method == "Conv":
            x_list = []
            for layer in self.tconvs:
                x_list.append(layer(x))
            for i in range(len(x_list)):
                x_list[i] = x_list[i][:, -x_list[-1].size(1):, ...]
            x = torch.cat(x_list, dim=-1)
            x = self.out(x)
            
        elif self.temp_method == "Attn":
            seq_len = x.shape[1]
            num_nodes = x.shape[2]
            in_channels = x.shape[3]
            temporal_in = x.reshape(-1, seq_len, num_nodes * in_channels)#(B, T, N*C)
            attn_out = self.temp_attn(temporal_in) #(B, T, embed_size)
            temporal_out = nn.Linear(self.embed_size, num_nodes * self.out_channels, device=self.device)(attn_out)
            x = temporal_out.reshape(-1, seq_len, num_nodes, self.out_channels)
        
        elif self.temp_method == "SAttn":
            seq_len = x.shape[1]
            num_nodes = x.shape[2]
            in_channels = x.shape[3]
            temporal_in = x.reshape(-1, seq_len, num_nodes * in_channels)#(B, T, N*C)
            attn_out = self.temp_attn(temporal_in) #(B, T, embed_size)
            temporal_out = nn.Linear(self.embed_size, num_nodes * self.out_channels, device=self.device)(attn_out)
            x = temporal_out.reshape(-1, seq_len, num_nodes, self.out_channels)    
            
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, int(in_channels / 2))
        self.linear2 = nn.Linear(int(in_channels / 2), int(in_channels / 4))
        self.linear3 = nn.Linear(int(in_channels / 4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_size, out_channels, is_scaled=True):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_channels / 4))
        self.linear2 = nn.Linear(int(out_channels / 4), int(out_channels / 2))
        self.linear3 = nn.Linear(int(out_channels / 2), out_channels)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        self.is_scaled = is_scaled

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        x = self.linear3(out)
        if self.is_scaled:
            x = self.sigmoid(x)
        return x
