import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import networkx as nx
from sklearn.metrics import roc_curve,roc_auc_score

import os
base_dir = os.getcwd()

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
    
def plot_history(history, model = "gat", mode = "train", data="swat"):
    fig = plt.figure()
    losses1 = [x[mode + '_loss1'] for x in history]
    losses2 = [x[mode + '_loss2'] for x in history]
    plt.plot(losses1, '-x', label="loss1")
    plt.plot(losses2, '-x', label="loss2")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.savefig(base_dir + "/" + data + "_" + model + "_"+ mode + "_history.pdf", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_history2(history, model = "gat", mode = "val",data="swat"):
    fig = plt.figure()
    losses1 = [x[mode + '_loss1'] for x in history]
    losses2 = [x[mode + '_loss2'] for x in history]
    losses3 = [x[mode + '_pred_loss'] for x in history]
    losses4 = [x[mode + '_ae_loss'] for x in history]
    losses5 = [x[mode + '_adv_loss'] for x in history]
    plt.plot(losses1, '-x', label="loss1")
    plt.plot(losses2, '-x', label="loss2")
    plt.plot(losses3, '-x', label="pred_loss")
    plt.plot(losses4, '-x', label="ae_loss")
    plt.plot(losses5, '-x', label="adv_loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.savefig(base_dir + "/" + data + "_" + model +"_"+ mode  + "_losses_history.pdf", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_history_pred(history, model = "gat",data="swat"):
    fig = plt.figure()
    train_loss = [x['train_loss'] for x in history]
    val_loss = [x['val_loss'] for x in history]

    plt.plot(train_loss, '-x', label="train")
    plt.plot(val_loss, '-x', label="val")

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.savefig(base_dir + "/" + model +"_losses_history.pdf", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    
def histogram(y_test,y_pred):
    plt.figure(figsize=(12,6))
    plt.hist([y_pred[y_test==0],
              y_pred[y_test==1]],
            bins=20,
            color = ['#82E0AA','#EC7063'],stacked=True)
    plt.title("Results",size=20)
    plt.grid()
    plt.show()
    
def ROC(y_test, y_pred):
    ### tr: thresholds
    fpr,tpr,tr = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    idx = np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.plot(fpr,1-fpr,'r:')
    plt.plot(fpr[idx],tpr[idx], 'ro')
    plt.legend(loc=4)
    plt.grid()
    plt.show()
    return tr[idx]
    
def confusion_matrix(target, predicted, perc=False):

    data = {'y_Actual':    target,
            'y_Predicted': predicted
            }
    df = pd.DataFrame(data, columns=['y_Predicted','y_Actual'])
    confusion_matrix = pd.crosstab(df['y_Predicted'], df['y_Actual'], rownames=['Predicted'], colnames=['Actual'])
    
    if perc:
        sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, fmt='.2%', cmap='Blues')
    else:
        sns.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.show()

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class MinMaxScaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min

### 多通道输入数据
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, mas):
        self.x = x
        self.mas = mas

    def __getitem__(self, index):
        x, mas = self.x[index], self.mas[index]
        return x, mas

    def __len__(self):
        return len(self.x) # len(self.data1) = len(self.data2)

# downsample by 10
def downsample(np_data, down_len, is_label = False):
    down_time_len = len(np_data) // down_len
    if is_label:
        d_data = np_data[:down_time_len * down_len].reshape(-1, down_len)
        # if exist anomalies, then this sample is abnormal
        d_data = np.round(np.max(d_data, axis=1))
    else:
        orig_len, col_num = np_data.shape

        down_time_len = orig_len // down_len

        np_data = np_data.transpose()

        d_data = np_data[:, :down_time_len*down_len].reshape(col_num, -1, down_len)
        d_data = np.median(d_data, axis=2).reshape(col_num, -1)

        d_data = d_data.transpose()

    return d_data

def concate_list(data_list):
    return np.concatenate(data_list)

def concate_results(data_results):
    return [concate_list(data_list) for data_list in data_results]

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def np_ma(data,w):
    _,feature_num = data.shape
    res = np.zeros(data.shape)
    res[:w-1,:] = data[:w-1,:]
    for i in range(feature_num):
        res[w-1:,i] = moving_average(data[:,i], w)
    return res

def to_dict(name_list):
    name2indices = dict(zip(name_list,list(range(len(name_list)))))
    indices2name = dict(zip(list(range(len(name_list))),name_list))
    return name2indices,indices2name

def get_topk_name(indices,indices2name):
    return [indices2name.get(i) for i in indices]


def plot_gc(adj, threshold=0.1):
    G = nx.Graph()

    N = len(adj)
    neighbor = []
    for i in range(N):
        topk = np.argwhere(adj[i] >= threshold).squeeze()

        topk = topk[:2]
        for k in topk:
            G.add_edge(str(i), str(k), weight=np.random.random())

            # if k not in [5, 9]:
            #     G.add_edge(str(i), str(k), weight=np.random.random())
            #     if k == 11:
            #         neighbor.append(i)

    plt.figure(figsize=(8, 8))

    pos = nx.spring_layout(G)

    val_map = {'5': "blue",
               #            '9': "green",
               '11': "red"}

    value1 = [100 if node not in val_map else 200 for node in G.nodes()]
    value2 = [val_map.get(node, "indigo") for node in G.nodes()]

    # Specify the edges you want here
    red_edges = [(str(i), "11") for i in neighbor]
    edge_colours = ['black' if not edge in red_edges else 'red'
                    for edge in G.edges()]
    black_edges = [edge for edge in G.edges() if edge not in red_edges]

    nodes = nx.draw_networkx_nodes(G, pos, node_size=value1, node_color=value2)

    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='red', arrows=True, width=2, )
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)

    # nx.draw_networkx_labels(G, pos)

    ax = plt.gca()
    ax.set_axis_off()

    # plt.savefig(r"C:\Users\14565\Desktop\gc.png", dpi=400, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    normal = np.arange(20)
    window_size = 10
    normal = normal[np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size + 1)[:, None]]
    print(normal)