
import os

import pandas as pd
from sklearn import preprocessing
import pickle

from lib.utils import *
#from utils import *

def preprocessTrainingData(file, sep=None, min_max_scaler = None, training = True):
    # === Normal period ====
    normal = pd.read_csv(file, sep= sep)  # , nrows=1000)
    if "Attack" in normal.columns:
        normal = normal.drop(["Timestamp", "Attack"], axis=1)
    else:
        normal = normal.drop(["Timestamp", 'Normal/Attack'], axis=1)

    # Transform all columns into float64
    for i in list(normal):
        normal[i] = normal[i].apply(lambda x: str(x).replace(",", "."))
    normal = normal.astype(float)

    normal = normal.fillna(normal.mean())
    normal = normal.fillna(0)

    return normal.values, min_max_scaler

def preprocessTestingData(file, sep=None, min_max_scaler = None, training=False):
    # === Normal period ====
    attack = pd.read_csv(file, sep= sep)  # , nrows=1000)

    if "Attack" in attack.columns:
        labels = attack["Attack"].values
        attack = attack.drop(["Timestamp", "Attack"], axis=1)
    else:
        labels = [float(label != 'Normal') for label in attack["Normal/Attack"].values]
        attack = attack.drop(["Timestamp", 'Normal/Attack'], axis=1)

    # Transform all columns into float64
    for i in list(attack):
        attack[i] = attack[i].apply(lambda x: str(x).replace(",", "."))
    attack = attack.astype(float)

    attack = attack.fillna(attack.mean())
    attack = attack.fillna(0)

    return attack.values, labels


def load_data(train_filename, test_filename, device = "gpu", window_size = 12, val_ratio = 0.2, batch_size = 64, is_down_sample = False, down_len=10):

    ## EDA - Data Pre-Processing
    normal, min_max_scaler = preprocessTrainingData(train_filename, sep=None, min_max_scaler=None, training=True)  # , nrows=1000)
    attack, labels = preprocessTestingData(test_filename, sep=None, min_max_scaler=min_max_scaler, training=False)  # , nrows=1000)
    ### Add Moving Average (MA)

    # normal: (495000, 45)
    # attack: (449919, 45)
    normal = normal[21600:,:]
    ## down sample
    if is_down_sample:
        normal = downsample(normal, down_len=down_len, is_label=False)
        attack = downsample(attack, down_len=down_len, is_label=False)
        labels = downsample(labels, down_len=down_len, is_label=True)

    # ## nomalization
    # min = normal.min(axis=0)##axis=0
    # max = normal.max(axis=0)##axis=0
    # min_max_scaler = MinMaxScaler(min, max)
    # normal = min_max_scaler.transform(normal)

    min_max_scaler = preprocessing.MinMaxScaler()
    normal = min_max_scaler.fit_transform(normal)
    attack = min_max_scaler.transform(attack)
    

    # windows_attack = min_max_scaler.transform(windows_attack)

    windows_normal = normal[np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size + 1)[:, None]]
    # print(windows_normal.shape)  # (494988, 12, 51)

    windows_attack = attack[np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]
    # print(windows_attack.shape)  # (449907, 12, 51)

    ### train/val/test
    windows_normal_train = windows_normal[:int(np.floor((1-val_ratio) * windows_normal.shape[0]))]
    windows_normal_val = windows_normal[int(np.floor((1-val_ratio) * windows_normal.shape[0])):]

    ## reshape: [B, T, N ,C]
    windows_normal_train = windows_normal_train.reshape(windows_normal_train.shape + (1,))
    windows_normal_val = windows_normal_val.reshape(windows_normal_val.shape + (1,))
    windows_attack = windows_attack.reshape(windows_attack.shape + (1,))

    ## train
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(windows_normal_train).float().to(device))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    ## val
    val_data = torch.utils.data.TensorDataset(torch.from_numpy(windows_normal_val).float().to(device))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    ## test
    test_data = torch.utils.data.TensorDataset(torch.from_numpy(windows_attack).float().to(device))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    ## test labels
    windows_labels = []
    for i in range(len(labels) - window_size + 1):
        windows_labels.append(list(np.int_(labels[i:i + window_size])))

    y_test_labels = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]

    print("train: ", type(windows_normal_train.shape), windows_normal_train.shape)
    print("val: ", windows_normal_val.shape)
    print("test: ", windows_attack.shape)
    print("test labels: ", len(y_test_labels))

    return train_loader, val_loader, test_loader, y_test_labels, min_max_scaler

def load_data2(train_filename, test_filename, device = "gpu", window_size = 12, val_ratio = 0.2, batch_size = 64, is_down_sample = False, down_len=10):

    ## EDA - Data Pre-Processing
    normal, min_max_scaler = preprocessTrainingData(train_filename, sep=None, min_max_scaler=None, training=True)  # , nrows=1000)
    attack, labels = preprocessTestingData(test_filename, sep=None, min_max_scaler=min_max_scaler, training=False)  # , nrows=1000)
    labels = np.array(labels)
    print("attack: ", attack.shape)
    print("labels: ", labels.shape)

    ### Add Moving Average (MA)
    window_sizes = [3,5,10,20]
    normal_mas = []
    attack_mas = []
    for w in window_sizes:
        normal_ma = np_ma(normal, w)
        normal_mas.append(normal_ma)

        attack_ma = np_ma(attack, w)
        attack_mas.append(attack_ma)

    # normal: (495000, 45)
    # attack: (449919, 45)
    normal = normal[21600:,:]
    for i in range(len(window_sizes)):
        normal_mas[i] = normal_mas[i][21600:,:]

    W = np.max(window_sizes)
    attack = attack[W:, :]
    labels = labels[W:]
    for i in range(len(window_sizes)):
        attack_mas[i] = attack_mas[i][W:, :]

    ## down sample
    if is_down_sample:
        normal = downsample(normal, down_len=down_len, is_label=False)
        attack = downsample(attack, down_len=down_len, is_label=False)
        labels = downsample(labels, down_len=down_len, is_label=True)

        for i in range(len(window_sizes)):
            normal_mas[i] = downsample(normal_mas[i], down_len=down_len, is_label=False)
            attack_mas[i] = downsample(attack_mas[i], down_len=down_len, is_label=False)

    # ## nomalization
    # min = normal.min()##axis=0
    # max = normal.max()##axis=0
    # min_max_scaler = MinMaxScaler(min, max)

    min_max_scaler = preprocessing.MinMaxScaler()
    normal = min_max_scaler.fit_transform(normal)
    attack = min_max_scaler.transform(attack)

    for i in range(len(window_sizes)):
        normal_mas[i] = min_max_scaler.transform(normal_mas[i])
        attack_mas[i] = min_max_scaler.transform(attack_mas[i])
        
    print(np.array(normal_mas).shape, np.array(attack_mas).shape)

    windows_normal = normal[np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size + 1)[:, None]]
    #print(windows_normal.shape)  # (494988, 12, 51)

    windows_attack = attack[np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]
    #print(windows_attack.shape)  # (449907, 12, 51)

    for i in range(len(window_sizes)):
        normal_mas[i] = normal_mas[i][np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size + 1)[:, None]]
        attack_mas[i] = attack_mas[i][np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]

    windows_normal_mas = np.stack(normal_mas, axis=-1)
    windows_attack_mas = np.stack(attack_mas, axis=-1)
    


    ### train/val/test
    windows_normal_train = windows_normal[:int(np.floor((1-val_ratio) * windows_normal.shape[0]))]
    windows_normal_val = windows_normal[int(np.floor((1-val_ratio) * windows_normal.shape[0])):]

    windows_normal_mas_train = windows_normal_mas[:int(np.floor((1-val_ratio) * windows_normal.shape[0]))]
    windows_normal_mas_val = windows_normal_mas[int(np.floor((1-val_ratio) * windows_normal.shape[0])):]


    ## reshape: [B, T, N ,C]
    windows_normal_train = windows_normal_train.reshape(windows_normal_train.shape + (1,))
    windows_normal_val = windows_normal_val.reshape(windows_normal_val.shape + (1,))
    windows_attack = windows_attack.reshape(windows_attack.shape + (1,))

    print("windows_normal_train: ", windows_normal_train.shape)
    print("windows_normal_mas_train: ", windows_normal_mas_train.shape)
    print("windows_normal_val: ", windows_normal_val.shape)
    print("windows_normal_mas_val: ", windows_normal_mas_val.shape)
    print("windows_attack: ", windows_attack.shape)
    print("windows_attack_mas: ", windows_attack_mas.shape)

    ## train
    train_data_tensor = torch.from_numpy(windows_normal_train).float().to(device)
    train_mas_data_tensor = torch.from_numpy(windows_normal_mas_train).float().to(device)

    train_dataset = MyDataset(train_data_tensor,train_mas_data_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    ## val
    val_data_tensor = torch.from_numpy(windows_normal_val).float().to(device)
    val_mas_data_tensor = torch.from_numpy(windows_normal_mas_val).float().to(device)

    val_dataset = MyDataset(val_data_tensor, val_mas_data_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    ## test
    test_data_tensor = torch.from_numpy(windows_attack).float().to(device)
    test_mas_data_tensor = torch.from_numpy(windows_attack_mas).float().to(device)

    test_dataset = MyDataset(test_data_tensor, test_mas_data_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    ## test labels
    windows_labels = []
    for i in range(len(labels) - window_size + 1):
        windows_labels.append(list(np.int_(labels[i:i + window_size])))

    y_test_labels = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]

    print("train: ", type(windows_normal_train.shape), windows_normal_train.shape)
    print("val: ", windows_normal_val.shape)
    print("test: ", windows_attack.shape)
    print("test labels: ", len(y_test_labels))

    return train_loader, val_loader, test_loader, y_test_labels, min_max_scaler

def load_data3(normal, attack, labels, device = "gpu", window_size = 12, val_ratio = 0.2, batch_size = 64, is_down_sample = False, down_len=10):
    #用有标签数据（含异常）训练和测试

    labels = np.array(labels)
    print("normal: ", normal.shape)
    print("attack: ", attack.shape)
    print("labels: ", labels.shape)

    ### Add Moving Average (MA)
    window_sizes = [3,5,10,20]
    normal_mas = []
    attack_mas = []
    for w in window_sizes:
        normal_ma = np_ma(normal, w)
        normal_mas.append(normal_ma)

        attack_ma = np_ma(attack, w)
        attack_mas.append(attack_ma)
    '''
    normal = normal[21600:,:]
    for i in range(len(window_sizes)):
        normal_mas[i] = normal_mas[i][21600:,:]
    '''
    W = np.max(window_sizes)
    attack = attack[W:, :]
    labels = labels[W:]
    for i in range(len(window_sizes)):
        attack_mas[i] = attack_mas[i][W:, :]

    ## down sample
    if is_down_sample:
        normal = downsample(normal, down_len=down_len, is_label=False)
        attack = downsample(attack, down_len=down_len, is_label=False)
        labels = downsample(labels, down_len=down_len, is_label=True)

        for i in range(len(window_sizes)):
            normal_mas[i] = downsample(normal_mas[i], down_len=down_len, is_label=False)
            attack_mas[i] = downsample(attack_mas[i], down_len=down_len, is_label=False)

    # ## nomalization
    # min = normal.min()##axis=0
    # max = normal.max()##axis=0
    # min_max_scaler = MinMaxScaler(min, max)

    min_max_scaler = preprocessing.MinMaxScaler()
    normal = min_max_scaler.fit_transform(normal)
    attack = min_max_scaler.transform(attack)

    for i in range(len(window_sizes)):
        normal_mas[i] = min_max_scaler.transform(normal_mas[i])
        attack_mas[i] = min_max_scaler.transform(attack_mas[i])
    

    windows_normal = normal[np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size + 1)[:, None]]
    

    windows_attack = attack[np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]
    

    for i in range(len(window_sizes)):
        normal_mas[i] = normal_mas[i][np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size + 1)[:, None]]
        attack_mas[i] = attack_mas[i][np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]

    windows_normal_mas = np.stack(normal_mas, axis=-1)
    windows_attack_mas = np.stack(attack_mas, axis=-1)


    ### train/val/test
    windows_normal_train = windows_normal[:int(np.floor((1-val_ratio) * windows_normal.shape[0]))]
    windows_normal_val = windows_normal[int(np.floor((1-val_ratio) * windows_normal.shape[0])):]

    windows_normal_mas_train = windows_normal_mas[:int(np.floor((1-val_ratio) * windows_normal.shape[0]))]
    windows_normal_mas_val = windows_normal_mas[int(np.floor((1-val_ratio) * windows_normal.shape[0])):]


    ## reshape: [B, T, N ,C]
    windows_normal_train = windows_normal_train.reshape(windows_normal_train.shape + (1,))
    windows_normal_val = windows_normal_val.reshape(windows_normal_val.shape + (1,))
    windows_attack = windows_attack.reshape(windows_attack.shape + (1,))

    print("windows_normal_train: ", windows_normal_train.shape)
    print("windows_normal_mas_train: ", windows_normal_mas_train.shape)
    print("windows_normal_val: ", windows_normal_val.shape)
    print("windows_normal_mas_val: ", windows_normal_mas_val.shape)
    print("windows_attack: ", windows_attack.shape)
    print("windows_attack_mas: ", windows_attack_mas.shape)

    ## train
    train_data_tensor = torch.from_numpy(windows_normal_train).float().to(device)
    train_mas_data_tensor = torch.from_numpy(windows_normal_mas_train).float().to(device)

    train_dataset = MyDataset(train_data_tensor,train_mas_data_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    ## val
    val_data_tensor = torch.from_numpy(windows_normal_val).float().to(device)
    val_mas_data_tensor = torch.from_numpy(windows_normal_mas_val).float().to(device)

    val_dataset = MyDataset(val_data_tensor, val_mas_data_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    ## test
    test_data_tensor = torch.from_numpy(windows_attack).float().to(device)
    test_mas_data_tensor = torch.from_numpy(windows_attack_mas).float().to(device)

    test_dataset = MyDataset(test_data_tensor, test_mas_data_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    ## test labels
    windows_labels = []
    for i in range(len(labels) - window_size + 1):
        windows_labels.append(list(np.int_(labels[i:i + window_size])))

    y_test_labels = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]

    print("train: ", type(windows_normal_train.shape), windows_normal_train.shape)
    print("val: ", windows_normal_val.shape)
    print("test: ", windows_attack.shape)
    print("test labels: ", len(y_test_labels))

    return train_loader, val_loader, test_loader, y_test_labels, min_max_scaler


def load_data_unsup_train(attack, labels, device = "gpu", window_size = 12, val_ratio = 0.2, batch_size = 64, is_down_sample = False, down_len=10):
    #接收无监督方法输出的数据和标签，筛选正常窗口，作为训练集

    labels = np.array(labels)
    print("attack_train: ", attack.shape)
    print("labels: ", labels.shape)

    ### Add Moving Average (MA)
    window_sizes = [3,5,10,20]
    attack_mas = []
    for w in window_sizes:

        attack_ma = np_ma(attack, w)
        attack_mas.append(attack_ma)
        
    W = np.max(window_sizes)
    attack = attack[W:, :]
    labels = labels[W:]
    for i in range(len(window_sizes)):
        attack_mas[i] = attack_mas[i][W:, :]

    ## down sample
    if is_down_sample:
        attack = downsample(attack, down_len=down_len, is_label=False)
        labels = downsample(labels, down_len=down_len, is_label=True)

        for i in range(len(window_sizes)):
            attack_mas[i] = downsample(attack_mas[i], down_len=down_len, is_label=False)

    # ## nomalization
    # min = normal.min()##axis=0
    # max = normal.max()##axis=0
    # min_max_scaler = MinMaxScaler(min, max)

    min_max_scaler = preprocessing.MinMaxScaler()
    attack = min_max_scaler.fit_transform(attack)

    for i in range(len(window_sizes)):
        attack_mas[i] = min_max_scaler.transform(attack_mas[i])
    

    windows_attack = attack[np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]
    

    for i in range(len(window_sizes)):
        
        attack_mas[i] = attack_mas[i][np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]

    
    windows_attack_mas = np.stack(attack_mas, axis=-1)
    
    ## window labels
    windows_labels = []
    for i in range(len(labels) - window_size + 1):
        windows_labels.append(list(np.int_(labels[i:i + window_size])))

    y_train_labels = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]
    
    
    ## select window
    windows_attack_ = []
    windows_attack_mas_ = []
    for i in range(len(y_train_labels)):
        if y_train_labels[i] == 0:
            windows_attack_.append(windows_attack[i])
            windows_attack_mas_.append(windows_attack_mas[i])
    windows_attack = np.array(windows_attack_)
    windows_attack_mas = np.array(windows_attack_mas_)


    ### train/val/test
    windows_train = windows_attack[:int(np.floor((1-val_ratio) * windows_attack.shape[0]))]
    windows_val = windows_attack[int(np.floor((1-val_ratio) * windows_attack.shape[0])):]

    windows_mas_train = windows_attack_mas[:int(np.floor((1-val_ratio) * windows_attack.shape[0]))]
    windows_mas_val = windows_attack_mas[int(np.floor((1-val_ratio) * windows_attack.shape[0])):]


    ## reshape: [B, T, N ,C]
    windows_train = windows_train.reshape(windows_train.shape + (1,))
    windows_val = windows_val.reshape(windows_val.shape + (1,))

    print("windows_train: ", windows_train.shape)
    print("windows_mas_train: ", windows_mas_train.shape)
    print("windows_val: ", windows_val.shape)
    print("windows_mas_val: ", windows_mas_val.shape)

    ## train
    train_data_tensor = torch.from_numpy(windows_train).float().to(device)
    train_mas_data_tensor = torch.from_numpy(windows_mas_train).float().to(device)

    train_dataset = MyDataset(train_data_tensor,train_mas_data_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    ## val
    val_data_tensor = torch.from_numpy(windows_val).float().to(device)
    val_mas_data_tensor = torch.from_numpy(windows_mas_val).float().to(device)

    val_dataset = MyDataset(val_data_tensor, val_mas_data_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    print("train: ", type(windows_train.shape), windows_train.shape)
    print("val: ", windows_val.shape)

    return train_loader, val_loader, min_max_scaler

def load_data_tsne(attack, labels, scaler, is_down_sample = True, down_len=50):
    #为tsne可视化输出降采样后的数据和标签

    labels = np.array(labels)
    print("attack_train: ", attack.shape)
    print("labels: ", labels.shape)

    ## down sample
    if is_down_sample:
        attack = downsample(attack, down_len=down_len, is_label=False)
        labels = downsample(labels, down_len=down_len, is_label=True)

    attack = scaler.fit_transform(attack)


    return attack, labels



def preprocessData(file, sep=None, min_max_scaler = None, val_ratio = 0.2, training=True):
    # === Normal period ====
    normal = pd.read_csv(file, header=None)
    normal = normal.astype(float).values

    train = normal[:int(np.floor((1-val_ratio*2) * normal.shape[0]))]
    val = normal[int(np.floor((1-val_ratio*2) * normal.shape[0])): int(np.floor((1-val_ratio) * normal.shape[0]))]
    test = normal[int(np.floor((1-val_ratio) * normal.shape[0])):]

    ## nomalization
    min_max_scaler = preprocessing.MinMaxScaler()

    train_scaled = min_max_scaler.fit_transform(train)
    val_scaled = min_max_scaler.transform(val)
    test_scaled = min_max_scaler.transform(test)

    return min_max_scaler, train_scaled, val_scaled, test_scaled

def load_data4GAT(data_filename, device = "gpu", window_size = 12, val_ratio = 0.2, batch_size = 64):
    ## EDA - Data Pre-Processing
    min_max_scaler, train_scaled, val_scaled, test_scaled = preprocessData(data_filename, sep=None, min_max_scaler = None, val_ratio = 0.2, training=True)
    # normal: (495000, 51)
    # attack: (449919, 51)
    windows_train = train_scaled.values[np.arange(window_size)[None, :] + np.arange(train_scaled.shape[0] - window_size + 1)[:, None]]
    windows_val = val_scaled.values[np.arange(window_size)[None, :] + np.arange(val_scaled.shape[0] - window_size + 1)[:, None]]
    windows_test = test_scaled.values[np.arange(window_size)[None, :] + np.arange(test_scaled.shape[0] - window_size + 1)[:, None]]

    ## train
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(windows_train).float().view(windows_train.shape + (1,)).to(device))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    ## val
    val_data = torch.utils.data.TensorDataset(torch.from_numpy(windows_val).float().view(windows_val.shape + (1,)).to(device))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    ## test
    test_data = torch.utils.data.TensorDataset(torch.from_numpy(windows_test).float().view(windows_test.shape + (1,)).to(device))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    print("train: ", type(windows_train.shape), windows_train.shape)
    print("val: ", windows_val.shape)
    print("test: ", windows_test.shape)

    return train_loader, val_loader, test_loader,  min_max_scaler


def preprocess4GDN(input_file, output_file, sep=None):
    # === Normal period ====
    if sep:
        normal = pd.read_csv(input_file, sep=sep)  # , nrows=1000)
    else:
        normal = pd.read_csv(input_file)

    normal = normal.rename(columns = {"Normal/Attack": "attack"})

    # print(list(normal))
    ## ['Timestamp', 'FIT101', 'LIT101', 'MV101', 'P101', 'P102', 'AIT201', 'AIT202', 'AIT203', 'FIT201',
    # 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302',
    # 'MV303', 'MV304', 'P301', 'P302', 'AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401', 'P402', 'P403', 'P404',
    # 'UV401', 'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502',
    # 'PIT501', 'PIT502', 'PIT503', 'FIT601', 'P601', 'P602', 'P603', 'attack']

    lookup = {
        "Normal": 0,
        "Attack": 1
    }

    # Transform all columns into float64
    for i in list(normal):
        if i not in ['Timestamp','attack']:
            normal[i] = normal[i].apply(lambda x: float(str(x).replace(",", ".")))
        if i == "attack":
            normal[i] = normal[i].apply(lookup.get)
    normal.to_csv(output_file, header=normal.columns)


def save_column_to_pkl(filename, outputName):
    normal = pd.read_csv(filename, sep= None)  # , nrows=1000)

    if "Attack" in normal.columns:
        normal = normal.drop(["Timestamp", "Attack"], axis=1)
    else:
        normal = normal.drop(["Timestamp", 'Normal/Attack'], axis=1)

    column2index = dict(zip(normal.columns,list(range(len(normal.columns)))))
    # for index,column in enumerate(normal.columns):
    #     column2index[column] = index

    with open(outputName+".pkl","wb")as f:
        pickle.dump(column2index,f)

    with open(outputName+".txt","w")as f:
        f.write(" ".join(normal.columns))

def save_data_to_unsupervise(test_filename):#向无监督框架中传数据，这部分数据是有标签的，所以从测试集里面选
    attack, labels = preprocessTestingData(test_filename, sep=None, min_max_scaler=None, training=False)  # , nrows=1000)
    np.savez("/home/chenty/STAT-AD/data/SWaT/test_data_swat.npz", a=attack, b=labels)

if __name__ == '__main__':
    # Read lib
    '''
    data_dir = r"F:\GitHub\SGAT-AE\data\SWaT"

    filename = data_dir + "/SWaT_Dataset_normal.csv"
    outputName = "swat_columns"

    save_column_to_pkl(filename,outputName)

    with open(outputName+".pkl", "rb") as f:
        data = pickle.load(f)
    print(data)

    with open(outputName+".txt","r")as f:
        data = f.readline()
    print(data.split(" "))
    '''
    file_path = '/home/chenty/STAT-AD/data/SWaT' + "/SWaT_Dataset_attack.csv"
    save_data_to_unsupervise(file_path)


