
import numpy as np
import os
import pickle
from sklearn import preprocessing
import sys
from lib.utils import *
#from utils import *

base_dir = os.getcwd()
prefix = os.path.join(base_dir, "data")
#prefix='/home/chenty/STAT-AD/data'

data_dim ={
    "SMD": 38,
    "SMAP": 25,
    "MSL": 55
}

def preprocess(df):
    """returns normalized and standardized data."""
    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num()

    # normalize data
    df = preprocessing.MinMaxScaler().fit_transform(df)
    print('Data normalized')

    return df

def get_data(dataset, max_train_size=None, max_test_size=None, train_start=0, test_start=0):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print('load data of:', dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = data_dim.get(dataset)
    f = open(os.path.join(prefix, dataset, dataset + '_train.pkl'), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()
    try:
        f = open(os.path.join(prefix, dataset, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    # if do_preprocess:
    #     train_data = preprocess(train_data)
    #     test_data = preprocess(test_data)

    # print("train set shape: ", train_data.shape)
    # print("test set shape: ", test_data.shape)
    # print("test set label shape: ", test_label.shape)
    return (train_data, None), (test_data, test_label)


def load_data(dataset, device = "gpu", window_size = 12, val_ratio = 0.2, batch_size = 64, is_down_sample = False, down_len=10):

    ## EDA - Data Pre-Processing
    (normal, _), (attack, labels) = get_data(dataset, max_train_size=None, max_test_size=None, train_start=0,
                                                        test_start=0)

    print("normal: ", normal.shape)
    print("attack: ", attack.shape)
    print("labels: ", labels.shape)

    # normal: (495000, 51)
    # attack: (449919, 51)
    # normal = normal[21600:,:]
    ## down sample
    if is_down_sample:
        normal = downsample(normal, down_len=down_len, is_label=False)
        attack = downsample(attack, down_len=down_len, is_label=False)
        labels = downsample(labels, down_len=down_len, is_label=True)

    # ## nomalization
    min = normal.min()##axis=0
    max = normal.max()##axis=0
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

def load_data2(dataset, device = "gpu", window_size = 12, val_ratio = 0.2, batch_size = 64, is_down_sample = False, down_len=10):

    ## EDA - Data Pre-Processing
    (normal, _), (attack, labels) = get_data(dataset, max_train_size=None, max_test_size=None, train_start=0,
                                            test_start=0)
    print("normal: ", normal.shape)
    print("attack: ", attack.shape)
    print("labels: ", labels.shape)
    
    #np.savez("/home/chenty/STAT-AD/data/SMD/test_data_smd.npz", a=attack, b=labels)

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
    # normal = normal[21600:,:]
    # for i in range(len(window_sizes)):
    #     normal_mas[i] = normal_mas[i][21600:,:]

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
    min = normal.min()##axis=0
    max = normal.max()##axis=0
    # min_max_scaler = MinMaxScaler(min, max)

    min_max_scaler = preprocessing.MinMaxScaler()
    normal = min_max_scaler.fit_transform(normal)
    attack = min_max_scaler.transform(attack)

    for i in range(len(window_sizes)):
        normal_mas[i] = min_max_scaler.transform(normal_mas[i])
        attack_mas[i] = min_max_scaler.transform(attack_mas[i])

    windows_normal = normal[np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size + 1)[:, None]]
    # print(windows_normal.shape)  # (494988, 12, 51)

    windows_attack = attack[np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]
    # print(windows_attack.shape)  # (449907, 12, 51)

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



if __name__ == '__main__':

    dataset = "MSL"
    train_start = 0
    train_end = None
    # x_dim = data_dim.get(dataset)
    # f = open(os.path.join(prefix, dataset, dataset + '_train.pkl'), "rb")
    # train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    # f.close()
    # print("train_data: ", train_data.shape)

    # (train_data, _), (test_data, test_label) = get_data(dataset, max_train_size=None, max_test_size=None, train_start=0, test_start=0)

    train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = load_data2(dataset, device = "cpu", window_size = 12, val_ratio = 0.2, batch_size = 64, is_down_sample = False, down_len=10)
