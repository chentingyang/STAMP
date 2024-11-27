import numpy as np
import pyod
from pyod.models.copod import COPOD
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.cblof import CBLOF
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP
from pyod.models.kpca import KPCA
from pyod.models.abod import ABOD
from pyod.models.loda import LODA
from pyod.models.rod import ROD
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.dif import DIF
from pyod.models.mcd import MCD
from pyod.models.cd import CD
from pyod.models.inne import INNE
from pyod.models.kde import KDE
from pyod.models.lunar import LUNAR
from sklearn.decomposition import PCA as PCA_dec

import pysnooper
from utils import evaluate
from datetime import datetime
from matplotlib import pyplot as plt
import pickle


np.random.seed(10)


test_path_SWaT = '/home/chenty/STAT-AD/data/SWaT/test_data_swat.npz'
#node_weight_path = '/home/chenty/STAT-AD/weights/node_weights_SWAT_unsup_train.npz'
#test_path = '/home/chenty/STAT-AD/data/WADI/test_data_wadi.npz'
#node_weight_path = '/home/chenty/STAT-AD/weights/node_weights_WADI_unsup_train.npz'
#test_path = '/home/chenty/STAT-AD/data/SMD/test_data_smd.npz'
#test_path = '/home/chenty/STAT-AD/data/MSL/test_data_msl.npz'
#test_path_SMD = '/home/chenty/STAT-AD/data/SMD/test_data_smd_unsup.npz'

test_data_path = '/home/chenty/STAT-AD/data/SMD/generalization/machine-3-5_test.pkl'
test_label_path = '/home/chenty/STAT-AD/data/SMD/generalization/machine-3-5_test_label.pkl'

def load_data(path, num):
    data = np.load(path)
    x = data['a']
    y = data['b']
    print(x.shape, y.shape)
    return x, y

def load_pkl(datapath):
    f = open(datapath, 'rb')
    return pickle.load(f)

#X, Y = load_data(test_path_SWaT, -1)
X = load_pkl(test_data_path)
Y = load_pkl(test_label_path)
print(X.shape, Y.shape)
print(len(np.where(Y==0)[0]))



X_full = X[:15000]
Y_full = Y[:15000]
X_val = X[15000:]
Y_val = Y[15000:]
print(len(np.where(Y_full==1)[0]))
print(len(np.where(Y_val==1)[0]))

#np.savez('/home/chenty/STAT-AD/data/SMD/test_data_smd_unsup.npz', a=X, b=Y)

print("load_done!")
    
#print(X_train.shape, Y_train.shape)

def train(x_train, *args, **kwargs):
    
    clf = args[0]

    clf.fit(x_train)
    
    return clf

def predict(x_test, *args, **kwargs):
    
    clf = args[0]
    
    if kwargs['conf']:
        labels, conf = clf.predict(x_test, return_confidence=True)
    else:
        labels = clf.predict(x_test, return_confidence=False)
        conf = None
    
    if kwargs['proba']:
        proba = clf.predict_proba(x_test)
    else:
        proba = None
    
    return labels, conf, proba

def method_evaluate(pred_labels, true_labels, pa=True):
    
    if pa:
        true_labels, pred_labels = evaluate.point_adjustment(pred_labels, true_labels)
    accuracy, precision, recall, f_score = evaluate.get_score(true_labels, pred_labels)
    print(len(np.where(pred_labels==0)[0]))
    
    return accuracy, precision, recall, f_score


def check_window(data, window_size, num):#根据窗口中检测出的正常数据的比例判定正常窗口
    window_data = np.ndarray((data.shape[0] - window_size + 1, window_size))
    for i in range(data.shape[0] - window_size + 1):
        window_data[i] = data[i:i+window_size]
    count = 0
    pivot = []
    for i in range(len(window_data)):
        if len(np.where(window_data[i]==0)[0]) >= len(window_data[i]) - num:
            count += 1
            pivot.append(i)                                                

    return count, pivot

def adjust_labels(labels, window_size, pivot):#将判定为正常的窗口中所有数据置为正常
    ad_labels = np.array([1 for _ in range(len(labels))])
    for t in range(len(labels)):
        ad_labels[t] = labels[t]
    for i in pivot:
        ad_labels[i:i+window_size] = 0
    return ad_labels

def bf_search(start, end, step_num, display_freq=1, verbose=True):
    if step_num is None or end is None:
        end = start
        step_num = 1
        
    search_step, search_range, search_lower_bound = step_num, end - start, start
    
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
        
    parameter = search_lower_bound
    best_recall = 0
    
    for i in range(search_step):
        #parameter += search_range / float(search_step)
        parameter += int(search_range / int(search_step))#对于整数参数
        
        clf = train(X_full, ABOD(n_neighbors=parameter, contamination=0.163))
        labels, conf, proba = predict(X_test, clf, conf=False, proba=False)
        
        accuracy, precision, recall, f_score = method_evaluate(labels, Y_test, False)
        print("param=: " + str(parameter))
        
        if recall > best_recall:
            best_recall = recall
            best_parameter = parameter
            print("best_parameter = : " + str(best_parameter) + ", best_score = : " + str(best_recall))
            
    print("best_parameter = : " + str(best_parameter) + ", best_score = : " + str(best_recall))
    
def bf_search_PCA(start, end, step_num, method, inverse_transform=False, display_freq=1, verbose=True):
    if step_num is None or end is None:
        end = start
        step_num = 1
        
    search_step, search_range, search_lower_bound = step_num, end - start, start
    
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
        
    parameter = search_lower_bound
    best_recall = 0
    
    for i in range(search_step):
        #parameter += search_range / float(search_step)
        parameter += int(search_range / int(search_step))#对于整数参数
        if inverse_transform:
            pca = PCA_dec(n_components = parameter)
            X_train_dec = pca.fit_transform(X_train)
            X_train_dec = pca.inverse_transform(X_train_dec)
            X_test_dec = pca.transform(X_test)
            X_test_dec = pca.inverse_transform(X_test_dec)
        else:
            pca = PCA_dec(n_components = parameter)
            X_train_dec = pca.fit_transform(X_train)
            X_test_dec = pca.transform(X_test)
        
        clf = train(X_train_dec, method)
        labels, conf, proba = predict(X_test_dec, clf, conf=False, proba=False)
        
        accuracy, precision, recall, f_score = method_evaluate(labels, Y_test, False)
        print("param=: " + str(parameter))
        
        if recall > best_recall:
            best_recall = recall
            best_parameter = parameter
            print("best_parameter = : " + str(best_parameter) + ", best_score = : " + str(best_recall))
            
    print("best_parameter = : " + str(best_parameter) + ", best_score = : " + str(best_recall))

def output_window(data, window_size, pivot):#输出判定为正常的窗口
    
    window_data = np.ndarray((data.shape[0] - window_size + 1, window_size, data.shape[1]))
    
    for i in range(data.shape[0] - window_size + 1):
        window_data[i] = data[i:i+window_size, :]
    
    window_data = window_data[pivot]
    return window_data

def plot_labels(pred, true):
    plt.xlabel('slot')
    plt.ylabel('label')
    plt_pred = pred
    plt_true = true
    plt.plot(range(len(pred)), plt_pred)
    plt.plot(range(len(true)), plt_true)
    plt.legend(['pred', 'true'])
    plt.ylim(bottom=0, top=3)
    plt.show()

#bf_search(0,0.5,50)
#bf_search_PCA(10,51,40,LOF(contamination=0.05, n_neighbors=100, n_jobs=-1), True)


detector_list = [
                 IForest(random_state=33, max_features=45, contamination=0.163), 
                 COPOD(contamination=0.163), LOF(contamination=0.163), 
                 KNN(contamination=0.08), ECOD(contamination=0.163), 
                 CBLOF(contamination=0.163, alpha=0.2), PCA(contamination=0.163, random_state=33, standardization=True)]

a = datetime.now()
'''
clf = train(X_full, LOF(n_neighbors=100, contamination=0.163, n_jobs=-1))
b = datetime.now()
print((b-a).seconds)

labels, conf, proba = predict(X_full, clf, conf=False, proba=False)
'''
#bf_search(1, 501, 50)
#labels = IForest(random_state=33, max_features=45, n_estimators= 100,contamination=0.1198).fit_predict(X_full)
#labels = IForest(random_state=33, n_estimators=100, max_features=38, contamination=0.0517).fit_predict(X)
#labels = COPOD(contamination=0.0517).fit_predict(X)
labels = LUNAR(contamination=0.012).fit_predict(X_val)
b = datetime.now()
print((b-a).seconds)

window_len = 15
c, p = check_window(labels, window_len, 0)
#accuracy, precision, recall, f_score = method_evaluate(labels, Y_full, False)#指标
accuracy, precision, recall, f_score = method_evaluate(labels, Y_val, True)#指标
print(c)#窗口数量
#print(np.array(labels).shape)
#np.savez("/home/chenty/STAT-AD/data/SMD/selected_data//ECOD/result_base", a=X_full, b=labels, c=X_val, d=Y_val)

'''
ad_labels = adjust_labels(labels, window_len, p)#标签调整
accuracy, precision, recall, f_score = method_evaluate(ad_labels, Y_full, False)#调整后指标
c, p = check_window(ad_labels, window_len, 1)
print(c)#调整后窗口数量



#windows = output_window(X_test, window_len, p)

plot_labels(labels, Y_test)
'''









 