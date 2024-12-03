import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pyod
from pyod.models.copod import COPOD
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.cblof import CBLOF
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
from sklearn.decomposition import PCA as PCA_dec
from pyod.models.loda import LODA
from pyod.models.hbos import HBOS
from sklearn.cluster import DBSCAN, KMeans
from sklearn import preprocessing
import pysnooper
from utils import evaluate
from datetime import datetime
import train_test_cluster
import pickle

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


def check_window(data, window_size, num):
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

def adjust_labels(labels, window_size, pivot):
    ad_labels = np.array([1 for _ in range(len(labels))])
    for t in range(len(labels)):
        ad_labels[t] = labels[t]
    for i in pivot:
        ad_labels[i:i+window_size] = 0
    return ad_labels

### grid searching the number of retained features 
def bf_search_feature_selection_pyod(start, end, step_num, feature_importance, display_freq=1, verbose=True):
    if step_num is None or end is None:
        end = start
        step_num = 1
        
    search_step, search_range, search_lower_bound = step_num, end - start, start
    
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
        
    parameter = search_lower_bound
    best_recall = 0
    
    for i in range(search_step):
        parameter += int(search_range / int(search_step)) # for integers
        
        trainfea = X_full[:, feature_importance[:parameter]]
        testfea = X_full[:, feature_importance[:parameter]]
        
        clf = train(trainfea, IForest(random_state=33, max_features=parameter, n_estimators= 100, contamination=0.1))
        labels, conf, proba = predict(testfea, clf, conf=False, proba=False)
        
        accuracy, precision, recall, f_score = method_evaluate(labels, Y_full, False)
        print("param=: " + str(parameter))
        
        if recall > best_recall:
            best_recall = recall
            best_parameter = parameter
            print("best_parameter = : " + str(best_parameter) + ", best_score = : " + str(best_recall))
            
    print("best_parameter = : " + str(best_parameter) + ", best_score = : " + str(best_recall))
    
### grid searching the number of retained features
def bf_search_feature_selection_cluster(start, end, step_num, feature_importance, method, display_freq=1, verbose=True):
    if step_num is None or end is None:
        end = start
        step_num = 1
        
    search_step, search_range, search_lower_bound = step_num, end - start, start
    
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
        
    parameter = search_lower_bound
    best_recall = 0
    
    for i in range(search_step):
        parameter += int(search_range / int(search_step))
        
        trainfea = X_full[:, feature_importance[:parameter]]
        testfea = X_full[:, feature_importance[:parameter]]

        
        if method in ['DBSCAN']:
            tmp_labels = train_test_cluster.fit_predict(trainfea, DBSCAN(min_samples=1, eps=0.4))
            _, _, sort_categories = train_test_cluster.check_labels(tmp_labels, method)
            labels = train_test_cluster.get_labels_by_sort(tmp_labels, sort_categories, 3)
            
        elif method in ['KMeans']:
            #kmeans = KMeans(n_clusters=20, random_state=33).fit(X_test)
            kmeans = KMeans(n_clusters=10, random_state=33).fit(trainfea)
            tmp_labels = kmeans.labels_
            centers = kmeans.cluster_centers_ #(clusters, features)
            _, _, sort_categories = train_test_cluster.check_labels(tmp_labels, method='KMeans')
            labels = train_test_cluster.get_labels_bydist(tmp_labels, 'KMeans', sort_categories, 3, centers=centers)
            
            
        accuracy, precision, recall, f_score = method_evaluate(labels, Y_full, False)
        print("param=: " + str(parameter))
        
        if recall > best_recall:
            best_recall = recall
            best_parameter = parameter
            print("best_parameter = : " + str(best_parameter) + ", best_score = : " + str(best_recall))
            
    print("best_parameter = : " + str(best_parameter) + ", best_score = : " + str(best_recall))



test_path_SMD = '/home/chenty/STAT-AD/data/SMD/test_data_smd_unsup.npz'
feature_importance_path_smd = '/home/chenty/STAT-AD/expe/node_weights_SMD_unsup_train_STAMP.npz' # saved model-derived information


def load_data(path):
    data = np.load(path)
    x = data['a']
    y = data['b']
    print(x.shape, y.shape)#(449919, 45)
    return x, y


X, Y = load_data(test_path_SMD)



X_full = X[:15000]
Y_full = Y[:15000] # 1.11% anomalies
print(np.sum(Y_full))
X_val = X[15000:]
Y_val = Y[15000:] # 2.98% anomalies


print("load_done!")


node_feature_weight = np.load(feature_importance_path_smd)
sort_graph_weight_out = node_feature_weight['a'] # information type OD
sort_graph_weight_in = node_feature_weight['b'] # information type ID
sort_score_weight = node_feature_weight['c'] # # information type OD


# search for best parameters of retained features 
bf_search_feature_selection_pyod(0, 38, 38, sort_graph_weight_out)
bf_search_feature_selection_pyod(0, 38, 38, sort_graph_weight_in)
bf_search_feature_selection_pyod(0, 38, 38, sort_score_weight)

'''
bf_search_feature_selection_cluster(18, 38, 10, sort_graph_weight_out, "DBSCAN")
bf_search_feature_selection_cluster(18, 38, 10, sort_graph_weight_in, "DBSCAN")
bf_search_feature_selection_cluster(18, 38, 10, sort_score_weight, "DBSCAN")
'''

# retained features
parameter = [29, 13, 21]


trainfea_out = X_full[:, sort_graph_weight_out[:parameter[0]]]
# put the screening method below 
labels_out = IForest(random_state=33, max_features=parameter[0], n_estimators= 100, contamination=0.1).fit_predict(trainfea_out)
np.savez("/home/chenty/STAT-AD/data/SMD/selected_data/Iforest/result_method1", a=X_full, b=labels_out, c=X_val, d=Y_val)


trainfea_in = X_full[:, sort_graph_weight_in[:parameter[1]]]   
# put the screening method below 
labels_in = IForest(random_state=33, max_features=parameter[1], n_estimators= 100, contamination=0.1).fit_predict(trainfea_in)
np.savez("/home/chenty/STAT-AD/data/SMD/selected_data/Iforest/result_method2", a=X_full, b=labels_in, c=X_val, d=Y_val)


trainfea_score = X_full[:, sort_score_weight[:parameter[2]]]  
# put the screening method below  
labels_score = IForest(random_state=33, max_features=parameter[2], n_estimators= 100, contamination=0.1).fit_predict(trainfea_score)
np.savez("/home/chenty/STAT-AD/data/SMD/selected_data/Iforest/result_method3", a=X_full, b=labels_score, c=X_val, d=Y_val)


