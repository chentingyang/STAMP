import numpy as np
import random
from utils import evaluate
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
import lightgbm as lgb
import sklearn.linear_model as lm
import sklearn.preprocessing as sp
import sklearn.pipeline as pl
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.neural_network as nn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import shap
from xgboost import plot_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectFromModel
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
from sklearn import preprocessing
import pysnooper
from utils import evaluate
from datetime import datetime
import train_test_cluster
import pickle


def normalize(data):
    mean_val = np.mean(data, axis=0)
    std_val = np.std(data, axis=0)
    for i in range(len(std_val)):
        if std_val[i] == 0:
            std_val[i] = 1
    result = []
    for d in data:
        result.append((np.array(d) - mean_val) / std_val)
    return np.array(result)

class lr_model:
    def __init__(self):
        self.model = lm.LogisticRegression(max_iter=1000)

    def train(self, trainfea, trainlab):
        self.model.fit(trainfea, trainlab)
        self.fea_len = len(trainfea[0])

    def predict(self, testfea):
        return self.model.predict(testfea)
    
    def get_feature_importance(self):
        return [abs(x) for id, x in enumerate(self.model.coef_[0])]
    
    def get_shap_importance(self, trainfea):
        explainer = shap.LinearExplainer(self.model, trainfea)
        shap_values = explainer(trainfea)
        return np.mean(np.abs(shap_values[:,:].values), axis=0)#在所有样本间的shap值取平均，shape=(features)

class polynomial_model:
    def __init__(self):
        self.model = pl.make_pipeline(
            sp.PolynomialFeatures(1),  
            sp.StandardScaler(),
            lm.LogisticRegression(max_iter=2000)  
        )

    def train(self, trainfea, trainlab):
        self.model.fit(trainfea, trainlab)
        self.fea_len = len(trainfea[0])

    def predict(self, testfea):
        return self.model.predict(testfea)

    def get_feature_importance(self):
        # TODO
        return [random.random() for _ in range(self.fea_len)]
    
    def get_shap_importance(self, trainfea):
        # NOT FEASABLE
        return [random.random() for _ in range(self.fea_len)]
        # X100 = shap.utils.sample(trainfea, 100)
        # explainer = shap.PermutationExplainer(self.model.named_steps['logisticregression'].predict, self.model[:-1].transform(X100))
        # shap_values = explainer.shap_values(self.model[:-1].transform(X100))
        # return np.average(np.abs(shap_values[:,:].values), axis=1)

class dt_model:
    def __init__(self):
        self.model = tr.DecisionTreeClassifier(criterion='gini', max_depth=15,random_state=42)

    def train(self, trainfea, trainlab):
        self.model.fit(trainfea, trainlab)

    def predict(self, testfea):
        return self.model.predict(testfea)

    def get_feature_importance(self):
        return list(self.model.feature_importances_)

    def get_shap_importance(self, trainfea):
        explainer = shap.Explainer(self.model)
        shap_values = explainer(trainfea)
        vals = shap_values[:,:].values
        return np.mean(np.abs(vals[:,:,1]), axis=0)

class rf_model:
    def __init__(self):
        self.model = es.RandomForestClassifier(
            criterion='gini', max_depth=30, random_state=42)

    def train(self, trainfea, trainlab):
        self.model.fit(trainfea, trainlab)

    def predict(self, testfea):
        return self.model.predict(testfea)

    def get_feature_importance(self):
        return list(self.model.feature_importances_)

    def get_shap_importance(self, trainfea):
        explainer = shap.Explainer(self.model)
        shap_values = explainer(trainfea)
        vals = shap_values[:,:].values
        return np.mean(np.abs(vals[:,:,1]), axis=0)

class xgb_model:
    def __init__(self):
        self.model = XGBClassifier(verbose=10, random_state=33)

    def train(self, trainfea, trainlab):
        self.model.fit(trainfea, trainlab)

    def predict(self, testfea, testlab):
        y_pred = self.model.predict(testfea)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(testlab,predictions)
        pre = precision_score(testlab,predictions)
        rec = recall_score(testlab,predictions)
        f1 = f1_score(testlab,predictions)
        print("Accuracy:%.2f%%"%(accuracy*100.0))
        print("Precision:%.2f%%"%(pre*100.0))
        print("Recall:%.2f%%"%(rec*100.0))
        print("F1_score:%.2f%%"%(f1*100.0))


    def get_feature_importance(self):
        return np.array(list(self.model.feature_importances_))

    def get_shap_importance(self, trainfea):
        explainer = shap.Explainer(self.model)
        shap_values = explainer(trainfea)
        return np.mean(np.abs(shap_values[:,:].values), axis=0)
    

class lgb_model:
    def __init__(self):
        self.model = lgb.LGBMClassifier(random_state=42)

    def train(self, trainfea, trainlab):
        self.model.fit(trainfea, trainlab)

    def predict(self, testfea):
        return self.model.predict(testfea)

    def get_feature_importance(self):
        return list(self.model.feature_importances_)

    def get_shap_importance(self, trainfea):
        explainer = shap.Explainer(self.model)
        shap_values = explainer(trainfea)
        vals = shap_values[:,:].values
        return np.mean(np.abs(vals[:,:,1]), axis=0)

class mlp_model:
    def __init__(self):
        self.model = nn.MLPClassifier()

    def train(self, trainfea, trainlab):
        self.model.fit(trainfea, trainlab)
        self.fea_len = len(trainfea[0])

    def predict(self, testfea):
        return self.model.predict(testfea)
    
    def get_feature_importance(self):
        # TODO
        return [random.random() for _ in range(self.fea_len)]
    
    def get_shap_importance(self, trainfea):
        # NOT FEASABLE
        return [random.random() for _ in range(self.fea_len)]

class if_model:
    def __init__(self):
        self.model = IsolationForest(random_state=42)

    def train(self, trainfea):
        self.model.fit(trainfea)

    def predict(self, testfea):
        result = self.model.predict(testfea)
        return [round(x) for x in result]

class ocs_model:
    def __init__(self):
        self.model = OneClassSVM(kernel='linear')

    def train(self, trainfea):
        self.model.fit(trainfea)

    def predict(self, testfea):
        result = self.model.predict(testfea)
        return [round(x) for x in result]
    
class dbs_model:
    def __init__(self):
        self.model = DBSCAN()

    def train(self, trainfea):
        self.model.fit(trainfea)

    def predict(self, testfea):
        result = self.model.predict(testfea)
        return [round(x) for x in result]

class lof_model:
    def __init__(self):
        self.model = LocalOutlierFactor(novelty=True)

    def train(self, trainfea):
        self.model.fit(trainfea)

    def predict(self, testfea):
        result = self.model.predict(testfea)
        return [round(x) for x in result]

class ee_model:
    def __init__(self):
        self.model = EllipticEnvelope()

    def train(self, trainfea):
        self.model.fit(trainfea)

    def predict(self, testfea):
        result = self.model.predict(testfea)
        return [round(x) for x in result]

class svdd_model:
    def __init__(self):
        self.model = OneClassSVM()

    def train(self, trainfea):
        self.model.fit(trainfea)

    def predict(self, testfea):
        result = self.model.predict(testfea)
        return [round(x) for x in result]
    # def __init__(self):
    #     self.model = BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='off')

    # def train(self, trainfea):
    #     self.model.fit(trainfea)

    # def predict(self, testfea):
    #     result = self.model.predict(testfea)
    #     return [round(x.max()) for x in result]
    
    
class blank_model:
    def __init__(self):
        pass

    def train(self, trainfea, trainlab):
        self.fea_len = len(trainfea[0])

    def predict(self, testfea):
        return [random.randint(0, 1) for _ in range(len(testfea))]
    
    def get_feature_importance(self):
        return [random.random() for _ in range(self.fea_len)]

    def get_shap_importance(self, trainfea):
        return [random.random() for _ in range(self.fea_len)]

def get_importance(model, X, Y):
    model.train(X, Y)
    feature_importance = model.get_feature_importance()
    shap_importance = model.get_shap_importance(X)
    sort_feature_importance = np.argsort(feature_importance)[::-1]#降序
    sort_shap_importance = np.argsort(shap_importance)[::-1]
    return sort_feature_importance, sort_shap_importance

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
        parameter += search_range / float(search_step)
        #parameter += int(search_range / int(search_step))#对于整数参数
        
        clf = train(X_train, PCA(random_state=33, contamination=parameter, standardization=True))
        labels, conf, proba = predict(X_test, clf, conf=False, proba=False)
        
        accuracy, precision, recall, f_score = method_evaluate(labels, Y_test, False)
        print("param=: " + str(parameter))
        
        if recall > best_recall:
            best_recall = recall
            best_parameter = parameter
            print("best_parameter = : " + str(best_parameter) + ", best_score = : " + str(best_recall))
            
    print("best_parameter = : " + str(best_parameter) + ", best_score = : " + str(best_recall))

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
        parameter += int(search_range / int(search_step))#对于整数参数
        
        trainfea = X_full[:, feature_importance[:parameter]]
        testfea = X_full[:, feature_importance[:parameter]]
        
        #labels = PCA(random_state=33, contamination=0.163, standardization=True).fit_predict(trainfea)
        clf = train(trainfea, IForest(random_state=33, max_features=parameter, n_estimators= 100, contamination=0.1))
        labels, conf, proba = predict(testfea, clf, conf=False, proba=False)
        
        accuracy, precision, recall, f_score = method_evaluate(labels, Y_full, False)
        print("param=: " + str(parameter))
        
        if recall > best_recall:
            best_recall = recall
            best_parameter = parameter
            print("best_parameter = : " + str(best_parameter) + ", best_score = : " + str(best_recall))
            
    print("best_parameter = : " + str(best_parameter) + ", best_score = : " + str(best_recall))
    
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
        parameter += int(search_range / int(search_step))#对于整数参数
        
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

np.random.seed(10)
random.seed(10)



#train_path = r'F:\异常检测\多维数据集\多维数据集\swat\train.csv'
#test_path = r'F:\异常检测\多维数据集\多维数据集\swat\test.csv'
test_path_SWaT = '/home/chenty/STAT-AD/data/SWaT/test_data_swat.npz'
test_path_SMD = '/home/chenty/STAT-AD/data/SMD/test_data_smd_unsup.npz'
#node_weight_path_swat = '/home/chenty/STAT-AD/weights/node_weights_SWAT_unsup_train_STAM.npz'
node_weight_path_smd = '/home/chenty/STAT-AD/weights/node_weights_SMD_unsup_train_GDN.npz'

'''
def load_data(path, num):
    data = np.loadtxt(path, delimiter=',', dtype=float, skiprows=1)
    x_data = data[:, 1:-1][:num]
    y_label = data[:, -1][:num]
    
    return (x_data, y_label)
'''
def load_data(path):
    data = np.load(path)
    x = data['a']
    y = data['b']
    print(x.shape, y.shape)#(449919, 45)
    return x, y


X, Y = load_data(test_path_SMD)

'''
# SWaT
X_full = X[:300000]
Y_full = Y[:300000]#16.3%异常率
X_val = X[300000:]
Y_val = Y[300000:]#3.8%异常率
print(np.sum(Y_full))
print(np.sum(Y_val))
'''

# SMD
X_full = X[:15000]
Y_full = Y[:15000]#1.11%异常率
print(np.sum(Y_full))
X_val = X[15000:]
Y_val = Y[15000:]#2.98%异常率
print(np.sum(Y_val))
print(Y_val.shape)

print("load_done!")


'''
sort_feature_importance, sort_shap_importance = get_importance(xgb_model(), X_train, Y_train)
print(sort_feature_importance)
'''
'''
node_feature_weight = np.load(node_weight_path_swat)#读取
sort_graph_weight_out = node_feature_weight['a']#节点图结构的出度权重降序排序索引
sort_graph_weight_in = node_feature_weight['b']#节点图结构的入度权重降序排序索引
sort_score_weight = node_feature_weight['c']#节点的异常得分降序排序索引
print(sort_graph_weight_out.shape)
'''
'''
sort_graph_weight_in = np.load('/home/chenty/STAT-AD/weights/node_weights_SWAT_unsup_train_GDN_graph_in.npy')
sort_graph_weight_out = np.load('/home/chenty/STAT-AD/weights/node_weights_SWAT_unsup_train_GDN_graph_out.npy')
sort_score_weight = np.load('/home/chenty/STAT-AD/weights/node_weights_SWAT_unsup_train_GDN_score.npy')
print(sort_graph_weight_in.shape, sort_graph_weight_out.shape, sort_score_weight.shape)
'''


node_feature_weight = np.load(node_weight_path_smd)#读取
sort_graph_weight_out = node_feature_weight['a']#节点图结构的出度权重降序排序索引
sort_graph_weight_in = node_feature_weight['b']#节点图结构的入度权重降序排序索引
sort_score_weight = node_feature_weight['c']#节点的异常得分降序排序索引
print(sort_graph_weight_out.shape)


#bf_search_feature_selection_pyod(0, 38, 38, sort_graph_weight_out)
#bf_search_feature_selection_pyod(0, 38, 38, sort_graph_weight_in)
#bf_search_feature_selection_pyod(0, 38, 38, sort_score_weight)
#bf_search_feature_selection_cluster(18, 38, 10, sort_graph_weight_out, "DBSCAN")
#bf_search_feature_selection_cluster(18, 38, 10, sort_graph_weight_in, "DBSCAN")
#bf_search_feature_selection_cluster(18, 38, 10, sort_score_weight, "DBSCAN")


parameter = [29, 13, 21]

trainfea_out = X_full[:, sort_graph_weight_out[:parameter[0]]]   
labels_out = IForest(random_state=33, max_features=parameter[0], n_estimators= 100, contamination=0.1).fit_predict(trainfea_out)

window_len = 15
c, p = check_window(labels_out, window_len, 0)
accuracy, precision, recall, f_score = method_evaluate(labels_out, Y_full, False)#指标
print(c)#窗口数量
print(np.array(labels_out).shape)
np.savez("/home/chenty/STAT-AD/data/SMD/selected_data/Iforest/result_method1_GDN", a=X_full, b=labels_out, c=X_val, d=Y_val)
#np.save('/home/chenty/STAT-AD/data/GDN_unsup/SWaT/COPOD/method1/train.npy', X_full)
#np.save('/home/chenty/STAT-AD/data/GDN_unsup/SWaT/COPOD/method1/train_labels.npy', labels_out)

trainfea_in = X_full[:, sort_graph_weight_in[:parameter[1]]]   
labels_in = IForest(random_state=33, max_features=parameter[1], n_estimators= 100, contamination=0.1).fit_predict(trainfea_in)

window_len = 15
c, p = check_window(labels_in, window_len, 0)
accuracy, precision, recall, f_score = method_evaluate(labels_in, Y_full, False)#指标
print(c)#窗口数量
print(np.array(labels_in).shape)
np.savez("/home/chenty/STAT-AD/data/SMD/selected_data/Iforest/result_method2_GDN", a=X_full, b=labels_in, c=X_val, d=Y_val)
#np.save('/home/chenty/STAT-AD/data/GDN_unsup/SWaT/COPOD/method2/train.npy', X_full)
#np.save('/home/chenty/STAT-AD/data/GDN_unsup/SWaT/COPOD/method2/train_labels.npy', labels_in)

trainfea_score = X_full[:, sort_score_weight[:parameter[2]]]   
labels_score = IForest(random_state=33, max_features=parameter[2], n_estimators= 100, contamination=0.1).fit_predict(trainfea_score)

window_len = 15
c, p = check_window(labels_score, window_len, 0)
accuracy, precision, recall, f_score = method_evaluate(labels_score, Y_full, False)#指标
print(c)#窗口数量
print(np.array(labels_score).shape)
np.savez("/home/chenty/STAT-AD/data/SMD/selected_data/Iforest/result_method3_GDN", a=X_full, b=labels_score, c=X_val, d=Y_val)
#np.save('/home/chenty/STAT-AD/data/GDN_unsup/SWaT/COPOD/method3/train.npy', X_full)
#np.save('/home/chenty/STAT-AD/data/GDN_unsup/SWaT/COPOD/method3/train_labels.npy', labels_score)

'''
method = 'DBSCAN'
trainfea_out = X_full[:, sort_graph_weight_out[:parameter[0]]] 
tmp_labels = train_test_cluster.fit_predict(trainfea_out, DBSCAN(min_samples=1, eps=0.4))
_, _, sort_categories = train_test_cluster.check_labels(tmp_labels, method)
labels_out = train_test_cluster.get_labels_by_sort(tmp_labels, sort_categories, 3)

window_len = 15
c, p = check_window(labels_out, window_len, 0)
accuracy, precision, recall, f_score = method_evaluate(labels_out, Y_full, False)#指标
print(c)#窗口数量
print(np.array(labels_out).shape)
np.savez("/home/chenty/STAT-AD/data/SMD/selected_data/DBSCAN/result_method1_SGATAE", a=X_full, b=labels_out, c=X_val, d=Y_val)

trainfea_in = X_full[:, sort_graph_weight_in[:parameter[1]]] 
tmp_labels = train_test_cluster.fit_predict(trainfea_in, DBSCAN(min_samples=1, eps=0.4))
_, _, sort_categories = train_test_cluster.check_labels(tmp_labels, method)
labels_in = train_test_cluster.get_labels_by_sort(tmp_labels, sort_categories, 3)

window_len = 15
c, p = check_window(labels_in, window_len, 0)
accuracy, precision, recall, f_score = method_evaluate(labels_in, Y_full, False)#指标
print(c)#窗口数量
print(np.array(labels_in).shape)
np.savez("/home/chenty/STAT-AD/data/SMD/selected_data/DBSCAN/result_method2_SGATAE", a=X_full, b=labels_in, c=X_val, d=Y_val)

trainfea_score = X_full[:, sort_score_weight[:parameter[2]]] 
tmp_labels = train_test_cluster.fit_predict(trainfea_score, DBSCAN(min_samples=1, eps=0.4))
_, _, sort_categories = train_test_cluster.check_labels(tmp_labels, method)
labels_score = train_test_cluster.get_labels_by_sort(tmp_labels, sort_categories, 3)

window_len = 15
c, p = check_window(labels_score, window_len, 0)
accuracy, precision, recall, f_score = method_evaluate(labels_score, Y_full, False)#指标
print(c)#窗口数量
print(np.array(labels_score).shape)
np.savez("/home/chenty/STAT-AD/data/SMD/selected_data/DBSCAN/result_method3_SGATAE", a=X_full, b=labels_score, c=X_val, d=Y_val)
'''
'''
method="KMeans"
trainfea_out = X_full[:, sort_graph_weight_out[:parameter[0]]] 
kmeans = KMeans(n_clusters=10, random_state=33).fit(trainfea_out)
tmp_labels = kmeans.labels_
centers = kmeans.cluster_centers_ #(clusters, features)
_, _, sort_categories = train_test_cluster.check_labels(tmp_labels, method='KMeans')
labels_out = train_test_cluster.get_labels_bydist(tmp_labels, 'KMeans', sort_categories, 3, centers=centers)

window_len = 15
c, p = check_window(labels_out, window_len, 0)
accuracy, precision, recall, f_score = method_evaluate(labels_out, Y_full, False)#指标
print(c)#窗口数量
print(np.array(labels_out).shape)
np.savez("/home/chenty/STAT-AD/data/SMD/selected_data/KMeans/result_method1_STAM", a=X_full, b=labels_out, c=X_val, d=Y_val)

trainfea_in = X_full[:, sort_graph_weight_in[:parameter[1]]] 
kmeans = KMeans(n_clusters=10, random_state=33).fit(trainfea_in)
tmp_labels = kmeans.labels_
centers = kmeans.cluster_centers_ #(clusters, features)
_, _, sort_categories = train_test_cluster.check_labels(tmp_labels, method='KMeans')
labels_in = train_test_cluster.get_labels_bydist(tmp_labels, 'KMeans', sort_categories, 3, centers=centers)

window_len = 15
c, p = check_window(labels_in, window_len, 0)
accuracy, precision, recall, f_score = method_evaluate(labels_in, Y_full, False)#指标
print(c)#窗口数量
print(np.array(labels_in).shape)
np.savez("/home/chenty/STAT-AD/data/SMD/selected_data/KMeans/result_method2_STAM", a=X_full, b=labels_in, c=X_val, d=Y_val)

trainfea_score = X_full[:, sort_score_weight[:parameter[2]]] 
kmeans = KMeans(n_clusters=10, random_state=33).fit(trainfea_score)
tmp_labels = kmeans.labels_
centers = kmeans.cluster_centers_ #(clusters, features)
_, _, sort_categories = train_test_cluster.check_labels(tmp_labels, method='KMeans')
labels_score = train_test_cluster.get_labels_bydist(tmp_labels, 'KMeans', sort_categories, 3, centers=centers)

window_len = 15
c, p = check_window(labels_score, window_len, 0)
accuracy, precision, recall, f_score = method_evaluate(labels_score, Y_full, False)#指标
print(c)#窗口数量
print(np.array(labels_score).shape)
np.savez("/home/chenty/STAT-AD/data/SMD/selected_data/KMeans/result_method3_STAM", a=X_full, b=labels_score, c=X_val, d=Y_val)
'''