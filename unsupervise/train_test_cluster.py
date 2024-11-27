import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from utils import hierarchicalcluster
import pysnooper
from utils import evaluate
from datetime import datetime
from matplotlib import pyplot as plt
import pickle


np.random.seed(10)

test_path = '/home/chenty/STAT-AD/data/SWaT/test_data_swat.npz'
node_weight_path = '/home/chenty/STAT-AD/weights/node_weights_SWAT_unsup_train.npz'

test_data_path = '/home/chenty/STAT-AD/data/SMD/generalization/machine-3-5_test.pkl'
test_label_path = '/home/chenty/STAT-AD/data/SMD/generalization/machine-3-5_test_label.pkl'

def load_data(path, num):
    data = np.load(test_path)
    x = data['a']
    y = data['b']
    print(x.shape, y.shape)#(449919, 45)
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

def fit_predict(inp, *args, **kwargs):
    
    clf = args[0]

    labels = clf.fit_predict(inp)
        
    return labels

def fit_predict_withcenter(inp, *args, **kwargs):
    
    clf = args[0]

    labels = clf.fit_predict(inp)
    
    centers = clf.cluster_centers_
    
    return labels, centers

def check_labels(labels, method):#按数量对类别排序
    
    ar, num = np.unique(labels, return_counts=True)
    categories = len(ar)#类别数量
    sort_num = num#各类样本数
    if method in ['DBSCAN']:#-1类为噪声
        sort_categories = num.argsort()[::-1] - 1#根据样本数量大小对类别排序
    elif method in ['KMeans', "Hier"]:
        sort_categories = num.argsort()[::-1]
    print('categories: ' + str(categories))
    print('sort_nums: ' + str(sort_num))
    print('sort_categories: ' + str(sort_categories))
    
    return categories, sort_num, sort_categories

def get_labels_bydist(labels, method, sort_cats, cat_choose, **kwargs):#按距离对类别排序
    
    pred_labels = np.array([1 for _ in range(len(labels))])
    distances = []#各类到主类的距离
    
    if method in ['KMeans']:
        centers = kwargs['centers']#(categories, dim)
        main_cat = sort_cats[0]#主类索引
        main_center = centers[main_cat]#主类簇中心
        for i in centers:
            distances.append(np.linalg.norm(i - main_center))
    print(distances)
            
    sort_distance = np.array(distances).argsort()#对距离降序排序获得索引
    
    for j in range(cat_choose):
        pivot = sort_distance[j]
        if pivot != -1:
            c = np.where(labels==pivot)[0]
            pred_labels[c] = 0
            
    return pred_labels
        
    
    return categories, sort_num, sort_categories

def get_labels_by_sort(labels, sort_cats, cat_choose=1):#cat_choose代表选择多少个主要类别（按类别排序）评估为正常标签
    pred_labels = np.array([1 for _ in range(len(labels))])
    for i in range(cat_choose):
        pivot = sort_cats[i]
        if pivot != -1:
            c = np.where(labels==pivot)[0]
            pred_labels[c] = 0
    return pred_labels


def method_evaluate(pred_labels, true_labels, pa=True):
    
    if pa:
        true_labels, pred_labels = evaluate.point_adjustment(pred_labels, true_labels)
    accuracy, precision, recall, f_score = evaluate.get_score(true_labels, pred_labels)
    
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
    
    
def bf_search(start, end, step_num, method, display_freq=1, verbose=True):
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
        
        if method in ['DBSCAN']:
            tmp_labels = fit_predict(X_full, DBSCAN(eps=parameter, min_samples=20, n_jobs=-1))#DBSCAN
            #tmp_labels = fit_predict(X_full, DBSCAN(eps=0.24, min_samples=parameter, n_jobs=-1))
            _, _, sort_categories = check_labels(tmp_labels, method)
            labels = get_labels_by_sort(tmp_labels, sort_categories, 3)
            
            
        elif method in ['KMeans']:
            kmeans = KMeans(n_clusters=20, random_state=33).fit(X_full)
            tmp_labels = kmeans.labels_
            centers = kmeans.cluster_centers_ #(clusters, features)
            _, _, sort_categories = check_labels(tmp_labels, method='KMeans')
            labels = get_labels_bydist(tmp_labels, 'KMeans', sort_categories, parameter, centers=centers)
            
        elif method in ['Hier']:
            #kmeans = KMeans(n_clusters=10, random_state=33).fit(X_test)
            tmp_labels = hierarchicalcluster.hierarchical_clustering(X_full, min_clusters=parameter)
            _, _, sort_categories = check_labels(tmp_labels, method='Hier')
            labels = get_labels_by_sort(tmp_labels, sort_categories, 1)
            
        accuracy, precision, recall, f_score = method_evaluate(labels, Y_full, False)
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
            pca = PCA(n_components = parameter)
            X_dec = pca.fit_transform(X_test)
            X_dec = pca.inverse_transform(X_dec)
        else:
            X_dec = PCA(n_components = parameter).fit_transform(X_test)
        
        if method in ['DBSCAN']:
            #tmp_labels = fit_predict(X_test, DBSCAN(eps=parameter, min_samples=5, n_jobs=-1))#DBSCAN
            tmp_labels = fit_predict(X_dec, DBSCAN(eps=0.24, min_samples=5, n_jobs=-1))
            _, _, sort_categories = check_labels(tmp_labels, method)
            labels = get_labels_by_sort(tmp_labels, sort_categories, 1)
            
        elif method in ['KMeans']:
            #kmeans = KMeans(n_clusters=10, random_state=33).fit(X_test)
            kmeans = KMeans(n_clusters=8, random_state=33).fit(X_dec)
            tmp_labels = kmeans.labels_
            centers = kmeans.cluster_centers_ #(clusters, features)
            _, _, sort_categories = check_labels(tmp_labels, method='KMeans')
            labels = get_labels_bydist(tmp_labels, 'KMeans', sort_categories, 1, centers=centers)
        
        accuracy, precision, recall, f_score = method_evaluate(labels, Y_test, False)
        print("param=: " + str(parameter))
        
        if recall > best_recall:
            best_recall = recall
            best_parameter = parameter
            print("best_parameter = : " + str(best_parameter) + ", best_score = : " + str(best_recall))
            
    print("best_parameter = : " + str(best_parameter) + ", best_score = : " + str(best_recall))

#bf_search(0, 15, 15, 'KMeans')

if __name__ == "__main__":
    a = datetime.now()

    #decomposition
    #X_dec = PCA(n_components = 20).fit_transform(X_test)
    
    #pca = PCA(n_components = parameter)
    #X_dec = pca.fit_transform(X_test)
    #X_dec = pca.inverse_transform(X_dec)
    

    #tmp_labels = fit_predict(X_full, DBSCAN(min_samples=1, eps=0.4))
    
    kmeans = KMeans(n_clusters=10, random_state=33).fit(X_full)
    tmp_labels = kmeans.labels_
    centers = kmeans.cluster_centers_ #(clusters, features)
    
    #tmp_labels = hierarchicalcluster.hierarchical_clustering(X_test, min_clusters=3)

    #_, _, sort_categories = check_labels(tmp_labels, method='DBSCAN')
    _, _, sort_categories = check_labels(tmp_labels, method='KMeans')

    #labels = get_labels_by_sort(tmp_labels, sort_categories, 3)
    labels = get_labels_bydist(tmp_labels, 'KMeans', sort_categories, 3, centers=centers)
    print(len(np.where(labels==0)[0]))
    b = datetime.now()
    print((b-a).seconds)

    #window_len = 15
    #c, p = check_window(labels, window_len, 1)
    #print(c)

    #ad_labels = adjust_labels(labels, window_len, p)#标签调整

    #accuracy, precision, recall, f_score = method_evaluate(ad_labels, Y_full, False)
    accuracy, precision, recall, f_score = method_evaluate(labels, Y_full, False)
    
    np.savez("/home/chenty/STAT-AD/data/SMD/selected_data//KMeans/result_base", a=X_full, b=labels, c=X_val, d=Y_val)

    #windows = output_window(X_test, window_len, p)

    #plot_labels(labels, Y_test)










