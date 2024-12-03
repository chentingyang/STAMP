# from util.data import *
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from scipy.stats import rankdata, iqr, trim_mean
from sklearn.metrics import f1_score, mean_squared_error
from numpy import percentile
import heapq
import math

'''
**************************************************************************
evaluate methods from STAMP
'''

def get_err_median_and_iqr(predicted, groundtruth):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)

    return err_median, err_iqr


def get_err_median_and_quantile(predicted, groundtruth, percentage):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    # err_iqr = iqr(np_arr)
    err_delta = percentile(np_arr, int(percentage*100)) - percentile(np_arr, int((1-percentage)*100))

    return err_median, err_delta


def get_err_mean_and_quantile(predicted, groundtruth, percentage):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = trim_mean(np_arr, percentage)
    # err_iqr = iqr(np_arr)
    err_delta = percentile(np_arr, int(percentage*100)) - percentile(np_arr, int((1-percentage)*100))

    return err_median, err_delta


def get_err_mean_and_std(predicted, groundtruth):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_mean = np.mean(np_arr)
    err_std = np.std(np_arr)

    return err_mean, err_std


def get_test_err_scores(test_result, option=1):
    ## [test_predicted_list, test_ground_list, construction_list]
    np_test_result = np.array(test_result)
    all_scores = None
    feature_num = np_test_result.shape[-1]

    
    for i in range(feature_num):
        test_re_list = np_test_result[:2, :, i]
        
        scores = get_err_scores(test_re_list, option=option)
        if all_scores is None:
            all_scores = scores
        else:
            all_scores = np.vstack((all_scores, scores))

    return all_scores


def get_err_scores(test_res, option=1):
    test_predict, test_gt = test_res

    
    if option == 1:
        n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)
    elif option == 2:
        n_err_mid, n_err_iqr = get_err_mean_and_std(test_predict, test_gt)
    elif option == 3:
        n_err_mid, n_err_iqr = get_err_mean_and_quantile(test_predict, test_gt, 0.25)
    else:
        n_err_mid, n_err_iqr = get_err_median_and_quantile(test_predict, test_gt, 0.25)

    test_delta = np.abs(np.subtract(np.array(test_predict).astype(np.float64),np.array(test_gt).astype(np.float64)))

    epsilon = 1e-2

    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(0, len(err_scores)):
        if i <before_num:
            smoothed_err_scores[i] = err_scores[i]
        else:
            smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])

    return smoothed_err_scores


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def adjust_predicts(score, label, threshold=None, pred=None, calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    #point adjust
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return predict,t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return predict,calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    m_predict = None
    for i in range(search_step):
        threshold += search_range / float(search_step)
        predict,target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
            m_predict = predict
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t, m_predict


def get_Test_scores_err_max(test_ae_result, option = 1, method = "max"):
    _, w, total_features = test_ae_result[0].shape
    test_ae_scores, normal_ae_scores = [], []
    for i in range(w):
        test_result = [data[:, i, :] for data in test_ae_result]

        test_ae_score = get_test_err_scores(test_result, option=option)

        test_ae_scores.append(test_ae_score.tolist())
    if method in ["max", "MAX"]:
        return np.array(test_ae_scores).max(axis=0)
    elif method in ["mean", "MEAN"]:
        return np.array(test_ae_scores).mean(axis=0)
    else:
        return np.array(test_ae_scores).sum(axis=0)


def get_score_PredAndAE(test_pred_result, test_ae_result,  test_generate_result, topk = 1, option = 1, method="max", alpha =0.4, beta=0.3, gamma = 0.3):
    print('=========================** Option: {} **============================\n'.format(str(option)))
    
    test_pred_scores, test_ae_scores, test_generate_scores = 0, 0, 0
    if alpha > 0:
        test_pred_scores = get_Test_scores_err_max(test_pred_result, option=option, method = method)
    if beta > 0:
        test_ae_scores = get_Test_scores_err_max(test_ae_result, option=option, method = method)
    if gamma > 0:
        test_generate_scores = get_Test_scores_err_max(test_generate_result, option=option, method = method)

    test_scores = alpha*test_pred_scores + beta*test_ae_scores + gamma*test_generate_scores
    print("test_scores: ", test_scores.shape) ### (feature_num, Batch)

    total_features = test_scores.shape[0]
    topk_indices = np.argpartition(test_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]

    total_topk_err_scores = np.sum(np.take_along_axis(test_scores, topk_indices, axis=0), axis=0)
    return test_scores, total_topk_err_scores

def get_topk_err(test_scores, topk = 3):
    print("test_scores: ", test_scores.shape) ### (feature_num, Batch)
    total_features = test_scores.shape[0]
    topk_indices = np.argpartition(test_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]

    print("topk_indices: ", topk_indices.shape) ### (topk, Batch)
    return topk_indices


def get_final_result(test_pred_results, test_ae_results,  test_generate_results, y_test_labels, topk = 1, option = 1, method="max", alpha =0.4, beta=0.3, gamma = 0.3, search_steps=500):

    test_scores,total_topk_err_scores = get_score_PredAndAE(test_pred_results, test_ae_results, test_generate_results, topk=topk, option=option,
                                    method=method, alpha=alpha, beta=beta, gamma=gamma)

    ### Find the best-f1 score by searching best `threshold` in [`start`, `end`)
    # get best f1
    start = total_topk_err_scores.min()
    end = total_topk_err_scores.max()
    t, th, predict = bf_search(np.array(total_topk_err_scores), np.array(y_test_labels), start=start, end=end, step_num=search_steps, display_freq=50)
    info = {
        'best-f1': t[0],
        'precision': t[1],
        'recall': t[2],
        'TP': t[3],
        'TN': t[4],
        'FP': t[5],
        'FN': t[6],
        'latency': t[-1],
        'threshold': th
    }
    return info,test_scores,predict


mean = np.inf 
std = np.inf

def get_threshold_from_traindata_pr(Y_predict, Y_train, k, scaler=False):
    """
    Returns:
        list: list for topk single prediction error value
    
    """
    if scaler:
        Y_predict = (Y_predict - mean) / std
        Y_train = (Y_train - mean) / std
        
    maxvalue = []
    for x, y in zip(Y_predict, Y_train):
        maxvalue.append(np.max(np.abs(x - y)))
    maxvalue = np.sort(maxvalue)[::-1]
    return maxvalue[:k]

def get_threshold_from_traindata_pr_sumk(Y_predict, Y_train, k1, k2, scaler=False):
    
    if scaler:
        Y_predict = (Y_predict - mean) / std
        Y_train = (Y_train - mean) / std
    
    maxvalue = []
    for x, y in zip(Y_predict, Y_train):
        absvalue = np.abs(x-y)
        maxvalue.append(np.sum(np.sort(absvalue)[::-1][:k1]))
    maxvalue = np.sort(maxvalue)[::-1]
    return maxvalue[:k2]
    
def get_error_list_pr(Y_true, Y_predict, threshold, k, scaler=False):
    
    if scaler:
        Y_predict = (Y_predict - mean) / std
        Y_true = (Y_true - mean) / std
        
    pred = []
    indexs = []
    if k == 1:
        for x, y in zip(Y_true, Y_predict):
            if np.max(np.abs(x - y)) > threshold:
                pred.append(1)
                indexs.append(np.where(np.abs(x - y) > threshold)[0])
            else:
                pred.append(0)
                indexs.append(None)
    else:
        for x, y in zip(Y_true, Y_predict):
            absvalue = np.abs(x-y)
            targ = np.sum(np.sort(absvalue)[::-1][:k]) / k
            if targ > threshold:
                pred.append(1)
                index = heapq.nlargest(k, range(len(list(absvalue))), list(absvalue).__getitem__)
                indexs.append(index)
            else:
                pred.append(0)
                indexs.append(None)
        
    return pred, indexs   
    
def get_topk_errors_from_label_pr(labels, predictions, true_values, k):
    output = []
    for i in range(len(labels)):
        if labels[i] == 1:
            pred = predictions[i]
            tv = true_values[i]
            targ = np.abs(pred - tv)#(d)
            targ = list(targ)
            errors = heapq.nlargest(k, targ)
            indexs = heapq.nlargest(k, range(len(targ)), targ.__getitem__)
            output.append([i,indexs,errors])
    return output

def get_hitrate_errors_from_label_pr(labels, index_labels, predictions, true_values, hitrate):
    output = []
    for i in range(len(labels)):
        if labels[i] == 1:
            pred = predictions[i]
            tv = true_values[i]
            targ = np.abs(pred - tv)#(d)
            targ = list(targ)
            targ_indexs = index_labels[i]
            k = math.ceil(len(targ_indexs) * hitrate)  #向上取整
            errors = heapq.nlargest(k, targ)
            indexs = heapq.nlargest(k, range(len(targ)), targ.__getitem__)
            output.append([i,indexs,errors])
    return output

def point_adjustment(pred, gt):
    #point_adjustment
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
            
    gt = np.array(gt)
    pred = np.array(pred)    
    return gt, pred


def get_score(gt, pred):
    '''
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN
    '''
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                            average='binary')
    print(
        "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))
    
    return accuracy, precision, recall, f_score


def bf_search_pr(y_pred, y_test, label, start, end, step_num, k, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Args:
        matrix: y_pred and y_test
        list: true label
        
    Returns:
        list: list for results
        float: the `threshold` for best-f1 and best-f1
    values:
        step_num: search steps
        k: get thresholds and reasons from topk instances
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
        
    search_step, search_range, search_lower_bound = step_num, end - start, start
    
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
        
    threshold = search_lower_bound
    best_score = 0
    best_threshold = 0
    
    for i in range(search_step):
        threshold += search_range / float(search_step)
        pred, _ = get_error_list_pr(y_test, y_pred, threshold, k, scaler=False)
        gt, pred = point_adjustment(pred, label)
        accuracy, precision, recall, f_score = get_score(gt, pred)
        print("threshold=: " + str(threshold))
        
        if f_score > best_score:
            best_score = f_score
            best_threshold = threshold
            print("best_threshold = : " + str(best_threshold) + ", best_score = : " + str(best_score))
            
    best_pred, reasons = get_error_list_pr(y_test, y_pred, best_threshold, k, scaler=False)
    _, best_pred_labels_after_pa = point_adjustment(best_pred, label)
    
    return best_score, best_threshold, best_pred, best_pred_labels_after_pa, reasons
    
def get_results(y_pred, y_test, label, start, end, step_num, k):
    best_score, best_threshold, best_pred_labels, best_pred_labels_after_pa, _ =\
        bf_search_pr(y_pred, y_test, label, start, end, step_num, k)
    accuracy, precision, recall, _ = get_score(np.array(label), best_pred_labels_after_pa)
    info = {
        'best-f1': best_score,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'threshold': best_threshold
    }
    return info, best_pred_labels

