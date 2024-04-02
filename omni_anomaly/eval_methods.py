# -*- coding: utf-8 -*-
import numpy as np

from omni_anomaly.spot import SPOT
from sklearn.metrics import roc_auc_score

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
    epsilon = 0. if TP + FP != 0 else 0.00001
    precision = TP / (TP + FP + epsilon)
    epsilon = 0. if TP + FN != 0 else 0.00001
    recall = TP / (TP + FN + epsilon)
    epsilon = 0. if precision + recall != 0 else 0.00001
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    roc_auc = roc_auc_score(actual, predict)
    return f1, precision, recall, TP, TN, FP, FN, roc_auc

def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False,
                    return_original_pred=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    (Basically just marks entire regions of anomaly labels as correctly identified if a single region is correctly identified)

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
        return_original_pred: Return unadjusted pred value

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict_og = score < threshold
        predict = predict_og.copy()
    else:
        predict_og = pred
        predict = predict_og.copy()
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
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
        if return_original_pred:
            return predict, latency / (anomaly_count + 1e-4), predict_og
        else:
            return predict, latency / (anomaly_count + 1e-4)
    else:
        if return_original_pred:
            return predict, predict_og
        else:
            return predict

def calc_seq(score, label, threshold):
    """
    Find the f1 score by from provided threshold.
    Method from MTAD-GAT (https://github.com/ML4ITS/mtad-gat-pytorch)
    """
    predict, latency = adjust_predicts(score, label, threshold, calc_latency=True)
    return calc_point2point(predict, label), latency

def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Method from MTAD-GAT (https://github.com/ML4ITS/mtad-gat-pytorch)
    """

    print(f"Finding best f1-score by searching for threshold..")
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1.0, -1.0, -1.0)
    m_t = 0.0
    m_l = 0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target, latency = calc_seq(score, label, threshold)
        if target[0] > m[0]:
            m_t = threshold
            m = target
            m_l = latency
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)

    return {
        "bf_f1": m[0],
        "bf_precision": m[1],
        "bf_recall": m[2],
        "bf_TP": m[3],
        "bf_TN": m[4],
        "bf_FP": m[5],
        "bf_FN": m[6],
        'bf_ROC/AUC': m[7],
        "bf_threshold": m_t,
        "bf_latency": m_l,
    }

def pot_eval(init_score, score, label, q=1e-3, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            For `OmniAnomaly`, it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            For `OmniAnomaly`, it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t

    Returns:
        dict: pot result dict
    """
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)  # data import
    s.initialize(level=level, min_extrema=True)  # initialization step
    ret = s.run(dynamic=False)  # run
    print(len(ret['alarms']))
    print(len(ret['thresholds']))
    pot_th = -np.mean(ret['thresholds'])
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    p_t = calc_point2point(pred, label)
    print('POT result: ', p_t, pot_th, p_latency)
    return {
        'pot-f1': p_t[0],
        'pot-precision': p_t[1],
        'pot-recall': p_t[2],
        'pot-TP': p_t[3],
        'pot-TN': p_t[4],
        'pot-FP': p_t[5],
        'pot-FN': p_t[6],
        'pot-threshold': pot_th,
        'pot-latency': p_latency
    }
