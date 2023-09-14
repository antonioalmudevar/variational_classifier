import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def auc(normal, anomaly, p=1):
    if len(normal)==0 or len(anomaly)==0:
        return 0.5
    else:
        labels = np.concatenate((np.zeros(len(normal)), np.ones(len(anomaly))))
        pred = np.concatenate((normal, anomaly))
        return roc_auc_score(labels, pred, max_fpr=p)


def log_eps(x, eps=1e-6):
    return torch.log(x+eps)


def log_a(x, log_0=-5):
    x = x + (x==0)*np.exp(log_0)
    return torch.log(x)


def onehot2int(labels):
    return labels.argmax(axis=1)


def mean_squared_error(preds, target):
    loss = torch.mean((preds-target)**2)
    return loss.mean()


def accuracy(preds, target, topk=(1,)):
    """
    https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = preds.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

"""
def bin_predictions_and_accuracies(probabilities, ground_truth, bins=10):
  if len(probabilities) != len(ground_truth):
    raise ValueError(
        'Probabilies and ground truth must have the same number of elements.')

  if [v for v in ground_truth if v not in [0., 1., True, False]]:
    raise ValueError(
        'Ground truth must contain binary labels {0,1} or {False, True}.')

  if isinstance(bins, int):
    num_bins = bins
  else:
    num_bins = bins.size - 1

  probabilities = np.where(probabilities == 0, 1e-8, probabilities)
  counts, bin_edges = np.histogram(probabilities, bins=bins, range=[0., 1.])
  indices = np.digitize(probabilities, bin_edges, right=True)
  accuracies = np.array([np.mean(ground_truth[indices == i])
                         for i in range(1, num_bins + 1)])
  return bin_edges, accuracies, counts


def bin_centers_of_mass(probabilities, bin_edges):
    probabilities = np.where(probabilities == 0, 1e-8, probabilities)
    indices = np.digitize(probabilities, bin_edges, right=True)
    return np.array(
        [np.mean(probabilities[indices == i]) for i in range(1, len(bin_edges))]
    )


def ece(preds, target, bins=15):
    preds = preds.flatten()
    target = target.flatten()
    bin_edges, accuracies, counts = bin_predictions_and_accuracies(
        preds, target, bins)
    bin_centers = bin_centers_of_mass(preds, bin_edges)
    num_examples = np.sum(counts)

    ece = np.sum([(counts[i] / float(num_examples)) * np.sum(
        np.abs(bin_centers[i] - accuracies[i]))
                    for i in range(bin_centers.size) if counts[i] > 0])
    return ece
"""

def ece(preds, target, n_bins=10):
    #https://github.com/sirius8050/Expected-Calibration-Error/blob/master/ECE.py
    preds = np.array(preds)
    target = np.array(target)
    if target.ndim > 1:
        target = np.argmax(target, axis=1)
    preds_index = np.argmax(preds, axis=1)
    preds_value = []
    for i in range(preds.shape[0]):
        preds_value.append(preds[i, preds_index[i]])
    preds_value = np.array(preds_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(preds.shape[0]):
            if preds_value[i] > a and preds_value[i] <= b:
                Bm[m] += 1
                if preds_index[i] == target[i]:
                    acc[m] += 1
                conf[m] += preds_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)