import sklearn
import numpy as np
from sklearn import metrics

# cloned and modified from https://github.com/rinikerlab/GHOST

# learning optimal decision bdry threshold
def opt_decision_bdry_thres(y_pred_probs, labels_train, thresholds, ThOpt_metrics):
    # Optmize the decision threshold based on the Cohen's Kappa coefficient
    if ThOpt_metrics == 'Kappa':
        tscores = []
        # evaluate the score on the y_preds using different thresholds
        for thresh in thresholds:
            scores = [1 if x>=thresh else -1 for x in y_pred_probs]
            kappa = metrics.cohen_kappa_score(labels_train,scores)
            tscores.append((np.round(kappa,3),thresh))
        # select the threshold providing the highest kappa score as optimal
        tscores.sort(reverse=True)
        imb_mcc_kap = tscores[0][0]
        thresh = tscores[0][-1]
    # Optmize the decision threshold based on the MCC
    elif ThOpt_metrics == 'MCC':
        tscores = []
        # evaluate the score on the y_preds using different thresholds
        for thresh in thresholds:
            scores = [1 if x>=thresh else -1 for x in y_pred_probs]
            mcc = metrics.matthews_corrcoef(labels_train,scores)
            tscores.append((np.round(mcc,3),thresh))
        # select the threshold providing the highest mcc score as optimal
        tscores.sort(reverse=True)
        imb_mcc_kap = tscores[0][0]
        thresh = tscores[0][-1]
    return imb_mcc_kap, thresh