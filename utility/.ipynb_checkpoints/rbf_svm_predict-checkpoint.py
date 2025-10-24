

from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score)

from utility.calc_met import calc_metrics
from utility.opt_dec_bdry_thres import opt_decision_bdry_thres
from utility.sv_fig import savefig


def model_svm_predict(X_train,y_train,X_test,y_test,X_val,y_val,C,gamma,n_splits,K_R_M,kl):
    model = SVC(C = C, 
                gamma = gamma, 
                kernel=kl,
                probability=True)
    model.fit(X_train, y_train)
    
    # predict
    y_pred = model.predict(X_test)
    
    # Get prediction probabilities for the test set
    test_probs = model.predict_proba(X_test)[:,1]
    
    # Get prediction probabilities for the Ext Val set
    val_probs = model.predict_proba(X_val)[:,1]
    
#     # new update to code (generating confidence interval on the testing and Ext valid cohort)   
#     hgbc_adult_scores = cross_val_score(hgbc, test_probs, y_test, cv=5, n_jobs=8)
    
    fpr1, tpr1, _ = roc_curve(y_test,test_probs)
    fpr2, tpr2, _ = roc_curve(y_val,val_probs)
    auc1 = round(roc_auc_score(y_test,test_probs),4)
    auc2 = round(roc_auc_score(y_val,val_probs),4)
    
    tpr1_mean = np.mean(tpr1); tpr1_std = np.std(tpr1)
    tpr2_mean = np.mean(tpr2); tpr2_std = np.std(tpr2)
    
     ###################################################################################################################
    # Print confusion matrix and classification metrics (test set)
    cm_test_50=calc_metrics(y_test, test_probs, threshold = 0.5,ThOpt_metrics = K_R_M)
    
    # Print confusion matrix and classification metrics (val set)
    cm_val_50=calc_metrics(y_val, val_probs, threshold = 0.5,ThOpt_metrics = K_R_M)
    
    # extract y_pred prediction probabilities from the trained svm model
    y_pred_probs = model.predict_proba(X_train)[:,1] # this shd be an oob from the training set
    
    # optmize the threshold 
    thresholds = np.round(np.arange(0.01,0.99,0.001),4)
    mcc_opt,threshold_Opt = opt_decision_bdry_thres(y_pred_probs, y_train, thresholds, ThOpt_metrics = K_R_M)
    
    # Print confusion matrix and classification metrics (test set)
    cm_test_Opt=calc_metrics(y_test, test_probs, threshold = threshold_Opt,ThOpt_metrics = K_R_M)
    
    # Print confusion matrix and classification metrics (val set)
    cm_val_Opt=calc_metrics(y_val, val_probs, threshold = threshold_Opt,ThOpt_metrics = K_R_M)
    
    ###################################################################################################################
    fig1, (ax1, ax2) = plt.subplots(figsize=(13, 7), ncols=2)
    # subplot 1
    ax1.plot(fpr1,tpr1, label="Test Data, AUC={:.3f}".format(auc1),lw=2,alpha=0.8)
    ax1.plot(fpr2,tpr2, label="External Validation, AUC={:.3f}".format(auc2),lw=2,alpha=0.8)
    ax1.set_ylabel("True positive rate (Sensitivity)", fontsize=20)
    ax1.set_xlabel("False positive rate (1 - Specificity)", fontsize=20)
    ax1.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05])
    xx = np.linspace(0,1,100)
    yy = xx
    ax1.plot(xx,yy,'k--')
    ax1.legend();
    
    # subplot 2
    #ax2.plot(fpr1,tpr1_mean, label="Test Data, AUC={:.3f}".format(aucs1_mean))
    ax2.plot(fpr1,tpr1, label="Test Data, AUC={:.3f}".format(auc1),lw=2,alpha=0.8)
    ax2.fill_between(fpr1,
                     tpr1-(1.96*tpr1_std/np.sqrt(tpr1.shape[0])),
                     tpr1+(1.96*tpr1_std/np.sqrt(tpr1.shape[0])),
                     alpha=0.2)
    #ax2.plot(fpr2,tpr2_mean, label="External Validation, AUC={:.3f}".format(aucs2_mean))
    ax2.plot(fpr2,tpr2, label="External Validation, AUC={:.3f}".format(auc2),lw=2,alpha=0.8)
    ax2.fill_between(fpr2,
                     tpr2-(1.96*tpr2_std/np.sqrt(tpr2.shape[0])),
                     tpr2+(1.96*tpr2_std/np.sqrt(tpr2.shape[0])),
                     alpha=0.2)
    ax2.set_ylabel("True positive rate (Sensitivity)", fontsize=20)
    ax2.set_xlabel("False positive rate (1 - Specificity)", fontsize=20)
    ax2.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05])
    
    xx = np.linspace(0,1,100)
    yy = xx
    ax2.plot(xx,yy,'k--')
    ax2.legend();

    fig1.tight_layout()
    

    return mcc_opt,threshold_Opt,y_pred,y_test,test_probs,val_probs,cm_test_50,cm_test_Opt,cm_val_50,cm_val_Opt
    
