

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from utility.sv_fig import savefig


def param_svm_search(X_train,y_train,n_splits,hyper_params,Cs,gammas,filename1,filename2,kl,rnd_st):
    
    # creating a KFold object with n splits
    folds = KFold(n_splits = n_splits, shuffle = True, random_state = rnd_st)
    model = SVC(kernel=kl)  # specify model
    
    # set up GridSearchCV()
    model_cv = GridSearchCV(estimator = model, param_grid = hyper_params, \
                        scoring= 'accuracy', cv = folds, verbose = 1,\
                        return_train_score=True, n_jobs = -1)
    # fit the model
    model_cv.fit(X_train, y_train)
    
    # obtaining the optimal accuracy score and hyperparameters
    best_score = model_cv.best_score_
    best_hyperparams = model_cv.best_params_
    
    best_C = best_hyperparams['C']
    best_gamma = best_hyperparams['gamma']
    
    cv_res = pd.DataFrame(model_cv.cv_results_)
    cv_res[["param_C", "param_gamma"]] = cv_res[["param_C", "param_gamma"]].astype(np.float64)
    
    # reshape values into a matrix with 'X' and 'Y' grid equals 'C' and 'gamma' respt
    test_scores_matrix = cv_res.pivot(
        index="param_gamma", columns="param_C", values="mean_test_score")
    
    train_scores_matrix = cv_res.pivot(
        index="param_gamma", columns="param_C", values="mean_train_score")
    
    # # values corresponding to 'best_C' and 'best_gamma' 
    # z1 = test_scores_matrix.loc[best_C,best_gamma]   
    # z2 = train_scores_matrix.loc[best_C,best_gamma]  
    
    ###########################################################################################
    
    
    # fig1, (ax1, ax2) = plt.subplots(figsize=(12, 6), ncols=2)
    
    # # plot imshow test accuracy
    # Z1 = test_scores_matrix
    # im1 = ax1.imshow(Z1, interpolation='none', cmap=cm.coolwarm,
    #                origin='lower', extent=[Cs[0], Cs[-1], gammas[0], gammas[-1]], aspect=6)  # cmap='seismic', cmap=cm.coolwarm
    # ax1.plot(best_C, best_gamma, marker='*', color="black")
    # ax1.annotate(
    #     f'best param \n ({best_C:.3f}, {best_gamma:.3f})',
    #     xy=(best_C, best_gamma),
    #     xytext=(best_C+0.2, best_gamma+0.03),
    #     arrowprops=None, fontsize=10)
    # ax1.set_xlabel("C", fontsize=15)
    # ax1.set_ylabel(r'$\gamma$', fontsize=15)
    # ax1.set_title("Test Accuracy \n Kernel = {} \n cv = {:d}".format(kl,n_splits), fontsize=15)
    # bar1 = plt.colorbar(im1)
    
    # # plot imshow training accuracy
    # Z2 = train_scores_matrix
    # im2 = ax2.imshow(Z2, interpolation='none', cmap=cm.coolwarm,
    #                origin='lower', extent=[Cs[0], Cs[-1], gammas[0], gammas[-1]], aspect=6)  # cmap='seismic', cmap=cm.coolwarm
    # ax2.plot(best_C, best_gamma, marker='*', color="black")
    # ax2.annotate(
    #     f'best param \n ({best_C:.3f}, {best_gamma:.3f})',
    #     xy=(best_C, best_gamma),
    #     xytext=(best_C+0.2, best_gamma+0.03),
    #     arrowprops=None, fontsize=10)
    # ax2.set_xlabel("C", fontsize=15)
    # ax2.set_ylabel(r'$\gamma$', fontsize=15)
    # ax2.set_title("Train Accuracy \n Kernel = {} \n cv = {:d}".format(kl,n_splits), fontsize=15)
    # bar2 = plt.colorbar(im2)
    
    # fig1.tight_layout
    # # plt.gca().format_coord = fmt

    def plot_accuracy(ax, Z, title, best_C, best_gamma, Cs, gammas, kernel, cv):
        """Plot accuracy heatmap with best parameters highlighted."""
        im = ax.imshow(
            Z, interpolation='none', cmap=cm.coolwarm, origin='lower', 
            extent=[Cs[0], Cs[-1], gammas[0], gammas[-1]], aspect=6
        )
        ax.plot(best_C, best_gamma, marker='*', color="black")
		# f'best param \n ({best_C:.3f}, {best_gamma:.3f})',
        ax.annotate(
            f'param \n ({best_C:.3f}, {best_gamma:.3f})',
            xy=(best_C, best_gamma),
            xytext=(best_C + 0.2, best_gamma + 0.03),
            fontsize=10
        )
        ax.set_xlabel("C", fontsize=15)
        ax.set_ylabel(r'$\gamma$', fontsize=15)
        ax.set_title(f"{title} \n Kernel = {kernel} \n cv = {cv}", fontsize=15)
        return im

    # Create figure and axes
    fig1, (ax1, ax2) = plt.subplots(figsize=(12, 5), ncols=2)

    # Plot test and train accuracy
    im1 = plot_accuracy(ax1, test_scores_matrix, "Test Accuracy", best_C, best_gamma, Cs, gammas, kl, n_splits)
    im2 = plot_accuracy(ax2, train_scores_matrix, "Train Accuracy", best_C, best_gamma, Cs, gammas, kl, n_splits)

    # Add colorbars
    fig1.colorbar(im1, ax=ax1)
    fig1.colorbar(im2, ax=ax2)

    fig1.tight_layout()
    savefig(filename1)
    
    #############################################################################################
    
    fig2, (ax1, ax2) = plt.subplots(subplot_kw={"projection": "3d"},figsize=(12, 5), ncols=2)
    
    # Plot the surface for test accuracy.
    X1 = Cs; Y1 = gammas; 
    X1, Y1 = np.meshgrid(X1, Y1)
    Z1 = test_scores_matrix
    surf1 = ax1.plot_surface(X1, Y1, Z1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    
    # ax1.text(best_C, best_gamma, z1, "*", color='black')
    ax1.set_xlabel("C", fontsize=15)
    ax1.set_ylabel(r'$\gamma$', fontsize=15)
    ax1.yaxis._axinfo['label']['space_factor'] = 3.0
    ax1.zaxis.set_rotate_label(True)
    ax1.set_zlabel('Accuracy', fontsize=15, rotation = -90)
    ax1.set_title("Test Accuracy \n Kernel = {} \n cv = {:d}".format(kl,n_splits), fontsize=15)
    bar1 = plt.colorbar(surf1, shrink=0.6, pad=0.05,location='left')
    
    # Plot the surface for train accuracy.
    X2 = Cs; Y2 = gammas; 
    X2, Y2 = np.meshgrid(X2, Y2)
    Z2 = train_scores_matrix
    surf2 = ax2.plot_surface(X2, Y2, Z2, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    
    # ax2.text(best_C, best_gamma, z2, "*", color='black')
    ax2.set_xlabel("C", fontsize=15)
    ax2.set_ylabel(r'$\gamma$', fontsize=15)
    ax2.yaxis._axinfo['label']['space_factor'] = 3.0
    ax2.zaxis.set_rotate_label(True)
    ax2.set_zlabel('Accuracy', fontsize=15, rotation = -90)
    ax2.set_title("Train Accuracy \n Kernel = {} \n cv = {:d}".format(kl,n_splits), fontsize=15)
    bar2 = plt.colorbar(surf2, shrink=0.6, pad=0.05,location='left')

    fig2.tight_layout
    savefig(filename2)

    # plt.show()
    
    return best_score, best_hyperparams

#, z1, z2
