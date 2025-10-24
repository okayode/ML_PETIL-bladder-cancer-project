import numpy as np
import matplotlib.pyplot as plt

from utility.sv_fig import savefig

def plot_results(gamma_rslt, c, gamma, k,filename):
    plt.figure(figsize=(5, 4))

    plt.plot(c, gamma_rslt["mean_test_score"])
    plt.plot(c, gamma_rslt["mean_train_score"])
    
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    title = "Gamma = " + str(gamma) + " and k = " + str(k)
    plt.title(title)
    
    # setting x and y axis range
    plt.yticks(np.arange(0, 1.1, 0.10))
    plt.xticks(np.arange(0, 6, 1))
    
    plt.legend(['validation accuracy', 'train accuracy'], loc='best')
    
    savefig(filename)