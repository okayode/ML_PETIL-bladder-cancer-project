import numpy as np
import matplotlib.pyplot as plt
from utility.make_cm import make_confusion_matrix

# visualize confusion matrix with seaborn heatmap
def my_cm(c_matrix):
    labels = ['TP','FN','FP','TN']
    categories = ['Yes TIL', 'No TIL']
    make_confusion_matrix(c_matrix,
                          group_names=labels,
                          categories=categories,
                          cmap='Blues')