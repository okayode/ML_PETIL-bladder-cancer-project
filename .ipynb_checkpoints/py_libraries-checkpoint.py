# # load libraies

import tensorflow as tf
from tensorflow import keras
# import eli5
import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import splev, splrep
import sklearn
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.linear_model import RidgeCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import (LabelEncoder, PolynomialFeatures, label_binarize, LabelBinarizer, StandardScaler)
from sklearn.metrics import (precision_recall_curve, roc_curve, auc, roc_auc_score, RocCurveDisplay, make_scorer, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix, classification_report, matthews_corrcoef,
                             mean_squared_error)
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from IPython.display import display
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from matplotlib.ticker import LinearLocator
import collections
import time
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier,GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

from sklearn.feature_selection import SelectFromModel, r_regression
from tqdm import tqdm

# ipympl can be install via pip or conda.
# %matplotlib widget

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
# from fairlearn.datasets import fetch_adult
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_consistent_length

import warnings
warnings.filterwarnings('ignore')

# loading utility files

from utility.sv_fig import savefig
from utility.make_cm import make_confusion_matrix
from utility.get_g_result import get_gamma_results
from utility.plt_result import plot_results

def savefig(filename, crop = True):
    plt.savefig('{}.pdf'.format(filename))





# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

