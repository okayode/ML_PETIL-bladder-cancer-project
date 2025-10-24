import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# from sklearn.feature_selection import r_regression
# from sklearn.model_selection import train_test_split
import seaborn as sns

from utility.data_spl import data_split
from utility.sv_fig import savefig

# (B001 - B131) 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)

# load data
data = pd.read_excel('data/BladderCancer_132P_TIL_prd1.xlsx')
data = data.loc[0:130]

Ndata = data[['ID','Surgeon','Age_at_Surgery','Race','Surgery','Smoker','BMI',\
              'NAC','cT','pT','cT_or_pT','pN','Bx_Histology','Histology',\
              'Sample_weight_g_tumor','Tumor_digest_count_primary_tumor','Number_of_fragments_plated_tumor',\
              'Overall_TIL_growth']].copy()

Ndata = Ndata.replace(to_replace="Yes",value="Yes TIL")
Ndata = Ndata.replace(to_replace="No",value="No TIL")

Ndata = Ndata.replace(r'^\s*$', np.nan, regex=True) # Replace Blank values with DataFrame.replace() methods.

Extracted_col0 = Ndata.iloc[:,0:1]  # "ID"
Extracted_col1 = Ndata['Surgeon']
Extracted_col2 = Ndata['Age_at_Surgery']
Extracted_col3 = Ndata['BMI']
Extracted_col4 = Ndata['cT_or_pT']
# Extracted_col5 = Ndata['Histology'] # may remove
Extracted_col6 = Ndata['Sample_weight_g_tumor']
Extracted_col7 = Ndata['Tumor_digest_count_primary_tumor']
Extracted_col8 = Ndata['Number_of_fragments_plated_tumor']
Extracted_col9 = Ndata['Overall_TIL_growth']

ndata_sel = Extracted_col0
ndata_sel = ndata_sel.join(Extracted_col1)
ndata_sel = ndata_sel.join(Extracted_col2)
ndata_sel = ndata_sel.join(Extracted_col3)
ndata_sel = ndata_sel.join(Extracted_col4)
# ndata_sel = ndata_sel.join(Extracted_col5) # may remove
ndata_sel = ndata_sel.join(Extracted_col6)
ndata_sel = ndata_sel.join(Extracted_col7)
ndata_sel = ndata_sel.join(Extracted_col8)
ndata_sel = ndata_sel.join(Extracted_col9)

ndata_sel = ndata_sel.rename(columns={'Overall_TIL_growth': 'OverallTILGrowth'})

Cols = ['Surgeon','Age at Surgery','BMI',\
        'cT or pT',\
        'Sample weight (g) tumor','Tumor digest count (primary tumor)','Number of fragments plated (tumor)',\
        'OverallTILGrowth']

feats = ['Surgeon','Age at Surgery','Race','Surgery','Smoker','BMI',\
        'NAC','cT','pT','cT or pT','pN','Bx Histology','Histology',\
        'Sample weight (g) tumor','Tumor digest count (primary tumor)','Number of fragments plated (tumor)']

feat_labels = feats

# Filter all rows for which has NaN (7 Robust & Predictive Features)
ndata_sel.dropna(inplace=True) # drop rows with Nan, no entries
# ndata_filt = ndata_sel[ndata_sel['OverallTILGrowth'].notna()]

ndata_filt = ndata_sel
# print(ndata_filt.shape)

ndata_filt = ndata_filt.replace(to_replace='Yes TIL',value='1')
ndata_filt = ndata_filt.replace(to_replace='No TIL',value='-1')

# convert column "OverallTILGrowth" of Ndata to numerics
ndata_filt["OverallTILGrowth"] = pd.to_numeric(ndata_filt["OverallTILGrowth"])

# ndata_filt.head()
# print(ndata_filt.shape)

# remove the Ext validation data from the dataset
n_X_ExtVal_ID = ['B035', 'B023', 'B089', 'B102', 'B091', 'B093', 'B002', 'B008', 'B004', 'B015', 'B020', 'B014', 'B099', 'B083', 'B012', 'B028', 'B021', 'B052', 'B054', 'B027']
n_X_ExtVal = ndata_filt[ndata_filt['ID'].isin(n_X_ExtVal_ID)]
# n_X_ExtVal.head()
y_ExtVal = n_X_ExtVal.iloc[:, -1] #.values

def remove_ExtVal_from_df(df1,df2):
    values_to_remove = list(df1['ID'].values)
    n_set = len(values_to_remove)
    
    for i in range(n_set):
        tmp = df2[df2['ID'] == values_to_remove[i]].index
        df2.drop(tmp,inplace=True)

    return df2

Xy_filt = remove_ExtVal_from_df(n_X_ExtVal,ndata_filt)
# print(Xy_filt.shape)

Xy_filt = Xy_filt.drop('ID', axis=1)
Xy_filt = Xy_filt.apply(pd.to_numeric) # convert all columns of Ndata to numerics

X = Xy_filt.iloc[:,:-1]
y = Xy_filt.iloc[:, -1] #.values
# X.head()

# print(X.shape)
# print(y.shape)

# data (B001 - B131)
X_train, X_test, y_train, y_test = data_split(X,y,rnd_st=1234,tst_sz=0.30)        # working best for now

# X_train.head()
# X_test.head()
scaler = StandardScaler().fit(X_train) # build a scaler for the training data

# scaled x_train
X_train_sc = scaler.transform(X_train) # use the scaler to transform the training data

# scaled x_test
X_test_sc = scaler.transform(X_test) # use the scaler to transform the testing data

# Testing on the Ext Validation set

n_X_ExtVal_r = n_X_ExtVal[['Surgeon','Age_at_Surgery','BMI','cT_or_pT','Sample_weight_g_tumor',\
                          'Tumor_digest_count_primary_tumor','Number_of_fragments_plated_tumor']].copy()

n_X_ExtVal_r = n_X_ExtVal_r.apply(pd.to_numeric) # convert all columns of Ndata to numerics

# scaled x_ExtVal
X_extVal_sc = scaler.transform(n_X_ExtVal_r)

# Boxplot for training data, testing data, ext validation data
bx_trn = pd.concat([X_train,y_train], axis=1)
bx_tst = pd.concat([X_test,y_test], axis=1)
bx_ext = pd.concat([n_X_ExtVal_r,y_ExtVal], axis=1)

bx_trn['OverallTILGrowth'] = bx_trn['OverallTILGrowth'].replace(to_replace=1,value="Yes TIL")
bx_trn['OverallTILGrowth'] = bx_trn['OverallTILGrowth'].replace(to_replace=-1,value="No TIL")

bx_tst['OverallTILGrowth'] = bx_tst['OverallTILGrowth'].replace(to_replace=1,value="Yes TIL")
bx_tst['OverallTILGrowth'] = bx_tst['OverallTILGrowth'].replace(to_replace=-1,value="No TIL")

bx_ext['OverallTILGrowth'] = bx_ext['OverallTILGrowth'].replace(to_replace=1,value="Yes TIL")
bx_ext['OverallTILGrowth'] = bx_ext['OverallTILGrowth'].replace(to_replace=-1,value="No TIL")

# print(bx_trn)
# print(bx_tst)
# print(bx_ext)

# print(bx_trn.shape)
# print(bx_tst.shape)
# print(bx_ext.shape)

# bx_trn.head()
# bx_tst.head()
# bx_ext.head()


# Boxplots of Training, Testing, Ext Validation

f1, axes = plt.subplots(1, 3, figsize=(12, 5))

PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'k'},
    'medianprops':{'color':'k'},
    'whiskerprops':{'color':'k'},
    'capprops':{'color':'k'}
}

g0=sns.boxplot(x='OverallTILGrowth',y="Surgeon" ,data=bx_trn, ax=axes[0],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
g0.set_yticks(range(10))
sns.stripplot(data=bx_trn, x="OverallTILGrowth", y="Surgeon",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[0],jitter=0.35, s=8.5)
g0.set_ylabel('Surgeon', fontdict={'size': 15})
g0.set_ylim(0,10)
g0.set(xlabel=None)
g0.set_title('Training')

g1=sns.boxplot(x='OverallTILGrowth',y="Surgeon" ,data=bx_tst, ax=axes[1],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
g1.set_yticks(range(10))
sns.stripplot(data=bx_tst, x="OverallTILGrowth", y="Surgeon",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[1],jitter=0.35, s=8.5)
# g1.set_ylabel('Surgeon', fontdict={'size': 15})
g1.set_ylim(0,10)
g1.set(xlabel=None)
g1.set(ylabel=None)
g1.set_title('Testing')

g2=sns.boxplot(x='OverallTILGrowth',y="Surgeon" ,data=bx_ext, ax=axes[2],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
g2.set_yticks(range(10))
sns.stripplot(data=bx_ext, x="OverallTILGrowth", y="Surgeon",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[2],jitter=0.35, s=8.5)
# g2.set_ylabel('Surgeon', fontdict={'size': 15})
g2.set_ylim(0,10)
g2.set(xlabel=None)
g2.set(ylabel=None)
g2.set_title('External validation')

f1.tight_layout()
# savefig('./figs/surgeon_bxplot')


f2, axes = plt.subplots(1, 3, figsize=(12, 5))

PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'k'},
    'medianprops':{'color':'k'},
    'whiskerprops':{'color':'k'},
    'capprops':{'color':'k'}
}

g0=sns.boxplot(x='OverallTILGrowth',y="Age_at_Surgery" ,data=bx_trn, ax=axes[0],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
sns.stripplot(data=bx_trn, x="OverallTILGrowth", y="Age_at_Surgery",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[0],jitter=0.25, s=8.5)
g0.set_ylabel('Age at Surgery', fontdict={'size': 15})
g0.set_ylim(25,95)
g0.set(xlabel=None)
g0.set_title('Training')

g1=sns.boxplot(x='OverallTILGrowth',y="Age_at_Surgery" ,data=bx_tst, ax=axes[1],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
sns.stripplot(data=bx_tst, x="OverallTILGrowth", y="Age_at_Surgery",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[1],jitter=0.25, s=8.5)
g1.set_ylim(25,95)
g1.set(xlabel=None)
g1.set(ylabel=None)
g1.set_title('Testing')

g2=sns.boxplot(x='OverallTILGrowth',y="Age_at_Surgery" ,data=bx_ext, ax=axes[2],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
sns.stripplot(data=bx_ext, x="OverallTILGrowth", y="Age_at_Surgery",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[2],jitter=0.25, s=8.5)
g2.set_ylim(25,95)
g2.set(xlabel=None)
g2.set(ylabel=None)
g2.set_title('External validation')

f2.tight_layout()
# savefig('./figs/Age_at_surgery_bxplot')


f3, axes = plt.subplots(1, 3, figsize=(12, 5))

PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'k'},
    'medianprops':{'color':'k'},
    'whiskerprops':{'color':'k'},
    'capprops':{'color':'k'}
}

g0=sns.boxplot(x='OverallTILGrowth',y="BMI" ,data=bx_trn, ax=axes[0],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
sns.stripplot(data=bx_trn, x="OverallTILGrowth", y="BMI",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[0],jitter=0.25, s=8.5)
g0.set_ylabel('BMI', fontdict={'size': 15})
g0.set_ylim(0,60)
g0.set(xlabel=None)
g0.set_title('Training')

g1=sns.boxplot(x='OverallTILGrowth',y="BMI" ,data=bx_tst, ax=axes[1],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
sns.stripplot(data=bx_tst, x="OverallTILGrowth", y="BMI",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[1],jitter=0.25, s=8.5)
g1.set_ylim(0,60)
g1.set(xlabel=None)
g1.set(ylabel=None)
g1.set_title('Testing')

g2=sns.boxplot(x='OverallTILGrowth',y="BMI" ,data=bx_ext, ax=axes[2],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
sns.stripplot(data=bx_ext, x="OverallTILGrowth", y="BMI",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[2],jitter=0.25, s=8.5)
g2.set_ylim(0,60)
g2.set(xlabel=None)
g2.set(ylabel=None)
g2.set_title('External validation')

f3.tight_layout()
# savefig('./figs/bmi_bxplot')


f4, axes = plt.subplots(1, 3, figsize=(12, 5))

PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'k'},
    'medianprops':{'color':'k'},
    'whiskerprops':{'color':'k'},
    'capprops':{'color':'k'}
}

g0=sns.boxplot(x='OverallTILGrowth',y="cT_or_pT" ,data=bx_trn, ax=axes[0],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
g0.set_yticks(range(13))
sns.stripplot(data=bx_trn, x="OverallTILGrowth", y="cT_or_pT",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[0],jitter=0.25, s=8.5)
g0.set_ylabel('cT or pT', fontdict={'size': 15})
g0.set_ylim(0,13)
g0.set(xlabel=None)
g0.set_title('Training')

g1=sns.boxplot(x='OverallTILGrowth',y="cT_or_pT" ,data=bx_tst, ax=axes[1],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
g1.set_yticks(range(13))
sns.stripplot(data=bx_tst, x="OverallTILGrowth", y="cT_or_pT",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[1],jitter=0.25, s=8.5)
g1.set_ylim(0,13)
g1.set(xlabel=None)
g1.set(ylabel=None)
g1.set_title('Testing')

g2=sns.boxplot(x='OverallTILGrowth',y="cT_or_pT" ,data=bx_ext, ax=axes[2],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
g2.set_yticks(range(13))
sns.stripplot(data=bx_ext, x="OverallTILGrowth", y="cT_or_pT",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[2],jitter=0.25, s=8.5)
g2.set_ylim(0,13)
g2.set(xlabel=None)
g2.set(ylabel=None)
g2.set_title('External validation')

f4.tight_layout()
# savefig('./figs/ct_pt_bxplot')

f5, axes = plt.subplots(1, 3, figsize=(12, 5))

PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'k'},
    'medianprops':{'color':'k'},
    'whiskerprops':{'color':'k'},
    'capprops':{'color':'k'}
}

g0=sns.boxplot(x='OverallTILGrowth',y="Sample_weight_g_tumor" ,data=bx_trn, ax=axes[0],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
sns.stripplot(data=bx_trn, x="OverallTILGrowth", y="Sample_weight_g_tumor",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[0],jitter=0.25, s=8.5)
g0.set_ylabel('Sample weight(g)tumor', fontdict={'size': 15})
g0.set_ylim(-5,20)
g0.set(xlabel=None)
g0.set_title('Training')

g1=sns.boxplot(x='OverallTILGrowth',y="Sample_weight_g_tumor" ,data=bx_tst, ax=axes[1],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
sns.stripplot(data=bx_tst, x="OverallTILGrowth", y="Sample_weight_g_tumor",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[1],jitter=0.25, s=8.5)
g1.set_ylim(-5,20)
g1.set(xlabel=None)
g1.set(ylabel=None)
g1.set_title('Testing')

g2=sns.boxplot(x='OverallTILGrowth',y="Sample_weight_g_tumor" ,data=bx_ext, ax=axes[2],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
sns.stripplot(data=bx_ext, x="OverallTILGrowth", y="Sample_weight_g_tumor",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[2],jitter=0.25, s=8.5)
g2.set_ylim(-5,20)
g2.set(xlabel=None)
g2.set(ylabel=None)
g2.set_title('External validation')

f5.tight_layout()
#savefig('./figs/sampl_wt_bxplot')

f6, axes = plt.subplots(1, 3, figsize=(12, 5))

PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'k'},
    'medianprops':{'color':'k'},
    'whiskerprops':{'color':'k'},
    'capprops':{'color':'k'}
}

g0=sns.boxplot(x='OverallTILGrowth',y="Tumor_digest_count_primary_tumor" ,data=bx_trn, ax=axes[0],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
sns.stripplot(data=bx_trn, x="OverallTILGrowth", y="Tumor_digest_count_primary_tumor",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[0],jitter=0.25, s=8.5)
g0.set_ylabel('Tumor digest count', fontdict={'size': 15})
#g0.set_ylim(0,2.5*10**8)
g0.set(xlabel=None)
g0.set_title('Training')

g1=sns.boxplot(x='OverallTILGrowth',y="Tumor_digest_count_primary_tumor" ,data=bx_tst, ax=axes[1],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
sns.stripplot(data=bx_tst, x="OverallTILGrowth", y="Tumor_digest_count_primary_tumor",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[1],jitter=0.25, s=8.5)
#g1.set_ylim(0,2.5*10**8)
g1.set(xlabel=None)
g1.set(ylabel=None)
g1.set_title('Testing')

g2=sns.boxplot(x='OverallTILGrowth',y="Tumor_digest_count_primary_tumor" ,data=bx_ext, ax=axes[2],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
sns.stripplot(data=bx_ext, x="OverallTILGrowth", y="Tumor_digest_count_primary_tumor",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[2],jitter=0.25, s=8.5)
#g2.set_ylim(0,2.5*10**8)
g2.set(xlabel=None)
g2.set(ylabel=None)
g2.set_title('External validation')

f6.tight_layout()
# savefig('./figs/tumor_dg_ct_bxplot')

f7, axes = plt.subplots(1, 3, figsize=(12, 5))

PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'k'},
    'medianprops':{'color':'k'},
    'whiskerprops':{'color':'k'},
    'capprops':{'color':'k'}
}

g0=sns.boxplot(x='OverallTILGrowth',y="Number_of_fragments_plated_tumor" ,data=bx_trn, ax=axes[0],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
sns.stripplot(data=bx_trn, x="OverallTILGrowth", y="Number_of_fragments_plated_tumor",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[0],jitter=0.25, s=8.5)
g0.set_ylabel('# of fragments plated', fontdict={'size': 15})
g0.set_ylim(-5,35)
g0.set(xlabel=None)
g0.set_title('Training')

g1=sns.boxplot(x='OverallTILGrowth',y="Number_of_fragments_plated_tumor" ,data=bx_tst, ax=axes[1],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
sns.stripplot(data=bx_tst, x="OverallTILGrowth", y="Number_of_fragments_plated_tumor",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[1],jitter=0.25, s=8.5)
g1.set_ylim(-5,35)
g1.set(xlabel=None)
g1.set(ylabel=None)
g1.set_title('Testing')

g2=sns.boxplot(x='OverallTILGrowth',y="Number_of_fragments_plated_tumor" ,data=bx_ext, ax=axes[2],order=['No TIL','Yes TIL'],\
               showfliers=False, linewidth=2.5, **PROPS)
sns.stripplot(data=bx_ext, x="OverallTILGrowth", y="Number_of_fragments_plated_tumor",hue="OverallTILGrowth",order=['No TIL','Yes TIL'],\
              palette={"Yes TIL": "darkorange", "No TIL": "steelblue"}, legend=False, ax=axes[2],jitter=0.25, s=8.5)
g2.set_ylim(-5,35)
g2.set(xlabel=None)
g2.set(ylabel=None)
g2.set_title('External validation')

f7.tight_layout()
# savefig('./figs/frag_plate_bxplot')

plt.show()





