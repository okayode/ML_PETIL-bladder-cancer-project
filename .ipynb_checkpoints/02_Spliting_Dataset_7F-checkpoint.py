import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import r_regression
from sklearn.model_selection import train_test_split
import seaborn as sns

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
Extracted_col1 = Ndata['Surgeon'];                           Extracted_col2 = Ndata['Age_at_Surgery']
Extracted_col3 = Ndata['Race'];                              Extracted_col4 = Ndata['Surgery']
Extracted_col5 = Ndata['Smoker'];                            Extracted_col6 = Ndata['BMI']
Extracted_col7 = Ndata['NAC'];                               Extracted_col8 = Ndata['cT']
Extracted_col9 = Ndata['pT'];                                Extracted_col10 = Ndata['cT_or_pT']
Extracted_col11 = Ndata['pN'];                               Extracted_col12 = Ndata['Bx_Histology']
Extracted_col13 = Ndata['Histology'];                        Extracted_col14 = Ndata['Sample_weight_g_tumor']
Extracted_col15 = Ndata['Tumor_digest_count_primary_tumor']; Extracted_col16 = Ndata['Number_of_fragments_plated_tumor']
Extracted_col17 = Ndata['Overall_TIL_growth']

ndata_sel = Extracted_col0
ndata_sel = ndata_sel.join(Extracted_col1); ndata_sel = ndata_sel.join(Extracted_col2)
ndata_sel = ndata_sel.join(Extracted_col3); ndata_sel = ndata_sel.join(Extracted_col4)
ndata_sel = ndata_sel.join(Extracted_col5); ndata_sel = ndata_sel.join(Extracted_col6)
ndata_sel = ndata_sel.join(Extracted_col7); ndata_sel = ndata_sel.join(Extracted_col8)
ndata_sel = ndata_sel.join(Extracted_col9); ndata_sel = ndata_sel.join(Extracted_col10)
ndata_sel = ndata_sel.join(Extracted_col11); ndata_sel = ndata_sel.join(Extracted_col12)
ndata_sel = ndata_sel.join(Extracted_col13); ndata_sel = ndata_sel.join(Extracted_col14)
ndata_sel = ndata_sel.join(Extracted_col15); ndata_sel = ndata_sel.join(Extracted_col16)
ndata_sel = ndata_sel.join(Extracted_col17)

ndata_sel = ndata_sel.rename(columns={'Overall_TIL_growth': 'OverallTILGrowth'})

Cols = ['Surgeon','Age at Surgery','Race','Surgery','Smoker','BMI',\
        'NAC','cT','pT','cT or pT','pN','Bx Histology','Histology',\
        'Sample weight (g) tumor','Tumor digest count (primary tumor)','Number of fragments plated (tumor)',\
        'OverallTILGrowth']

feats = ['Surgeon','Age at Surgery','Race','Surgery','Smoker','BMI',\
        'NAC','cT','pT','cT or pT','pN','Bx Histology','Histology',\
        'Sample weight (g) tumor','Tumor digest count (primary tumor)','Number of fragments plated (tumor)']

feat_labels = feats

# print(ndata_sel)
# print(ndata_sel.shape)

# Filter all rows which has NaN 
ndata_sel.dropna(inplace=True) # drop rows with Nan, no entries
# ndata_filt = ndata_sel[ndata_sel['OverallTILGrowth'].notna()]
ndata_filt = ndata_sel
# print(ndata_filt.shape)

# creating the External dataset
ndata_filt = ndata_filt.replace(to_replace='Yes TIL',value='1')
ndata_filt = ndata_filt.replace(to_replace='No TIL',value='-1')

# convert column "OverallTILGrowth" of Ndata to numerics
ndata_filt["OverallTILGrowth"] = pd.to_numeric(ndata_filt["OverallTILGrowth"])

X_16F = ndata_filt.iloc[:,:-1]
y_16F = ndata_filt.iloc[:, -1] #.values

XX, X_ExtVal, yy, y_ExtVal = train_test_split(X_16F, y_16F, test_size = 0.33, \
                                                        random_state=1234,stratify=y_16F)


# print(y_ExtVal)
# print(X_ExtVal.iloc[:,0].tolist())

n_X_ExtVal = X_ExtVal
# print(n_X_ExtVal)
# n_X_ExtVal.head()


# Training and Testing dataset (7 robust predictive features) 
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

# (7 Robust & Predictive Features)

NoTIL_pred2_idx0 = []
TIL_pred2_idx0 = []

for i in range(ndata_filt.shape[0]):
    if ndata_filt.iloc[i,-1]=='Yes TIL':
        TIL_pred2_idx0.append(ndata_filt.iloc[i,0])
        
for i in range(ndata_filt.shape[0]):
    if ndata_filt.iloc[i,-1]=='No TIL':
        NoTIL_pred2_idx0.append(ndata_filt.iloc[i,0])

print(' ')
print ("ID list for the Yes TIL class (7 Features):  "+str(TIL_pred2_idx0))
print ("size of the Yes TIL class (7 Features):  "+str(len(TIL_pred2_idx0)))

print(' ')
print ("ID list for the No TIL class (7 Features):  "+str(NoTIL_pred2_idx0))
print ("size of the No TIL class (7 Features):  "+str(len(NoTIL_pred2_idx0)))

print(' ')
print ("ID list for the External validation set:  "+str(X_ExtVal.iloc[:,0].tolist()))

ndata_filt = ndata_filt.replace(to_replace='Yes TIL',value='1')
ndata_filt = ndata_filt.replace(to_replace='No TIL',value='-1')

# convert column "OverallTILGrowth" of Ndata to numerics
ndata_filt["OverallTILGrowth"] = pd.to_numeric(ndata_filt["OverallTILGrowth"])
# ndata_filt.head()

# print(ndata_filt.shape)


# ## remove the Ext validation data from the dataset
def remove_ExtVal_from_df(df1,df2):
    values_to_remove = list(df1['ID'].values)
    n_set = len(values_to_remove)
    
    for i in range(n_set):
        tmp = df2[df2['ID'] == values_to_remove[i]].index
        df2.drop(tmp,inplace=True)

    return df2

# 
Xy_filt = remove_ExtVal_from_df(n_X_ExtVal,ndata_filt)
# print(Xy_filt.shape)

# (dataset minus Ext validation data, 7 Robust & Predictive Features)
NoTIL_pred3_idx0 = []
TIL_pred3_idx0 = []

for i in range(Xy_filt.shape[0]):
    if Xy_filt.iloc[i,-1]==1:
        TIL_pred3_idx0.append(Xy_filt.iloc[i,0])
        
for i in range(Xy_filt.shape[0]):
    if Xy_filt.iloc[i,-1]==-1:
        NoTIL_pred3_idx0.append(Xy_filt.iloc[i,0])

print(' ')
print ("ID list for the Yes TIL class (7 Features, Training/Testing cohort):  "+str(TIL_pred3_idx0))
print ("size of the Yes TIL class (7 Features, Training/Testing cohort):  "+str(len(TIL_pred3_idx0)))

print(' ')
print ("ID list for the No TIL class (7 Features, Training/Testing cohort):  "+str(NoTIL_pred3_idx0))
print ("size of the No TIL class (7 Features, Training/Testing cohort):  "+str(len(NoTIL_pred3_idx0)))

print(' ')

# 
Xy_filt = Xy_filt.drop('ID', axis=1)
Xy_filt = Xy_filt.apply(pd.to_numeric) # convert all columns of Ndata to numerics

X = Xy_filt.iloc[:,:-1]
y = Xy_filt.iloc[:, -1] #.values
# X.head()
# print(X.shape)
# print(y.shape)

fig = plt.figure(1,figsize=(10, 8))

sns.heatmap(X.corr(), annot=True, fmt=".1f")
plt.title('Correlation between (7 Robust & Predictive) features')
fig.tight_layout()

# savefig('./figs/corr_02')
plt.show()
