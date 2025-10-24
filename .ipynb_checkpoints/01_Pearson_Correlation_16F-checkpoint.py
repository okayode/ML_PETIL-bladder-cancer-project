import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import r_regression
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
# print(ndata_sel.shape)

# check corelation of each of the 16 features with 'Overall TIL Growth'
# corelation feature and target

ndata_sely = ndata_sel

ndata_sely.dropna(inplace=True) # drop rows with Nan, no entries
# print(ndata_sely.shape)

X_cor = ndata_sely.drop('OverallTILGrowth', axis=1)
X_cor = X_cor.drop('ID', axis=1)

ndata_sely = ndata_sely.replace(to_replace='Yes TIL',value='1')
ndata_sely = ndata_sely.replace(to_replace='No TIL',value='-1')

# convert column "OverallTILGrowth" of Ndata to numerics
ndata_sely['OverallTILGrowth'] = pd.to_numeric(ndata_sely['OverallTILGrowth'])
y_cor = ndata_sely['OverallTILGrowth']

# print(y_cor)
# print(X_cor)

feat_cols = list(X_cor.columns)

feat_cor_targ = r_regression(X_cor, y_cor, center=True, force_finite=True)
feat_cor_targ = list(feat_cor_targ)

# print(feat_cols)
# print(feat_cor_targ)

print("Corelation with OverallTILGrowth")
for feature in zip(feat_cols, feat_cor_targ):
    print(feature)

# # ### individual corelation

# # Surgeon
# x1 = ndata_sely['Surgeon']
# print(x1.corr(y_cor))

# # Age_at_Surgery
# x2 = ndata_sely['Age_at_Surgery']
# print(x2.corr(y_cor))

# # Race
# x3 = ndata_sely['Race']
# print(x3.corr(y_cor))

# # Surgery
# x4 = ndata_sely['Surgery']
# print(x4.corr(y_cor))

# # Smoker
# x5 = ndata_sely['Smoker']
# print(x5.corr(y_cor))

# # BMI
# x6 = ndata_sely['BMI']
# print(x6.corr(y_cor))

# # NAC
# x7 = ndata_sely['NAC']
# print(x7.corr(y_cor))

# # cT
# x8 = ndata_sely['cT']
# print(x8.corr(y_cor))

# # pT
# x9 = ndata_sely['pT']
# print(x9.corr(y_cor))

# # cT_or_pT
# x10 = ndata_sely['cT_or_pT']
# print(x10.corr(y_cor))

# # pN
# x11 = ndata_sely['pN']
# print(x11.corr(y_cor))

# # Bx_Histology
# x12 = ndata_sely['Bx_Histology']
# print(x12.corr(y_cor))

# # Histology
# x13 = ndata_sely['Histology']
# print(x13.corr(y_cor))

# # Sample_weight_g_tumor
# x14 = ndata_sely['Sample_weight_g_tumor']
# print(x14.corr(y_cor))

# # Tumor_digest_count_primary_tumor
# x15 = ndata_sely['Tumor_digest_count_primary_tumor']
# print(x15.corr(y_cor))

# # Number_of_fragments_plated_tumor
# x16 = ndata_sely['Number_of_fragments_plated_tumor']
# print(x16.corr(y_cor))

# computing corelation matrix of all features
fig = plt.figure(1,figsize=(10, 8))
X_cor = ndata_sel.drop('OverallTILGrowth', axis=1)
X_cor = X_cor.drop('ID', axis=1)
X_cor = X_cor.drop('Surgery', axis=1)

sns.heatmap(X_cor.corr(), annot=True, fmt=".1f")
plt.title('Correlation (16 - 1 features)')
fig.tight_layout()

# plt.show()
# savefig('./figs/corr_01')

# Filter all rows which has NaN 
ndata_sel.dropna(inplace=True) # drop rows with Nan, no entries
# ndata_filt = ndata_sel[ndata_sel['OverallTILGrowth'].notna()]


ndata_filt = ndata_sel
# print(ndata_filt.shape)

NoTIL_pred1_idx0 = []
TIL_pred1_idx0 = []

for i in range(ndata_filt.shape[0]):
    if ndata_filt.iloc[i,-1]=='Yes TIL':
        TIL_pred1_idx0.append(ndata_filt.iloc[i,0])
        
for i in range(ndata_filt.shape[0]):
    if ndata_filt.iloc[i,-1]=='No TIL':
        NoTIL_pred1_idx0.append(ndata_filt.iloc[i,0])

print(' ')

# print(TIL_pred1_idx0)
# print(len(TIL_pred1_idx0))
print ("ID list for the Yes TIL class:  "+str(TIL_pred1_idx0))
print ("size of the Yes TIL class:  "+str(len(TIL_pred1_idx0)))

print(' ')

# print(NoTIL_pred1_idx0)
# len(NoTIL_pred1_idx0)
print ("ID list for the No TIL class:  "+str(NoTIL_pred1_idx0))
print ("size of the No TIL class:  "+str(len(NoTIL_pred1_idx0)))


# count class imbalance # data (16 Features)
fig = plt.figure(2,figsize=(8, 6))
ndata_filt.OverallTILGrowth.value_counts()/len(ndata_filt.index)

sns.countplot(x='OverallTILGrowth',hue='OverallTILGrowth', data=ndata_filt,palette='hls')
# savefig('./figs/distri')
fig.tight_layout()
plt.show()




