# ML-PETIL: Machine Learning Predictor of the Expansion of Tumor Infiltrating Lymphocytes

One major advance for treating solid tumors is the success of adoptive cell therapy (ACT) during which autologous tumor-infiltrating lymphocytes (TILs) are expanded and activated ex vivo and then reinfused into the cancer patient. 

ML-PETIL is a tool that can first learn from patient and tumor data already collected in the clinic (local data) which data features are important for predicting TIL expansion, without the need to predefine which data categories to consider. Then, this tool predicts a possible TIL expansion for individual patients (personalized predictions) allowing to determine whether ACTTIL therapy could potentially treat an individual bladder cancer patient.


## ML-PETIL needs the following libries

numpy
sklearn
matplotlib
seaborn
pandas
tensorflow
statsmodels
scipy


## Implementing ML-PETIL

ML-PETIL is implemented in the following order (without imputation):

01_Pearson_Correlation_16F.ipynb
02_Feature_selection.ipynb
03_Spliting_Dataset_7F.ipynb
04_Boxplots.ipynb
05_Optimal_hyp_search.ipynb
06_Performance_Analysis.ipynb


### Authors

Kayode Olumoyin kayode.olumoyin@moffitt.org, Katarzyna Rejniak 


### Source Code
https://github.com/okayode/ML_PETIL-bladder-cancer-project


### Acknowledgements

This work was supported by the grants NIH R01-CA259387, DoD-W81XWH2210340, Moffitt Cancer Center Support Grant P30-CA076292.