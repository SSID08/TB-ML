from sklearn.compose import make_column_selector,make_column_transformer
from sklearn.preprocessing import OneHotEncoder
import pickle
import numpy as np
import os

'''
This script performs one-hot-encoding transformation for categorical features\
included in the feature matrix and saves the resultant dataframe with the updated\
feature names
'''

for file in os.listdir('##path'):
    f=os.path.join('##path',file)
    pd_df=pickle.load(open(f,"rb"))
    #cols=pd_df.columns
    out_cols=[str(i) for i in pd_df.columns]
    pd_df.columns=out_cols
    pheno=np.array(pd_df['Phenotype'])
    pd_df=pd_df.drop(columns=['Phenotype'])
    categorical_preprocessor = OneHotEncoder(drop=['nan'],dtype='int')
    categorical_columns_selector = make_column_selector(dtype_include=object)
    numerical_columns_selector=make_column_selector(dtype_exclude=object)
    categorical_columns = categorical_columns_selector(pd_df)
    numerical_columns=numerical_columns_selector(pd_df)
    #numerical_columns=
    try:
        ct=make_column_transformer(('passthrough',numerical_columns),(categorical_preprocessor,categorical_columns),sparse_threshold=0)
        transformed_pd_df=ct.fit_transform(pd_df)
        pickle.dump({'Matrix':transformed_pd_df,'Columns':ct.get_feature_names_out(),'Phenotype':pheno},open(f"##path/{file}","wb"))

    except Exception as e:
        print(f"{file} and Exception is  : {e}")