import pickle
import argparse
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer,recall_score,precision_score
import numpy as np
import os
import re


'''This is a CLI tool to build xgBoost models on user input\
feature matrices
'''

parser=argparse.ArgumentParser(description='Run ML training')

parser.add_argument('--input',help="pickle input matrix and classification labels",required=True)
parser.add_argument('--out_folder',help="output folder path",required=True)
parser.add_argument('--feature_set',help="identifier for your input feature set",required=True)
parser.add_argument('--store_model',action='store_true')
parser.add_argument('--learning_rate',required=True)
parser.add_argument('--n_estimators',required=True)

args=parser.parse_args()
DR_name=re.sub('.pkl','',os.path.basename(args.input))

items=pickle.load(open(args.input,"rb"))
X=np.array(items["Matrix"])
Y=np.array(items["Phenotype"])

lr=float(args.learning_rate)
n_estim=int(args.n_estimators)
xgb_classifier=xgb.XGBClassifier(objective='binary:logistic',learning_rate=lr,booster='gbtree',n_estimators=n_estim,reg_lambda=1,base_score=.5)
cv=StratifiedKFold(n_splits=5,shuffle=True)
specificity_scorer=make_scorer(recall_score,pos_label=0)
NPV_scorer=make_scorer(precision_score,pos_label=0)
scorer={'recall':'recall','precision':'precision','specificity':specificity_scorer,'NPV':NPV_scorer,'accuracy':'accuracy','roc_auc':'roc_auc'}
cv_results=cross_validate(xgb_classifier,X,Y,scoring=scorer,cv=cv,return_train_score=True)
params=xgb_classifier.get_params()
print(cv_results)
if args.store_model:
    xgb_classifier.fit(X,Y)
    with open(os.path.join(args.out_folder,'Models',f"XGB_Boost_{DR_name}_{args.feature_set}.pkl"),'wb') as f:
        pickle.dump({"Model":xgb_classifier},f)

with open(os.path.join(args.out_folder,'Metrics',f"XGB_boost_{DR_name}_{args.feature_set}"+".txt"),'w') as f:
    f.write(str(X.shape)+"\n")
    f.write(f"Mean Recall on Test set:{round(np.mean(cv_results['test_recall']),3)}\
        Standard Deviation : {round(np.std(cv_results['test_recall']),3)} \n")
    f.write(f"Mean Specificity on Test set: {round(np.mean(cv_results['test_specificity']),3)}\
        Standard Deviation : {round(np.std(cv_results['test_specificity']),3)} \n ")
    f.write(f"Mean PPV on Test set: {round(np.mean(cv_results['test_precision']),3)}\
        Standard Deviation : {round(np.std(cv_results['test_precision']),3)}. \n ")
    f.write(f"Mean NPV on Test set: {round(np.mean(cv_results['test_NPV']),3)}.\
        Standard Deviation : {round(np.std(cv_results['test_NPV']),3)}. \n ")
    f.write(f"Mean ROC on Test set: {round(np.mean(cv_results['test_roc_auc']),3)}\
        Standard Deviation : {round(np.std(cv_results['test_roc_auc']),3)} \n")
    f.write(f"Mean Accuracy on Test set: {round(np.mean(cv_results['test_accuracy']),3)}\
        Standard Deviation: {round(np.std(cv_results['test_accuracy']),3)} \n ")
    for key in params.keys():
        f.write(key+ ": " + str(params[key]) + "\n")
    f.write(str(X.shape)+"\n")
    for key in cv_results.keys():
        f.write(key+ ": " + str(cv_results[key])+ "\n")
