import pickle
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer,recall_score,precision_score
import numpy as np
import os
import re


'''This is a CLI tool to build Random Forest models on user input\
feature matrices
'''

parser=argparse.ArgumentParser(description='Run ML training')

parser.add_argument('--input',help="pickle input matrix and classification labels",required=True)
parser.add_argument('--out_folder',help="output folder path",required=True)
parser.add_argument('--feature_set',help="identifier for your input feature set",required=True)
parser.add_argument('--store_model',action='store_true')


args=parser.parse_args()
DR_name=re.sub('.pkl','',os.path.basename(args.input))

items=pickle.load(open(args.input,"rb"))
X=np.array(items["Matrix"])
Y=np.array(items["Phenotype"])
rf=RandomForestClassifier(n_estimators=100,max_features=0.5,max_samples=0.5,min_samples_split=0.0001,min_samples_leaf=0.0001,class_weight='balanced')
cv=StratifiedKFold(n_splits=5,shuffle=True)
specificity_scorer=make_scorer(recall_score,pos_label=0)
NPV_scorer=make_scorer(precision_score,pos_label=0)
scorer={'recall':'recall','precision':'precision','specificity':specificity_scorer,'NPV':NPV_scorer,'accuracy':'accuracy','roc_auc':'roc_auc'}
cv_results=cross_validate(rf,X,Y,scoring=scorer,cv=cv,return_train_score=True)
if args.store_model:
    rf.fit(X,Y)
    with open(os.path.join(args.out_folder,'Models',f"RF_{DR_name}_{args.feature_set}.pkl"),'wb') as f:
        pickle.dump({"Model":rf},f)

params=rf.get_params()
print(cv_results)

with open(os.path.join(args.out_folder,'Metrics',f"RF_{DR_name}_{args.feature_set}"+".txt"),'w') as f:
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
        Standard Deviation: {round(np.std(cv_results['test_accuracy']),3)} \n")
    for key in params.keys():
        f.write(key+ ": " + str(params[key]) + "\n")
    for key in cv_results.keys():
        f.write(key+ ": " + str(cv_results[key])+ "\n")



