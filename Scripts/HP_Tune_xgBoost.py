import pickle
import argparse
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score,recall_score,accuracy_score,make_scorer,balanced_accuracy_score,roc_auc_score
import numpy as np
import os
import re

'''
Tune xgBoost algorithm
'''

parser=argparse.ArgumentParser(description='Run hyper-parameter tuning for xgBoost')

parser.add_argument('--input',help="pickle input matrix and classification labels",required=True)
parser.add_argument('--out_folder',help="output folder path",required=True)
parser.add_argument('--feature_set',help="identifier for your input feature set",required=True)

args=parser.parse_args()
DR_name=re.sub('.pkl','',os.path.basename(args.input))

items=pickle.load(open(args.input,"rb"))
X=items["Matrix"]
Y=items["Phenotype"]

xgb_classifier=xgb.XGBClassifier(subsample=.7,objective='binary:logistic',booster='gbtree')
param_distribution={"n_estimators":[100,200,500,1000],
                    'learning_rate':[0.1,0.3,0.5,0.7]}

X_train,X_test,Y_train,y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,shuffle=True,random_state=42)
print(X_train.shape)

cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=41)
tune1=RandomizedSearchCV(xgb_classifier,
                          param_grid=param_distribution,cv=cv,scoring='roc_auc',n_jobs=-1,refit=False).fit(X_train,Y_train)
print('First tune done')
best_params=tune1.best_params_
n_estimators=best_params['n_estimators']
lr=best_params['learning_rate']

depth_distribution={'max_depth':[6,30,50,100,200]}

xgb_classifier=xgb.XGBClassifier(subsample=.7,objective='binary:logistic',booster='gbtree',n_estimators=n_estimators,
                                 learning_rate=lr)
tune2=RandomizedSearchCV(xgb_classifier,param_grid=depth_distribution,cv=cv,scoring='roc_auc',n_jobs=-1,refit=False).fit(X_train,Y_train)
best_depth=tune2.best_params_['max_depth']
print('Second tune done')
# 3. Define thresholds
thresholds = np.arange(0.3, 0.8, 0.1)
xgb_classifier=xgb.XGBClassifier(objective='binary:logistic',booster='gbtree',n_estimators=n_estimators,
                                            learning_rate=lr,max_depth=best_depth)
# 4. Perform CV
cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=56)
bal_accs = []
for threshold in thresholds:
    print('Threshold entered')
    accs=[]
    for i, (train_index, test_index) in enumerate(cv.split(X_train, Y_train)):
        print(f'{i}th iteration')
        model_fit=xgb_classifier.fit(X_train[train_index,:],Y_train[train_index])
        y_scores = model_fit.predict_proba(X_train[test_index,:])[:,1]
        y_pred = (y_scores >= threshold).astype(int)
        accs.append(balanced_accuracy_score(Y_train[test_index], y_pred))
    bal_accs.append(accs)

bal_accs=np.array(bal_accs)

# 5. Find the optimal threshold
optimal_threshold_index = np.argmax(np.mean(bal_accs,axis=1))
optimal_threshold = thresholds[optimal_threshold_index]

print(optimal_threshold)

xgb_classifier.fit(X_train,Y_train)
pred=(xgb_classifier.predict_proba(X_test)[:, 1] >= optimal_threshold).astype(int)

with open(os.path.join(args.out_folder,f"MODEL_XGB_{DR_name}_{args.feature_set}.pkl"),'wb') as f:
        pickle.dump({"Model":xgb_classifier},f)

specificity_scorer=make_scorer(recall_score,pos_label=0)
NPV_scorer=make_scorer(precision_score,pos_label=0)
spec=recall_score(y_test,pred,pos_label=0)
NPV=precision_score(y_test,pred,pos_label=0)
recall=recall_score(y_test,pred)
precision=precision_score(y_test,pred)
acc_score=accuracy_score(y_test,pred)
auc_score=roc_auc_score(y_test,pred)

scorer={'recall':'recall','precision':'precision','specificity':specificity_scorer,'NPV':NPV_scorer,'accuracy':'accuracy','roc_auc':'roc_auc'}

cv_results=cross_validate(xgb_classifier,X,Y,scoring=scorer,cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=43))
print(DR_name)
print(f'Best CV score: {str(tune2.best_score_)}; Best N_estimators: {str(n_estimators)}; Best_Learning_Rate : {str(lr)}; Best Depth={str(best_depth)}')
print(f'Recall score on hold-out set: {recall}')
print(f'Specificity score on hold-out set: {spec}')
print(f'NPV on hold-out set: {NPV}')
print(f'PPV score on hold-out set : {precision}')
print(f'Accuracy score on hold-out set: {acc_score}')
print(f'ROC-AUC score on hold-out set: {auc_score}')

with open(os.path.join(args.out_folder,f"xgBoost_{DR_name}_{args.feature_set}"+".txt"),'w') as f:
    f.write(f'Best CV score: {str(tune2.best_score_)}; Best N_estimators: {str(n_estimators)}; Best_Learning_Rate : {str(lr)}; Best Depth={str(best_depth)} \n')
    f.write(f'Recall score on Test set: {recall}\n Specificity score on Test set {spec}\n')
    f.write(f'NPV on Test set: {NPV}\nPPV score on Test set : {precision}\n')
    f.write(f'Accuracy score on test set: {acc_score}\n')
    f.write(f'ROC-AUC score on hold-out set: {auc_score}\n')
    f.write(f'Best Threshold: {optimal_threshold}\n')
    f.write(f'CV Results: \n\n')
    f.write(f"Mean Recall on validation set:{round(np.mean(cv_results['test_recall']),3)}\
        Standard Deviation : {round(np.std(cv_results['test_recall']),3)} \n")
    f.write(f"Mean Specificity on Validation set: {round(np.mean(cv_results['test_specificity']),3)}\
        Standard Deviation : {round(np.std(cv_results['test_specificity']),3)} \n ")
    f.write(f"Mean PPV on Validation set: {round(np.mean(cv_results['test_precision']),3)}\
        Standard Deviation : {round(np.std(cv_results['test_precision']),3)}. \n ")
    f.write(f"Mean NPV on Validation set: {round(np.mean(cv_results['test_NPV']),3)}.\
        Standard Deviation : {round(np.std(cv_results['test_NPV']),3)}. \n ")
    f.write(f"Mean ROC on Validation set: {round(np.mean(cv_results['test_roc_auc']),3)}\
        Standard Deviation : {round(np.std(cv_results['test_roc_auc']),3)} \n")
    f.write(f"Mean Accuracy on Validation set: {round(np.mean(cv_results['test_accuracy']),3)}\
        Standard Deviation: {round(np.std(cv_results['test_accuracy']),3)} \n")
    for key in cv_results.keys():
        f.write(key+ ": " + str(cv_results[key])+ "\n")