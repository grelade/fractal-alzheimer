import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from sklearn.preprocessing import StandardScaler
import sklearn.svm
import sklearn.neighbors
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import permutation_test_score
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.pipeline import Pipeline
from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler

import lightgbm as lgbm

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model_type',default='svc',type=str,choices=['svc','lgbm','knn'])
args = parser.parse_args()

# load data
dfX = pd.read_csv('data_hursts.csv',index_col=0)
dfY = pd.read_csv('data_labels.csv',index_col=0)

dfY = dfY[dfY['CDR']!=2]
Y = dfY['CDR'].dropna()
y = Y.to_numpy()
y = (y*2).astype(int)
X = dfX.loc[Y.index].to_numpy()

model_type = args.model_type #svc, lgbm, knn

verbose = False
random_seed = 1234
cv_param_folds = 5
cv_select_folds = 5

n_jobs = -1

pipeline = []

pipeline.append(('scaler',StandardScaler()))
pipeline.append(('resampler',SMOTE(random_state = random_seed)))

if model_type == 'svc':
    pipeline.append(('model',sklearn.svm.SVC(verbose=verbose)))
    param_grid = [{'C': 10.**np.arange(-3,3,2),
                   'gamma': 10.**np.arange(-3,1,1)}]
elif model_type == 'knn':
    pipeline.append(('model',sklearn.neighbors.KNeighborsClassifier()))
    param_grid = [{'n_neighbors': [2,3,4,5,6],
                   'weights': ['uniform', 'distance']}]
elif model_type == 'lgbm':
    pipeline.append(('model',lgbm.LGBMClassifier(verbose=int(verbose),n_jobs=n_jobs)))
    param_grid = [{'num_leaves': [15,30,45],
               'max_depth': [5,10,15],
               'n_estimators': [20,40,60,80,100]}]

model_pipeline = Pipeline(pipeline)


def pipeline_param_grid(param_grid,model_name):
    pg = []
    for x in param_grid:
        d = {}
        for k,v in x.items():
            d[model_name+'__'+k] = v
        pg += [d]

    return pg

param_grid = pipeline_param_grid(param_grid,'model')

mcc = sklearn.metrics.make_scorer(sklearn.metrics.matthews_corrcoef)

inner_cv = StratifiedKFold(n_splits=cv_param_folds,
                           shuffle=True,
                           random_state=random_seed)

clf = GridSearchCV(estimator = model_pipeline,
                param_grid = param_grid,
                cv = inner_cv,
                verbose = verbose,
                scoring = dict(accuracy='accuracy',
                                f1_macro='f1_macro',
                                mcc=mcc),
                refit = 'mcc')

outer_cv = StratifiedKFold(n_splits=cv_select_folds,
                           shuffle=True,
                           random_state=random_seed)

cv_select_results = cross_validate(estimator = clf,
                                       X = X,
                                       y = y,
                                       cv = outer_cv,
                                       n_jobs = n_jobs,
                                       verbose = verbose,
                                       return_estimator = True,
                                       return_train_score = True,
                                       scoring = dict(accuracy='accuracy',
                                                      f1_macro='f1_macro',
                                                      mcc=mcc))

perm_scores = permutation_test_score(estimator = clf, X = X,y = y, cv = outer_cv, scoring = mcc)
perm_mcc = perm_scores[2]
test_mcc2 = perm_scores[0]

test_acc = cv_select_results['test_accuracy'].mean()
test_mcc = cv_select_results['test_mcc'].mean()
test_f1 = cv_select_results['test_f1_macro'].mean()

print(f'{model_type} | test_acc = {test_acc:.3f}; test_mcc = {test_mcc:.3f}; test_f1 = {test_f1:.3f}; test_mcc2 = {test_mcc2}; perm_mcc = {perm_mcc}')
