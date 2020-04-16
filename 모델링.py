#!/usr/bin/env python
# coding: utf-8

##### is Main Process title
### is Detail
# is Remark

# 해당 코드는 정리용입니다.  
# 3개의 test 데이터 중 특이한 것을 제외한 test3에 대해서만 진행합니다.     
# 다른 데이터를 사용할 경우 test를 수정해주시고 test의 특징을 반영하기 위해 train에서 특징을 나타내는 변수를 제거해주세요.  
# 추가적으로 파라미터 튜닝 역시 LGBM 모델을 대표적으로 진행합니다. 5개의 코드 모두 구현할 시에는 해당 코드를 복사하여 모델을 변경하면 됩니다. 그리고 모델링에 많은 시간이 걸릴 수 있으니 "RUN ALL"을 진행할 때 주의바랍니다.

##### 0. 데이터 로드 및 세팅
### 데이터 로드
import pandas as pd
train = pd.read_csv('final_train.csv', encoding = 'cp949')
test = pd.read_csv('final_test.csv', encoding = 'cp949')

### Test Seting
test1 = test.query('FLO != "M"')

a = list(train.FLT.unique())
b = list(test.FLT.unique())

f=[]
for i in b:
    if i in a:
        pass
    else:
        f.append(b)

test3 = test1.query('FLT != @f')

### Train & Test split / Scaling
X_train = train.drop(['DLY','index'], axis = 1)
y_train = train['DLY']
X_test = test3.drop(['DLY','index'], axis = 1)
y_test = test3['DLY']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train2 = scaler.fit_transform(X_train)
X_test2 = scaler.transform(X_test)

from sklearn.model_selection import train_test_split
X_train2_train, X_train2_val, y_train_train, y_train_val = train_test_split(X_train2, y_train, test_size=0.3, random_state=0)

##### 1. Basic Modeling (모델 채택)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

########### warning this code!!!
# Running this in .py
get_ipython().run_cell_magic('time', '', '\nclfs = []\nlog=LogisticRegression(random_state=0); clfs.append(log)\ntree=DecisionTreeClassifier(random_state=0); clfs.append(tree)\nnn=MLPClassifier(random_state=0); clfs.append(nn)\nknn=KNeighborsClassifier(); clfs.append(knn)\nrf=RandomForestClassifier(random_state=0); clfs.append(rf)\ngbm=GradientBoostingClassifier(random_state=0); clfs.append(gbm)\nlgbm=LGBMClassifier(random_state=0); clfs.append(lgbm)\nxgb=XGBClassifier(); clfs.append(xgb)\n\ncv_results = []\npred_results = []\nfor clf in clfs :\n    cv_results.append(cross_val_score(clf, X_train2, y_train, scoring="roc_auc", cv=10, n_jobs=-1))\n    clf.fit(X_train2, y_train)\n    pred_results.append(pd.Series(clf.predict_proba(X_test2)[:,1], name=type(clf).__name__))\n\ncv_means = []\ncv_std = []\nfor cv_result in cv_results:\n    cv_means.append(cv_result.mean())\n    cv_std.append(cv_result.std())\n    \ncv_res = pd.DataFrame({"CrossValMeans":cv_means, "CrossValerrors": cv_std, \n                       "Algorithm": [type(i).__name__ for i in clfs]})\nensemble_results = pd.concat(pred_results, axis=1)\n\nplt.figure(figsize = (8,6))\ng = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{\'xerr\':cv_std})\ng.set_xlabel("Mean Accuracy")\ng.set_title("Cross validation scores")\nplt.show()\n\nplt.figure(figsize = (8,6))\ng = sns.heatmap(ensemble_results.corr(),annot=True)\ng.set_title("Correlation between models")\nplt.show()')

# Running this in .ipynb
%%time

clfs = []
log=LogisticRegression(random_state=0); clfs.append(log)
tree=DecisionTreeClassifier(random_state=0); clfs.append(tree)
nn=MLPClassifier(random_state=0); clfs.append(nn)
knn=KNeighborsClassifier(); clfs.append(knn)
rf=RandomForestClassifier(random_state=0); clfs.append(rf)
gbm=GradientBoostingClassifier(random_state=0); clfs.append(gbm)
lgbm=LGBMClassifier(random_state=0); clfs.append(lgbm)
xgb=XGBClassifier(); clfs.append(xgb)

cv_results = []
pred_results = []
for clf in clfs :
    cv_results.append(cross_val_score(clf, X_train2, y_train, scoring="roc_auc", cv=10, n_jobs=-1))
    clf.fit(X_train2, y_train)
    pred_results.append(pd.Series(clf.predict_proba(X_test2)[:,1], name=type(clf).__name__))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
cv_res = pd.DataFrame({"CrossValMeans":cv_means, "CrossValerrors": cv_std, 
                       "Algorithm": [type(i).__name__ for i in clfs]})
ensemble_results = pd.concat(pred_results, axis=1)

plt.figure(figsize = (8,6))
g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g.set_title("Cross validation scores")
plt.show()

plt.figure(figsize = (8,6))
g = sns.heatmap(ensemble_results.corr(),annot=True)
g.set_title("Correlation between models")
plt.show()
########### warning until!!!

cv_res.sort_values(by="CrossValMeans",ascending=False)
# 평균적으로 score가 높은 Logistic, MLP, XGB, GBM, LGBM 채택

##### 2. Imbalance Learning
train.DLY.value_counts()

### TomekLink
from imblearn.under_sampling import TomekLinks
tl = TomekLinks(random_state=0)
X_train2_tl, y_train_tl = tl.fit_sample(X_train2, y_train)

X_train2_tl_train, X_train2_tl_val, y_train_tl_train, y_train_tl_val = train_test_split(X_train2_tl, y_train_tl, test_size=0.3, random_state=0)

lgbm = LGBMClassifier(random_state=0).fit(X_train2_tl_train, y_train_tl_train)
roc_auc_score(y_train_tl_val, lgbm.predict_proba(X_train2_tl_val)[:,1])

### EasyEnsemble
from imblearn.ensemble import EasyEnsemble
ee = EasyEnsemble(random_state=0)
X_train2_ee, y_train_ee = ee.fit_sample(X_train2, y_train)

X_train2_ee = X_train2_ee.reshape((-1,X_train2_train.shape[1]))
y_train_ee = y_train_ee.reshape(-1)

X_train2_ee_train, X_train2_ee_val, y_train_ee_train, y_train_ee_val = train_test_split(X_train2_ee, y_train_ee, test_size=0.3, random_state=0)

lgbm = LGBMClassifier(random_state=0).fit(X_train2_ee_train, y_train_ee_train)
roc_auc_score(y_train_ee_val, lgbm.predict_proba(X_train2_ee_val)[:,1])

##### 3. Paramerter Tuning
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

kfold = StratifiedKFold(n_splits=10)

### 기본 데이터
lgbm = LGBMClassifier()

lgb_param_grid = {'num_leaves' : np.arange(2,50,1),
                'min_data_in_leaf' : np.arange(50,500,50),
                 'max_depth' : np.arange(1,10,1)}

LGB = RandomizedSearchCV(lgbm, param_distributions = lgb_param_grid, cv=kfold, scoring="roc_auc", n_jobs= 3, verbose = 1)
LGB.fit(X_train2, y_train)

LGB_best = LGB.best_estimator_ ;display(LGB.best_score_, LGB_best)

LGB_best = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
        importance_type='split', learning_rate=0.1, max_depth=2,
        min_child_samples=20, min_child_weight=0.001, min_data_in_leaf=400,
        min_split_gain=0.0, n_estimators=100, n_jobs=-1, num_leaves=32,
        objective=None, random_state=None, reg_alpha=0.0, reg_lambda=0.0,
        silent=True, subsample=1.0, subsample_for_bin=200000,
        subsample_freq=0)

### Tomek 적용
lgbm = LGBMClassifier()

lgb_param_grid = {'num_leaves' : np.arange(2,50,1),
                'min_data_in_leaf' : np.arange(50,500,50),
                 'max_depth' : np.arange(1,10,1)}

LGB_tl = RandomizedSearchCV(lgbm, param_distributions = lgb_param_grid, cv=kfold, scoring="roc_auc", n_jobs= 3, verbose = 1)
LGB_tl.fit(X_train2_tl, y_train_tl)

LGB_best_tl = LGB_tl.best_estimator_ ;display(LGB_tl.best_score_, LGB_best_tl)

LGB_best_tl = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
        importance_type='split', learning_rate=0.1, max_depth=6,
        min_child_samples=20, min_child_weight=0.001, min_data_in_leaf=150,
        min_split_gain=0.0, n_estimators=100, n_jobs=-1, num_leaves=18,
        objective=None, random_state=None, reg_alpha=0.0, reg_lambda=0.0,
        silent=True, subsample=1.0, subsample_for_bin=200000,
        subsample_freq=0)

### Easy 적용
lgbm = LGBMClassifier()

lgb_param_grid = {'num_leaves' : np.arange(2,50,1),
                'min_data_in_leaf' : np.arange(50,500,50),
                 'max_depth' : np.arange(1,10,1)}

LGB_ee = RandomizedSearchCV(lgbm,param_distributions = lgb_param_grid, cv=kfold, scoring="roc_auc", n_jobs= 3, verbose = 1)
LGB_ee.fit(X_train2_ee, y_train_ee)

LGB_best_ee = LGB_ee.best_estimator_ ;display(LGB_ee.best_score_, LGB_best_ee)

LGB_best_ee = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
        importance_type='split', learning_rate=0.1, max_depth=9,
        min_child_samples=20, min_child_weight=0.001, min_data_in_leaf=300,
        min_split_gain=0.0, n_estimators=100, n_jobs=-1, num_leaves=48,
        objective=None, random_state=None, reg_alpha=0.0, reg_lambda=0.0,
        silent=True, subsample=1.0, subsample_for_bin=200000,
        subsample_freq=0)

##### 4. Ensemble
### Soft_Voting
from sklearn.ensemble import VotingClassifier

voting_basic  = VotingClassifier(estimators=[('GBC_best', GBC_best), ('XGB_best', XGB_best), ('LGB_best', LGB_best),('LR_best',LR_best),('NN_best',NN_best)], voting='soft')
pred_basic = voting_basic.fit(X_train2, y_train).predict_proba(X_test2)[:,1]

voting_tl  = VotingClassifier(estimators=[('GBC_best_tl', GBC_best_tl), ('XGB_best_tl', XGB_best_tl), ('LGB_best_tl', LGB_best_tl),('LR_best_tl',LR_best_tl),('NN_best_tl',NN_best_tl)], voting='soft')
pred_tl = voting_tl.fit(X_train2_tl, y_train_tl).predict_proba(X_test2)[:,1]

voting_ee  = VotingClassifier(estimators=[('GBC_best_ee', GBC_best_ee), ('XGB_best_ee', XGB_best_ee), ('LGB_best_ee', LGB_best_ee),('LR_best_ee',LR_best_ee),('NN_best_ee',NN_best_ee)], voting='soft')
pred_ee = voting_ee.fit(X_train2_ee, y_train_ee).predict_proba(X_test2)[:,1]

### Gmean
from scipy.stats.mstats import gmean
pred_final = gmean([pred_basic, pred_tl, pred_ee], axis = 0)

##### 5. 제출
y_test = pd.DataFrame(pred_final).rename(columns = {0: 'DLY_RATE'})

test1 = pd.concat([X_test, y_test], axis = 1)
test2 = pd.concat([X_test, y_test], axis = 1)
test3 = pd.concat([X_test, y_test], axis = 1)

test.drop(['DLY', 'DLY_RATE'], axis = 1)

feature = ['SDT_YY', 'SDT_MM', 'SDT_DD', 'SDT_DY', 'ARP', 'ODP', 'FLO', 'FLT', 'AOD', 'STT']

test = pd.merge(test, test1, how = 'left', left_on = feature, right_on = feature)
test = pd.merge(test, test2, how = 'left', left_on = feature, right_on = feature)
test = pd.merge(test, test3, how = 'left', left_on = feature, right_on = feature)

def DLY(x):
    if x >= 0.5:
        return 1
    else:
        return 0
test['DLY'] = test['DLY_RATE'].apply(DLY)

test.to_csv('submission.csv', index = False, encoding = 'cp949')
