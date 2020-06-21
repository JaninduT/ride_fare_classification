import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=12,4

pd.set_option('display.max_columns',None)
data=pd.read_csv('train.csv')
predict_data=pd.read_csv('test.csv')

data.pickup_time=pd.to_datetime(data.pickup_time,format='%m/%d/%Y %H:%M')
data.drop_time=pd.to_datetime(data.drop_time,format='%m/%d/%Y %H:%M')

data['pickup_year']=data.apply(lambda row: row.pickup_time.year,axis=1)
data['pickup_month']=data.apply(lambda row: row.pickup_time.month,axis=1)
data['pickup_day']=data.apply(lambda row: row.pickup_time.day,axis=1)
data['pickup_hour']=data.apply(lambda row: row.pickup_time.hour,axis=1)
data['pickup_minute']=data.apply(lambda row: row.pickup_time.minute,axis=1)

data['drop_year']=data.apply(lambda row: row.drop_time.year,axis=1)
data['drop_month']=data.apply(lambda row: row.drop_time.month,axis=1)
data['drop_day']=data.apply(lambda row: row.drop_time.day,axis=1)
data['drop_hour']=data.apply(lambda row: row.drop_time.hour,axis=1)
data['drop_minute']=data.apply(lambda row: row.drop_time.minute,axis=1)

data=data.drop(columns=['tripid','pickup_time','drop_time'])

predict_data.pickup_time=pd.to_datetime(predict_data.pickup_time,format='%m/%d/%Y %H:%M')
predict_data.drop_time=pd.to_datetime(predict_data.drop_time,format='%m/%d/%Y %H:%M')

predict_data['pickup_year']=predict_data.apply(lambda row: row.pickup_time.year,axis=1)
predict_data['pickup_month']=predict_data.apply(lambda row: row.pickup_time.month,axis=1)
predict_data['pickup_day']=predict_data.apply(lambda row: row.pickup_time.day,axis=1)
predict_data['pickup_hour']=predict_data.apply(lambda row: row.pickup_time.hour,axis=1)
predict_data['pickup_minute']=predict_data.apply(lambda row: row.pickup_time.minute,axis=1)

predict_data['drop_year']=predict_data.apply(lambda row: row.drop_time.year,axis=1)
predict_data['drop_month']=predict_data.apply(lambda row: row.drop_time.month,axis=1)
predict_data['drop_day']=predict_data.apply(lambda row: row.drop_time.day,axis=1)
predict_data['drop_hour']=predict_data.apply(lambda row: row.drop_time.hour,axis=1)
predict_data['drop_minute']=predict_data.apply(lambda row: row.drop_time.minute,axis=1)

predict_data=predict_data.drop(columns=['pickup_time','drop_time'])
predict_index=predict_data.pop('tripid')

train_data=data.sample(frac=0.75,random_state=200)
test_data=data.drop(train_data.index)

label = {'correct':1, 'incorrect':0}
data.label=[label[item] for item in data.label]
train_data.label=[label[item] for item in train_data.label]
test_data.label=[label[item] for item in test_data.label]

data=data.dropna()
train_data=train_data.dropna()
test_data=test_data.dropna()
target='label'
data_label=data.pop('label')
train_label=train_data.pop('label')
test_label=test_data.pop('label')

all_vars=data.columns.to_list()
top_vars=['duration','fare','meter_waiting','meter_waiting_till_pickup',
          'drop_lon','pick_lat','pick_lon','drop_lat','meter_waiting_fare',
          'additional_fare','pick_hour','pick_minute']
top_vars_reduced=['duration','fare','meter_waiting','meter_waiting_till_pickup',
          'meter_waiting_fare','additional_fare']
bottom_vars=[cols for cols in all_vars if cols not in top_vars]
bottom_vars_reduced=[cols for cols in all_vars if cols not in top_vars_reduced]

data=data.drop(bottom_vars,axis=1)
train_data=train_data.drop(bottom_vars,axis=1)
test_data=test_data.drop(bottom_vars,axis=1)
predict_data=predict_data.drop(bottom_vars,axis=1)

def modelfit(alg, dtrain, dtest, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='error', early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    
    alg.fit(dtrain[predictors], dtrain['label'],eval_metric='error')
        
    
    dtest_predictions = alg.predict(dtest[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:,1]
        
  
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtest['label'].values, dtest_predictions))
    print("F1 score : %f" % metrics.f1_score(dtest['label'].values, dtest_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtest['label'], dtest_predprob))
                    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
   

##predictors = [x for x in data.columns if x not in [target]]
##xgb1 = XGBClassifier(
## learning_rate =0.1,
## n_estimators=175,
## max_depth=9,
## min_child_weight=2,
## gamma=0.0,
## subsample=0.8,
## colsample_bytree=0.8,
## reg_alpha=5e-05,
## objective= 'binary:logistic',
## nthread=4,
## scale_pos_weight=1,
## seed=27)
##modelfit(xgb1, train_data, test_data, predictors)

##param_test1 = {
## 'max_depth':range(3,20,2),
## 'min_child_weight':range(1,10,2)
##}
##gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=381, max_depth=5,
## min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
## objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
## param_grid = param_test1, scoring='f1_macro',n_jobs=4,iid=False, cv=5)
##gsearch1.fit(train_data[predictors],train_data[target])
##print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)

##param_test2 = {
## 'max_depth':[8,9,10],
## 'min_child_weight':[2,3,4]
##}
##gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=381, max_depth=5,
## min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
## objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
## param_grid = param_test2, scoring='f1_macro',n_jobs=4,iid=False, cv=5)
##gsearch2.fit(train_data[predictors],train_data[target])
##print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_)

##param_test2b = {
## 'min_child_weight':[1,2,3,4]
##}
##gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=381, max_depth=9,
## min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
## objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
## param_grid = param_test2b, scoring='f1_macro',n_jobs=4,iid=False, cv=5)
##gsearch2b.fit(train_data[predictors],train_data[target])
##print(gsearch2b.cv_results_, gsearch2b.best_params_, gsearch2b.best_score_)

##param_test3 = {
## 'gamma':[i/10.0 for i in range(0,5)]
##}
##gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=381, max_depth=9,
## min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
## objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
## param_grid = param_test3, scoring='f1_macro',n_jobs=4,iid=False, cv=5)
##gsearch3.fit(train_data[predictors],train_data[target])
##print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)

##param_test4 = {
## 'subsample':[i/10.0 for i in range(6,10)],
## 'colsample_bytree':[i/10.0 for i in range(6,10)]
##}
##gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=234, max_depth=9,
## min_child_weight=2, gamma=0.0, subsample=0.8, colsample_bytree=0.8,
## objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
## param_grid = param_test4, scoring='f1_macro',n_jobs=4,iid=False, cv=5)
##gsearch4.fit(train_data[predictors],train_data[target])
##print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)

##param_test5 = {
## 'subsample':[i/100.0 for i in range(75,90,5)],
## 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
##}
##gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=234, max_depth=9,
## min_child_weight=2, gamma=0.0, subsample=0.8, colsample_bytree=0.8,
## objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
## param_grid = param_test5, scoring='f1_macro',n_jobs=4,iid=False, cv=5)
##gsearch5.fit(train_data[predictors],train_data[target])
##print(gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_)

##param_test6 = {
## 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
##}
##gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=234, max_depth=9,
## min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
## objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
## param_grid = param_test6, scoring='f1_macro',n_jobs=4,iid=False, cv=5)
##gsearch6.fit(train_data[predictors],train_data[target])
##print(gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_)

##param_test7 = {
## 'reg_alpha':[1e-7, 1e-6, 0.05e-5, 1e-5, 1e-4, 0.5e-4]
##}
##gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=234, max_depth=9,
## min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
## objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
## param_grid = param_test7, scoring='f1_macro',n_jobs=4,iid=False, cv=5)
##gsearch7.fit(train_data[predictors],train_data[target])
##print(gsearch7.cv_results_, gsearch7.best_params_, gsearch7.best_score_)

xgb_model=XGBClassifier(learning_rate =0.1,
 n_estimators=175,
 max_depth=9,
 min_child_weight=2,
 gamma=0.0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=5e-05,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
xgb_model.fit(data,data_label)

print(xgb_model)

print("score : "+str(xgb_model.score(train_data,train_label)))
pred=pd.DataFrame(xgb_model.predict(test_data))
print("Accuracy : "+str(metrics.accuracy_score(test_label,pred)))
print("F1 score : "+str(metrics.f1_score(test_label,pred)))
