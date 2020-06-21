import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

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

train_data=data.sample(frac=0.8,random_state=200)
test_data=data.drop(train_data.index)

label = {'correct':1, 'incorrect':0}
data.label=[label[item] for item in data.label]
train_data.label=[label[item] for item in train_data.label]
test_data.label=[label[item] for item in test_data.label]

data=data.dropna()
train_data=train_data.dropna()
test_data=test_data.dropna()

data_label=data.pop('label')
train_label=train_data.pop('label')
test_label=test_data.pop('label')

all_vars=data.columns.to_list()
top_vars=['duration','fare','meter_waiting','meter_waiting_till_pickup',
          'drop_lon','pick_lat','pick_lon','drop_lat','meter_waiting_fare',
          'additional_fare']
top_vars_reduced=['duration','fare','meter_waiting','meter_waiting_till_pickup',
          'meter_waiting_fare','additional_fare']
bottom_vars=[cols for cols in all_vars if cols not in top_vars]
bottom_vars_reduced=[cols for cols in all_vars if cols not in top_vars_reduced]

data=data.drop(bottom_vars,axis=1)
train_data=train_data.drop(bottom_vars,axis=1)
test_data=test_data.drop(bottom_vars,axis=1)
predict_data=predict_data.drop(bottom_vars,axis=1)



random_forest=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=90, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=310,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)


n_estimators=[int(x) for x in np.linspace(start=100,stop=300,num=5)]
max_features=['auto','sqrt']
max_depth=[int(x) for x in np.linspace(1,16,num=3)]
min_samples_split=[4,5,6,7,8,9,10]

grid={'n_estimators':n_estimators,
             'max_features':max_features,
             'max_depth':max_depth,
             'min_samples_split':min_samples_split}

rf_grid=GridSearchCV(random_forest,grid,
                       cv = 10,verbose=2,n_jobs = -1, scoring='f1_macro')

rf_grid.fit(train_data, train_label)

print(rf_grid.best_params_)
fin=rf_grid.best_estimator_
pred=pd.DataFrame(fin.predict(test_data))
print("Accuracy : "+str(metrics.accuracy_score(test_label,pred)))
print("F1 score : "+str(metrics.f1_score(test_label,pred)))
