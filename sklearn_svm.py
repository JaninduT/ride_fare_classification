import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
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
train_data.label=[label[item] for item in train_data.label]
test_data.label=[label[item] for item in test_data.label]

train_data=train_data.dropna()
test_data=test_data.dropna()

train_label=train_data.pop('label')
test_label=test_data.pop('label')

svm_model=SVC(probability=True)
svm_model=svm_model.fit(train_data,train_label)

predictions = svm_model.predict(predict_data)

output = pd.DataFrame({'tripid': predict_index, 'prediction': predictions})
output.to_csv('sample_submission_svm.csv', index=False)
