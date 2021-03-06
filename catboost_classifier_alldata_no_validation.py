import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score

pd.set_option('display.max_columns',None)
data=pd.read_csv('train.csv')
predict_data=pd.read_csv('test.csv')
predict_index=predict_data.pop('tripid')

data=data.drop(columns=['tripid'])
train_data=data.sample(frac=0.8,random_state=200)
test_data=data.drop(train_data.index)

label = {'correct':1, 'incorrect':0}
data.label=[label[item] for item in data.label]
train_data.label=[label[item] for item in train_data.label]
test_data.label=[label[item] for item in test_data.label]

data_label=data.pop('label')
train_label=train_data.pop('label')
test_label=test_data.pop('label')

##all_vars=data.columns.to_list()
##top_vars_reduced=['duration','fare','meter_waiting','meter_waiting_till_pickup',
##          'meter_waiting_fare','additional_fare','pickup_time','drop_time']
##bottom_vars_reduced=[cols for cols in all_vars if cols not in top_vars_reduced]
##data=data.drop(bottom_vars_reduced,axis=1)
##train_data=train_data.drop(bottom_vars_reduced,axis=1)
##test_data=test_data.drop(bottom_vars_reduced,axis=1)
###predict_data=predict_data.drop(bottom_vars,axis=1)


##train_data.fillna(-999,inplace=True)
##test_data.fillna(-999,inplace=True)

data.dropna()
train_data.dropna()
test_data.dropna()

categorical_features_indices = np.where(train_data.dtypes != np.float)[0]

#train_pool = Pool(train_data, train_label, cat_features=categorical_features_indices)
train_pool = Pool(data, data_label, cat_features=categorical_features_indices)
validate_pool = Pool(test_data, test_label, cat_features=categorical_features_indices)

model = CatBoostClassifier()

model.fit(train_pool,
    logging_level='Verbose',
    plot=False
)


predictions = model.predict(predict_data)

output = pd.DataFrame({'tripid': predict_index, 'prediction': predictions})
output.to_csv('sample_submission_catboost_alldata_no_eval_mat.csv', index=False)

pred=pd.DataFrame(model.predict(test_data))
print("Accuracy : "+str(metrics.accuracy_score(test_label,pred)))
print("F1 score : "+str(metrics.f1_score(test_label,pred)))
