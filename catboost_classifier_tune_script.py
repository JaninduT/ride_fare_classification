import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
import hyperopt
from numpy.random import RandomState

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

params = {
    'iterations': 1000,
    'learning_rate': 0.1,
    'eval_metric': 'F1',
    'random_seed': 42,
    'logging_level': 'Silent',
    'use_best_model': True
}
train_pool = Pool(train_data, train_label, cat_features=categorical_features_indices)
validate_pool = Pool(test_data, test_label, cat_features=categorical_features_indices)

def hyperopt_objective(params):
    model = CatBoostClassifier(
        l2_leaf_reg=int(params['l2_leaf_reg']),
        learning_rate=params['learning_rate'],
        iterations=1000,
        eval_metric='F1',
        random_seed=42,
        verbose=False,
        loss_function='Logloss',
    )
    
    cv_data = cv(
        Pool(data, data_label, cat_features=categorical_features_indices),
        model.get_params()
    )
    best_f1 = np.max(cv_data['test-F1-mean'])
    
    return 1 - best_f1

params_space = {
    'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 2, 1),
    'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
}

trials = hyperopt.Trials()

best = hyperopt.fmin(
    hyperopt_objective,
    space=params_space,
    algo=hyperopt.tpe.suggest,
    max_evals=25,
    trials=trials,
    rstate=RandomState(123)
)

print(best)


model = CatBoostClassifier(
    l2_leaf_reg=int(best['l2_leaf_reg']),
    learning_rate=best['learning_rate'],
    iterations=1000,
    eval_metric='F1',
    random_seed=42,
    verbose=False,
    loss_function='Logloss'
)
cv_data = cv(Pool(data, data_label, cat_features=categorical_features_indices), model.get_params())
print('Precise validation f1 score: {}'.format(np.max(cv_data['test-F1-mean'])))


#{'l2_leaf_reg': 1.0, 'learning_rate': 0.16129990013229004}
