import xgboost as xgb
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

XGB_PARAMS = {
    'max_depth':10, 
    'subsample':1,
    'min_child_weight':0.5,
    'eta':0.4, 
    'num_round':100,
    'seed':1,
    'silent':0,
    'eval_metric':'rmse'
}

def xgbRegressor():
    return xgb.XGBRegressor(
        max_depth=10,
        learning_rate=0.0025,
        n_estimators=5000
    )

def price_diff_over(train, y_test, y_pred, over):
    d = train.join(y_test).join(pd.DataFrame(y_pred).set_index(y_test.index).rename({0: 'Predicted'}, axis=1))
    d['SalePrice'] = np.exp(d['SalePrice'])
    d['Predicted'] = np.exp(d['Predicted'])
    d['DiffRatio'] = ((d['SalePrice'] - d['Predicted']).abs() / d['SalePrice']) * 100
    return d[d['DiffRatio'] > over].sort_values(by='DiffRatio', ascending=False)[['DiffRatio']]

def missing_barplot(nulls_count):
    p = sns.barplot(nulls_count.index, y=nulls_count['Percent missing'])
    for item in p.get_xticklabels():
        item.set_rotation(45)
    return p

def correlation_barplot(data, target, num):
    corrs = data.assign(SalePrice=target).corr()['SalePrice'].abs().sort_values(ascending=False)[:num+1]
    corrs2 = data.assign(SalePrice=target).corr()['SalePrice'].loc[corrs.index].sort_values(ascending=False)
    corrs2 = corrs2.drop(['SalePrice'])
    plt.figure(figsize=(20,8))
    p = sns.barplot(corrs2.index, y=corrs2)
    for item in p.get_xticklabels():
        item.set_rotation(45)
    p