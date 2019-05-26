# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: ml_py36
#     language: python
#     name: ml_py36
# ---

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import json
import ast
import eli5

# +
TRAIN_PATH = "/Users/zakopuro/Code/python_code/kaggle_TMDB/input/train.csv"
TEST_PATH = "/Users/zakopuro/Code/python_code/kaggle_TMDB/input/test.csv"

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

# +
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

def str_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: None if pd.isna(x) else ast.literal_eval(x) )
    return df

df_train = str_to_dict(df_train)
df_test = str_to_dict(df_test)
# -

df_test.head()

df_train.head()

df_train.shape,df_test.shape

df_train['has_homepage'] = 0
df_test['has_homepage'] = 0

# ホームページが存在するかしないかでcolumnを作成
df_train.loc[df_train['homepage'].isnull() == False, 'has_homepage'] = 1
df_test.loc[df_test['homepage'].isnull() == False, 'has_homepage'] = 1
df_train = df_train.drop(['homepage'],axis=1)
df_test = df_test.drop(['homepage'],axis=1)

df_train['genres'].apply(lambda x: len(x) if x != None else 0).value_counts()

df_train['genres_num'] =  df_train['genres'].apply(lambda x: len(x) if x != None else 0)
df_test['genres_num'] =  df_test['genres'].apply(lambda x: len(x) if x != None else 0)

df_train['genres'].apply(lambda x: [i['id'] for i in x] if x != None else 0)

# +
list_genres_num = []
list_revunes_genresu_num = []
for i,_ in enumerate(df_train['genres_num'].value_counts()):
    list_genres_num.append(i)
    list_revunes_genresu_num.append(df_train[df_train['genres_num'] == i]['revenue'].mean()) 
    print(list_revunes_genresu_num)
    
plt.bar(list_genres_num,list_revunes_genresu_num)
plt.title('genres_num/revenus')
plt.xlabel('gneres_num')
plt.ylabel('revenue_median')

# +
df_train['collection_name'] = df_train['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != None else 0)
df_train['has_collection'] = df_train['belongs_to_collection'].apply(lambda x: len(x) if x != None else 0)

df_test['collection_name'] = df_test['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != None else 0)
df_test['has_collection'] = df_test['belongs_to_collection'].apply(lambda x: len(x) if x != None else 0)

df_train = df_train.drop(['belongs_to_collection'], axis=1)
df_test = df_test.drop(['belongs_to_collection'], axis=1)
# -

df_train['production_companies'].apply(lambda x: len(x) if x != None else 0).value_counts()

df_train['companies_num'] =  df_train['production_companies'].apply(lambda x: len(x) if x != None else 0)


df_train['original_language'].value_counts()

le = LabelEncoder()
le.fit(list(df_train['original_language']))
df_train['original_language'] = le.transform(df_train['original_language'])

le = LabelEncoder()
le.fit(list(df_test['original_language']))
df_test['original_language'] = le.transform(df_test['original_language'])

df_train = df_train.drop(['imdb_id', 'homepage','genres','status','poster_path','overview','title','original_title'],axis=1)
df_test = df_test.drop(['imdb_id', 'homepage','genres','status','poster_path','overview','title','original_title'],axis=1)

df_train.head()

df_train['has_collection'] = 0
df_test['has_collection'] = 0

df_train['has_collection'] = df_train['belongs_to_collection'].apply(lambda x: len(x) if x != None else 0)
df_test['has_collection'] = df_test['belongs_to_collection'].apply(lambda x: len(x) if x != None else 0)

df_train = df_train.drop(['belongs_to_collection'],axis=1)
df_test = df_test.drop(['belongs_to_collection'],axis=1)

df_train['companies_num'] = 0
df_test['companies_num'] = 0

df_train['companies_num'] = df_train['production_companies'].apply(lambda x: len(x) if x != None else 0)
df_test['companies_num'] = df_test['production_companies'].apply(lambda x: len(x) if x != None else 0)

df_train = df_train.drop(['production_companies'],axis=1)
df_test = df_test.drop(['production_companies'],axis=1)

df_train['country_num'] = 0
df_test['country_num'] = 0

df_train['country_num'] = df_train['production_countries'].apply(lambda x: len(x) if x != None else 0)
df_test['country_num'] = df_test['production_countries'].apply(lambda x: len(x) if x != None else 0)

df_train = df_train.drop(['production_countries'],axis=1)
df_test = df_test.drop(['production_countries'],axis=1)

df_train.head()

df_train['spoken_languages_num'] = 0
df_test['spoken_languages_num'] = 0

df_train['spoken_languages_num'] = df_train['spoken_languages'].apply(lambda x: len(x) if x != None else 0)
df_test['spoken_languages_num'] = df_test['spoken_languages'].apply(lambda x: len(x) if x != None else 0)

df_train = df_train.drop(['spoken_languages'],axis=1)
df_test = df_test.drop(['spoken_languages'],axis=1)

df_train = df_train.drop(['tagline'],axis=1)
df_test = df_test.drop(['tagline'],axis=1)

df_train['Keyword_num'] = 0
df_test['Keyword_num'] = 0

df_train['Keyword_num'] = df_train['Keywords'].apply(lambda x: len(x) if x != None else 0)
df_test['Keyword_num'] = df_test['Keywords'].apply(lambda x: len(x) if x != None else 0)

df_train = df_train.drop(['Keywords'],axis=1)
df_test = df_test.drop(['Keywords'],axis=1)

df_train['cast_num'] = 0
df_test['cast_num'] = 0

df_train['cast_num'] = df_train['cast'].apply(lambda x: len(x) if x != None else 0)
df_test['cast_num'] = df_test['cast'].apply(lambda x: len(x) if x != None else 0)

df_train = df_train.drop(['cast'],axis=1)
df_test = df_test.drop(['cast'],axis=1)

df_train['crew_num'] = 0
df_test['crew_num'] = 0

df_train['crew_num'] = df_train['crew'].apply(lambda x: len(x) if x != None else 0)
df_test['crew_num'] = df_test['crew'].apply(lambda x: len(x) if x != None else 0)

df_train = df_train.drop(['crew'],axis=1)
df_test = df_test.drop(['crew'],axis=1)


def chg_date_year(date):
    if date == 'NaN':
        year = 'NaN'
        print('debug')
    else:
        if date.year >= 2020:
            # 2020年以降はおかしいので−１００年する
            year = date.year - 100
        else:
            year = date.year
    return year


df_train['release_date'] = pd.to_datetime(df_train['release_date'])
df_test['release_date'] = pd.to_datetime(df_test['release_date'])
df_train['release_date_year'] = df_train['release_date'].apply(lambda x: chg_date_year(x))
df_test['release_date_year'] = df_test['release_date'].apply(lambda x: chg_date_year(x))

df_train = df_train.drop(['release_date'],axis=1)
df_test = df_test.drop(['release_date'],axis=1)

df_train = df_train.drop(['id'],axis=1)
df_test = df_test.drop(['id'],axis=1)

df_train = df_train.fillna(df_train.mean())
df_test = df_test.fillna(df_test.mean())

X = df_train.drop(['revenue'],axis =1)
y = df_train['revenue']

X_test = df_test

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3,random_state=12)


def rmsle(y_pred, y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))


# model_svr = SVR()
# model_svr.fit(X_train,y_train)
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_valid_norm = scaler.transform(X_valid)

#リッジ回帰
Ridge_params = {"alpha":np.logspace(-2,4,24)}
gridsearch = GridSearchCV(Ridge(),Ridge_params,scoring= 'neg_mean_squared_error',return_train_score=True)
gridsearch.fit(X_train_norm,y_train)
print('最適パラメータ',gridsearch.best_params_,'neg_mean_squared_error',gridsearch.best_score_)
ridge = Ridge(alpha = gridsearch.best_params_['alpha']).fit(X_train_norm,y_train)
Ridge_score = rmsle(ridge.predict(X_valid_norm),y_valid)
print(Ridge_score)

#SVR
# params_cnt = 20
svr_params = {"C":[10000000,1000000000], "epsilon":[0.000000001,0.0000001]}
gridsearch = GridSearchCV(SVR(kernel="linear"),svr_params,cv=5,scoring= 'neg_mean_squared_error',return_train_score=True)
gridsearch.fit(X_train_norm,y_train)
print('最適パラメータ',gridsearch.best_params_,'neg_mean_squared_error',gridsearch.best_score_)
svr = SVR(kernel="linear", C=gridsearch.best_params_["C"], epsilon=gridsearch.best_params_["epsilon"])
svr.fit(X_train_norm,y_train)
svr_score = rmsle(svr.predict(X_valid_norm),y_valid)
print(svr_score)

svr_result = pd.DataFrame(gridsearch.cv_results_)
display(svr_result.head())

lgb_params = {'num_leaves': 30,
         'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': 7,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2,
         "verbosity": -1}
lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators = 20000, nthread = 4, n_jobs = -1)
lgb_model.fit(X_train, y_train, 
        eval_set=[(X_train_norm, y_train), (X_valid_norm, y_valid)], eval_metric='rmse',
        verbose=1000, early_stopping_rounds=200)
lgb_score = rmsle(lgb_model.predict(X_valid_norm,num_iteration=lgb_model.best_iteration_),y_valid)
print(lgb_score)

# +
xgb_params = {'eta': 0.01,
              'objective': 'reg:linear',
              'max_depth': 5,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'eval_metric': 'rmse',
              'seed': 11,
              'silent': True}
train_data = xgb.DMatrix(data=X_train, label=y_train)
valid_data = xgb.DMatrix(data=X_valid, label=y_valid)

watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
xgb_model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, verbose_eval=500, params=xgb_params)
y_pred_valid = xgb_model.predict(xgb.DMatrix(X_valid), ntree_limit=xgb_model.best_ntree_limit)
xgb_score = rmsle(y_pred_valid,y_valid)
print(xgb_score)
# -
sub = pd.read_csv('/Users/zakopuro/Code/python_code/kaggle_TMDB/input/sample_submission.csv')
X_test = scaler.transform(df_test)
prediction_ridge = ridge.predict(X_test)
sub['revenue'] = prediction_ridge
sub.to_csv('output/ridge.csv',index=False)


svr_predict = svr.predict(df_test)
sub = pd.read_csv('/Users/zakopuro/Code/python_code/kaggle_TMDB/input/sample_submission.csv')
X_test = scaler.transform(df_test)
prediction_svr = svr.predict(X_test)
sub['revenue'] = prediction_svr
sub.to_csv('output/svr.csv',index=False)
