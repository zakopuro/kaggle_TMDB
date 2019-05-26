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
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import collections
import json
import math
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

df_train['belongs_to_collection'].apply(lambda x: len(x) if x != None else 0).value_counts()

df_train['has_collection'] = df_train['belongs_to_collection'].apply(lambda x: len(x) if x != None else 0)
df_test['has_collection'] = df_test['belongs_to_collection'].apply(lambda x: len(x) if x != None else 0)

df_train['collection_name'] = df_train['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != None else 0)
df_test['collection_name'] = df_test['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != None else 0)

df_train = df_train.drop(['belongs_to_collection'], axis=1)
df_test = df_test.drop(['belongs_to_collection'], axis=1)

df_train['genres'].apply(lambda x: len(x) if x != None else 0).value_counts()

df_train['genres_num'] =  df_train['genres'].apply(lambda x: len(x) if x != None else 0)
df_test['genres_num'] =  df_test['genres'].apply(lambda x: len(x) if x != None else 0)

list_of_genres = list(df_train['genres'].apply(lambda x: [i['name'] for i in x] if x != None else []).values)
genres_all = ','.join([i for j in list_of_genres for i in j])
list_genres_all = genres_all.split(',')
print(collections.Counter(list_genres_all))

df_train['all_genres'] = df_train['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != None else '')
df_test['all_genres'] = df_test['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != None else '')

for gen in list_genres_all:
    df_train['genre_' + gen] = df_train['all_genres'].apply(lambda x: 1 if gen in x else 0)
    df_test['genre_' + gen] = df_test['all_genres'].apply(lambda x: 1 if gen in x else 0)
df_train = df_train.drop(['all_genres'],axis=1)
df_test = df_test.drop(['all_genres'],axis=1)

df_train = df_train.drop(['genres'],axis=1)
df_test = df_test.drop(['genres'],axis=1)

df_train['has_homepage'] = 0
df_test['has_homepage'] = 0
df_train.loc[df_train['homepage'].isnull() == False, 'has_homepage'] = 1
df_test.loc[df_test['homepage'].isnull() == False, 'has_homepage'] = 1
df_train = df_train.drop(['homepage'],axis=1)
df_test = df_test.drop(['homepage'],axis=1)

df_train = df_train.drop(['imdb_id'],axis=1)
df_test = df_test.drop(['imdb_id'],axis=1)

list_top10_original_lang = list(collections.Counter(df_train['original_language']).most_common(10))
for lang in list_top10_original_lang:
    df_train['original_language_' + lang[0]] = df_train['original_language'].apply(lambda x: 1 if lang[0] in x else 0)
    df_test['original_language_' + lang[0]] = df_test['original_language'].apply(lambda x: 1 if lang[0] in x else 0)

df_train = df_train.drop(['original_language'],axis=1)
df_test = df_test.drop(['original_language'],axis=1)

df_train = df_train.drop(['original_title'],axis=1)
df_test = df_test.drop(['original_title'],axis=1)

df_train = df_train.drop(['overview'],axis=1)
df_test = df_test.drop(['overview'],axis=1)

df_train = df_train.drop(['poster_path'],axis=1)
df_test = df_test.drop(['poster_path'],axis=1)

df_train['production_companies_num'] =  df_train['production_companies'].apply(lambda x: len(x) if x != None else 0)
df_test['production_companies_num'] =  df_test['production_companies'].apply(lambda x: len(x) if x != None else 0)

# +
list_of_production_companies = list(df_train['production_companies'].apply(lambda x: [i['name'] for i in x] if x != None else []).values)
production_companies_all = ','.join([i for j in list_of_production_companies for i in j])
list_production_companies_all = production_companies_all.split(',')
list_top15_production_companies = collections.Counter(list_production_companies_all).most_common(15)

df_train['all_production_companies'] = df_train['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != None else '')
df_test['all_production_companies'] = df_test['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != None else '')
# -

for company in list_top15_production_companies:
    df_train['production_companies_' + company[0]] = df_train['all_production_companies'].apply(lambda x: 1 if company[0] in x else 0)
    df_test['production_companies_' + company[0]] = df_test['all_production_companies'].apply(lambda x: 1 if company[0] in x else 0)

df_train['production_countries_num'] =  df_train['production_countries'].apply(lambda x: len(x) if x != None else 0)
df_test['production_countries_num'] =  df_test['production_countries'].apply(lambda x: len(x) if x != None else 0)

# +
list_of_production_countries = list(df_train['production_countries'].apply(lambda x: [i['name'] for i in x] if x != None else []).values)
production_countries_all = ','.join([i for j in list_of_production_countries for i in j])
list_production_countries_all = production_countries_all.split(',')
list_top20_production_countries = collections.Counter(list_production_countries_all).most_common(20)

df_train['all_production_countries'] = df_train['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != None else '')
df_test['all_production_countries'] = df_test['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != None else '')
# -

for country in list_top20_production_countries:
    df_train['production_country_' + country[0]] = df_train['all_production_countries'].apply(lambda x: 1 if country[0] in x else 0)
    df_test['production_country_' + country[0]] = df_test['all_production_countries'].apply(lambda x: 1 if country[0] in x else 0)

df_train = df_train.drop(['all_production_countries','all_production_companies','production_companies','production_countries'],axis=1)
df_test = df_test.drop(['all_production_countries','all_production_companies','production_companies','production_countries'],axis=1)


def chg_date_year(date):
    if date.year >= 2020:
        # 2020年以降はおかしいので−１００年する
        year = date.year - 100
    else:
        year = date.year
    return int(year)


df_test.loc[df_test['release_date'].isnull() == True, 'release_date'] = '01/05/00'
df_train['release_date'] = pd.to_datetime(df_train['release_date'])
df_test['release_date'] = pd.to_datetime(df_test['release_date'])
df_train['release_date_year'] = df_train['release_date'].apply(lambda x: chg_date_year(x))
df_test['release_date_year'] = df_test['release_date'].apply(lambda x: chg_date_year(x))

df_train['release_date_month'] = df_train['release_date'].apply(lambda x: x.month)
df_test['release_date_month'] = df_test['release_date'].apply(lambda x: x.month)

df_train['runtime'] = df_train['runtime'].fillna(df_train['runtime'].mean())
df_test['runtime'] = df_test['runtime'].fillna(df_test['runtime'].mean())

df_train['spoken_languages_num'] =  df_train['spoken_languages'].apply(lambda x: len(x) if x != None else 0)
df_test['spoken_languages_num'] =  df_test['spoken_languages'].apply(lambda x: len(x) if x != None else 0)

# +
list_of_spoken_languages = list(df_train['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != None else []).values)
spoken_languages_all = ','.join([i for j in list_of_spoken_languages for i in j])
list_all_spoken_languages = spoken_languages_all.split(',')
list_top20_spoken_languages = collections.Counter(list_all_spoken_languages).most_common(20)
df_train['all_spoken_languages'] = df_train['spoken_languages'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != None else '')
df_test['all_spoken_languages'] = df_test['spoken_languages'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != None else '')

for lang in list_top20_spoken_languages:
    df_train['spoken_languages_' + lang[0]] = df_train['all_spoken_languages'].apply(lambda x: 1 if lang[0] in x else 0)
    df_test['spoken_languages_' + lang[0]] = df_test['all_spoken_languages'].apply(lambda x: 1 if lang[0] in x else 0)
df_train = df_train.drop(['all_spoken_languages'],axis=1)
df_test = df_test.drop(['all_spoken_languages'],axis=1)
# -

df_train = df_train.drop(['spoken_languages'],axis=1)
df_test = df_test.drop(['spoken_languages'],axis=1)

df_train = df_train.drop(['release_date'],axis=1)
df_test = df_test.drop(['release_date'],axis=1)

df_train = df_train.drop(['status'],axis=1)
df_test = df_test.drop(['status'],axis=1)

df_train = df_train.drop(['tagline'],axis=1)
df_test = df_test.drop(['tagline'],axis=1)

df_train = df_train.drop(['title'],axis=1)
df_test = df_test.drop(['title'],axis=1)

df_train['Keywords_num'] = df_train['Keywords'].apply(lambda x: len(x) if x != None else 0)
df_test['Keywords_num'] = df_test['Keywords'].apply(lambda x: len(x) if x != None else 0)

# +
list_of_Keywords = list(df_train['Keywords'].apply(lambda x: [i['name'] for i in x] if x != None else []).values)
Keywords_all = ','.join([i for j in list_of_Keywords for i in j])
list_Keywords_all = Keywords_all.split(',')
list_top30_Keywords = collections.Counter(list_Keywords_all).most_common(30)

df_train['all_Keywords'] = df_train['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != None else '')
df_test['all_Keywords'] = df_test['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != None else '')

for Keyword in list_top30_Keywords:
    df_train['Keywords_' + Keyword[0]] = df_train['all_Keywords'].apply(lambda x: 1 if Keyword[0] in x else 0)
    df_test['Keywords_' + Keyword[0]] = df_test['all_Keywords'].apply(lambda x: 1 if Keyword[0] in x else 0)
df_train = df_train.drop(['all_Keywords'],axis=1)
df_test = df_test.drop(['all_Keywords'],axis=1)
# -

df_train = df_train.drop(['Keywords'],axis=1)
df_test = df_test.drop(['Keywords'],axis=1)

df_train['cast_num'] = df_train['cast'].apply(lambda x: len(x) if x != None else 0)
df_test['cast_num'] = df_test['cast'].apply(lambda x: len(x) if x != None else 0)

# +
list_of_cast = list(df_train['cast'].apply(lambda x: [i['name'] for i in x] if x != None else []).values)
cast_all = ','.join([i for j in list_of_cast for i in j])
list_cast_all = cast_all.split(',')
list_top30_cast = collections.Counter(list_cast_all).most_common(31)[1:31]
df_train['all_cast'] = df_train['cast'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != None else '')
df_test['all_cast'] = df_test['cast'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != None else '')

for cast in list_top30_cast:
    df_train['cast_' + cast[0]] = df_train['all_cast'].apply(lambda x: 1 if cast[0] in x else 0)
    df_test['cast_' + cast[0]] = df_test['all_cast'].apply(lambda x: 1 if cast[0] in x else 0)
df_train = df_train.drop(['all_cast'],axis=1)
df_test = df_test.drop(['all_cast'],axis=1)
# -

df_train = df_train.drop(['cast'],axis=1)
df_test = df_test.drop(['cast'],axis=1)

# +
list_of_crew = list(df_train['crew'].apply(lambda x: [i['name'] for i in x] if x != None else []).values)
crew_all = ','.join([i for j in list_of_crew for i in j])
list_crew_all = crew_all.split(',')
list_top15_crew = collections.Counter(list_crew_all).most_common(15)
df_train['all_crew'] = df_train['crew'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != None else '')
df_test['all_crew'] = df_test['crew'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != None else '')

for crew in list_top15_crew:
    df_train['crew_' + crew[0]] = df_train['all_crew'].apply(lambda x: 1 if crew[0] in x else 0)
    df_test['crew_' + crew[0]] = df_test['all_crew'].apply(lambda x: 1 if crew[0] in x else 0)
df_train = df_train.drop(['all_crew'],axis=1)
df_test = df_test.drop(['all_crew'],axis=1)
# -

df_train = df_train.drop(['crew'],axis=1)
df_test = df_test.drop(['crew'],axis=1)

le = LabelEncoder()
le.fit(list(df_train['collection_name'].fillna('')) + list(df_test['collection_name'].fillna('')))
df_train['collection_name'] = le.transform(df_train['collection_name'].fillna('').astype(str))
df_test['collection_name'] = le.transform(df_test['collection_name'].fillna('').astype(str))

df_train['log_revenue'] = np.log1p(df_train['revenue'])

df_train['log_budget'] = np.log1p(df_train['budget'])
df_test['log_budget'] = np.log1p(df_test['budget'])

X = df_train.drop(['id','revenue','log_revenue'],axis=1)
y = df_train['log_revenue']
X_test = df_test.drop(['id'],axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3)

n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=12)

# +
lgb_params = {'num_leaves': 30,
         'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': 5,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2,
         "verbosity": -1}
scores = []
prediction = np.zeros(X_test.shape[0])

lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators = 20000, nthread = 4, n_jobs = -1)
lgb_model.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
            verbose=1000, early_stopping_rounds=200)
            
y_pred_valid = lgb_model.predict(X_valid)
y_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration_)
# -

sub = pd.read_csv('/Users/zakopuro/Code/python_code/kaggle_TMDB/input/sample_submission.csv')
sub['revenue'] = np.expm1(y_pred)
sub.to_csv("output/lgb2.csv", index=False)


