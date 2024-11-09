#!/usr/bin/env python
# coding: utf-8
import pdb
import sys
# In[1]:

from time import time

from sklearn.utils import shuffle

from BufferToLogger import BufferToLogger
from tools import OUT_PATH, open_meta_df, USE_POINTS
import pandas as pd
import numpy as np
from skopt import BayesSearchCV
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from skopt.space import Integer, Categorical, Real
from pprint import pprint
from warnings import filterwarnings
from joblib import dump
from KMeansCustomEstimator import KMeansCustomEstimator
import xgboost as xgb
from functools import reduce
from operator import mul
from loguru import logger
from SyntheticGen import SyntheticGen
from collections import Counter

filterwarnings("ignore")

# In[2]:


model_name = 'XGB'
logger.add(f"model_{model_name}_{{time}}.log")


real_stdout = sys.stdout
real_stderr = sys.stderr

BASE_REGEX = r'^(?:Fitting|Epoch |\d+/\d+ - \d+\w+ - loss:)'

sys.stdout = BufferToLogger('INFO', BASE_REGEX, real_stdout)
sys.stderr = BufferToLogger('INFO', BASE_REGEX, real_stderr)


class CustomXGB(xgb.XGBClassifier):
    def __init__(self, device='gpu',
                 n_estimators=10,
                 max_depth=1,
                 learning_rate=0.01,
                 min_child_weight=1,
                 subsample=1,
                 colsample_bytree=1,
                 booster='gbtree',
                 objective='binary:logistic',
                 eval_metric='mgloss',
                 early_stopping_rounds=1,
                 xg_n_neighbors=5, proportion_gen=1.):
        super().__init__(device=device,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         min_child_weight=min_child_weight,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         booster=booster,
                         objective=objective,
                         eval_metric=eval_metric,
                         early_stopping_rounds=early_stopping_rounds,
                         xg_n_neighbors=xg_n_neighbors,
                         proportion_gen=proportion_gen)
        self.gen = SyntheticGen(xg_n_neighbors)
        self.proportion_gen = proportion_gen
        self.xg_n_neighbors = xg_n_neighbors

    def fit(self, X, y):
        X, y = self.gen.fit(X, y).generate(X, y, int(X.shape[0] * self.proportion_gen))
        for train, test in tqdm(list(StratifiedKFold(n_splits=3, random_state=42, shuffle=True).split(X, y)),
                                desc='[pipXGB] K-Fold'):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            super().fit(X_train, y_train, eval_set=[(X_test, y_test)])

    def get_params(self, deep=True):
        return {
            'device': self.device,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'booster': self.booster,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'early_stopping_rounds': self.early_stopping_rounds,
            'xg_n_neighbors': self.xg_n_neighbors,
            'proportion_gen': self.proportion_gen,
        }

    def set_params(self, **parameters):
        update_gen = False
        if 'proportion_gen' in parameters:
            self.proportion_gen = parameters.pop('proportion_gen')

        if 'xg_n_neighbors' in parameters:
            self.xg_n_neighbors = parameters.pop('xg_n_neighbors')
            update_gen = True

        if update_gen:
            self.gen = SyntheticGen(self.xg_n_neighbors)

        super().set_params(**parameters)
        return self


# In[3]:


meta_df = open_meta_df()
video_id = np.load(OUT_PATH / 'video_id.npy')
landmarks = np.load(OUT_PATH / 'landmarks.npy')[:, USE_POINTS, :]
train_idx = np.load(OUT_PATH / 'train_idx.npy')
test_idx = np.load(OUT_PATH / 'test_idx.npy')
meta_df.head()

# In[4]:


logger.info("Iniciado com os landmarks tendo o formato {}.", landmarks.shape)

# In[5]:


observations = reduce(mul, landmarks.shape[1:])

stacked_train_landmarks = [landmarks[video_id == i].reshape((-1, observations)) for i in train_idx]
stacked_test_landmarks = [landmarks[video_id == i].reshape((-1, observations)) for i in test_idx]

classes = meta_df['pose_id'].values - 1

y_train = classes[train_idx]
y_test = classes[test_idx]
logger.info("Alterado para ter {} observações.", observations)

# In[6]:


logger.debug("Checagem de integridade {} + {} = {} = {}", len(y_train), len(y_test), len(y_train) + len(y_test),
             len(classes))

# In[7]:


kmeans_keys = frozenset(['n_clusters'])

shortest_video = min([len(x) for x in stacked_train_landmarks + stacked_test_landmarks])
logger.debug("Menor vídeo tem {} frames", shortest_video)

estimator_keys = frozenset([
    "device",
    'n_estimators',
    'max_depth',
    'learning_rate',
    'min_child_weight',
    'subsample',
    'colsample_bytree',
    'booster',
    'objective',
    'eval_metric',
    'early_stopping_rounds',
    'xg_n_neighbors',
    'proportion_gen',
])

opt = BayesSearchCV(
    KMeansCustomEstimator(
        CustomXGB,
        two_dimensions=False,
        device='cuda',
        kmeans_keys=kmeans_keys,
        estimator_keys=estimator_keys,
        n_clusters=8,
        n_estimators=50,
        max_depth=3,
        learning_rate=0.01,
        min_child_weight=1.,
        subsample=.5,
        colsample_bytree=.5,
        booster='gbtree',
        objective='multi:softmax',
        eval_metric='mlogloss',
        tree_method='auto',
        early_stopping_rounds=10,
        xg_n_neighbors=5,
        proportion_gen=1.,
    ),
    {
        'estimator': Categorical([CustomXGB]),
        'two_dimensions': Categorical([False]),
        'kmeans_keys': Categorical([kmeans_keys]),
        'estimator_keys': Categorical([estimator_keys]),
        'n_clusters': Integer(1, shortest_video - 1),
        'device': Categorical(['cuda']),
        'n_estimators': Integer(50, 500),
        'max_depth': Integer(2, 10),
        'learning_rate': Real(0.01, 0.3),
        'min_child_weight': Real(0.01, 10.),
        'subsample': Real(0.1, 1.),
        'colsample_bytree': Real(0.1, 1.),
        'booster': Categorical(['gbtree', 'gblinear', 'dart']),
        'objective': Categorical(['multi:softmax', 'multi:softprob']),
        'eval_metric': Categorical(['mlogloss', 'merror']),
        'early_stopping_rounds': Integer(1, 50),
        'xg_n_neighbors': Integer(2, 10),
        'proportion_gen': Real(.1, 5.),
    },
    n_iter=100,
    random_state=42,
    cv=3,
    n_jobs=None,
    verbose=1,
)
opt.fit(stacked_train_landmarks, y_train)
logger.info("Melhor score de validação {} com os parâmetros {}.", opt.best_score_, opt.best_params_)

# In[9]:


estimator = clone(opt.best_estimator_)

# In[10]:

start_train_score = time()

estimator.fit(stacked_train_landmarks, y_train)

train_score = estimator.score(stacked_train_landmarks, y_train)
test_score = estimator.score(stacked_test_landmarks, y_test)

end_train_score = time()

# In[11]:


logger.info('Acurácia de treino: {}\nAcurácia de teste: {}\nTempo para treino e teste: {}', train_score, test_score,
            end_train_score - start_train_score)

# In[13]:


dump(opt.cv_results_, OUT_PATH / f'scores/{model_name}_scores.h5', compress=9)
dump(estimator, OUT_PATH / f'Models/{model_name}.h5', compress=9)
