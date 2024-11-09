#!/usr/bin/env python
# coding: utf-8

# In[1]:

from time import time

from tools import OUT_PATH, open_meta_df
import pandas as pd
import numpy as np
from skopt import BayesSearchCV
from sklearn.base import clone

from tqdm.auto import tqdm
from skopt.space import Integer, Categorical, Real
from pprint import pprint
from warnings import filterwarnings
from joblib import dump
from KMeansInterpolateCustomEstimator import (KMeansInterpolateCustomEstimator, MAX_INTERPOLATION_RATIO,
                                              MIN_INTERPOLATION_RATIO)
from functools import reduce
from operator import mul
from loguru import logger
filterwarnings("ignore")


CUDA = False

if CUDA:
    from cuml.ensemble import RandomForestClassifier

    additional_optimize_params = {
        'split_criterion': Categorical([0, 1]),
        'max_leaves': Categorical([-1, 1, 2, 3, 4, 5, 6, 7, 10, 100]),
        'n_bins': Integer(100, 1000),
        'output_type': Categorical(['numpy']),
        'feature_selector_model': Categorical(['chi2', 'f_classif', 'mutual_info_classif']),
    }
    estimator_keys_0 = [
        'split_criterion',
        'max_leaves',
        'n_bins',
        'output_type',
    ]
else:
    from sklearn.ensemble import RandomForestClassifier

    additional_optimize_params = {
        'criterion': Categorical(['gini', 'entropy', 'log_loss']),
        'max_leaf_nodes': Categorical([None, 2, 3, 4, 5, 6, 7, 10, 20 , 30, 40, 50, 60 , 70, 100]),
        'n_jobs': Categorical([-1]),
        'feature_selector_model': Categorical(['chi2', 'f_classif', 'mutual_info_classif', 'bypass']),
    }
    estimator_keys_0 = [
        'criterion',
        'max_leaf_nodes',
        'n_jobs',
    ]


# In[2]:


model_name = 'RF_Interpolate'
logger.add(f"model_{model_name}_{{time}}.log")


# In[3]:


meta_df = open_meta_df()
video_id = np.load(OUT_PATH / 'video_id.npy')
landmarks = np.load(OUT_PATH / 'landmarks.npy')
train_idx = np.load(OUT_PATH / 'train_idx.npy')
test_idx = np.load(OUT_PATH / 'test_idx.npy')
meta_df.head()


# In[4]:


logger.info("Iniciado com os landmarks tendo o formato {}.", landmarks.shape)


# In[5]:


observations = reduce(mul, landmarks.shape[1:])

stacked_train_landmarks = [landmarks[video_id == i].reshape((-1, observations)) for i in train_idx]
stacked_test_landmarks = [landmarks[video_id == i].reshape((-1, observations)) for i in test_idx]

classes = meta_df['pose_id'].values

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
    'n_estimators',
    'max_depth',
    'min_samples_split',
    'min_samples_leaf',
    'bootstrap',
    'max_features',
] + estimator_keys_0)

opt = BayesSearchCV(
    KMeansInterpolateCustomEstimator(
        interpolation_ratio=MIN_INTERPOLATION_RATIO,
        interpolation_kind='linear',
        estimator=RandomForestClassifier,
        two_dimensions=False,
        kmeans_keys=kmeans_keys,
        estimator_keys=estimator_keys,
        n_clusters=8,
        n_estimators=100,
        max_depth=100,
        min_samples_split=1,
        min_samples_leaf=1,
        bootstrap=True,
        max_features=1.0,
        split_criterion='gini',
        max_leaves=-1,
        n_bins=128,
        output_type='numpy',
        criterion='gini',
        max_leaf_nodes=None,
        n_jobs=-1,
        cuda=CUDA,
        feature_selector_model='chi2',
        features_selected=10,
    ),
    {
        'interpolation_ratio': Real(MIN_INTERPOLATION_RATIO, 5),
        'interpolation_kind': Categorical(['zero', 'slinear', 'quadratic', 'cubic', 'linear', 'nearest',
                                           'previous', 'next']),
        'estimator': Categorical([RandomForestClassifier]),
        'two_dimensions': Categorical([False]),
        'kmeans_keys': Categorical([kmeans_keys]),
        'n_clusters': Integer(1, 100),
        'estimator_keys': Categorical([estimator_keys]),
        'n_estimators': Integer(100, 20_000),
        'max_depth': Integer(10, 100_000),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 20),
        'bootstrap': Categorical([True, False]),
        'max_features': Real(.01, 1.0),
        'max_samples': Real(.1, 1.),
        'cuda': Categorical([CUDA]),
        'features_selected': Integer(5, 99),
        **additional_optimize_params,
    },
    n_iter=100,
    random_state=42,
    cv=3,
    n_jobs=None,
    verbose=1,
)
logger.catch(reraise=True, message='Erro ao dar fit no otimizador')(opt.fit)(stacked_train_landmarks, y_train)
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
            end_train_score-start_train_score)



# In[13]:


dump(opt.cv_results_, OUT_PATH / f'scores/{model_name}_scores.h5', compress=9)
dump(estimator, OUT_PATH / f'Models/{model_name}.h5', compress=9)


