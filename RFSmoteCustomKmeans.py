#!/usr/bin/env python
# coding: utf-8

# In[1]:

from time import time

from tools import OUT_PATH, open_meta_df, USE_POINTS
import pandas as pd
import numpy as np
from skopt import BayesSearchCV
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm
from skopt.space import Integer, Categorical, Real
from pprint import pprint
from warnings import filterwarnings
from joblib import dump
from KMeansCustomEstimator import KMeansCustomEstimator
from functools import reduce
from operator import mul
from loguru import logger
from SyntheticGen import SyntheticGen

filterwarnings("ignore")

# In[2]:


model_name = 'RFV3'
logger.add(f"model_{model_name}_{{time}}.log")


class CustomRF(RandomForestClassifier):
    def __init__(self, proportion_gen=1, rf_n_neighbors=5, n_estimators=100,
                 *,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features="sqrt",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None,
                 ):
        super().__init__(n_estimators=n_estimators,
                         criterion=criterion,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes,
                         min_impurity_decrease=min_impurity_decrease,
                         bootstrap=bootstrap,
                         oob_score=oob_score,
                         n_jobs=n_jobs,
                         random_state=random_state,
                         verbose=verbose,
                         warm_start=warm_start,
                         class_weight=class_weight,
                         ccp_alpha=ccp_alpha,
                         max_samples=max_samples,
                         )
        self.gen = SyntheticGen(rf_n_neighbors)
        self.proportion_gen = proportion_gen
        self.rf_n_neighbors = rf_n_neighbors

    def fit(self, X, y, sample_weight=None):
        X_resampled, y_resampled = self.gen.fit(X, y).generate(X, y, int(X.shape[0] * self.proportion_gen))
        return super().fit(X_resampled, y_resampled, sample_weight)


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
    'bootstrap',
    'max_depth',
    'max_features',
    'min_samples_leaf',
    'min_samples_split',
    'n_estimators',
    "n_jobs",
])

opt = BayesSearchCV(
    KMeansCustomEstimator(
        CustomRF,
        two_dimensions=False,
        kmeans_keys=kmeans_keys,
        estimator_keys=estimator_keys,
        n_clusters=8,
        n_neighbors=5,
        weights='uniform',
        bootstrap=True,
        max_depth=None,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=100,
        n_jobs=-1,
        proportion_gen=1,
        rf_n_neighbors=5,
    ),
    {
        'estimator': Categorical([CustomRF]),
        'two_dimensions': Categorical([False]),
        'kmeans_keys': Categorical([kmeans_keys]),
        'estimator_keys': Categorical([estimator_keys]),
        'n_clusters': Integer(1, shortest_video - 1),
        'bootstrap': Categorical([True, False]),
        'max_depth': Categorical([None]),
        'max_features': Categorical(["sqrt"]),
        'min_samples_leaf': Categorical([1]),
        'min_samples_split': Categorical([2]),
        'n_estimators': Integer(2, 20_000),
        'n_jobs': Categorical([-1]),
        'proportion_gen': Real(.1, 10.),
        'rf_n_neighbors': Integer(2, 30)
    },
    n_iter=200,
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
