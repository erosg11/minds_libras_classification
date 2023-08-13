#!/usr/bin/env python
# coding: utf-8

# In[1]:
from datetime import datetime
from time import time

from tools import OUT_PATH, open_meta_df
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from warnings import filterwarnings
from joblib import dump
from KMeansCustomEstimator import KMeansCustomEstimator
from functools import reduce
from operator import mul
import sys
from BufferToLogger import BufferToLogger
from loguru import logger

from RNCustomKmeans import create_mlp_model

filterwarnings("ignore")

real_stdout = sys.stdout
real_stderr = sys.stderr

BASE_REGEX = r'^(?:Fitting|Epoch |\d+/\d+ - \d+\w+ - loss:)'

sys.stdout = BufferToLogger('INFO', BASE_REGEX, real_stdout)
sys.stderr = BufferToLogger('INFO', BASE_REGEX, real_stderr)

if __name__ == '__main__':
    model_name = 'RF_Eval'
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

    features = reduce(mul, landmarks.shape[1:])

    stacked_train_landmarks = [landmarks[video_id == i].reshape((-1, features)) for i in train_idx]
    stacked_test_landmarks = [landmarks[video_id == i].reshape((-1, features)) for i in test_idx]

    classes = meta_df['pose_id'].values - 1  # type: np.ndarray

    y_train = classes[train_idx]
    y_test = classes[test_idx]
    logger.info("Alterado para ter {} observações.", features)

    # In[6]:

    logger.debug("Checagem de integridade {} + {} = {} = {}", len(y_train), len(y_test), len(y_train) + len(y_test),
                 len(classes))

    # In[7]:

    kmeans_keys = frozenset(['n_clusters'])

    estimator_keys = frozenset([
        'bootstrap',
        'max_depth',
        'max_features',
        'min_samples_leaf',
        'min_samples_split',
        'n_estimators',
        "n_jobs",
    ])

    # In[9]:

    estimator = KMeansCustomEstimator(
        RandomForestClassifier,
        two_dimensions=False,
        kmeans_keys=kmeans_keys,
        estimator_keys=estimator_keys,
        n_clusters=70,
        weights='uniform',
        bootstrap=False,
        max_depth=None,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=18785,
        n_jobs=-1,
    )  # type: KMeansCustomEstimator

    # In[10]:

    estimator.fit(stacked_train_landmarks, y_train)

    start_train_score = time()

    train_score = estimator.score(stacked_train_landmarks, y_train)
    test_score = estimator.score(stacked_test_landmarks, y_test)

    end_train_score = time()

    # In[11]:

    delta = end_train_score - start_train_score

    logger.info('Acurácia de treino: {}\nAcurácia de teste: {}\nTempo para avaliar treino e teste: {}, média de {}'
                ' vídeos/s',
                train_score, test_score,
                delta, delta / (len(stacked_train_landmarks) + len(stacked_test_landmarks)))

    # In[13]:
    dump(estimator, OUT_PATH / f'Models/{model_name}.h5', compress=9)
