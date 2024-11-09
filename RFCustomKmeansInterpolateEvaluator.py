#!/usr/bin/env python
# coding: utf-8

# In[1]:
from datetime import datetime
from time import time

from KMeansInterpolateCustomEstimator import KMeansInterpolateCustomEstimator
from tools import OUT_PATH, open_meta_df
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from warnings import filterwarnings
from joblib import dump
from KMeansCustomEstimator import KMeansCustomEstimator
from functools import reduce
from operator import mul
import sys
from BufferToLogger import BufferToLogger
from loguru import logger

# from RNCustomKmeans import create_mlp_model

filterwarnings("ignore")



CUDA = False

if CUDA:
    from cuml.ensemble import RandomForestClassifier

    estimator_keys_0 = [
        'split_criterion',
        'max_leaves',
        'n_bins',
        'output_type',
    ]
else:
    from sklearn.ensemble import RandomForestClassifier

    estimator_keys_0 = [
        'criterion',
        'max_leaf_nodes',
        'n_jobs',
    ]

real_stdout = sys.stdout
real_stderr = sys.stderr

BASE_REGEX = r'^(?:Fitting|Epoch |\d+/\d+ - \d+\w+ - loss:)'

sys.stdout = BufferToLogger('INFO', BASE_REGEX, real_stdout)
sys.stderr = BufferToLogger('INFO', BASE_REGEX, real_stderr)

if __name__ == '__main__':
    model_name = 'RF_Interpolate_Eval'
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
                                   'n_estimators',
                                   'max_depth',
                                   'min_samples_split',
                                   'min_samples_leaf',
                                   'bootstrap',
                                   'max_features',
                               ] + estimator_keys_0)

    # In[9]:

    estimator = KMeansInterpolateCustomEstimator(
        **{**dict(interpolation_ratio=1,
        estimator=RandomForestClassifier,
        two_dimensions=False,
        kmeans_keys=kmeans_keys,
        estimator_keys=estimator_keys,
        n_jobs=-1,
        cuda=CUDA),
           **dict([
               ('bootstrap', False),
               ('criterion', 'entropy'),
               ('feature_selector_model', 'bypass'),
               ('features_selected', 56),
               ('interpolation_kind', 'next'),
               ('interpolation_ratio', 3.3468721432271575),
               ('max_depth', 40836),
               ('max_features', 0.01),
               ('max_leaf_nodes', 100),
               ('max_samples', 0.9879238804427334),
               ('min_samples_leaf', 1),
               ('min_samples_split', 2),
               ('n_clusters', 58),
               ('n_estimators', 20000)])},
    )

    # In[10]:

    estimator.fit(stacked_train_landmarks, y_train)

    start_train_score = time()

    predict_train = estimator.predict(stacked_train_landmarks)
    predict_test = estimator.predict(stacked_test_landmarks)

    end_train_score = time()

    train_score = accuracy_score(y_train, predict_train)
    test_score = accuracy_score(y_test, predict_test)

    # In[11]:

    delta = end_train_score - start_train_score

    logger.info('Acurácia de treino: {}\nAcurácia de teste: {}\nTempo para avaliar treino e teste: {}, média de {}'
                ' s/vídeos',
                train_score, test_score,
                delta, delta / (len(stacked_train_landmarks) + len(stacked_test_landmarks)))

    # In[13]:
    predictions_dir = OUT_PATH / f'Predictions/{model_name}'

    predictions_dir.mkdir(exist_ok=True)

    dump(predict_train, predictions_dir / 'trains.h5')
    dump(predict_test, predictions_dir / 'tests.h5')

    dump(estimator, OUT_PATH / f'Models/{model_name}.h5')
