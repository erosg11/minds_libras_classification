#!/usr/bin/env python
# coding: utf-8

# In[1]:
from datetime import datetime
from time import time

import pandas as pd

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
from tabulate import tabulate
from tqdm.auto import tqdm

# from RNCustomKmeans import create_mlp_model

filterwarnings("ignore")

real_stdout = sys.stdout
real_stderr = sys.stderr

BASE_REGEX = r'^(?:Fitting|Epoch |\d+/\d+ - \d+\w+ - loss:)'

sys.stdout = BufferToLogger('INFO', BASE_REGEX, real_stdout)
sys.stderr = BufferToLogger('INFO', BASE_REGEX, real_stderr)

if __name__ == '__main__':
    model_name = 'RF_Eval_1MenOut'
    logger.add(f"model_{model_name}_{{time}}.log")


    # In[3]:

    meta_df = open_meta_df()
    video_id = np.load(OUT_PATH / 'video_id.npy')
    landmarks = np.load(OUT_PATH / 'landmarks.npy')
    logger.info("Iniciado com os landmarks tendo o formato {}.", landmarks.shape)
    features = reduce(mul, landmarks.shape[1:])
    classes = meta_df['pose_id'].values - 1  # type: np.ndarray
    logger.info("Alterado para ter {} observações.", features)
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
    table = []
    add_row = table.append
    for sinalizador, sub_df in tqdm(list(meta_df.groupby('sinalizador')), desc='Sinalizadores'):
        logger.info('Excluindo sinalizador {}.', sinalizador)
        train_idx = np.array(list(set(meta_df.index) - set(sub_df.index)))
        test_idx = sub_df.index



        stacked_train_landmarks = [landmarks[video_id == i].reshape((-1, features)) for i in train_idx]
        stacked_test_landmarks = [landmarks[video_id == i].reshape((-1, features)) for i in test_idx]


        y_train = classes[train_idx]
        y_test = classes[test_idx]

        logger.debug("Checagem de integridade {} + {} = {} = {}", len(y_train), len(y_test), len(y_train) + len(y_test),
                     len(classes))

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


        estimator.fit(stacked_train_landmarks, y_train)

        start_train_score = time()

        predict_train = estimator.predict(stacked_train_landmarks)
        predict_test = estimator.predict(stacked_test_landmarks)

        end_train_score = time()

        train_score = accuracy_score(y_train, predict_train)
        test_score = accuracy_score(y_test, predict_test)
        add_row([sinalizador, train_score, test_score])

        delta = end_train_score - start_train_score

        logger.info('[Sinalizador {}] Acurácia de treino: {}\nAcurácia de teste: {}\nTempo para avaliar treino e '
                    'teste: {}, média de {} s/vídeos', sinalizador,
                    train_score, test_score,
                    delta, delta / (len(stacked_train_landmarks) + len(stacked_test_landmarks)))

        predictions_dir = OUT_PATH / f'Predictions/{model_name}/Sinalizador {sinalizador:02}'

        predictions_dir.mkdir(exist_ok=True, parents=True)

        dump(predict_train, predictions_dir / 'trains.h5')
        dump(predict_test, predictions_dir / 'tests.h5')

    print('Resultado final')
    print(tabulate(table, headers=['Sinalizador', 'Train', 'Test'], floatfmt='.2f', tablefmt="grid"))
    df_result = pd.DataFrame(table, columns=['Sinalizador', 'Train', 'Test']).set_index('Sinalizador')
    result_dir = OUT_PATH / f'{model_name}'
    result_dir.mkdir(exist_ok=True, parents=True)
    df_result.to_csv(result_dir / 'results.csv')
    print(df_result.describe())