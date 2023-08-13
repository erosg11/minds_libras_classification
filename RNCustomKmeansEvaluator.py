#!/usr/bin/env python
# coding: utf-8

# In[1]:
from datetime import datetime
from time import time

from tools import OUT_PATH, open_meta_df
import numpy as np
from KerasCustomModel import KerasCustomModel
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

BATCH_SIZE = 128
EPOCHS = 1000
MULTI_GPU = True

if __name__ == '__main__':
    model_name = 'RN_Eval'
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
        'n_clusters',
        'model_generator',
        'batch_size',
        'epochs',
        'early_stopping',
        'early_stopping_min_delta',
        'early_stopping_patience',
        'reduce_lro_plateau',
        'reduce_lro_plateau_monitor',
        'reduce_lro_plateau_factor',
        'reduce_lro_plateau_patience',
        'reduce_lro_plateau_min_delta',
        'reduce_lro_plateau_cooldown',
        'reduce_lro_plateau_min_lr',
        'verbose',
        'units',
        'hidden_layers',
        'activation',
        # 'negative_slope_relu',
        # 'max_value_relu',
        # 'threshold_relu',
        'out_activation',
        # 'elu_alpha',
        # 'leaky_relu_alpha',
        'n_classes',
        'batch_normalization',
        'optimizer',
        'adam_learning_rate',
        'adam_beta_1',
        'adam_beta_2',
        'adam_epsilon',
        'adam_amsgrad',
        'sgd_learning_rate',
        'sgd_momentum',
        'sgd_nesterov',
        'rmsprop_learning_rate',
        'rmsprop_rho',
        'rmsprop_epsilon',
        'rmsprop_centered',
    ])

    # In[9]:

    out_checkpoints_folder = OUT_PATH / f'{model_name}.{datetime.now():%Y%m%d%H%M%S}'
    out_checkpoints_folder.mkdir(exist_ok=True, parents=True)

    estimator = KMeansCustomEstimator(
        **{'estimator': KerasCustomModel,
           'two_dimensions': False,
           'kmeans_keys': frozenset({'n_clusters'}),
           'estimator_keys': frozenset(
               {'sgd_nesterov', 'model_generator', 'batch_normalization', 'units', 'reduce_lro_plateau_min_lr',
                'early_stopping', 'reduce_lro_plateau_factor', 'reduce_lro_plateau_cooldown', 'rmsprop_epsilon',
                'rmsprop_rho', 'reduce_lro_plateau_min_delta', 'verbose', 'rmsprop_learning_rate',
                'early_stopping_patience', 'reduce_lro_plateau', 'sgd_momentum', 'adam_learning_rate',
                'reduce_lro_plateau_patience', 'adam_beta_1', 'adam_amsgrad', 'activation',
                'reduce_lro_plateau_monitor',
                'batch_size', 'n_clusters', 'epochs', 'out_activation', 'hidden_layers', 'optimizer',
                'sgd_learning_rate',
                'adam_beta_2', 'rmsprop_centered', 'early_stopping_min_delta', 'adam_epsilon', 'n_classes',
                'tensorboard', 'tensorboard_log_dir_format',
                'model_checkpoint', 'model_checkpoint_format', 'data_features'}),
           'fit_estimator_keys': None,
           'n_clusters': 70,
           'model_generator': create_mlp_model,
           'batch_size': 31,
           'epochs': 1000,
           'early_stopping': False,
           'early_stopping_min_delta': 59.363370150935054,
           'early_stopping_patience': 75,
           'reduce_lro_plateau': True,
           'reduce_lro_plateau_monitor': 'val_loss',
           'reduce_lro_plateau_factor': 0.5577047493819142,
           'reduce_lro_plateau_patience': 0,
           'reduce_lro_plateau_min_delta': 40.9087392652479,
           'reduce_lro_plateau_cooldown': 73,
           'reduce_lro_plateau_min_lr': 0.9211766935186267,
           'verbose': 2,
           'units': 855,
           'hidden_layers': 0,
           'activation': 'relu',
           'out_activation': 'softmax',
           'n_classes': 20,
           'batch_normalization': False,
           'optimizer': 'adam',
           'adam_learning_rate': 0.04365510082176306,
           'adam_beta_1': 0.9085584803515259,
           'adam_beta_2': 0.7064117693708993,
           'adam_epsilon': 0.015640708235555995,
           'adam_amsgrad': True,
           'sgd_learning_rate': 0.2478239233120042,
           'sgd_momentum': 0.5993324145775742,
           'sgd_nesterov': False,
           'rmsprop_learning_rate': 0.012571443288409631,
           'rmsprop_rho': 0.5460736798660093,
           'rmsprop_epsilon': 0.11707484392445998,
           'rmsprop_centered': True,

           # Acompanhar o treinamento, não mexer
           'tensorboard': True,
           'tensorboard_log_dir_format': f'logs.{model_name}.{{datetime}}/',
           'model_checkpoint': True,
           'model_checkpoint_format': f'{out_checkpoints_folder}/weights.{{epoch:02d}}-{{val_loss:.2f}}.hdf5',
           'keras_tqdm': True,
           'data_features': features,
           }
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
