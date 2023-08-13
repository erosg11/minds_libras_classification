#!/usr/bin/env python
# coding: utf-8

# In[1]:
from datetime import datetime
from time import time

from tools import OUT_PATH, open_meta_df
from pathlib import Path
import pandas as pd
import numpy as np
from skopt import BayesSearchCV
from sklearn.base import clone
from KerasCustomModel import KerasCustomModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Softmax, Activation, ELU, LeakyReLU, Input, Dropout
from tensorflow.keras.activations import selu, sigmoid
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.backend import clear_session
from tqdm.auto import tqdm
from skopt.space import Integer, Categorical, Real
from pprint import pprint
from warnings import filterwarnings
from joblib import dump
from KMeansCustomEstimator import KMeansCustomEstimator
from functools import reduce
from operator import mul
import sys
from BufferToLogger import BufferToLogger
from loguru import logger

filterwarnings("ignore")

real_stdout = sys.stdout
real_stderr = sys.stderr

BASE_REGEX = r'^(?:Fitting|Epoch |\d+/\d+ - \d+\w+ - loss:)'

sys.stdout = BufferToLogger('INFO', BASE_REGEX, real_stdout)
sys.stderr = BufferToLogger('INFO', BASE_REGEX, real_stderr)

BATCH_SIZE = 128
EPOCHS = 5000
MULTI_GPU = True


def create_mlp_model(
        units=64,
        hidden_layers=5,
        n_clusters=5,
        activation='relu',
        negative_slope_relu=0.,
        max_value_relu=None,
        threshold_relu=0.,
        out_activation='sigmoid',
        elu_alpha=1,
        leaky_relu_alpha=0.3,
        n_classes=10,
        batch_normalization=False,
        optimizer='adam',
        adam_learning_rate=0.001,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        adam_epsilon=1e-07,
        adam_amsgrad=False,
        adam_weight_decay=None,
        adam_clipnorm=None,
        adam_clipvalue=None,
        adam_global_clipnorm=None,
        adam_use_ema=False,
        adam_ema_momentum=0.99,
        adam_ema_overwrite_frequency=None,
        sgd_learning_rate=0.01,
        sgd_momentum=0.0,
        sgd_nesterov=False,
        sgd_weight_decay=None,
        sgd_clipnorm=None,
        sgd_clipvalue=None,
        sgd_global_clipnorm=None,
        sgd_use_ema=False,
        sgd_ema_momentum=0.99,
        sgd_ema_overwrite_frequency=None,
        rmsprop_learning_rate=0.001,
        rmsprop_rho=0.9,
        rmsprop_momentum=0.0,
        rmsprop_epsilon=1e-07,
        rmsprop_centered=False,
        rmsprop_weight_decay=None,
        rmsprop_clipnorm=None,
        rmsprop_clipvalue=None,
        rmsprop_global_clipnorm=None,
        rmsprop_use_ema=False,
        rmsprop_ema_momentum=0.99,
        rmsprop_ema_overwrite_frequency=100,
        data_features=0,
        dropout=0.0,
):
    clear_session()
    if MULTI_GPU:
        try:
            from tensorflow.config import list_physical_devices
            if len(list_physical_devices('GPU')) > 1:
                import tensorflow as tf
                strategy = tf.distribute.MirroredStrategy()
                logger.info('Trabalhando com {} dispositivos!', strategy.num_replicas_in_sync)
                context = strategy.scope
            else:
                from contextlib import nullcontext
                context = nullcontext
        except:
            logger.exception('Erro ao tentar iniciar contexto multigpu')
            from contextlib import nullcontext
            context = nullcontext
    else:
        from contextlib import nullcontext
        context = nullcontext
    with context():
        if activation == 'relu':
            activation_layer = ReLU(max_value_relu, negative_slope_relu, threshold_relu)
        elif activation == 'elu':
            activation_layer = ELU(elu_alpha)
        elif activation == 'leaky_relu':
            activation_layer = LeakyReLU(leaky_relu_alpha)
        elif activation == 'selu':
            activation_layer = Activation(selu)
        else:
            raise KeyError(f"Ativação {activation} inválida!")

        if out_activation == 'sigmoid':
            out_activation_layer = Activation(sigmoid)
        elif out_activation == 'softmax':
            out_activation_layer = Softmax()
        else:
            raise KeyError(f"Ativação de saída {out_activation} inválida!")

        if optimizer == 'adam':
            optimizer_model = Adam(
                learning_rate=adam_learning_rate,
                beta_1=adam_beta_1,
                beta_2=adam_beta_2,
                epsilon=adam_epsilon,
                amsgrad=adam_amsgrad,
                weight_decay=adam_weight_decay,
                clipnorm=adam_clipnorm,
                clipvalue=adam_clipvalue,
                global_clipnorm=adam_global_clipnorm,
                use_ema=adam_use_ema,
                ema_momentum=adam_ema_momentum,
                ema_overwrite_frequency=adam_ema_overwrite_frequency,
            )
        elif optimizer == 'sgd':
            optimizer_model = SGD(
                learning_rate=sgd_learning_rate,
                momentum=sgd_momentum,
                nesterov=sgd_nesterov,
                weight_decay=sgd_weight_decay,
                clipnorm=sgd_clipnorm,
                clipvalue=sgd_clipvalue,
                global_clipnorm=sgd_global_clipnorm,
                use_ema=sgd_use_ema,
                ema_momentum=sgd_ema_momentum,
                ema_overwrite_frequency=sgd_ema_overwrite_frequency,
            )
        elif optimizer == 'rmsprop':
            optimizer_model = RMSprop(
                learning_rate=rmsprop_learning_rate,
                rho=rmsprop_rho,
                momentum=rmsprop_momentum,
                epsilon=rmsprop_epsilon,
                centered=rmsprop_centered,
                weight_decay=rmsprop_weight_decay,
                clipnorm=rmsprop_clipnorm,
                clipvalue=rmsprop_clipvalue,
                global_clipnorm=rmsprop_global_clipnorm,
                use_ema=rmsprop_use_ema,
                ema_momentum=rmsprop_ema_momentum,
                ema_overwrite_frequency=rmsprop_ema_overwrite_frequency,
            )
        else:
            raise KeyError(f"Otimizador {optimizer} inválido!")

        if batch_normalization:
            bn = lambda x_: BatchNormalization()(x_)
        else:
            bn = lambda x_: x_

        inputs = Input(shape=(n_clusters * data_features,))
        x = inputs
        for _ in range(hidden_layers):
            x = Dense(units=units, activation=activation_layer)(x)
            x = Dropout(rate=dropout)(x)
            x = bn(x)
        outputs = Dense(n_classes, activation=out_activation_layer)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer_model, loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model


@logger.catch(message='Erro ao dar fit no modelo', onerror=lambda _: sys.exit(1))
def fit_model():
    n_classes = classes.max(initial=0) + 1
    opt = BayesSearchCV(
        KMeansCustomEstimator(
            KerasCustomModel,
            two_dimensions=False,
            kmeans_keys=kmeans_keys,
            estimator_keys=estimator_keys,
            n_clusters=8,
            model_generator=create_mlp_model,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            early_stopping=True,
            early_stopping_min_delta=0,
            early_stopping_patience=0,
            reduce_lro_plateau=True,
            reduce_lro_plateau_monitor="val_loss",
            reduce_lro_plateau_factor=0.1,
            reduce_lro_plateau_patience=10,
            reduce_lro_plateau_min_delta=0.0001,
            reduce_lro_plateau_cooldown=0,
            reduce_lro_plateau_min_lr=0,
            verbose=2,
            units=64,
            hidden_layers=5,
            activation='relu',
            # negative_slope_relu=0.,
            # max_value_relu=None,
            # threshold_relu=0.,
            out_activation='sigmoid',
            # elu_alpha=1,
            # leaky_relu_alpha=0.3,
            n_classes=n_classes,
            batch_normalization=False,
            optimizer='adam',
            adam_learning_rate=0.001,
            adam_beta_1=0.9,
            adam_beta_2=0.999,
            adam_epsilon=1e-07,
            adam_amsgrad=False,
            sgd_learning_rate=0.01,
            sgd_momentum=0.0,
            sgd_nesterov=False,
            rmsprop_learning_rate=0.001,
            rmsprop_rho=0.9,
            rmsprop_epsilon=1e-07,
            rmsprop_centered=False,
            dropout=0.0,
        ),
        {
            'estimator': Categorical([KerasCustomModel]),
            'two_dimensions': Categorical([False]),
            'kmeans_keys': Categorical([kmeans_keys]),
            'estimator_keys': Categorical([estimator_keys]),
            'n_clusters': Integer(1, shortest_video - 1),
            'model_generator': Categorical([create_mlp_model]),
            'batch_size': Integer(5, BATCH_SIZE),
            'epochs': Integer(5, EPOCHS),
            'early_stopping': Categorical([True, False]),
            'early_stopping_min_delta': Real(0, 100),
            'early_stopping_patience': Integer(0, 100),
            'reduce_lro_plateau': Categorical([True, False]),
            'reduce_lro_plateau_monitor': Categorical(['val_loss']),
            'reduce_lro_plateau_factor': Real(1e-10, 1 - 1e-10),
            'reduce_lro_plateau_patience': Integer(0, 100),
            'reduce_lro_plateau_min_delta': Real(0, 100),
            'reduce_lro_plateau_cooldown': Integer(0, 100),
            'reduce_lro_plateau_min_lr': Real(0, 1 - 1e-10),
            'verbose': Categorical([2]),
            'units': Integer(1, 7200),
            'hidden_layers': Integer(0, 20),
            'activation': Categorical(['relu', 'elu', 'selu', 'leaky_relu']),
            # 'negative_slope_relu': Real(0, 1e3),
            # 'max_value_relu': Real(0, 1e3),
            # 'threshold_relu': Real(0, 1e3),
            'out_activation': Categorical(['sigmoid', 'softmax']),
            # 'elu_alpha': Real(-1e3, 1e3),
            # 'leaky_relu_alpha': Real(0, 1e3),
            'n_classes': Categorical([n_classes]),
            'batch_normalization': Categorical([True, False]),
            'optimizer': Categorical(['adam', 'sgd', 'rmsprop']),
            'adam_learning_rate': Real(1e-15, 1e-1),
            'adam_beta_1': Real(.5, 1 - 1e-15),
            'adam_beta_2': Real(.5, 1 - 1e-15),
            'adam_epsilon': Real(1e-15, 0.3),
            'adam_amsgrad': Categorical([True, False]),
            'sgd_learning_rate': Real(1e-15, 0.3),
            'sgd_momentum': Real(0, 1),
            'sgd_nesterov': Categorical([True, False]),
            'rmsprop_learning_rate': Real(1e-15, .3),
            'rmsprop_rho': Real(.5, 1 - 1e-15),
            'rmsprop_epsilon': Real(1e-15, 0.3),
            'rmsprop_centered': Categorical([True, False]),
            'data_features': Categorical([features]),
            'dropout': Real(0, 0.5),
        },
        n_iter=150,
        random_state=42,
        cv=3,
        n_jobs=None,
        verbose=1,
    )
    opt.fit(stacked_train_landmarks, y_train)
    return opt


# In[2]:

if __name__ == '__main__':

    model_name = 'RN'
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

    shortest_video = min([len(x) for x in stacked_train_landmarks + stacked_test_landmarks])
    logger.debug("Menor vídeo tem {} frames", shortest_video)

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
        'data_features'
    ])

    opt = fit_model()

    logger.info("Melhor score de validação {} com os parâmetros {}.", opt.best_score_, opt.best_params_)

    # In[9]:

    out_checkpoints_folder = OUT_PATH / f'{model_name}.{datetime.now():%Y%m%d%H%M%S}'
    out_checkpoints_folder.mkdir(exist_ok=True, parents=True)

    estimator = clone(opt.best_estimator_)  # type: KMeansCustomEstimator
    estimator.set_params(estimator_keys=estimator_keys | {'tensorboard', 'tensorboard_log_dir_format',
                                                          'model_checkpoint', 'model_checkpoint_format'},
                         tensorboard=True,
                         tensorboard_log_dir_format=f'logs.{model_name}.{{datetime}}/',
                         model_checkpoint=True,
                         model_checkpoint_format=f'{out_checkpoints_folder}/weights.{{epoch:02d}}-{{val_loss:.2f}}.hdf5',
                         )

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
