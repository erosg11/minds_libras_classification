from datetime import datetime
from typing import Optional, Union, Callable

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.base import BaseEstimator, ClassifierMixin
from loguru import logger

from tools import OUT_PATH


class KerasCustomModel(BaseEstimator, ClassifierMixin):

    model: Union[Model, Sequential]

    def __init__(self,
                 model_generator: Optional[Callable[..., Union[Model, Sequential]]] = None,
                 batch_size: int = 32,
                 epochs: int = 1000,
                 tensorboard: bool = False,
                 tensorboard_params: Optional[dict] = None,
                 tensorboard_log_dir_format: Optional[str] = None,
                 early_stopping: bool = False,
                 early_stopping_min_delta=0,
                 early_stopping_patience=0,
                 early_stopping_monitor='val_loss',
                 model_checkpoint: bool = False,
                 model_checkpoint_params: Optional[dict] = None,
                 model_checkpoint_format: Optional[str] = None,
                 reduce_lro_plateau: bool = False,
                 reduce_lro_plateau_monitor="val_loss",
                 reduce_lro_plateau_factor=0.1,
                 reduce_lro_plateau_patience=10,
                 reduce_lro_plateau_min_delta=0.0001,
                 reduce_lro_plateau_cooldown=0,
                 reduce_lro_plateau_min_lr=0,
                 keras_tqdm=False,
                 model_fit_params: Optional[dict] = None,
                 verbose: int = 0,
                 validation_split=0.3,
                 **model_generator_params: dict,
                 ):
        self.model_generator = model_generator
        self.model_generator_params = model_generator_params
        if self.model_generator:
            self.model = self.model_generator(**self.model_generator_params)
        else:
            self.model = None
        self.history_list = []
        self.append_history = self.history_list.append
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = []
        append_callback = self.callbacks.append
        self.tensorboard = tensorboard
        self.tensorboard_log_dir_format = tensorboard_log_dir_format
        self.tensorboard_params = tensorboard_params
        self.early_stopping = early_stopping
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_monitor = early_stopping_monitor
        self.reduce_lro_plateau_monitor = reduce_lro_plateau_monitor
        self.reduce_lro_plateau_factor = reduce_lro_plateau_factor
        self.reduce_lro_plateau_patience = reduce_lro_plateau_patience
        self.reduce_lro_plateau_min_delta = reduce_lro_plateau_min_delta
        self.reduce_lro_plateau_cooldown = reduce_lro_plateau_cooldown
        self.reduce_lro_plateau_min_lr = reduce_lro_plateau_min_lr
        self.keras_tqdm = keras_tqdm
        self.validation_split = validation_split
        self.model_checkpoint = model_checkpoint
        self.model_checkpoint_params = model_checkpoint_params
        self.model_checkpoint_format = model_checkpoint_format
        self.reduce_lro_plateau = reduce_lro_plateau
        if tensorboard:
            if tensorboard_log_dir_format:
                tensorboard_dir = tensorboard_log_dir_format.format(datetime=datetime.now())
            else:
                tensorboard_dir = OUT_PATH / f'logs_{datetime.now():%Y%m%d%H%M%S}'
            if tensorboard_params is None:
                tensorboard_params = {}
            tensorboard_params['log_dir'] = tensorboard_dir
            append_callback(TensorBoard(**tensorboard_params))
        if early_stopping:
            append_callback(EarlyStopping(self.early_stopping_monitor,
                                          self.early_stopping_min_delta,
                                          self.early_stopping_patience,
                                          0, restore_best_weights=True))
        if model_checkpoint:
            if model_checkpoint_format:
                model_checkpoint_format = model_checkpoint_format
            else:
                model_checkpoint_format = OUT_PATH / \
                    f'weights/weights.{datetime.now():%Y%m%d%H%M%S}.{{epoch:02d}}-{{val_loss:.2f}}.hdf5'
            if model_checkpoint_params is None:
                model_checkpoint_params = {}
            model_checkpoint_params['filepath'] = model_checkpoint_format
            append_callback(ModelCheckpoint(**model_checkpoint_params))
        if reduce_lro_plateau:
            append_callback(ReduceLROnPlateau(
                self.reduce_lro_plateau_monitor,
                self.reduce_lro_plateau_factor,
                self.reduce_lro_plateau_patience,
                0,
                min_delta=self.reduce_lro_plateau_min_delta,
                cooldown=self.reduce_lro_plateau_cooldown,
                min_lr=self.reduce_lro_plateau_min_lr,
            ))
        if keras_tqdm:
            try:
                from keras_tqdm import TQDMCallback
            except ImportError:
                logger.exception('Não foi possível importar o keras-tqdm')
            else:
                append_callback(TQDMCallback())
                del TQDMCallback
        if model_fit_params is None:
            self.model_fit_params = {}
        else:
            self.model_fit_params = model_fit_params
        self.verbose = verbose

    def get_params(self, deep=False):
        return {
            'model_generator': self.model_generator,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'tensorboard': self.tensorboard,
            'tensorboard_log_dir_format': self.tensorboard_log_dir_format,
            'tensorboard_params': self.tensorboard_params,
            'early_stopping': self.early_stopping,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_monitor': self.early_stopping_monitor,
            'model_checkpoint': self.model_checkpoint,
            'model_checkpoint_params': self.model_checkpoint_params,
            'model_checkpoint_format': self.model_checkpoint_format,
            'reduce_lro_plateau': self.reduce_lro_plateau,
            'reduce_lro_plateau_monitor': self.reduce_lro_plateau_monitor,
            'reduce_lro_plateau_factor': self.reduce_lro_plateau_factor,
            'reduce_lro_plateau_patience': self.reduce_lro_plateau_patience,
            'reduce_lro_plateau_min_delta': self.reduce_lro_plateau_min_delta,
            'reduce_lro_plateau_cooldown': self.reduce_lro_plateau_cooldown,
            'reduce_lro_plateau_min_lr': self.reduce_lro_plateau_min_lr,
            'model_fit_params': self.model_fit_params,
            'validation_split': self.validation_split,
            'verbose': self.verbose,
            **self.model_generator_params,
        }

    def set_params(self, **params):
        final_params = {**self.get_params(), **params}
        self.__init__(**final_params)
        return self

    def fit(self, x, y):
        self.append_history(
            self.model.fit(x, y, self.batch_size, self.epochs, verbose=self.verbose, callbacks=self.callbacks,
                           validation_split=self.validation_split, **self.model_fit_params))

    def predict(self, x):
        return self.model.predict(x, self.batch_size, self.verbose)

    def score(self, X, y, sample_weight=None):
        return self.model.evaluate(X, y, self.batch_size, verbose=self.verbose, sample_weight=sample_weight)[1]

    def __reduce_ex__(self, protocol):
        filepath = OUT_PATH / 'weigths'
        filepath.mkdir(exist_ok=True, parents=True)
        filepath /= f'{datetime.utcnow():%Y%m%d%H%M%S.h5}'
        self.model.save(filepath)
        return _keras_custom_from_file, (self.get_params(), self.history_list, filepath)

    def __reduce__(self):
        from pickle import DEFAULT_PROTOCOL
        return self.__reduce_ex__(DEFAULT_PROTOCOL)


def _keras_custom_from_file(params, history_list, weights_filename):
    estimator = KerasCustomModel(**params)
    estimator.history_list = history_list
    estimator.model.load_weights(weights_filename)
    return estimator
