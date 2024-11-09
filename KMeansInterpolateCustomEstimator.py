from typing import Literal

from KMeansCustomEstimator import KMeansCustomEstimator
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
import numpy as np
from loguru import logger

MIN_INTERPOLATION_RATIO = 1.2
MAX_INTERPOLATION_RATIO = 50


class KMeansInterpolateCustomEstimator(KMeansCustomEstimator):

    @staticmethod
    def _validate_interpolation_frames(interpolation_ratio):
        if interpolation_ratio > MAX_INTERPOLATION_RATIO:
            raise ValueError(f'A escala de interpolação deve ser menor que {MAX_INTERPOLATION_RATIO}!')
        if interpolation_ratio < MIN_INTERPOLATION_RATIO:
            raise ValueError(f'A escala de interpolação deve ser menor que {MIN_INTERPOLATION_RATIO}!')
        return interpolation_ratio

    def __init__(self, interpolation_ratio: float = 1.2,
                 interpolation_kind:
                 Literal['zero', 'slinear', 'quadratic', 'cubic', 'linear', 'nearest', 'previous', 'next'] |
                 int = 'linear',
                 estimator=None, two_dimensions=False, kmeans_keys=None, estimator_keys=None, fit_estimator_keys=None,
                 cuda=False, feature_selector_model='bypass', features_selected=100, **kwargs):
        self._interpolation_ratio = interpolation_ratio
        self.interpolation_kind = interpolation_kind
        super().__init__(estimator, two_dimensions, kmeans_keys, estimator_keys, fit_estimator_keys, cuda,
                         feature_selector_model, features_selected, **kwargs)
        self.interpolation_frames = int(np.ceil(self.kmeans.n_clusters *
                                                self._validate_interpolation_frames(interpolation_ratio)))
        logger.info('Usando {} quadros interpolados vs {} clusters.', self.interpolation_frames, self.kmeans.n_clusters)
        self._t = np.linspace(0, 1, self.interpolation_frames)

    def __start_cuda(self, cuda):
        super().__start_cuda(cuda)

    def get_params(self, deep=False):
        super_params = super().get_params()
        super_params.update({'interpolation_ratio': self._interpolation_ratio,
                             'interpolation_kind': self.interpolation_kind})
        return super_params

    def set_params(self, interpolation_ratio=None, interpolation_kind=None, estimator=None, two_dimensions=None,
                   kmeans_keys=None, estimator_keys=None, fit_estimator_keys=None, cuda=None,
                   feature_selector_model=None,
                   features_selected=None, **params):
        logger.debug(f'set_params(classe filha)={interpolation_ratio=}, {interpolation_kind=}')
        if interpolation_ratio is not None:
            self._interpolation_ratio = interpolation_ratio
        if interpolation_kind is not None:
            self.interpolation_kind = interpolation_kind
        super().set_params(estimator, two_dimensions, kmeans_keys, estimator_keys, fit_estimator_keys,
                           cuda, feature_selector_model, features_selected, **params)
        self.interpolation_frames = int(np.ceil(self.kmeans.n_clusters *
                                                self._validate_interpolation_frames(interpolation_ratio)))
        self._t = np.linspace(0, 1, self.interpolation_frames, cuda)
        logger.info('Usando {} quadros interpolados vs {} clusters.', self.interpolation_frames, self.kmeans.n_clusters)
        return self

    def _gen_clusters(self, X):
        logger.debug('Iniciando interpolação de {} videos, com formato semelhante a {}.', len(X), X[0].shape)
        new_X = []
        add_new_x = new_X.append
        for x in tqdm(X, desc='Fazendo interpolação'):
            x = np.asarray(x)
            len_x = len(x)
            t = np.linspace(0, 1, len_x)
            f = interp1d(t, x.reshape(len_x, -1), kind=self.interpolation_kind, axis=0)
            new_x = f(self._t).reshape([self.interpolation_frames] + list(x.shape[1:]))
            add_new_x(new_x)
        logger.debug('Feita interpolação de {} videos, saida com formato {}', len(new_X), new_X[0].shape)
        return super()._gen_clusters(new_X)
