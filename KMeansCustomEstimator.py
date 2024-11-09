import pdb

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from loguru import logger
from tqdm.auto import tqdm
import cupy as cp
# from cuml import KMeans
from sklearn.cluster import KMeans
from cuml.preprocessing import MinMaxScaler
import cudf
from pdb import set_trace as bp
import numpy_indexed as npi


class _NoneFeatureSelector:

    def __init__(self, *args, **kwargs):
        pass

    def set_params(self, *args, **kwargs):
        return self

    def fit_transform(self, X, y=None):
        return X

    def transform(self, X, y=None):
        return X


def _get_mutual_info_classifier_selector(k):
    return SelectKBest(mutual_info_classif, k=k)


def _get_f_classif_selector(k):
    return SelectKBest(f_classif, k=k)

def _get_chi2_selector(k):
    return SelectKBest(chi2, k=k)


_map_feature_selector = {
    'bypass': _NoneFeatureSelector,
    'mutual_info_classif': _get_mutual_info_classifier_selector,
    'f_classif': _get_f_classif_selector,
    'chi2': _get_chi2_selector
}



def groupby(data, groups, agg_func='median'):
    xp = cp.get_array_module(data)
    sorted_indices = xp.argsort(groups)
    sorted_data = data[sorted_indices]
    sorted_groups = groups[sorted_indices]
    unique_groups = xp.unique(sorted_groups)
    agg_func = agg_func if callable(agg_func) else getattr(xp, agg_func)
    aggs = xp.zeros(len(unique_groups))
    for i, group in enumerate(unique_groups):
        group_data = sorted_data[sorted_groups == group]
        aggs[i] = agg_func(group_data)
    return aggs


def _apply_func_multi_times(func):
    def wrap(*x):
        nonlocal func
        if len(x) == 1:
            return func(x[0])
        else:
            return [func(y) for y in x]

    return wrap


def _raise_type_error(x):
    raise TypeError(f'Type {type(x)!r} not supported, value: {x!r}.')

def to_numpy(x):
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    elif hasattr(x, 'to_numpy'):
        return x.to_numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f'Type {type(x)} not supported.')

class KMeansCustomEstimator(BaseEstimator, ClassifierMixin):


    def __init__(self, estimator=None, two_dimensions=False, kmeans_keys=None, estimator_keys=None,
                 fit_estimator_keys=None, cuda=False, feature_selector_model='bypass', features_selected=10, **kwargs):
        self.kmeans = KMeans(n_init='auto')
        if kmeans_keys:
            self.kmeans.set_params(**{x: kwargs[x] for x in kmeans_keys})
        if estimator:
            self.estimator = estimator()
            if estimator_keys:
                self.estimator.set_params(**{x: kwargs[x] for x in estimator_keys})
        else:
            self.estimator = None
        self.two_dimensions = two_dimensions
        self.estimator_keys = estimator_keys
        self.kmeans_keys = kmeans_keys
        self.fit_estimator_keys = fit_estimator_keys
        self.kwargs = kwargs
        self._start_feature_selection(feature_selector_model, features_selected)
        self.__start_cuda(cuda)
        self.scaler = MinMaxScaler()
        logger.info('Iniciado com parâmetros {}', self.get_params())

    def _start_feature_selection(self, feature_selector_model, features_selected):
        self._feature_selector_model = feature_selector_model
        self._features_selected = features_selected
        self._feature_selector = _map_feature_selector.get(feature_selector_model, 'bypass')(features_selected)

    def __start_cuda(self, cuda):
        self.cuda = cuda
        do_nothing = _apply_func_multi_times(lambda x: x)
        self.__convert_from_cuda = _apply_func_multi_times(to_numpy) if self.cuda else do_nothing
        self.__convert_to_cuda = _apply_func_multi_times(
            lambda x: cudf.DataFrame(x).astype('float32') if len(x.shape) > 1 else cudf.Series(x).astype('int32')) \
            if self.cuda else _apply_func_multi_times(to_numpy)
        self.__xp = cp if cuda else np

    @classmethod
    def de_reduce(cls, params):
        return cls(**params)

    def __reduce__(self):
        return type(self).de_reduce, (self.get_params(True),)

    def get_params(self, deep=False):
        return {
            'estimator': type(self.estimator),
            'two_dimensions': self.two_dimensions,
            'kmeans_keys': self.kmeans_keys,
            'estimator_keys': self.estimator_keys,
            'fit_estimator_keys': self.fit_estimator_keys,
            'cuda': self.cuda,
            'feature_selector_model': self._feature_selector_model,
            'features_selected': self._features_selected,
            **self.kwargs,
        }

    @logger.catch(message='Erro ao alterar parâmetros', reraise=True)
    def set_params(self, estimator=None, two_dimensions=None, kmeans_keys=None, estimator_keys=None,
                   fit_estimator_keys=None, cuda=None, feature_selector_model=None,
                   features_selected=None, **params):
        must_update_estimator = False
        logger.debug(f'set_params={estimator=}, {two_dimensions=}, {cuda=}, {feature_selector_model=}, '
                     f'{features_selected=}, {params=}')
        if estimator:
            self.estimator = estimator()
            must_update_estimator = True
        if estimator_keys:
            self.estimator_keys = estimator_keys
        if kmeans_keys:
            self.kmeans_keys = kmeans_keys
        if fit_estimator_keys:
            self.fit_estimator_keys = fit_estimator_keys
        update_estimator = {x: params[x] for x in self.estimator_keys if x in params}
        if update_estimator and not must_update_estimator:
            self.estimator.set_params(**update_estimator)
        update_kmeans = {x: params[x] for x in self.kmeans_keys if x in params}
        if update_kmeans:
            self.kmeans.set_params(**update_kmeans)
        if isinstance(two_dimensions, bool):
            self.two_dimensions = two_dimensions
        self.kwargs.update(params)
        if must_update_estimator:
            self.estimator.set_params(**{x: self.kwargs[x] for x in self.estimator_keys})
        if cuda is not None:
            self.__start_cuda(cuda)
        if feature_selector_model is not None or features_selected is not None:
            self._start_feature_selection(feature_selector_model or self._feature_selector_model,
                                          features_selected or self._features_selected)
        self.scaler = MinMaxScaler()
        logger.info('Atualizado para parâmetros {}', self.get_params())
        return self

    def _gen_clusters(self, X):
        len_clusters = len(X)
        logger.debug('Gerando {} clusters.', len_clusters)
        new_x = cp.zeros((len_clusters, self.kmeans.n_clusters, X[0].shape[1]))
        for i, x in enumerate(tqdm(X, desc='Gerando clusters')):
            kmeans = clone(self.kmeans, safe=True)
            # cu_x = cudf.DataFrame(x)
            cu_x = cp.asnumpy(x)
            # prediction = cp.asarray(kmeans.fit_predict(cu_x))
            # frame_index = cp.arange(x.shape[0])
            prediction = cp.asnumpy(kmeans.fit_predict(cu_x))
            frame_index = np.arange(x.shape[0])
            median_frame_per_centroid = npi.group_by(prediction).median(frame_index)[1]
            # median_frame_per_centroid = groupby(frame_index, prediction)
            # centroids = kmeans.cluster_centers_.to_cupy()
            centroids = cp.asarray(kmeans.cluster_centers_)
            new_x[i, :len(median_frame_per_centroid), :] = centroids[median_frame_per_centroid.argsort()]
        if not self.two_dimensions:
            new_x = cudf.DataFrame(new_x.reshape((len_clusters, -1)))
        logger.debug('Gerados clusters no seguinte formato: {}', new_x.shape)
        return new_x

    def _preprocess(self, X):
        # np_clusters = cudf.DataFrame(self._gen_clusters(X))
        # return to_numpy(self.scaler.fit(np_clusters).transform(np_clusters))
        return self._gen_clusters(X)

    @logger.catch(message='Erro ao fazer preprocessamento para predict', reraise=True)
    def _preprocess_to_predict(self, X):
        return self.__convert_to_cuda(self._feature_selector.transform(self._preprocess(X)))

    @logger.catch(message='Erro ao fazer preprocessamento para fit', reraise=True)
    def _preprocess_to_fit(self, X, y):
        return self.__convert_to_cuda(self._feature_selector.fit_transform(self._preprocess(X), y), y)

    def fit(self, X, y):
        logger.debug('Iniciando fit.')
        np_clusters, y = self._preprocess_to_fit(X, y)
        logger.debug('Treinando com X de formato: {} e y com formato: {}', np_clusters.shape, y.shape)
        try:
            self.estimator.fit(np_clusters, y)
        except:
            logger.exception("Erro ao dar fit com kwargs {}", self.kwargs)
            raise
        return self

    def predict(self, X):
        np_clusters = self._preprocess_to_predict(X)
        return self.estimator.predict(np_clusters)

    @logger.catch(message='Erro ao dar fit predict', reraise=True)
    def fit_predict(self, X, y):
        logger.info('Iniciando fit predict.')
        np_clusters, y = self._preprocess_to_fit(X, y)
        if hasattr(self.estimator, 'fit_predict'):
            try:
                return self.estimator.fit_predict(np_clusters, y)
            except:
                logger.exception("Erro ao dar fit_predict com kwargs {}", self.kwargs)
                raise
        try:
            self.estimator.fit(np_clusters, y)
        except:
            logger.exception("Erro ao dar fit_predict com kwargs {}", self.kwargs)
            raise
        return self.estimator.predict(np_clusters)

    def score(self, X, y, sample_weight=None):
        y = self.__convert_to_cuda(y)
        try:
            if hasattr(self.estimator, 'score'):
                np_clusters = self._preprocess_to_predict(X)
                if self.cuda:
                    result = self.estimator.score(np_clusters, y)
                else:
                    result = self.estimator.score(np_clusters, y, sample_weight)
            else:
                result = super().score(X, y, sample_weight)
        except:
            logger.exception('Erro ao fazer score')
            result = 0
        logger.info("Resultado do modelo: {}", result)
        return result
