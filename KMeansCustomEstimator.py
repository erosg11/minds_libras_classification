from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
from loguru import logger
import numpy_indexed as npi


class KMeansCustomEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator=None, two_dimensions=False, kmeans_keys=None, estimator_keys=None,
                 fit_estimator_keys=None, **kwargs):
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
        logger.info('Iniciado com parâmetros {}', self.get_params())

    def get_params(self, deep=False):
        return {
            'estimator': type(self.estimator),
            'two_dimensions': self.two_dimensions,
            'kmeans_keys': self.kmeans_keys,
            'estimator_keys': self.estimator_keys,
            'fit_estimator_keys': self.fit_estimator_keys,
            **self.kwargs,
        }

    @logger.catch(message='Erro ao alterar parâmetros', reraise=True)
    def set_params(self, estimator=None, two_dimensions=None, kmeans_keys=None, estimator_keys=None,
                   fit_estimator_keys=None, **params):
        must_update_estimator = False
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
        logger.info('Atualizado para parâmetros {}', self.get_params())
        return self

    def _gen_clusters(self, X):
        len_clusters = len(X)
        clusters = [None] * len_clusters
        for i, x in enumerate(X):
            kmeans = clone(self.kmeans, safe=True)
            prediction = kmeans.fit_predict(x)
            frame_index = np.arange(x.shape[0])
            median_frame_per_centroid = npi.group_by(prediction).median(frame_index)[1]
            centroids = kmeans.cluster_centers_
            clusters[i] = centroids[median_frame_per_centroid.argsort()]
        if self.two_dimensions:
            return np.array(clusters)
        else:
            return np.array(clusters).reshape((len_clusters, -1))

    def fit(self, X, y):
        np_clusters = self._gen_clusters(X)
        try:
            self.estimator.fit(np_clusters, y)
        except:
            logger.exception("Erro ao dar fit com kwargs {}", self.kwargs)
            raise
        return self

    def predict(self, X):
        np_clusters = self._gen_clusters(X)
        return self.estimator.predict(np_clusters)

    @logger.catch(message='Erro ao dar fit predict', reraise=True)
    def fit_predict(self, X, y):
        np_clusters = self._gen_clusters(X)
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
        try:
            if hasattr(self.estimator, 'score'):
                np_clusters = self._gen_clusters(X)
                result = self.estimator.score(np_clusters, y, sample_weight)
            else:
                result = super().score(X, y, sample_weight)
        except:
            logger.exception('Erro ao fazer score')
            result = 0
        logger.info("Resultado do modelo: {}", result)
        return result
