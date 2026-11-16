"""Scalar, replayable kernel families for native Gaussian-process estimators."""
import numbers

import numpy as np
from code_recording import recordable
from sklearn.gaussian_process.kernels import Hyperparameter, PairwiseKernel


@recordable
class SearchablePairwiseKernel(PairwiseKernel):
    """Keep discrete family/degree choices separate from continuous GP fitting.

    Native PairwiseKernel implements covariance and log-gamma gradients. Only
    gamma is optimized by GP; degree and coef0 are explicit outer-search values.
    """

    def __init__(self, metric='rbf', gamma=.5, degree=3, coef0=1.,
                 gamma_bounds=(1e-5, 1e5)):
        self.metric = metric
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.gamma_bounds = gamma_bounds

    @property
    def hyperparameter_gamma(self):
        return Hyperparameter('gamma', 'numeric',
                              'fixed' if self.metric == 'linear' else self.gamma_bounds)

    def _delegate(self):
        if self.metric not in {'rbf', 'poly', 'linear'}:
            raise ValueError('Choose rbf, poly or linear for the GP kernel family.')
        if self.metric != 'linear' and (isinstance(self.gamma, bool)
                or not isinstance(self.gamma, numbers.Real) or not np.isfinite(self.gamma) or self.gamma <= 0):
            raise ValueError('GP kernel gamma must be finite and positive.')
        if self.metric == 'poly':
            if isinstance(self.degree, bool) or not isinstance(self.degree, numbers.Integral) or self.degree < 1:
                raise ValueError('Polynomial degree must be a positive integer.')
            if not isinstance(self.coef0, numbers.Real) or not np.isfinite(self.coef0) or self.coef0 < 0:
                raise ValueError('GP polynomial offset must be finite and nonnegative.')
        return PairwiseKernel(gamma=self.gamma, metric=self.metric,
                              gamma_bounds='fixed' if self.metric == 'linear' else self.gamma_bounds,
                              pairwise_kernels_kwargs={'degree': self.degree, 'coef0': self.coef0})

    def __call__(self, X, Y=None, eval_gradient=False):
        return self._delegate()(X, Y, eval_gradient=eval_gradient)

    def diag(self, X):
        return self._delegate().diag(X)

    def __repr__(self):
        return (f'SearchablePairwiseKernel(metric={self.metric!r}, gamma={self.gamma!r}, '
                f'degree={self.degree!r}, coef0={self.coef0!r}, gamma_bounds={self.gamma_bounds!r})')


@recordable
def kernel_parameters(estimator, parameters):
    """Decode only an allowlisted GP descriptor; never evaluate code from JSON."""
    parameters = dict(parameters)
    descriptor = parameters.get('kernel')
    if not isinstance(descriptor, dict):
        return parameters
    if estimator not in {'sklearn.gaussian_process.GaussianProcessRegressor',
                         'sklearn.gaussian_process.GaussianProcessClassifier'}:
        raise ValueError('Kernel descriptors are supported only for Gaussian processes.')
    allowed = {'metric', 'gamma', 'degree', 'coef0', 'gamma_bounds'}
    if set(descriptor) - allowed:
        raise ValueError('Unknown GP kernel descriptor fields.')
    descriptor = dict(descriptor)
    if isinstance(descriptor.get('gamma_bounds'), list):
        descriptor['gamma_bounds'] = tuple(descriptor['gamma_bounds'])
    kernel = SearchablePairwiseKernel(**descriptor)
    kernel._delegate()  # Fail early on invalid recipe descriptors.
    parameters['kernel'] = kernel
    return parameters
