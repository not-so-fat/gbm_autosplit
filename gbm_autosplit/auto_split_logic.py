import logging

import numpy
from numpy import random

from gbm_autosplit import utils


logger = logging.getLogger(__name__)


def auto_split_fit(learner, x, y, **kwargs):
    if learner.early_stopping_rounds > 0:
        _tune_num_estimator(learner, x, y, **kwargs)
        if learner.n_estimators == learner.max_n_estimators:
            logger.warning("n_estimators reached max_n_estimators: {}".format(learner.n_estimators))
    # refit with full sample, and prevent splitting during `refit` in cv
    learner.set_params(early_stopping_rounds=None)


def _tune_num_estimator(learner, x, y, **kwargs):
    """
    Decide optimal number of estimator by validation data generated by splitting x
    """
    xtr, ytr, xva, yva = split_xy(x, y, learner.ratio_training)
    learner.call_parent_fit(xtr, ytr, eval_set=[(xva, yva)], eval_metric=learner.metric, verbose=False, **kwargs)
    learner.n_estimators = utils.get_n_estimators(learner)


def split_xy(x, y, training_ratio):
    sample_size = len(y)
    size_training = int(training_ratio * sample_size)
    indices_list = numpy.arange(x.shape[0])
    random.shuffle(indices_list)
    indices_training = indices_list[:size_training]
    indices_validation = indices_list[size_training:]
    return x[indices_training, :], y[indices_training], x[indices_validation, :], y[indices_validation]
