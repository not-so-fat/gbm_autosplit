# gbm_autosplit

LightGBM / XGBoost scikit learn interfaces which performs "early stopping" with single data set during `fit`.

## Motivation
"Early stopping" is great practice to tune the number of estimators for gradient boosting models. 
However it is not easy to use it in tuning module in scikit-learn such as RandomizedSearchCV / GridSearchCV
because to use early stopping module requires two data sets but scikit learn does not have such interface.

## Algorithm
To solve this situation, this interface performs following steps with in `fit`.
1. User instanciates Classifier / Regressor with additional hyper parameters `max_n_estimators`, `ratio_training`, and `eval_metric`.
2. User calls `fit` with `x` and `y` as usual
    1. Randomly split sample `(x, y)` into training and validation as ratio of sample size of training = `ratio_training`, 
    2. Call `fit` of original GBM, using early stopping with split training and validation for the metric `eval_metric` with `n_estimators` = `max_n_estimators`
    3. Get `best_n_estimators` as the number of trees of stopped model of step 2-2.
    4. Call `fit` of original GBM with entire `(x, y)` and `n_estimators` = `best_n_estimators` of step 2-3.


## Install

```
pip install gbm_autosplit
```

## Usage

```
import gbm_autosplit

estimator = gbm_autosplit.LGBMClassifier()
estimator.fit(x, y)
```
