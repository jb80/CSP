#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Revised XGBoost, SHAP, mutual-information, and exhaustive-interaction LASSO workflow.

This script is used after:

1. CSP FewShot Learner
2. Embedded Topic Model

The analysis uses model-estimated probabilities for:

class 0 = absence
class 1 = presence
class 2 = unclear or insufficient information

For presence-positive variables:

signed score = P(presence) - P(absence)

For conflict and inequality:

signed score = P(absence) - P(presence)

If P(missing) > MISSING_PROBABILITY_THRESHOLD:

signed score = 0

The score is then scaled from [-1, 1] to [0, 1].

XGBoost analysis:

1. Optimize hyperparameters with Optuna.
2. Train repeated models using different train-test splits.
3. Calculate model diagnostics.
4. Explain every model using the same complete dataset.
5. Aggregate signed SHAP values across documents and models.
6. Retain mean absolute SHAP values separately as magnitude measures.
7. Calculate SHAP interaction values.

Topic-specific XGBoost analysis:

1. Use topic probabilities from the ETM results.
2. Create seven topic-context clusters using KMeans.
3. Name clusters from their topic-probability centroids.
4. Repeat the XGBoost and SHAP analysis within each topic context.

LASSO analysis:

1. Generate every possible interaction-only combination of the eleven institutional predictors.
2. Fit LassoCV to identify candidate configurations.
3. Repeat LASSO on subsamples to assess coefficient and selection stability.
4. Preserve selected interactions, coefficients, and diagnostics.

With eleven predictors and interaction degree eleven:

number of candidate terms = 2^11 - 1 = 2047

SHAP signed sums are interpreted as net directional model attribution.

Mean absolute SHAP values are stored separately and represent attribution magnitude rather than direction.

SHAP results explain the trained predictive models and are not interpreted as causal effects.

@author: jbaggio
"""


'''
Packages
'''

import os
import re
import gc
import pickle
import random
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import xgboost as xgb
import optuna
import shap

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline

from scipy.spatial.distance import cosine



'''
Directories
'''

mainres = Path('/CommonsSynthProject/NLPSynth/Analysis/MainResults')
topicsout = Path('/CommonsSynthProject/NLPSynth/Analysis/Figures/Topics')
shapout = Path('/CommonsSynthProject/NLPSynth/Analysis/Shapley')
pslout = Path('//CommonsSynthProject/NLPSynth/Analysis/PSL')


allout = shapout / 'All_Together'
topicout = shapout / 'Per_Topic'
miout = shapout / 'ZZZMutual_Info'
lassoout = pslout / 'LASSO'

for directory in [mainres, topicsout, shapout, allout, topicout, miout, lassoout]:
    directory.mkdir(parents=True, exist_ok=True)


'''
Reproducibility and analysis settings
'''

RANDOM_SEED = 8081

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

warnings.filterwarnings('ignore')

MISSING_PROBABILITY_THRESHOLD = 0.66
MISSING_PROBABILITY_THRESHOLDS = [0.50, 0.66, 0.75]
PRIMARY_MISSING_PROBABILITY_THRESHOLD = 0.66

TEST_SIZE = 0.25

N_OPTUNA_TRIALS_AGGREGATE = 500
N_OPTUNA_TRIALS_TOPIC = 200

N_REPEATED_MODELS_AGGREGATE = 35
N_REPEATED_MODELS_TOPIC = 15

AGGREGATE_CV_FOLDS = 10
TOPIC_CV_FOLDS = 5


LASSO_REPEATS_AGGREGATE = 50
LASSO_REPEATS_TOPIC = 50
LASSO_DEGREE = 5 # above this, too many predictors 

LASSO_SUBSAMPLE_FRACTION = 0.50
LASSO_SELECTION_TOLERANCE = 1e-4
LASSO_MAX_ITER = 10000

SAVE_XGBOOST_MODELS = True
SAVE_SHAP_INTERACTIONS = True


'''
Variable definitions
'''

coding_variables = ['rescond', 'conflict', 'inequality', 'socbound', 'resbound', 'fit', 'prop', 
'participation', 'monitor', 'accmonitor', 'gradsanc', 'confres', 'autonomy', 'nested']

depcols = {'rescond': 'RSC', 'conflict': 'CFL', 'inequality': 'IEQ'}

indepcols = {'resbound': 'RSB', 'socbound': 'SCB', 'fit': 'FIT', 'prop': 'PRP', 
'participation': 'PRT', 'monitor': 'MON', 'accmonitor': 'ACM', 'gradsanc': 'GRS', 'confres': 'CFR', 
'autonomy': 'AUT', 'nested': 'NST'}

outcome_columns = ['RSC', 'CFL', 'IEQ']

feature_columns = ['SCB', 'RSB', 'FIT', 'PRP', 'PRT', 'MON', 'ACM', 'GRS', 'CFR', 'AUT', 'NST']

absence_positive_variables = {'conflict', 'inequality'}

#make sure the topic are the same as the one stated in CSP_4_EmbeddedTopics script

topic_probability_columns = ['Biodiversity Conservation','Forestry and Livelihoods','Institutions and Governance','Water Governance','Fisheries','Land Rights']

topic_context_names = ['Biodiversity Conservation','Forestry and Livelihoods','Institutions and Governance','Water Governance','Fisheries','Land Rights', 'Mixed']


'''
General functions
'''

def safe_filename(value):
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value))
    return value.strip('_')


def clear_memory():
    gc.collect()


def validate_columns(dataframe, required_columns, dataframe_name):
    missing_columns = [column for column in required_columns if column not in dataframe.columns]

    if missing_columns:
        raise ValueError(dataframe_name + ' is missing columns: ' + str(missing_columns))


def get_probability_column(dataframe, variable, class_value):
    explicit_column = variable + '_prob_' + str(class_value)

    if explicit_column in dataframe.columns:
        return dataframe[explicit_column].astype(float)

    legacy_column = variable + '_probs'

    if legacy_column not in dataframe.columns:
        raise ValueError('No probability output exists for ' + variable)

    print('WARNING: using legacy list-position probabilities for ' + variable)

    return dataframe[legacy_column].apply(lambda probabilities: float(probabilities[class_value]))


def make_signed_score(presence_probability, absence_probability, missing_probability, absence_is_positive=False, missing_threshold=MISSING_PROBABILITY_THRESHOLD):
    if absence_is_positive:
        signed_score = absence_probability - presence_probability
    else:
        signed_score = presence_probability - absence_probability

    signed_score = np.where(missing_probability > missing_threshold, 0.0, signed_score)

    return signed_score


def create_analysis_data_for_threshold(dfpreds_all, dftops, missing_threshold):
    score_dataframe = pd.DataFrame({'txt_title': dfpreds_all['txt_title'].astype(str)})
    certainty_dataframe = pd.DataFrame({'txt_title': dfpreds_all['txt_title'].astype(str)})

    for variable in coding_variables:
        presence_probability = dfpreds_all[variable + '_pres']
        absence_probability = dfpreds_all[variable + '_abs']
        missing_probability = dfpreds_all[variable + '_mis']

        score_dataframe[variable] = make_signed_score(presence_probability, absence_probability, missing_probability, absence_is_positive=variable in absence_positive_variables, missing_threshold=missing_threshold)
        certainty_dataframe[variable] = 1.0 - missing_probability

    score_dataframe = score_dataframe.drop(columns=['reschange'], errors='ignore')
    score_dataframe = score_dataframe.rename(columns=depcols)
    score_dataframe = score_dataframe.rename(columns=indepcols)

    complete_dataframe = pd.merge(left=score_dataframe, right=dftops, on='txt_title', how='inner', validate='one_to_one')

    dependent_variables = complete_dataframe[outcome_columns].astype(float).copy()
    independent_variables = complete_dataframe[feature_columns].astype(float).copy()

    dependent_variables_scaled = ((dependent_variables + 1.0) / 2.0).clip(lower=0.0, upper=1.0)
    independent_variables_scaled = ((independent_variables + 1.0) / 2.0).clip(lower=0.0, upper=1.0)

    return {
        'missing_probability_threshold': float(missing_threshold),
        'dfall': complete_dataframe,
        'depvars': dependent_variables,
        'indepvars': independent_variables,
        'depvars_scaled': dependent_variables_scaled,
        'indepvars_scaled': independent_variables_scaled,
        'certainty': certainty_dataframe
    }



'''
XGBoost and Optuna functions
'''

def xgboost_parameter_space(trial, topic_model=False):
    if topic_model:
        maximum_rounds = trial.suggest_int('n_estimators', 50, 1000, step=50)
        lambda_upper = 200.0
        alpha_upper = 200.0
        minimum_child_upper = 10.0
    else:
        maximum_rounds = trial.suggest_int('n_estimators', 20, 500, step=20)
        lambda_upper = 100.0
        alpha_upper = 100.0
        minimum_child_upper = 50.0

    parameters = {
        'tree_method': 'hist',
        'objective': 'reg:logistic',
        'booster': 'gbtree',
        'eta': trial.suggest_float('eta', 1e-4, 0.1, log=True),
        'lambda': trial.suggest_float('lambda', 1e-3, lambda_upper, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, alpha_upper, log=True),
        'subsample': trial.suggest_float('subsample', 0.25, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.20, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, minimum_child_upper),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'gamma': trial.suggest_float('gamma', 1e-3, 2.0, log=True),
        'eval_metric': 'rmse',
        'seed': RANDOM_SEED,
        'nthread': -1
    }

    return parameters, maximum_rounds


def optimize_xgboost(features, labels, number_of_trials, number_of_folds, study_name, topic_model=False):
    dtrain = xgb.DMatrix(data=features, label=labels, feature_names=list(features.columns))

    def objective(trial):
        parameters, maximum_rounds = xgboost_parameter_space(trial, topic_model=topic_model)

        cv_results = xgb.cv(params=parameters, dtrain=dtrain, num_boost_round=maximum_rounds, nfold=number_of_folds, stratified=False, shuffle=True, early_stopping_rounds=50, metrics='rmse', seed=RANDOM_SEED, as_pandas=True, verbose_eval=False)

        trial.set_user_attr('best_n_estimators', len(cv_results))

        return float(cv_results['test-rmse-mean'].min())

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='minimize', sampler=sampler, study_name=study_name)
    study.optimize(objective, n_trials=number_of_trials, gc_after_trial=True)

    best_parameters = study.best_trial.params.copy()
    best_parameters['n_estimators'] = study.best_trial.user_attrs['best_n_estimators']
    best_parameters.update({'tree_method': 'hist', 'objective': 'reg:logistic', 'booster': 'gbtree', 'eval_metric': 'rmse', 'random_state': RANDOM_SEED, 'n_jobs': -1})

    return study, best_parameters


def repeated_xgboost_shap(features, labels, best_parameters, number_of_models, number_of_cv_folds, analysis_name):
    features = features.reset_index(drop=True).copy()
    labels = labels.reset_index(drop=True).copy()

    model_dictionary = {}

    train_rmse = []
    test_rmse = []
    train_r2 = []
    test_r2 = []
    cv_rmse_mean = []
    cv_rmse_standard_deviation = []

    shap_values_list = []
    base_values_list = []
    interaction_values_list = []
    split_random_states = []

    for model_number in range(number_of_models):
        split_seed = RANDOM_SEED + model_number * 97
        split_random_states.append(split_seed)

        print(analysis_name + ', model ' + str(model_number + 1) + ' of ' + str(number_of_models))

        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=TEST_SIZE, random_state=split_seed)

        model = xgb.XGBRegressor(**best_parameters)
        model.fit(train_features, train_labels)

        model_dictionary[model_number] = model

        predicted_train = model.predict(train_features)
        predicted_test = model.predict(test_features)

        train_rmse.append(root_mean_squared_error(train_labels, predicted_train))
        test_rmse.append(root_mean_squared_error(test_labels, predicted_test))
        train_r2.append(r2_score(train_labels, predicted_train))
        test_r2.append(r2_score(test_labels, predicted_test))

        valid_cv_folds = min(number_of_cv_folds, len(train_features))
        cross_validation = KFold(n_splits=valid_cv_folds, shuffle=True, random_state=split_seed)
        cv_scores = cross_val_score(model, train_features, train_labels, cv=cross_validation, scoring='neg_root_mean_squared_error', n_jobs=-1)
        cv_scores = -cv_scores

        cv_rmse_mean.append(np.mean(cv_scores))
        cv_rmse_standard_deviation.append(np.std(cv_scores))

        # Every repeated model is explained on the same complete dataset.
        explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
        explanation = explainer(features)
        shap_values = np.asarray(explanation.values)

        shap_values_list.append(shap_values)
        base_values_list.append(np.asarray(explanation.base_values))

        if SAVE_SHAP_INTERACTIONS:
            interaction_values = explainer.shap_interaction_values(features)
            interaction_values_list.append(np.asarray(interaction_values))

        clear_memory()

    shap_values_array = np.stack(shap_values_list, axis=0)

    mean_local_shap = np.mean(shap_values_array, axis=0)
    standard_deviation_local_shap = np.std(shap_values_array, axis=0)

    # Net signed attribution across documents for each repeated model.
    signed_sum_by_model = np.sum(shap_values_array, axis=1)

    # Signed average attribution per document for each repeated model.
    signed_mean_by_model = np.mean(shap_values_array, axis=1)

    # Attribution magnitude without positive-negative cancellation.
    mean_absolute_by_model = np.mean(np.abs(shap_values_array), axis=1)

    mean_base_values = np.mean(np.stack(base_values_list, axis=0), axis=0)

    aggregated_explanation = shap.Explanation(values=mean_local_shap, base_values=mean_base_values, data=features.to_numpy(), feature_names=list(features.columns))

    if SAVE_SHAP_INTERACTIONS:
        interaction_values_array = np.stack(interaction_values_list, axis=0)
        mean_interactions = np.mean(interaction_values_array, axis=0)
        standard_deviation_interactions = np.std(interaction_values_array, axis=0)
    else:
        mean_interactions = None
        standard_deviation_interactions = None

    diagnostics = {
        'Train_RMSE_Mean': np.mean(train_rmse),
        'Train_RMSE_StDev': np.std(train_rmse),
        'Test_RMSE_Mean': np.mean(test_rmse),
        'Test_RMSE_StDev': np.std(test_rmse),
        'Train_R2_Mean': np.mean(train_r2),
        'Train_R2_StDev': np.std(train_r2),
        'Test_R2_Mean': np.mean(test_r2),
        'Test_R2_StDev': np.std(test_r2),
        'RMSE_CV_Mean': np.mean(cv_rmse_mean),
        'RMSE_CV_Between_Model_StDev': np.std(cv_rmse_mean),
        'RMSE_CV_Within_Model_StDev_Mean': np.mean(cv_rmse_standard_deviation),
        'N_Observations': len(features),
        'N_Models': number_of_models
    }

    return {
        'models': model_dictionary,
        'diagnostics': diagnostics,
        'aggregated_explanation': aggregated_explanation,
        'mean_local_shap': mean_local_shap,
        'std_local_shap': standard_deviation_local_shap,
        'signed_sum_by_model': signed_sum_by_model,
        'signed_mean_by_model': signed_mean_by_model,
        'mean_absolute_by_model': mean_absolute_by_model,
        'mean_interactions': mean_interactions,
        'std_interactions': standard_deviation_interactions,
        'split_random_states': split_random_states
    }


'''
LASSO functions
'''

def build_all_interactions(features):
    polynomial_model = PolynomialFeatures(degree=LASSO_DEGREE, interaction_only=True, include_bias=False)
    interaction_matrix = polynomial_model.fit_transform(features)
    interaction_names = polynomial_model.get_feature_names_out(features.columns)
    interaction_dataframe = pd.DataFrame(interaction_matrix, columns=interaction_names, index=features.index)

    return interaction_dataframe, polynomial_model


def fit_repeated_lasso_interactions(features, outcome, analysis_name, number_of_repeats, number_of_cv_folds):
    features = features.reset_index(drop=True).copy()
    outcome = outcome.reset_index(drop=True).copy()

    if len(features) != len(outcome):
        raise ValueError(analysis_name + ': features and outcome have different numbers of observations.')

    if len(features) < 3:
        raise ValueError(analysis_name + ': at least three observations are required for LASSO.')

    interaction_features, polynomial_model = build_all_interactions(features)

    print(analysis_name + ': number of LASSO terms = ' + str(interaction_features.shape[1]))

    valid_cv_folds = min(number_of_cv_folds, len(interaction_features))

    if valid_cv_folds < 2:
        raise ValueError(analysis_name + ': at least two cross-validation folds are required.')

    full_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', LassoCV(cv=valid_cv_folds, random_state=RANDOM_SEED, max_iter=LASSO_MAX_ITER, n_jobs=-1))
    ])

    full_pipeline.fit(interaction_features, outcome)

    full_lasso = full_pipeline.named_steps['lasso']
    full_coefficients = pd.Series(full_lasso.coef_, index=interaction_features.columns, name='full_coefficient')
    selected_full = np.abs(full_coefficients) > LASSO_SELECTION_TOLERANCE
    selected_features = full_coefficients[selected_full]

    cross_validation = KFold(n_splits=valid_cv_folds, shuffle=True, random_state=RANDOM_SEED)
    out_of_fold_predictions = cross_val_predict(full_pipeline, interaction_features, outcome, cv=cross_validation, n_jobs=-1)

    out_of_fold_mse = mean_squared_error(outcome, out_of_fold_predictions)
    out_of_fold_rmse = root_mean_squared_error(outcome, out_of_fold_predictions)
    out_of_fold_r2 = r2_score(outcome, out_of_fold_predictions)

    random_generator = np.random.RandomState(RANDOM_SEED)
    sample_size = max(2, int(np.floor(LASSO_SUBSAMPLE_FRACTION * len(interaction_features))))

    repeated_coefficients = []
    repeated_alphas = []

    for repeat_number in range(number_of_repeats):
        print(analysis_name + ', LASSO repeat ' + str(repeat_number + 1) + ' of ' + str(number_of_repeats))

        selected_indices = random_generator.choice(np.arange(len(interaction_features)), size=sample_size, replace=False)
        repeated_features = interaction_features.iloc[selected_indices].reset_index(drop=True)
        repeated_outcome = outcome.iloc[selected_indices].reset_index(drop=True)
        repeated_cv_folds = min(number_of_cv_folds, len(repeated_features))

        if repeated_cv_folds < 2:
            raise ValueError(analysis_name + ': fewer than two cross-validation folds in repeated LASSO.')

        repeated_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', LassoCV(cv=repeated_cv_folds, random_state=RANDOM_SEED + repeat_number, max_iter=LASSO_MAX_ITER, n_jobs=-1))
        ])

        repeated_pipeline.fit(repeated_features, repeated_outcome)

        repeated_lasso = repeated_pipeline.named_steps['lasso']
        repeated_coefficients.append(repeated_lasso.coef_)
        repeated_alphas.append(repeated_lasso.alpha_)

        clear_memory()

    repeated_coefficients = np.asarray(repeated_coefficients)
    selected_matrix = np.abs(repeated_coefficients) > LASSO_SELECTION_TOLERANCE

    selected_coefficient_mean = np.zeros(repeated_coefficients.shape[1], dtype=float)
    selected_coefficient_median = np.zeros(repeated_coefficients.shape[1], dtype=float)
    selected_coefficient_stdev = np.zeros(repeated_coefficients.shape[1], dtype=float)

    for coefficient_position in range(repeated_coefficients.shape[1]):
        selected_values = repeated_coefficients[selected_matrix[:, coefficient_position], coefficient_position]

        if len(selected_values) > 0:
            selected_coefficient_mean[coefficient_position] = np.mean(selected_values)
            selected_coefficient_median[coefficient_position] = np.median(selected_values)
            selected_coefficient_stdev[coefficient_position] = np.std(selected_values)

    stability_table = pd.DataFrame({
        'feature': interaction_features.columns,
        'interaction_order': [len(feature.split(' ')) for feature in interaction_features.columns],
        'full_coefficient': full_coefficients.values,
        'selection_frequency': np.mean(selected_matrix, axis=0),
        'positive_frequency': np.mean(repeated_coefficients > LASSO_SELECTION_TOLERANCE, axis=0),
        'negative_frequency': np.mean(repeated_coefficients < -LASSO_SELECTION_TOLERANCE, axis=0),
        'coefficient_mean': np.mean(repeated_coefficients, axis=0),
        'coefficient_median': np.median(repeated_coefficients, axis=0),
        'coefficient_stdev': np.std(repeated_coefficients, axis=0),
        'selected_coefficient_mean': selected_coefficient_mean,
        'selected_coefficient_median': selected_coefficient_median,
        'selected_coefficient_stdev': selected_coefficient_stdev
    })

    stability_table['dominant_sign'] = np.select(
        [
            stability_table['positive_frequency'] > stability_table['negative_frequency'],
            stability_table['negative_frequency'] > stability_table['positive_frequency']
        ],
        ['positive', 'negative'],
        default='mixed'
    )

    stability_table['sign_stability'] = np.maximum(stability_table['positive_frequency'], stability_table['negative_frequency'])

    stability_table = stability_table.sort_values(
        by=['selection_frequency', 'sign_stability', 'selected_coefficient_median'],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    diagnostics = {
        'analysis_name': analysis_name,
        'n_observations': len(features),
        'n_original_features': features.shape[1],
        'n_interaction_terms': interaction_features.shape[1],
        'n_selected_full_model': int(selected_full.sum()),
        'lasso_alpha_full': float(full_lasso.alpha_),
        'lasso_intercept_full': float(full_lasso.intercept_),
        'repeated_alpha_mean': float(np.mean(repeated_alphas)),
        'repeated_alpha_stdev': float(np.std(repeated_alphas)),
        'oof_mse': out_of_fold_mse,
        'oof_rmse': out_of_fold_rmse,
        'oof_r2': out_of_fold_r2,
        'number_of_repeated_fits': number_of_repeats,
        'subsample_fraction': LASSO_SUBSAMPLE_FRACTION,
        'selection_tolerance': LASSO_SELECTION_TOLERANCE
    }

    return {
        'pipeline': full_pipeline,
        'polynomial_model': polynomial_model,
        'full_coefficients': full_coefficients,
        'selected_features': selected_features,
        'repeated_coefficients': repeated_coefficients,
        'stability_table': stability_table,
        'diagnostics': diagnostics,
        'out_of_fold_predictions': out_of_fold_predictions
    }


'''
Load FewShot predictions
'''

with open(mainres / 'predicted_codes_cases.p', 'rb') as input_file:
    dfpreds = pickle.load(input_file)

validate_columns(dfpreds, ['txt_title', 'CaseStudy_p'], 'predicted_codes_cases.p')

dfpreds = dfpreds.drop_duplicates(subset=['txt_title'], keep='first').reset_index(drop=True)

print('Total papers:', len(dfpreds))


'''
Load topic-model results
'''


with open(mainres / 'TopicsMain.p', 'rb') as input_file:
    dftops = pickle.load(input_file)

dftops = dftops.rename(columns={"topic": "topname"})

dfprobs_cases = dfpreds.copy()
print('Hard-predicted case studies:', len(dfpreds))




'''
Probability diagnostics
'''

probability_diagnostics = []

for variable in coding_variables:
    probability_sum = dfpreds[variable + '_prob_0'] + dfpreds[variable + '_prob_1'] + dfpreds[variable + '_prob_2']

    probability_diagnostics.append({
        'variable': variable,
        'minimum': probability_sum.min(),
        'maximum': probability_sum.max(),
        'mean_absolute_deviation_from_one': np.mean(np.abs(probability_sum - 1.0))
    })

probability_diagnostics = pd.DataFrame(probability_diagnostics)
probability_diagnostics.to_csv(shapout / 'FewShot_Probability_Diagnostics.csv', index=False)


'''
Change name to abs pres and mis for 0, 1, and 2
'''

rename_dict = {}

for col in dfpreds.columns:

    if col.endswith('_prob_0'):
        rename_dict[col] = col.replace('_prob_0', '_abs')

    elif col.endswith('_prob_1'):
        rename_dict[col] = col.replace('_prob_1', '_pres')

    elif col.endswith('_prob_2'):
        rename_dict[col] = col.replace('_prob_2', '_mis')

dfpreds = dfpreds.rename(columns=rename_dict)


'''
Construct threshold-specific analysis datasets
'''

analysis_data_by_threshold = {}

for missing_threshold in MISSING_PROBABILITY_THRESHOLDS:
    threshold_key = format(missing_threshold, '.2f')
    analysis_data_by_threshold[threshold_key] = create_analysis_data_for_threshold(dfpreds, dftops, missing_threshold)

    threshold_dataframe = analysis_data_by_threshold[threshold_key]['dfall']
    threshold_suffix = threshold_key.replace('.', '')

    threshold_dataframe.to_csv(shapout / ('ATopic_Prob_Threshold_' + threshold_suffix + '.csv'), index=False)

    with open(shapout / ('Changed_Prob_Threshold_' + threshold_suffix + '.p'), 'wb') as output_file:
        pickle.dump(threshold_dataframe, output_file)

primary_threshold_key = format(PRIMARY_MISSING_PROBABILITY_THRESHOLD, '.2f')
primary_analysis_data = analysis_data_by_threshold[primary_threshold_key]

dfall = primary_analysis_data['dfall']
depvars = primary_analysis_data['depvars']
indepvars = primary_analysis_data['indepvars']
depvars_scaled = primary_analysis_data['depvars_scaled']
indepvars_scaled = primary_analysis_data['indepvars_scaled']
dfcertainty = primary_analysis_data['certainty']

dfall.to_csv(shapout / 'Changed_Prob.csv', index=False)

with open(shapout / 'Changed_Prob.p', 'wb') as output_file:
    pickle.dump(dfall, output_file)

print('Matched case-study papers:', len(dfall))
print('Primary missing-probability threshold:', PRIMARY_MISSING_PROBABILITY_THRESHOLD)


'''
Assess probabilty distribution per threshold per variable
'''

colors = {
    '0.50': 'steelblue',
    '0.66': 'darkorange',
    '0.75': 'grey'
}
   
fig, axes = plt.subplots(3, 5, figsize=(18, 10))

thresholds = ['0.50', '0.66', '0.75']

all_variables = outcome_columns + feature_columns

for i, var in enumerate(all_variables):

    ax = axes.flat[i]

    for th in thresholds:

        df = analysis_data_by_threshold[th]

        if var in df['indepvars_scaled'].columns:
            vals = df['indepvars_scaled'][var]
        else:
            vals = df['depvars_scaled'][var]

        ax.hist(vals, bins=25, density=True, alpha=0.15, color=colors[th], edgecolor=colors[th], linewidth=2,label=f'Th={th}')

    ax.set_title(var)

axes.flat[0].legend()
plt.tight_layout()
fig.savefig(mainres / 'VarsDistribution.pdf', bbox_inches='tight')




'''
Aggregate XGBoost and SHAP analysis
'''

xgboptuna = {}
xgb_best_parameters = {}
xgbmodels = {}

shapexpldict = {}
shapdict = {}
shapdict_std = {}

shap_signed_sum = {}
shap_signed_mean = {}
shap_mean_absolute = {}

shapinterdict = {}
shapinterdict_std = {}

diagnostic = {}

for outcome_name in outcome_columns:
    print('\nAggregate outcome:', outcome_name)

    outcome_values = depvars_scaled[outcome_name]

    study, best_parameters = optimize_xgboost(features=indepvars_scaled, labels=outcome_values, number_of_trials=N_OPTUNA_TRIALS_AGGREGATE, number_of_folds=AGGREGATE_CV_FOLDS, study_name='aggregate_' + outcome_name, topic_model=False)

    xgboptuna[outcome_name] = study
    xgb_best_parameters[outcome_name] = best_parameters

    results = repeated_xgboost_shap(features=indepvars_scaled, labels=outcome_values, best_parameters=best_parameters, number_of_models=N_REPEATED_MODELS_AGGREGATE, number_of_cv_folds=AGGREGATE_CV_FOLDS, analysis_name=outcome_name)

    xgbmodels[outcome_name] = results['models']
    diagnostic[outcome_name] = results['diagnostics']
    shapexpldict[outcome_name] = results['aggregated_explanation']
    shapdict[outcome_name] = results['mean_local_shap']
    shapdict_std[outcome_name] = results['std_local_shap']
    shap_signed_sum[outcome_name] = results['signed_sum_by_model']
    shap_signed_mean[outcome_name] = results['signed_mean_by_model']
    shap_mean_absolute[outcome_name] = results['mean_absolute_by_model']
    shapinterdict[outcome_name] = results['mean_interactions']
    shapinterdict_std[outcome_name] = results['std_interactions']


'''
Aggregate diagnostics
'''

dfdiagnostic = pd.DataFrame.from_dict(diagnostic, orient='index')
dfdiagnostic.to_csv(shapout / 'Diagnostic_All.csv')


'''
Aggregate signed SHAP summaries
'''

dfshap_signed_sum_mean = pd.DataFrame({outcome: np.mean(shap_signed_sum[outcome], axis=0) for outcome in outcome_columns}).T
dfshap_signed_sum_mean.columns = feature_columns

dfshap_signed_sum_median = pd.DataFrame({outcome: np.median(shap_signed_sum[outcome], axis=0) for outcome in outcome_columns}).T
dfshap_signed_sum_median.columns = feature_columns

dfshap_signed_sum_std = pd.DataFrame({outcome: np.std(shap_signed_sum[outcome], axis=0) for outcome in outcome_columns}).T
dfshap_signed_sum_std.columns = feature_columns

dfshap_signed_mean = pd.DataFrame({outcome: np.mean(shap_signed_mean[outcome], axis=0) for outcome in outcome_columns}).T
dfshap_signed_mean.columns = feature_columns

dfshap_absolute_mean = pd.DataFrame({outcome: np.mean(shap_mean_absolute[outcome], axis=0) for outcome in outcome_columns}).T
dfshap_absolute_mean.columns = feature_columns

# Preserve historical output names.
dfshap_signed_sum_std.to_csv(shapout / 'Shap_All_StDev.csv')
dfshap_signed_sum_median.to_csv(shapout / 'Shap_All_Median.csv')
dfshap_signed_sum_mean.to_csv(shapout / 'Shap_All_Mean.csv')

# Explicitly named outputs.
dfshap_signed_sum_mean.to_csv(shapout / 'SHAP_Signed_Sum_Mean.csv')
dfshap_signed_mean.to_csv(shapout / 'SHAP_Signed_Mean_Per_Document.csv')
dfshap_absolute_mean.to_csv(shapout / 'SHAP_Mean_Absolute.csv')



'''
Signed SHAP directional figure  --> Main Paper
'''

figure_outcome_order = ['RSC', 'IEQ', 'CFL']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for axis, outcome_name in zip(axes, figure_outcome_order):
    signed_values = dfshap_signed_sum_mean.loc[outcome_name].sort_values(ascending=False)
    colors = ['steelblue' if value >= 0 else 'indianred' for value in signed_values.values]

    axis.bar(signed_values.index, signed_values.values, color=colors)
    axis.axhline(0, color='black', linewidth=0.8)
    axis.set_title(outcome_name, fontsize=30)
    axis.set_xlabel('')
    axis.set_ylabel('')
    axis.tick_params(axis='x', rotation=90, labelsize=26)
    axis.tick_params(axis='y', labelsize=26)

fig.tight_layout()
fig.savefig(allout / 'Pos_Neg_OverallEffects2.pdf', bbox_inches='tight')
fig.savefig(allout / 'Pos_Neg_OverallEffects2.png', dpi=300, bbox_inches='tight')
plt.close(fig)


'''
Aggregate SHAP distribution figure
'''

long_frames = []

for outcome_name in outcome_columns:
    current_dataframe = pd.DataFrame(shap_signed_sum[outcome_name], columns=feature_columns)
    current_dataframe['DV'] = outcome_name
    long_frames.append(current_dataframe)

long_dataframe = pd.concat(long_frames, ignore_index=True)
melted_dataframe = long_dataframe.melt(id_vars=['DV'], var_name='Feature', value_name='SHAP Value')

fig, axis = plt.subplots(figsize=(14, 7))

sns.boxplot(data=melted_dataframe, x='Feature', y='SHAP Value', hue='DV', showmeans=True, meanprops={'marker': '+', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 5}, ax=axis)

axis.set_ylabel('Signed SHAP Sum', fontsize=28)
axis.set_xlabel('Feature', fontsize=28)
axis.tick_params(axis='x', rotation=90, labelsize=26)
axis.tick_params(axis='y', labelsize=26)
axis.grid(axis='y', linestyle='--', alpha=0.7)
axis.legend(title='Outcome', ncol=3)

fig.tight_layout()
fig.savefig(allout / 'Boxplot_All_DVs.pdf')
plt.close(fig)


'''
Aggregate SHAP heatmap
'''

cmap_colors = [(0.0, 'blue'), (0.45, 'lightblue'), (0.499, 'white'), (0.501, 'white'), (0.55, 'coral'), (1.0, 'red')]
cmapshap = LinearSegmentedColormap.from_list('custom_RdYlBu', cmap_colors)

dfshap_effects = dfshap_signed_sum_mean.loc[figure_outcome_order]
maximum_effect = np.max(np.abs(dfshap_effects.to_numpy()))

if maximum_effect == 0:
    maximum_effect = 1.0

fig, axis = plt.subplots(figsize=(10, 8))

sns.heatmap(dfshap_effects, cmap=cmapshap, annot=True, fmt='.2f', cbar=True, linewidths=0.5, center=0, vmin=-maximum_effect, vmax=maximum_effect, ax=axis)

axis.set_yticklabels(axis.get_yticklabels(), fontsize=26, rotation=0)
axis.set_xticklabels(axis.get_xticklabels(), fontsize=26, rotation=90)

fig.tight_layout()
fig.savefig(allout / 'ShapEffects_All.pdf')
plt.close(fig)


'''
Aggregate SHAP explanation plots
'''

for outcome_name in outcome_columns:
    explanation = shapexpldict[outcome_name]

    plt.figure()
    shap.plots.violin(explanation, max_display=len(feature_columns), show=False)
    plt.title(outcome_name)
    plt.savefig(allout / ('Violin_' + outcome_name + '.pdf'), bbox_inches='tight')
    plt.close()

    feature_clustering = shap.utils.hclust(indepvars_scaled, depvars_scaled[outcome_name], linkage='single')

    fig, axis = plt.subplots(figsize=(10, 8))
    shap.plots.bar(explanation, clustering=feature_clustering, clustering_cutoff=1, max_display=len(feature_columns), show=False, ax=axis)
    fig.suptitle(outcome_name)
    fig.savefig(allout / ('BarClust_' + outcome_name + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    
'''
Sensitivity Analysis for different threshold for the 0.5 assignment
'''

shap_signed_sum_threshold = {}
shap_absolute_threshold = {}
diagnostic_threshold = {}


for threshold_key in sorted(analysis_data_by_threshold.keys()):

    threshold_data = analysis_data_by_threshold[threshold_key]

    threshold_depvars = threshold_data['depvars_scaled']
    threshold_indepvars = threshold_data['indepvars_scaled']

    for outcome_name in outcome_columns:

        study, best_parameters = optimize_xgboost(
            features=threshold_indepvars,
            labels=threshold_depvars[outcome_name],
            number_of_trials=N_OPTUNA_TRIALS_AGGREGATE,
            number_of_folds=AGGREGATE_CV_FOLDS,
            study_name=f"threshold_{threshold_key}_{outcome_name}",
            topic_model=False
        )

        results = repeated_xgboost_shap(
            features=threshold_indepvars,
            labels=threshold_depvars[outcome_name],
            best_parameters=best_parameters,
            number_of_models=N_REPEATED_MODELS_AGGREGATE,
            number_of_cv_folds=AGGREGATE_CV_FOLDS,
            analysis_name=f"{threshold_key}_{outcome_name}"
        )

        analysis_key = (threshold_key, outcome_name)

        shap_signed_sum_threshold[analysis_key] = results['signed_sum_by_model']
        shap_absolute_threshold[analysis_key] = results['mean_absolute_by_model']
        diagnostic_threshold[analysis_key] = results['diagnostics']

dfdiagnostic_threshold = pd.DataFrame.from_dict(diagnostic_threshold, orient='index')

dfdiagnostic_threshold.index = pd.MultiIndex.from_tuples(dfdiagnostic_threshold.index, names=['Threshold', 'Outcome'])

dfdiagnostic_threshold.to_csv(shapout / 'Diagnostic_Thresholds.csv')

#create one dataframe for sensitivity
rows = []

for threshold_key in sorted(analysis_data_by_threshold.keys()):

    for outcome_name in outcome_columns:

        mean_signed = np.mean(
            shap_signed_sum_threshold[(threshold_key, outcome_name)],
            axis=0
        )

        for feature, value in zip(feature_columns, mean_signed):

            rows.append({
                'Threshold': threshold_key,
                'Outcome': outcome_name,
                'Feature': feature,
                'SignedSHAP': value
            })

dfshap_threshold = pd.DataFrame(rows)
dfshap_threshold.to_csv(shapout / 'SHAP_Threshold_Sensitivity.csv', index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, outcome_name in enumerate(depvars):

    ax = axes[i]

    subset = dfshap_threshold[dfshap_threshold['Outcome'] == outcome_name]

    sns.barplot(data=subset, x='Feature', y='SignedSHAP', hue='Threshold', ax=ax)

    ax.set_title(outcome_name, fontsize=30)
    ax.tick_params(axis='x', labelsize=26, rotation=90)
    ax.tick_params(axis='y', labelsize=26)

    ax.set_xlabel('')
    ax.set_ylabel('Mean Signed SHAP', fontsize=26)

plt.tight_layout()
fig.savefig(allout / ('Bar_sensitivity.pdf'), bbox_inches='tight')
plt.show()


'''
Topic-specific XGBoost and SHAP analysis
'''

xgboptuna_topic = {}
xgb_best_parameters_topic = {}
xgbmodels_topic = {}

shapexpldict_topic = {}
shapdict_topic = {}
shapdict_topic_std = {}

shap_signed_sum_topic = {}
shap_signed_mean_topic = {}
shap_mean_absolute_topic = {}

shapinterdict_topic = {}
shapinterdict_std_topic = {}

diagnostic_topic = {}

for topic_name in sorted(dfall['topname'].dropna().unique()):
    topic_mask = dfall['topname'] == topic_name
    topic_features = indepvars_scaled[topic_mask.to_numpy()].reset_index(drop=True)
    topic_outcomes = depvars_scaled[topic_mask.to_numpy()].reset_index(drop=True)

    if len(topic_features) < 20:
        print('Skipping small topic:', topic_name)
        continue

    valid_topic_folds = min(TOPIC_CV_FOLDS, len(topic_features))

    for outcome_name in outcome_columns:
        analysis_key = (topic_name, outcome_name)

        print('\nTopic analysis:', analysis_key)

        study, best_parameters = optimize_xgboost(features=topic_features, labels=topic_outcomes[outcome_name], number_of_trials=N_OPTUNA_TRIALS_TOPIC, number_of_folds=valid_topic_folds, study_name='topic_' + safe_filename(topic_name) + '_' + outcome_name, topic_model=True)

        xgboptuna_topic[analysis_key] = study
        xgb_best_parameters_topic[analysis_key] = best_parameters

        results = repeated_xgboost_shap(features=topic_features, labels=topic_outcomes[outcome_name], best_parameters=best_parameters, number_of_models=N_REPEATED_MODELS_TOPIC, number_of_cv_folds=valid_topic_folds, analysis_name=str(topic_name) + '_' + outcome_name)

        xgbmodels_topic[analysis_key] = results['models']
        diagnostic_topic[analysis_key] = results['diagnostics']
        shapexpldict_topic[analysis_key] = results['aggregated_explanation']
        shapdict_topic[analysis_key] = results['mean_local_shap']
        shapdict_topic_std[analysis_key] = results['std_local_shap']
        shap_signed_sum_topic[analysis_key] = results['signed_sum_by_model']
        shap_signed_mean_topic[analysis_key] = results['signed_mean_by_model']
        shap_mean_absolute_topic[analysis_key] = results['mean_absolute_by_model']
        shapinterdict_topic[analysis_key] = results['mean_interactions']
        shapinterdict_std_topic[analysis_key] = results['std_interactions']


'''
Topic-specific diagnostic tables
'''

if diagnostic_topic:
    dfdiagnostic_topic = pd.DataFrame.from_dict(diagnostic_topic, orient='index')
    dfdiagnostic_topic.index = pd.MultiIndex.from_tuples(dfdiagnostic_topic.index, names=['Topic', 'Outcome'])
else:
    dfdiagnostic_topic = pd.DataFrame()

dfdiagnostic_topic.to_csv(shapout / 'Diagnostic_Topic.csv')

if shap_signed_sum_topic:
    dfshap_topic_mean = pd.DataFrame({key: np.mean(values, axis=0) for key, values in shap_signed_sum_topic.items()}).T
    dfshap_topic_median = pd.DataFrame({key: np.median(values, axis=0) for key, values in shap_signed_sum_topic.items()}).T
    dfshap_topic_std = pd.DataFrame({key: np.std(values, axis=0) for key, values in shap_signed_sum_topic.items()}).T
    dfshap_topic_absolute = pd.DataFrame({key: np.mean(values, axis=0) for key, values in shap_mean_absolute_topic.items()}).T

    for dataframe in [dfshap_topic_mean, dfshap_topic_median, dfshap_topic_std, dfshap_topic_absolute]:
        dataframe.columns = feature_columns
        dataframe.index = pd.MultiIndex.from_tuples(dataframe.index, names=['Topic', 'Outcome'])
else:
    empty_topic_index = pd.MultiIndex.from_tuples([], names=['Topic', 'Outcome'])
    dfshap_topic_mean = pd.DataFrame(columns=feature_columns, index=empty_topic_index)
    dfshap_topic_median = pd.DataFrame(columns=feature_columns, index=empty_topic_index)
    dfshap_topic_std = pd.DataFrame(columns=feature_columns, index=empty_topic_index)
    dfshap_topic_absolute = pd.DataFrame(columns=feature_columns, index=empty_topic_index)

dfshap_topic_std.to_csv(shapout / 'Shap_Topic_StDev.csv')
dfshap_topic_median.to_csv(shapout / 'Shap_Topic_Median.csv')
dfshap_topic_mean.to_csv(shapout / 'Shap_Topic_Mean.csv')
dfshap_topic_absolute.to_csv(shapout / 'Shap_Topic_Absolute_Mean.csv')


'''
Within vs across outcome shap profile distances -> Main Paper
'''

records = []

for i in range(len(dfshap_topic_mean)):
    for j in range(i + 1, len(dfshap_topic_mean)):

        idx1 = dfshap_topic_mean.index[i]
        idx2 = dfshap_topic_mean.index[j]

        topic1, outcome1 = idx1
        topic2, outcome2 = idx2

        dist = cosine(dfshap_topic_mean.iloc[i], dfshap_topic_mean.iloc[j])

        comparison = ("Within outcome" if outcome1 == outcome2 else "Within S-E Context")

        records.append([topic1, outcome1, topic2, outcome2, dist, comparison])

dfdist = pd.DataFrame(records,columns=["Topic1","Outcome1", "Topic2", "Outcome2", "Distance", "Comparison"])

plt.figure(figsize=(6,6))

sns.violinplot(data=dfdist, x="Comparison", y="Distance",inner=None,cut=0,alpha=0.5)
sns.boxplot(data=dfdist, x="Comparison", y="Distance", width=0.5, showfliers=False)
#sns.stripplot(data=dfdist, x="Comparison", y="Distance", color="black", alpha=0.5,jitter=0.15)

plt.xlabel("Comparison type", fontsize=18)
plt.ylabel("Cosine distance", fontsize=18)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.xlabel("")
plt.tight_layout()

plt.savefig(shapout / "SHAP_Outcome_vs_Context_Distance.pdf", bbox_inches="tight")
plt.close()


''' 
Topic specific SHAP in one figure with panels
'''


topic_labels = {
    'Biodiversity Conservation': 'Biodiversity\nConservation',
    'Fisheries': 'Fisheries',
    'Forestry and Livelihoods': 'Forestry\nand Livelihoods',
    'Institutions and Governance': 'Institutions\nand Governance',
    'Land Rights': 'Land Rights',
    'Mixed': 'Mixed',
    'Water Governance': 'Water\nGovernance'
}

outcomes = figure_outcome_order
fig, axes = plt.subplots(len(topic_labels), len(outcomes), figsize=(18, 24), sharey=True)

for r, topic_name in enumerate(topic_labels):

    topic_table = dfshap_topic_mean.xs(topic_name, level='Topic')

    for c, outcome_name in enumerate(outcomes):

        ax = axes[r, c]

        if outcome_name not in topic_table.index:
            ax.axis("off")
            continue

        signed_values = topic_table.loc[outcome_name].sort_values(ascending=False)
        colors = ['steelblue' if v >= 0 else 'indianred' for v in signed_values.values]
        ax.bar(signed_values.index, signed_values.values, color=colors)
        ax.axhline(0, color='black', linewidth=0.8)

        if r == 0:
            ax.set_title(outcome_name, fontsize=20)
        if c == 0:
            ax.set_ylabel(topic_labels[topic_name], fontsize=18)

        ax.tick_params(axis='x', rotation=90, labelsize=20)

plt.tight_layout()
plt.savefig(shapout / "Topic_SHAP_AllPanels.pdf", bbox_inches="tight")
plt.close()



'''
Context specific signed SHAP figures - one per context
'''

for topic_name in dfshap_topic_mean.index.get_level_values('Topic').unique():
    topic_table = dfshap_topic_mean.xs(topic_name, level='Topic')
    existing_outcomes = [outcome_name for outcome_name in figure_outcome_order if outcome_name in topic_table.index]

    if not existing_outcomes:
        continue

    fig, axes = plt.subplots(1, len(existing_outcomes), figsize=(len(existing_outcomes) * 5, 5), squeeze=False)

    for axis, outcome_name in zip(axes[0], existing_outcomes):
        signed_values = topic_table.loc[outcome_name].sort_values(ascending=False)
        colors = ['steelblue' if value >= 0 else 'indianred' for value in signed_values.values]

        axis.bar(signed_values.index, signed_values.values, color=colors)
        axis.axhline(0, color='black', linewidth=0.8)
        axis.set_title(outcome_name, fontsize=28)
        axis.tick_params(axis='x', rotation=90, labelsize=26)

    fig.suptitle(topic_name, fontsize=26)
    fig.tight_layout()
    fig.savefig(shapout / ('Pos_Neg_OverallEffects_' + safe_filename(topic_name) + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    

'''
Context Boxplots
'''

topic_long_frames = []

for (topic_name, outcome_name), values in shap_signed_sum_topic.items():

    df_current = pd.DataFrame(
        values,
        columns=feature_columns
    )

    df_current["Topic"] = topic_name
    df_current["Outcome"] = outcome_name

    topic_long_frames.append(df_current)

topic_long = pd.concat(topic_long_frames, ignore_index=True)
topic_long = topic_long.melt(id_vars=["Topic", "Outcome"], var_name="Feature", value_name="SHAP Value")


fig, axes = plt.subplots(4, 2, figsize=(18, 18), sharey=True)
axes = axes.flatten()

for ax, topic_name in zip(axes, topic_labels.keys()):

    subset = topic_long[topic_long["Topic"] == topic_name]
                        
    sns.boxplot(data=subset, x="Feature", y="SHAP Value", hue="Outcome", showfliers=False, ax=ax)

    ax.set_title(topic_labels[topic_name], fontsize=18)
    ax.tick_params(axis="x", rotation=90, labelsize=18)
    ax.tick_params(axis="y", labelsize=10)

    ax.set_xlabel("")
    ax.set_ylabel("")
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()

for ax in axes:
    legend = ax.get_legend()
    if legend:
        legend.remove()

fig.legend(handles,labels, loc="upper center", ncol=3, fontsize=14)
plt.tight_layout()
fig.savefig(shapout / "Boxplot_Topic_SHAP.pdf", bbox_inches="tight")


'''
Mutual information analysis
'''

aggregate_mutual_information = {}

for outcome_name in outcome_columns:
    mutual_information_scores = mutual_info_regression(X=indepvars_scaled, y=depvars_scaled[outcome_name], random_state=RANDOM_SEED)
    aggregate_mutual_information[outcome_name] = dict(zip(feature_columns, mutual_information_scores))

dfinfo = pd.DataFrame.from_dict(aggregate_mutual_information)
dfinfo.to_csv(miout / 'Mutual_Information_Outcomes.csv')

fig, axis = plt.subplots(figsize=(8, 8))
sns.heatmap(dfinfo, cmap='coolwarm', annot=True, fmt='.3f', ax=axis)
fig.tight_layout()
fig.savefig(miout / 'InfoDeps_heatmap.pdf', bbox_inches='tight')
plt.close(fig)


'''
Pairwise predictor mutual information
'''

pairwise_mutual_information = pd.DataFrame(0.0, index=feature_columns, columns=feature_columns)

for first_feature in feature_columns:
    for second_feature in feature_columns:
        if first_feature == second_feature:
            pairwise_mutual_information.loc[first_feature, second_feature] = 0.0
        else:
            pairwise_mutual_information.loc[first_feature, second_feature] = mutual_info_regression(X=indepvars_scaled[[first_feature]], y=indepvars_scaled[second_feature], random_state=RANDOM_SEED)[0]

pairwise_mutual_information = (pairwise_mutual_information + pairwise_mutual_information.T) / 2.0
pairwise_mutual_information.to_csv(miout / 'Mutual_Information_Predictors.csv')

fig, axis = plt.subplots(figsize=(10, 8))
sns.heatmap(pairwise_mutual_information, cmap='coolwarm', annot=True, fmt='.3f', ax=axis)
fig.tight_layout()
fig.savefig(miout / 'InfoIndeps_heatmap.pdf', bbox_inches='tight')
plt.close(fig)



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

LASSO

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



'''
Aggregate interaction selection via LASSO
'''

lasso_dict = {}
lasso_models = {}
lasso_diagnostics = {}
lasso_stability_tables = {}
lasso_mse_dict = {}

for outcome_name in outcome_columns:
    print('\nAggregate LASSO:', outcome_name)

    lasso_results = fit_repeated_lasso_interactions(features=indepvars_scaled, outcome=depvars_scaled[outcome_name], analysis_name=outcome_name, number_of_repeats=LASSO_REPEATS_AGGREGATE, number_of_cv_folds=AGGREGATE_CV_FOLDS)

    lasso_models[outcome_name] = lasso_results['pipeline']
    lasso_diagnostics[outcome_name] = lasso_results['diagnostics']
    lasso_stability_tables[outcome_name] = lasso_results['stability_table']
    lasso_mse_dict[outcome_name] = lasso_results['diagnostics']['oof_mse']

    selected_coefficients = lasso_results['full_coefficients']
    selected_coefficients = selected_coefficients[np.abs(selected_coefficients) > LASSO_SELECTION_TOLERANCE]

    lasso_dict[outcome_name] = list(zip(selected_coefficients.index, selected_coefficients.values))

    lasso_results['stability_table'].to_csv(lassoout / ('LASSO_Stability_' + outcome_name + '.csv'), index=False)

df_lasso_diagnostics = pd.DataFrame.from_dict(lasso_diagnostics, orient='index')
df_lasso_diagnostics.to_csv(lassoout / 'LASSO_Diagnostics_All.csv')


'''
Topic-specific interaction selection via LASSO
'''

topic_lasso_dict = {}
topic_lasso_models = {}
topic_lasso_diagnostics = {}
topic_lasso_stability_tables = {}
topic_mse_dict = {}

for topic_name in sorted(dfall['topname'].dropna().unique()):
    topic_mask = dfall['topname'] == topic_name
    topic_features = indepvars_scaled[topic_mask.to_numpy()].reset_index(drop=True)
    topic_outcomes = depvars_scaled[topic_mask.to_numpy()].reset_index(drop=True)

    if len(topic_features) < 20:
        continue

    valid_topic_folds = min(TOPIC_CV_FOLDS, len(topic_features))

    for outcome_name in outcome_columns:
        analysis_key = (topic_name, outcome_name)
        analysis_name = safe_filename(topic_name) + '_' + outcome_name

        print('\nTopic LASSO:', analysis_key)

        lasso_results = fit_repeated_lasso_interactions(features=topic_features, outcome=topic_outcomes[outcome_name], analysis_name=analysis_name, number_of_repeats=LASSO_REPEATS_TOPIC, number_of_cv_folds=valid_topic_folds)

        topic_lasso_models[analysis_key] = lasso_results['pipeline']
        topic_lasso_diagnostics[analysis_key] = lasso_results['diagnostics']
        topic_lasso_stability_tables[analysis_key] = lasso_results['stability_table']
        topic_mse_dict[analysis_key] = lasso_results['diagnostics']['oof_mse']

        selected_coefficients = lasso_results['full_coefficients']
        selected_coefficients = selected_coefficients[np.abs(selected_coefficients) > LASSO_SELECTION_TOLERANCE]

        topic_lasso_dict[analysis_key] = list(zip(selected_coefficients.index, selected_coefficients.values))

        lasso_results['stability_table'].to_csv(lassoout / ('LASSO_Stability_' + analysis_name + '.csv'), index=False)

df_topic_lasso_diagnostics = pd.DataFrame.from_dict(topic_lasso_diagnostics, orient='index')

if len(df_topic_lasso_diagnostics) > 0:
    df_topic_lasso_diagnostics.index = pd.MultiIndex.from_tuples(df_topic_lasso_diagnostics.index, names=['Topic', 'Outcome'])

df_topic_lasso_diagnostics.to_csv(lassoout / 'LASSO_Diagnostics_Topic.csv')


'''
Missing-threshold sensitivity LASSO
'''

threshold_lasso_dict = {}
threshold_lasso_diagnostics = {}
threshold_lasso_stability_tables = {}

threshold_topic_lasso_dict = {}
threshold_topic_lasso_diagnostics = {}
threshold_topic_lasso_stability_tables = {}

for threshold_key in sorted(analysis_data_by_threshold.keys()):
    threshold_data = analysis_data_by_threshold[threshold_key]

    print('\nMissing threshold LASSO:', threshold_key)

    if threshold_key == primary_threshold_key:
        print('Reusing primary LASSO results for threshold:', threshold_key)

        for outcome_name in outcome_columns:
            threshold_analysis_key = (threshold_key, outcome_name)
            threshold_lasso_dict[threshold_analysis_key] = lasso_dict[outcome_name]
            threshold_lasso_diagnostics[threshold_analysis_key] = lasso_diagnostics[outcome_name]
            threshold_lasso_stability_tables[threshold_analysis_key] = lasso_stability_tables[outcome_name]

        for primary_topic_key in topic_lasso_dict:
            topic_name, outcome_name = primary_topic_key
            threshold_topic_key = (threshold_key, topic_name, outcome_name)

            threshold_topic_lasso_dict[threshold_topic_key] = topic_lasso_dict[primary_topic_key]
            threshold_topic_lasso_diagnostics[threshold_topic_key] = topic_lasso_diagnostics[primary_topic_key]
            threshold_topic_lasso_stability_tables[threshold_topic_key] = topic_lasso_stability_tables[primary_topic_key]

        continue

    threshold_dfall = threshold_data['dfall'].copy()
    threshold_depvars = threshold_data['depvars_scaled'].copy()
    threshold_indepvars = threshold_data['indepvars_scaled'].copy()

    if len(threshold_dfall) != len(threshold_depvars) or len(threshold_dfall) != len(threshold_indepvars):
        raise ValueError('Threshold ' + threshold_key + ' has inconsistent dataframe lengths.')

    for outcome_name in outcome_columns:
        analysis_key = (threshold_key, outcome_name)
        analysis_name = 'threshold_' + threshold_key.replace('.', '') + '_' + outcome_name

        lasso_results = fit_repeated_lasso_interactions(
            features=threshold_indepvars,
            outcome=threshold_depvars[outcome_name],
            analysis_name=analysis_name,
            number_of_repeats=LASSO_REPEATS_AGGREGATE,
            number_of_cv_folds=AGGREGATE_CV_FOLDS
        )

        threshold_lasso_diagnostics[analysis_key] = lasso_results['diagnostics']
        threshold_lasso_stability_tables[analysis_key] = lasso_results['stability_table']

        selected_coefficients = lasso_results['full_coefficients']
        selected_coefficients = selected_coefficients[np.abs(selected_coefficients) > LASSO_SELECTION_TOLERANCE]
        threshold_lasso_dict[analysis_key] = list(zip(selected_coefficients.index, selected_coefficients.values))

        lasso_results['stability_table'].to_csv(lassoout / ('LASSO_Stability_' + analysis_name + '.csv'), index=False)

    for topic_name in sorted(threshold_dfall['topname'].dropna().unique()):
        topic_mask = threshold_dfall['topname'] == topic_name
        threshold_topic_features = threshold_indepvars[topic_mask.to_numpy()].reset_index(drop=True)
        threshold_topic_outcomes = threshold_depvars[topic_mask.to_numpy()].reset_index(drop=True)

        if len(threshold_topic_features) < 20:
            print('Skipping small threshold topic:', threshold_key, topic_name)
            continue

        valid_topic_folds = min(TOPIC_CV_FOLDS, len(threshold_topic_features))

        for outcome_name in outcome_columns:
            analysis_key = (threshold_key, topic_name, outcome_name)
            analysis_name = 'threshold_' + threshold_key.replace('.', '') + '_' + safe_filename(topic_name) + '_' + outcome_name

            lasso_results = fit_repeated_lasso_interactions(
                features=threshold_topic_features,
                outcome=threshold_topic_outcomes[outcome_name],
                analysis_name=analysis_name,
                number_of_repeats=LASSO_REPEATS_TOPIC,
                number_of_cv_folds=valid_topic_folds
            )

            threshold_topic_lasso_diagnostics[analysis_key] = lasso_results['diagnostics']
            threshold_topic_lasso_stability_tables[analysis_key] = lasso_results['stability_table']

            selected_coefficients = lasso_results['full_coefficients']
            selected_coefficients = selected_coefficients[np.abs(selected_coefficients) > LASSO_SELECTION_TOLERANCE]
            threshold_topic_lasso_dict[analysis_key] = list(zip(selected_coefficients.index, selected_coefficients.values))

            lasso_results['stability_table'].to_csv(lassoout / ('LASSO_Stability_' + analysis_name + '.csv'), index=False)



'''
Save XGBoost and Optuna results
'''

with open(shapout / 'XGB_Optuna_All.p', 'wb') as output_file:
    pickle.dump(xgboptuna, output_file)

with open(shapout / 'XGB_Best_Parameters_All.p', 'wb') as output_file:
    pickle.dump(xgb_best_parameters, output_file)

with open(shapout / 'XGB_Optuna_Topic.p', 'wb') as output_file:
    pickle.dump(xgboptuna_topic, output_file)

with open(shapout / 'XGB_Best_Parameters_Topic.p', 'wb') as output_file:
    pickle.dump(xgb_best_parameters_topic, output_file)

if SAVE_XGBOOST_MODELS:
    with open(shapout / 'XGB_Models_All.p', 'wb') as output_file:
        pickle.dump(xgbmodels, output_file)

    with open(shapout / 'XGB_Models_Topic.p', 'wb') as output_file:
        pickle.dump(xgbmodels_topic, output_file)


'''
Save SHAP results
'''

with open(shapout / 'SHAP_Explanations_All.p', 'wb') as output_file:
    pickle.dump(shapexpldict, output_file)

with open(shapout / 'SHAP_Local_Mean_All.p', 'wb') as output_file:
    pickle.dump(shapdict, output_file)

with open(shapout / 'SHAP_Local_StDev_All.p', 'wb') as output_file:
    pickle.dump(shapdict_std, output_file)

with open(shapout / 'SHAP_Signed_Sums_All.p', 'wb') as output_file:
    pickle.dump(shap_signed_sum, output_file)

with open(shapout / 'SHAP_Signed_Means_All.p', 'wb') as output_file:
    pickle.dump(shap_signed_mean, output_file)

with open(shapout / 'SHAP_Mean_Absolute_All.p', 'wb') as output_file:
    pickle.dump(shap_mean_absolute, output_file)

with open(shapout / 'SHAP_Explanations_Topic.p', 'wb') as output_file:
    pickle.dump(shapexpldict_topic, output_file)

with open(shapout / 'SHAP_Signed_Sums_Topic.p', 'wb') as output_file:
    pickle.dump(shap_signed_sum_topic, output_file)

with open(shapout / 'SHAP_Mean_Absolute_Topic.p', 'wb') as output_file:
    pickle.dump(shap_mean_absolute_topic, output_file)

if SAVE_SHAP_INTERACTIONS:
    with open(shapout / 'SHAP_Interactions_All.p', 'wb') as output_file:
        pickle.dump(shapinterdict, output_file)

    with open(shapout / 'SHAP_Interactions_StDev_All.p', 'wb') as output_file:
        pickle.dump(shapinterdict_std, output_file)

    with open(shapout / 'SHAP_Interactions_Topic.p', 'wb') as output_file:
        pickle.dump(shapinterdict_topic, output_file)

    with open(shapout / 'SHAP_Interactions_StDev_Topic.p', 'wb') as output_file:
        pickle.dump(shapinterdict_std_topic, output_file)


'''
Save LASSO results
'''

with open(lassoout / 'LASSO_All.p', 'wb') as output_file:
    pickle.dump(lasso_dict, output_file)

with open(lassoout / 'LASSO_Models_All.p', 'wb') as output_file:
    pickle.dump(lasso_models, output_file)

with open(lassoout / 'LASSO_Stability_All.p', 'wb') as output_file:
    pickle.dump(lasso_stability_tables, output_file)

with open(lassoout / 'LASSO_All_MSE.p', 'wb') as output_file:
    pickle.dump(lasso_mse_dict, output_file)

with open(lassoout / 'LASSO_Topic.p', 'wb') as output_file:
    pickle.dump(topic_lasso_dict, output_file)

with open(lassoout / 'LASSO_Models_Topic.p', 'wb') as output_file:
    pickle.dump(topic_lasso_models, output_file)

with open(lassoout / 'LASSO_Stability_Topic.p', 'wb') as output_file:
    pickle.dump(topic_lasso_stability_tables, output_file)

with open(lassoout / 'LASSO_Topic_MSE.p', 'wb') as output_file:
    pickle.dump(topic_mse_dict, output_file)

'''
Save threshold LASSO results
'''

with open(lassoout / 'LASSO_Threshold_All.p', 'wb') as output_file:
    pickle.dump(threshold_lasso_dict, output_file)

with open(lassoout / 'LASSO_Threshold_Stability_All.p', 'wb') as output_file:
    pickle.dump(threshold_lasso_stability_tables, output_file)

with open(lassoout / 'LASSO_Threshold_Topic.p', 'wb') as output_file:
    pickle.dump(threshold_topic_lasso_dict, output_file)

with open(lassoout / 'LASSO_Threshold_Stability_Topic.p', 'wb') as output_file:
    pickle.dump(threshold_topic_lasso_stability_tables, output_file)

with open(lassoout / 'LASSO_Threshold_Diagnostics_All.p', 'wb') as output_file:
    pickle.dump(threshold_lasso_diagnostics, output_file)

with open(lassoout / 'LASSO_Threshold_Diagnostics_Topic.p', 'wb') as output_file:
    pickle.dump(threshold_topic_lasso_diagnostics, output_file)

'''
Save analysis data
'''

analysis_data = {
    'primary_missing_probability_threshold': PRIMARY_MISSING_PROBABILITY_THRESHOLD,
    'missing_probability_thresholds': MISSING_PROBABILITY_THRESHOLDS,
    'analysis_data_by_threshold': analysis_data_by_threshold,
    'dfall': dfall,
    'depvars': depvars,
    'indepvars': indepvars,
    'depvars_scaled': depvars_scaled,
    'indepvars_scaled': indepvars_scaled,
    'certainty': dfcertainty,
}

with open(mainres / 'Analysis_Data.p', 'wb') as output_file:
    pickle.dump(analysis_data, output_file)
    

'''
Validate outputs required by the PSL analysis
'''

expected_threshold_keys = {format(threshold, '.2f') for threshold in MISSING_PROBABILITY_THRESHOLDS}
actual_threshold_keys = set(analysis_data_by_threshold.keys())

if actual_threshold_keys != expected_threshold_keys:
    raise ValueError('Threshold analysis-data keys do not match settings. Expected ' + str(expected_threshold_keys) + ', found ' + str(actual_threshold_keys))

for threshold_key in sorted(expected_threshold_keys):
    for outcome_name in outcome_columns:
        aggregate_key = (threshold_key, outcome_name)

        if aggregate_key not in threshold_lasso_stability_tables:
            raise KeyError('Missing aggregate threshold LASSO stability key: ' + str(aggregate_key))

for primary_topic_key in topic_lasso_stability_tables:
    topic_name, outcome_name = primary_topic_key
    threshold_topic_key = (primary_threshold_key, topic_name, outcome_name)

    if threshold_topic_key not in threshold_topic_lasso_stability_tables:
        raise KeyError('Missing primary-threshold topic LASSO stability key: ' + str(threshold_topic_key))

print('\nThreshold interfaces validated successfully.')

    
'''
Summary
'''

print('\nMatched case-study papers:')
print(len(dfall))

print('\nPrimary missing-probability threshold:')
print(PRIMARY_MISSING_PROBABILITY_THRESHOLD)

print('\nAvailable threshold datasets:')
print(sorted(analysis_data_by_threshold.keys()))

print('\nAggregate XGBoost diagnostics:')
print(dfdiagnostic)

print('\nSigned SHAP sums:')
print(dfshap_signed_sum_mean)

print('\nMean absolute SHAP values:')
print(dfshap_absolute_mean)

print('\nAggregate LASSO diagnostics:')
print(df_lasso_diagnostics)

print('\nTopic LASSO diagnostics:')
print(df_topic_lasso_diagnostics)

print('\nThreshold aggregate LASSO models:')
print(len(threshold_lasso_stability_tables))

print('\nThreshold topic LASSO models:')
print(len(threshold_topic_lasso_stability_tables))

print('\nKey figure:')
print(allout / 'Pos_Neg_OverallEffects2.pdf')

print('\nKey PSL input:')
print(shapout / 'Analysis_Data.p')

print('\nDone')
        
        
        
        
