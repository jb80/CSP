#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PSL model construction, evaluation, rule ranking, and network functions.

This module:

1. Loads LASSO-selected interactions.
2. Creates PSL predicates.
3. Generates positive and negative candidate rules.
4. Supports coefficient-directed or bidirectional rule generation.
5. Filters configurations using support and body-strength criteria.
6. Injects continuous observations, truth values, and inference targets.
7. Matches learned PSL rules using canonical rule strings.
8. Evaluates inferred outcomes.
9. Calculates QCA-style rule coverage, consistency, and effect.
10. Constructs weighted condition co-occurrence networks.
11. Calculates network and community similarity.

LASSO interactions are used to screen candidate rule structures.

PSL subsequently learns the rule weights and evaluates the resulting soft
logical clauses over continuous truth values.

@author: jbaggio
"""


'''
Packages
'''

import re
import html
import itertools
import pickle
from collections import Counter, defaultdict


import numpy as np
import pandas as pd

from pslpython.model import Model
from pslpython.predicate import Predicate
from pslpython.rule import Rule
from pslpython.partition import Partition

from sklearn.metrics import mean_squared_error, adjusted_rand_score
from scipy.stats import spearmanr


import networkx as nx
import igraph as ig
import leidenalg
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors




'''
Settings
'''

RANDOM_SEED = 7879
RELATIVE_DIRECTION_THRESHOLD = 0.05 #if rules are positive and negative, chooose either one if the drelative ifference is > 0.05% or choose both if relative difference is < 0.05 

colors_pos = {0: 'aliceblue', 1: 'aqua', 2: 'aquamarine', 3: 'blue', 4: 'blueviolet', 5: 'cadetblue', 6: 'cornflowerblue', 7: 'cyan', 8: 'darkblue', 9: 'darkcyan'}

colors_neg = {0: 'darkgoldenrod', 1: 'darkorange', 2: 'gold', 3: 'goldenrod', 4: 'greenyellow', 5: 'lightgoldenrodyellow', 6: 'lightyellow', 7: 'orange', 8: 'orangered', 9: 'palegoldenrod'}


'''
LASSO rule loading
'''

def load_lasso_rules(lasso_file, depvar, context=None):
    with open(lasso_file, 'rb') as input_file:
        lasso_rules = pickle.load(input_file)

    key = depvar if context is None else (context, depvar)

    if key not in lasso_rules:
        raise KeyError('No LASSO interactions found for ' + str(key))

    return lasso_rules[key]


def rules_from_stability_table(stability_table, selection_threshold=0.50, sign_stability_threshold=0.50, use_full_model=False, coefficient_tolerance=1e-4):
    table = stability_table.copy()

    if use_full_model:
        table = table[np.abs(table['full_coefficient']) > coefficient_tolerance]
        coefficient_column = 'full_coefficient'
    else:
        table = table[(table['selection_frequency'] >= selection_threshold) & (table['sign_stability'] >= sign_stability_threshold)]

        if 'selected_coefficient_median' in table.columns:
            coefficient_column = 'selected_coefficient_median'
        else:
            coefficient_column = 'coefficient_median'

    table = table[np.abs(table[coefficient_column]) > coefficient_tolerance]

    return [(row['feature'], float(row[coefficient_column])) for _, row in table.iterrows()]

def get_candidate_rules(stability_dictionary, fallback_dictionary, key, specification, threshold_key=None, threshold_stability_dictionary=None, threshold_fallback_dictionary=None, coefficient_tolerance=1e-4):

    if threshold_key is not None:
        threshold_lookup_key = ((threshold_key,) + key if isinstance(key, tuple) else (threshold_key, key))

        if (threshold_stability_dictionary is not None and threshold_lookup_key in threshold_stability_dictionary):
            return rules_from_stability_table(threshold_stability_dictionary[threshold_lookup_key], selection_threshold=specification['LASSO_SELECTION_THRESHOLD'],
                sign_stability_threshold=specification['LASSO_SIGN_STABILITY_THRESHOLD'],
                use_full_model=specification['USE_FULL_LASSO_MODEL'],
                coefficient_tolerance=coefficient_tolerance
            )

        if (threshold_fallback_dictionary is not None and threshold_lookup_key in threshold_fallback_dictionary):
            return threshold_fallback_dictionary[threshold_lookup_key]

    if key in stability_dictionary:
        return rules_from_stability_table(stability_dictionary[key], selection_threshold=specification['LASSO_SELECTION_THRESHOLD'], 
            sign_stability_threshold=specification['LASSO_SIGN_STABILITY_THRESHOLD'],
            use_full_model=specification['USE_FULL_LASSO_MODEL'],
            coefficient_tolerance=coefficient_tolerance
        )

    if key in fallback_dictionary:
        return fallback_dictionary[key]

    return []
    
'''
Rule naming and parsing
'''

def sanitize_name(name):
    sanitized = re.sub(r'[^A-Za-z0-9_]', '_', str(name))
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')

    if not sanitized:
        sanitized = 'PREDICATE'

    if sanitized[0].isdigit():
        sanitized = 'P_' + sanitized

    return sanitized.upper()


def create_name_map(names):
    name_map = {name: sanitize_name(name) for name in names}
    reverse_map = defaultdict(list)

    for original, sanitized in name_map.items():
        reverse_map[sanitized].append(original)

    collisions = {sanitized: originals for sanitized, originals in reverse_map.items() if len(originals) > 1}

    if collisions:
        raise ValueError('Predicate-name collisions after sanitization: ' + str(collisions))

    return name_map


def split_term(term):
    term = str(term).replace('(', '').replace(')', '').strip()

    if '_AND_' in term:
        parts = term.split('_AND_')
    else:
        parts = term.split()

    return [part for part in parts if part]


def get_lower_order_terms(term):
    parts = split_term(term)
    return {' '.join(sorted(subset)) for size in range(1, len(parts)) for subset in itertools.combinations(parts, size)}


def canonicalize_rule(rule):
    rule_string = html.unescape(str(rule)).strip()

    if ':' in rule_string:
        prefix, remainder = rule_string.split(':', 1)

        try:
            float(prefix.strip())
            rule_string = remainder.strip()
        except ValueError:
            pass

    rule_string = re.sub(r'\s+', ' ', rule_string)
    rule_string = rule_string.replace(' -> ', '->')
    rule_string = rule_string.replace(' & ', '&')

    return rule_string


'''
Fuzzy conjunctions
'''

def lukas_and(values):
    values = np.asarray(values, dtype=float)

    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError('Lukasiewicz conjunction requires a nonempty two-dimensional array.')

    return np.maximum(0.0, np.sum(values, axis=1) - values.shape[1] + 1.0)

def lukas_avg_and(values):
    values = np.asarray(values, dtype=float)

    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError('Lukasiewicz conjunction requires a nonempty two-dimensional array.')

    n = values.shape[1]

    return np.maximum(0.0, np.mean(values, axis=1) - ((n - 1.0) / n))


def product_and(values):
    values = np.asarray(values, dtype=float)

    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError('Product conjunction requires a nonempty two-dimensional array.')

    return np.prod(values, axis=1)


def minimum_and(values):
    values = np.asarray(values, dtype=float)

    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError('Minimum conjunction requires a nonempty two-dimensional array.')

    return np.min(values, axis=1)


def calculate_body(values, operator='lukas_avg'):
    
    if operator == 'lukasiewicz':
        return lukas_and(values)

    if operator == 'lukas_avg':
        return lukas_avg_and(values)

    if operator == 'product':
        return product_and(values)

    if operator == 'minimum':
        return minimum_and(values)

    raise ValueError('Unknown fuzzy operator: ' + str(operator))

'''
PSL model construction
'''

def create_predicates(model, atomic_tokens, name_map):
    for variable in sorted(atomic_tokens):
        model.add_predicate(Predicate(name_map[variable], 1))


def add_rule_with_meta(model, name_map, depvar, body_parts, head_is_positive, weight, squared, origin, added_rules, rule_meta_list, coefficient=None, max_rules=None):
    if max_rules is not None and len(added_rules) >= max_rules:
        return False

    body_parts = tuple(sorted(body_parts))
    body = ' & '.join(name_map[part] + '(X)' for part in body_parts)
    head = name_map[depvar] + '(X)' if head_is_positive else '~' + name_map[depvar] + '(X)'
    rule_string = body + ' -> ' + head
    rule_key = canonicalize_rule(rule_string)

    if rule_key in added_rules:
        return False

    model.add_rule(Rule(rule_string, weight=float(weight), squared=bool(squared)))
    added_rules.add(rule_key)

    rule_meta_list.append({
        'rule': rule_string,
        'rule_key': rule_key,
        'origin': origin,
        'sign': 1 if head_is_positive else -1,
        'parts': body_parts,
        'k': len(body_parts),
        'coefficient': coefficient,
        'init_weight': float(weight),
        'squared': bool(squared)
    })

    return True


def rule_passes_support(indepvars, parts, config):
    if any(part not in indepvars.columns for part in parts):
        return False, 0.0, 0.0

    values = indepvars[list(parts)].astype(float).to_numpy()
    body = calculate_body(values, operator=config.get('SCREENING_OPERATOR', 'lukasiewicz'))
    support_fraction = float(np.mean(body > 0))
    mean_body = float(np.mean(body))
    passes = support_fraction >= config['MIN_SUPPORT_FRAC'] and mean_body >= config['MIN_MEAN_BODY']

    return passes, support_fraction, mean_body


def generate_exploratory_rules(model, indepvars, depvar, exploratory_terms, name_map, config, added_rules, rule_meta_list, minimum_weight):
    for term in sorted(exploratory_terms):
        parts = tuple(sorted(split_term(term)))

        if not parts:
            continue

        passes, _, _ = rule_passes_support(indepvars, parts, config)

        if not passes:
            continue

        add_rule_with_meta(model, name_map, depvar, parts, True, minimum_weight, False, 'exploratory', added_rules, rule_meta_list, max_rules=config.get('MAX_RULES'))
        add_rule_with_meta(model, name_map, depvar, parts, False, minimum_weight, False, 'exploratory', added_rules, rule_meta_list, max_rules=config.get('MAX_RULES'))


def generate_lasso_rules(model, indepvars, depvar, rule_list, name_map, config, added_rules, rule_meta_list, minimum_weight):
    for term, coefficient in rule_list or []:
        parts = tuple(sorted(split_term(term)))

        if not parts:
            continue

        maximum_k = len(parts) if config['MAX_K'] is None else min(len(parts), config['MAX_K'])

        if maximum_k < config['MIN_K']:
            continue

        for rule_size in range(config['MIN_K'], maximum_k + 1):
            for seed in itertools.combinations(parts, rule_size):
                passes, _, _ = rule_passes_support(indepvars, seed, config)

                if not passes:
                    continue

                base_weight = max(minimum_weight, round(abs(float(coefficient)), 6)) if config['LASSO_W'] else 1.0

                if config.get('LASSO_DIRECTION_MODE', 'both') == 'coefficient':
                    directions = [float(coefficient) > 0]
                else:
                    directions = [True, False]

                for head_is_positive in directions:
                    add_rule_with_meta(model, name_map, depvar, seed, head_is_positive, base_weight, False, 'lasso', added_rules, rule_meta_list, coefficient=float(coefficient), max_rules=config.get('MAX_RULES'))

                if config['SUBSET_INCL'] and rule_size > config['MIN_K']:
                    for subset_size in range(rule_size - 1, config['MIN_K'] - 1, -1):
                        for subset in itertools.combinations(seed, subset_size):
                            passes_subset, _, _ = rule_passes_support(indepvars, subset, config)

                            if not passes_subset:
                                continue

                            subset_weight = max(minimum_weight, round(base_weight * config['SUBSET_DECAY'] ** (rule_size - subset_size), 6))

                            for head_is_positive in directions:
                                add_rule_with_meta(model, name_map, depvar, subset, head_is_positive, subset_weight, False, 'subset', added_rules, rule_meta_list, coefficient=float(coefficient), max_rules=config.get('MAX_RULES'))


def inject_data(model, indepvars, depvar, depvars, name_map, config, truth_entities=None, target_entities=None):
    predicate_map = model.get_predicates()

    if truth_entities is None:
        truth_entities = list(depvars.index)

    if target_entities is None:
        target_entities = list(depvars.index)

    for variable in indepvars.columns:
        if variable not in name_map or name_map[variable] not in predicate_map:
            continue

        predicate = predicate_map[name_map[variable]]

        if config['ALL_IVOBS']:
            rows = [[str(entity), float(value)] for entity, value in indepvars[variable].items()]
        else:
            rows = [[str(entity), float(value)] for entity, value in indepvars[variable].items() if not np.isclose(value, 0.5)]

        if rows:
            predicate.add_data(Partition.OBSERVATIONS, rows)

    dependent_predicate = predicate_map[name_map[depvar]]
    dependent_predicate.add_data(Partition.TARGETS, [[str(entity)] for entity in target_entities])

    truth_rows = [[str(entity), float(depvars.loc[entity, depvar])] for entity in truth_entities if entity in depvars.index and not np.isclose(float(depvars.loc[entity, depvar]), 0.5)]

    if truth_rows:
        dependent_predicate.add_data(Partition.TRUTH, truth_rows)


def build_psl_model(*, depvar, indepvars, depvars, rule_list, filtered, config, truth_entities=None, target_entities=None):
    selected_terms = {str(term) for term, _ in rule_list} if rule_list else set()
    exploratory_terms = set()

    if config['USE_GLOBAL_EXPLORATORY']:
        for term in selected_terms:
            exploratory_terms.update(get_lower_order_terms(term))

        exploratory_terms = exploratory_terms - selected_terms

    atomic_tokens = set(indepvars.columns)
    atomic_tokens.add(depvar)

    for term in selected_terms.union(exploratory_terms):
        atomic_tokens.update(split_term(term))

    name_map = create_name_map(atomic_tokens)
    model = Model(name='psl_model_' + name_map[depvar])

    create_predicates(model, atomic_tokens, name_map)

    if config['ADD_NEGATIVE_DV_PRIOR']:
        model.add_rule(Rule('~' + name_map[depvar] + '(X)', weight=float(config['DV_PRIOR_WEIGHT']), squared=True))

    coefficient_values = [abs(float(coefficient)) for _, coefficient in rule_list or []]
    minimum_weight = max(config.get('MIN_RULE_WEIGHT', 1e-3), min(coefficient_values)) if config['LASSO_W'] and coefficient_values else 1.0

    added_rules = set()
    rule_meta_list = []

    if config['USE_GLOBAL_EXPLORATORY']:
        generate_exploratory_rules(model, indepvars, depvar, exploratory_terms, name_map, config, added_rules, rule_meta_list, minimum_weight)

    if filtered:
        for body_string, sign in rule_list:
            parts = tuple(sorted(split_term(body_string)))
            head_is_positive = float(sign) > 0
            passes, _, _ = rule_passes_support(indepvars, parts, config)

            if passes:
                add_rule_with_meta(model, name_map, depvar, parts, head_is_positive, 1.0, False, 'filtered', added_rules, rule_meta_list, coefficient=float(sign), max_rules=config.get('MAX_RULES'))
    else:
        generate_lasso_rules(model, indepvars, depvar, rule_list, name_map, config, added_rules, rule_meta_list, minimum_weight)

    inject_data(model, indepvars, depvar, depvars, name_map, config, truth_entities=truth_entities, target_entities=target_entities)

    return model, rule_meta_list, name_map


'''
Learned-rule handling
'''

def get_learned_rule_weights(model):
    learned_weights = {}

    for rule in model.get_rules():
        learned_weights[canonicalize_rule(rule)] = float(rule.weight())

    return learned_weights


def filter_rule_meta_by_weight(rule_list, model, tie_tolerance=RELATIVE_DIRECTION_THRESHOLD):
    learned_weights = get_learned_rule_weights(model)
    grouped_rules = defaultdict(dict)

    for metadata in rule_list:
        rule_key = metadata.get('rule_key', canonicalize_rule(metadata['rule']))
        weight = learned_weights.get(rule_key)

        if weight is None:
            continue

        body = rule_key.split('->', 1)[0].strip()
        direction = 'positive' if metadata['sign'] == 1 else 'negative'
        grouped_rules[body][direction] = (metadata, weight)

    filtered_metadata = []

    for _, directions in grouped_rules.items():
        positive = directions.get('positive')
        negative = directions.get('negative')

        if positive and negative:
            difference = abs(positive[1] - negative[1]) / max(positive[1], negative[1])

            if difference <= tie_tolerance:
                filtered_metadata.append(positive[0])
                filtered_metadata.append(negative[0])
                
            elif positive[1] > negative[1]:
                filtered_metadata.append(positive[0])
            else:
                filtered_metadata.append(negative[0])
        elif positive:
            filtered_metadata.append(positive[0])
        elif negative:
            filtered_metadata.append(negative[0])

    return filtered_metadata


'''
Inference normalization
'''

def predicate_matches(predicate, predicate_name):
    predicate_string = str(predicate).upper()
    requested_name = str(predicate_name).upper()

    if predicate_string == requested_name:
        return True

    if predicate_string.startswith(requested_name + '/'):
        return True

    if predicate_string.startswith(requested_name + '('):
        return True

    return requested_name in predicate_string


def normalize_inference(inferences, predicate_name=None):

    if len(inferences) == 1:
        dataframe = next(iter(inferences.values())).copy()
    else:
        matches = []

        for predicate, dataframe in inferences.items():
            if predicate_matches(predicate, predicate_name):
                matches.append(dataframe.copy())

        if len(matches) != 1:
            available_predicates = [str(predicate) for predicate in inferences.keys()]

            raise ValueError('Expected one inference result for '+ str(predicate_name) + ', found ' + str(len(matches)) + '. Available predicates: ' + str(available_predicates))

        dataframe = matches[0].copy()

    if dataframe.empty:
        return pd.DataFrame(columns=['entity', 'y_pred'])

    value_column_candidates = [column for column in dataframe.columns if str(column).lower() in ['truth', 'value', 'score']]

    if value_column_candidates:
        value_column = value_column_candidates[0]
    else:
        value_column = dataframe.columns[-1]

    entity_columns = [column for column in dataframe.columns if column != value_column]

    if not entity_columns:
        raise ValueError('Could not identify an entity column in inference output.')

    entity_column = entity_columns[0]

    result = dataframe.rename(columns={
            entity_column: 'entity',
            value_column: 'y_pred'
        }
    )

    result['entity'] = result['entity'].astype(str)
    result['y_pred'] = pd.to_numeric(result['y_pred'], errors='coerce')

    result = result.dropna(subset=['y_pred'])

    return result[['entity', 'y_pred']]


'''
Goodness of fit
'''

def gof_psl(depvar, inferences, predicate_name):
    predictions = normalize_inference(inferences, predicate_name)

    truth = depvar.rename('y_true').reset_index()
    entity_column = truth.columns[0]
    truth = truth.rename(columns={entity_column: 'entity'})
    truth['entity'] = truth['entity'].astype(str)
    truth['y_true'] = pd.to_numeric(truth['y_true'], errors='coerce')
    truth = truth.dropna(subset=['y_true'])
    truth = truth[~np.isclose(truth['y_true'], 0.5)]

    evaluation = truth.merge(predictions, on='entity', how='inner')

    if evaluation.empty:
        return np.nan, np.nan, evaluation

    mse = mean_squared_error(evaluation['y_true'], evaluation['y_pred'])
    rmse = np.sqrt(mse)

    return rmse, mse, evaluation

'''
Fit PSL 
'''
def fit_psl_model(depvar, independent_data, dependent_data, candidate_rules, specification):
    model, rule_metadata, name_map = build_psl_model(depvar=depvar, indepvars=independent_data, depvars=dependent_data, rule_list=candidate_rules, filtered=False, config=specification)

    model.learn()
    initial_inference = model.infer()

    filtered_metadata = filter_rule_meta_by_weight(rule_metadata, model)
    filtered_rule_list = [(' '.join(metadata['parts']), metadata['sign']) for metadata in filtered_metadata]

    filtered_model, filtered_rule_metadata, filtered_name_map = build_psl_model(depvar=depvar, indepvars=independent_data, depvars=dependent_data, rule_list=filtered_rule_list, filtered=True, config=specification)

    filtered_model.learn()
    filtered_inference = filtered_model.infer()

    rmse_initial, mse_initial, evaluation_initial = gof_psl(dependent_data[depvar], initial_inference, name_map[depvar])
    rmse_filtered, mse_filtered, evaluation_filtered = gof_psl(dependent_data[depvar], filtered_inference, filtered_name_map[depvar])

    result = {
        'model': model,
        'rule_metadata': rule_metadata,
        'name_map': name_map,
        'inference': initial_inference,
        'filtered_model': filtered_model,
        'filtered_rule_metadata': filtered_rule_metadata,
        'filtered_name_map': filtered_name_map,
        'filtered_inference': filtered_inference,
        'diagnostics': {
            'candidate_lasso_rules': len(candidate_rules),
            'initial_psl_rules': len(rule_metadata),
            'filtered_psl_rules': len(filtered_rule_metadata),
            'initial_rmse': rmse_initial,
            'initial_mse': mse_initial,
            'filtered_rmse': rmse_filtered,
            'filtered_mse': mse_filtered,
            'n_evaluation_initial': len(evaluation_initial),
            'n_evaluation_filtered': len(evaluation_filtered),
            'gof_interpretation': 'in_sample_reconstruction'
        }
    }

    positive_rules, negative_rules = rank_filtered_model(result, depvar, independent_data, dependent_data, specification)

    result['positive_rules'] = positive_rules
    result['negative_rules'] = negative_rules

    return result



'''
Rule ranking
'''

def extract_and_rank_rules(rulelist, model, indeps, deps, single_dep, inferred_res=None, predicate_name=None, top_n=None, direction='both', importance_metric='coverage_consistency', use_inferred=False, fuzzy_operator='lukasiewicz'):
    learned_weights = get_learned_rule_weights(model)

    if use_inferred:
        if inferred_res is None or predicate_name is None:
            raise ValueError('Inferred results and predicate name are required when use_inferred=True.')

        inference_dataframe = normalize_inference(inferred_res, predicate_name).set_index('entity')
        dv_values = inference_dataframe['y_pred'].astype(float)
    else:
        dv_values = deps[single_dep].astype(float).copy()
        dv_values = dv_values[~np.isclose(dv_values, 0.5)]
        dv_values.index = dv_values.index.astype(str)

    independent_data = indeps.copy()
    independent_data.index = independent_data.index.astype(str)

    ranked_rules = []

    for metadata in rulelist:
        rule_key = metadata.get('rule_key', canonicalize_rule(metadata['rule']))
        weight = learned_weights.get(rule_key)

        if weight is None:
            continue

        sign = int(metadata['sign'])

        if direction == 'positive' and sign != 1:
            continue

        if direction == 'negative' and sign != -1:
            continue

        parts = tuple(metadata['parts'])

        if any(part not in independent_data.columns for part in parts):
            continue

        shared_entities = sorted(set(dv_values.index).intersection(set(independent_data.index)))

        if not shared_entities:
            continue

        values = independent_data.loc[shared_entities, list(parts)].astype(float).to_numpy()
        body = calculate_body(values, operator=fuzzy_operator)
        outcome_values = dv_values.loc[shared_entities].astype(float).to_numpy()
        target_values = outcome_values if sign == 1 else 1.0 - outcome_values

        positive_body = body > 0
        coverage = float(np.mean(positive_body))
        consistency = float(np.mean(target_values[positive_body])) if np.any(positive_body) else np.nan
        weighted_consistency = float(np.average(target_values, weights=body)) if body.sum() > 0 else np.nan
        baseline = float(np.mean(target_values))
        effect = consistency - baseline if not np.isnan(consistency) else np.nan
        weighted_effect = weighted_consistency - baseline if not np.isnan(weighted_consistency) else np.nan

        metric_values = {
            'coverage': coverage,
            'consistency': 0.0 if np.isnan(consistency) else consistency,
            'weighted_consistency': 0.0 if np.isnan(weighted_consistency) else weighted_consistency,
            'coverage_consistency': 0.0 if np.isnan(consistency) else coverage * consistency,
            'cov_cons_eff':  0.0 if np.isnan(effect) else coverage * (consistency + effect),
            'coverage_effect': 0.0 if np.isnan(effect) else coverage * effect,
            'coverage_positive_effect': 0.0 if np.isnan(effect) else coverage * max(effect, 0.0),
            'weighted_effect': 0.0 if np.isnan(weighted_effect) else weighted_effect
        }

        importance = metric_values.get(importance_metric, metric_values['coverage_consistency'])

        ranked_rules.append({
            'rule': metadata['rule'],
            'rule_key': rule_key,
            'weight': weight,
            'origin': metadata['origin'],
            'sign': sign,
            'k': metadata['k'],
            'squared': metadata['squared'],
            'init_weight': metadata['init_weight'],
            'coefficient': metadata.get('coefficient'),
            'coverage': coverage,
            'consistency': consistency,
            'weighted_consistency': weighted_consistency,
            'baseline': baseline,
            'effect': effect,
            'weighted_effect': weighted_effect,
            'importance': importance
        })

    result = pd.DataFrame(ranked_rules)

    if not result.empty:
        result = result.sort_values('importance', ascending=False)

    if top_n is not None:
        result = result.head(top_n)

    return result.reset_index(drop=True)


def rank_filtered_model(result, depvar, independent_data, dependent_data, specification):
    positive_rules = extract_and_rank_rules(
        rulelist=result['filtered_rule_metadata'],
        model=result['filtered_model'],
        indeps=independent_data,
        deps=dependent_data,
        single_dep=depvar,
        inferred_res=result['filtered_inference'],
        predicate_name=result['filtered_name_map'][depvar],
        top_n=specification['TOP_N'],
        direction='positive',
        importance_metric=specification['IMPORTANCE_METRIC'],
        use_inferred=True,
        fuzzy_operator=specification['SCREENING_OPERATOR']
    )

    negative_rules = extract_and_rank_rules(
        rulelist=result['filtered_rule_metadata'],
        model=result['filtered_model'],
        indeps=independent_data,
        deps=dependent_data,
        single_dep=depvar,
        inferred_res=result['filtered_inference'],
        predicate_name=result['filtered_name_map'][depvar],
        top_n=specification['TOP_N'],
        direction='negative',
        importance_metric=specification['IMPORTANCE_METRIC'],
        use_inferred=True,
        fuzzy_operator=specification['SCREENING_OPERATOR']
    )

    return positive_rules, negative_rules


'''
Rule Robustness functions
'''


def summarize_rule_robustness(rule_dataframe, grouping_columns):
    if rule_dataframe.empty:
        return pd.DataFrame()

    dataframe = rule_dataframe.copy()
    dataframe['rank'] = dataframe.groupby(grouping_columns + ['Specification', 'type'])['importance'].rank(method='min', ascending=False)

    available_specifications = dataframe.groupby(grouping_columns)['Specification'].nunique().rename('available_specifications').reset_index()

    summary = dataframe.groupby(grouping_columns + ['rule_key', 'sign']).agg(
        specification_count=('Specification', 'nunique'),
        importance_mean=('importance', 'mean'),
        importance_median=('importance', 'median'),
        importance_stdev=('importance', 'std'),
        importance_min=('importance', 'min'),
        importance_max=('importance', 'max'),
        learned_weight_mean=('weight', 'mean'),
        learned_weight_median=('weight', 'median'),
        coverage_mean=('coverage', 'mean'),
        consistency_mean=('consistency', 'mean'),
        effect_mean=('effect', 'mean'),
        best_rank=('rank', 'min'),
        median_rank=('rank', 'median'),
        worst_rank=('rank', 'max')
    ).reset_index()

    summary = summary.merge(available_specifications, on=grouping_columns, how='left')
    summary['retention_frequency'] = summary['specification_count'] / summary['available_specifications']
    summary = summary.sort_values(['retention_frequency', 'importance_median'], ascending=[False, False])

    return summary


'''
Rule Direction Robustness
'''

def summarize_direction_robustness(rule_dataframe, grouping_columns):

    if rule_dataframe.empty:
        return pd.DataFrame()

    dataframe = rule_dataframe.copy()

    # remove outcome direction and keep only the body
    dataframe['body_key'] = (dataframe['rule_key'].str.split('->').str[0].str.strip()    )

    summary = dataframe.groupby(grouping_columns + ['body_key']).agg(specification_count=('Specification', 'nunique'),
        positive_frequency=('sign',lambda values: float(np.mean(values == 1))),
        negative_frequency=('sign',lambda values: float(np.mean(values == -1))),
        importance_median=('importance', 'median'), learned_weight_median=('weight', 'median')).reset_index()

    summary['dominant_direction'] = np.select(
        [summary['positive_frequency'] > summary['negative_frequency'],

            summary['negative_frequency'] > summary['positive_frequency']],
        [
            'positive',
            'negative'
        ],
        default='mixed'
    )

    summary['direction_stability'] = np.maximum(
        summary['positive_frequency'],
        summary['negative_frequency']
    )

    summary = summary.sort_values(
        ['direction_stability', 'importance_median'],
        ascending=[False, False]
    )

    return summary

'''
Rule set overlap
'''

def calculate_rule_set_overlap(rule_dataframe, grouping_columns):
    rows = []

    if rule_dataframe.empty:
        return pd.DataFrame()

    for group_key, group_dataframe in rule_dataframe.groupby(grouping_columns):
        specifications = sorted(group_dataframe['Specification'].unique())

        for first_specification in specifications:
            first_rules = set(group_dataframe[group_dataframe['Specification'] == first_specification]['rule_key'])

            for second_specification in specifications:
                second_rules = set(group_dataframe[group_dataframe['Specification'] == second_specification]['rule_key'])

                row = {
                    'Specification1': first_specification,
                    'Specification2': second_specification,
                    'Rule_Jaccard': jaccard_similarity(first_rules, second_rules)
                }

                if isinstance(group_key, tuple):
                    for column, value in zip(grouping_columns, group_key):
                        row[column] = value
                else:
                    row[grouping_columns[0]] = group_key

                rows.append(row)

    return pd.DataFrame(rows)


'''
Importance-ranking sensitivity
'''

def calculate_rank_correlations(rule_dataframe, grouping_columns):
    rows = []

    if rule_dataframe.empty:
        return pd.DataFrame()

    for group_key, group_dataframe in rule_dataframe.groupby(grouping_columns):
        specifications = sorted(group_dataframe['Specification'].unique())

        for first_specification in specifications:
            first = group_dataframe[group_dataframe['Specification'] == first_specification][['rule_key', 'importance']]
            first = first.groupby('rule_key', as_index=False)['importance'].mean()

            for second_specification in specifications:
                second = group_dataframe[group_dataframe['Specification'] == second_specification][['rule_key', 'importance']]
                second = second.groupby('rule_key', as_index=False)['importance'].mean()
                merged = first.merge(second, on='rule_key', suffixes=('_first', '_second'))

                if len(merged) >= 3 and merged['importance_first'].nunique() > 1 and merged['importance_second'].nunique() > 1:
                    correlation, p_value = spearmanr(merged['importance_first'], merged['importance_second'])
                else:
                    correlation, p_value = np.nan, np.nan

                row = {
                    'Specification1': first_specification,
                    'Specification2': second_specification,
                    'Shared_Rules': len(merged),
                    'Spearman': correlation,
                    'PValue': p_value
                }

                if isinstance(group_key, tuple):
                    for column, value in zip(grouping_columns, group_key):
                        row[column] = value
                else:
                    row[grouping_columns[0]] = group_key

                rows.append(row)

    return pd.DataFrame(rows)



'''
Rule network functions
'''

def clean_rule(rule):
    return html.unescape(str(rule)).replace('(X)', '').split('->')[0].strip()


def count_cooccurrences(dataframe, label, include_context=False):
    expected_columns = ['rule', 'importance', 'type', 'DV']

    if include_context:
        expected_columns.append('Context')

    if dataframe is None or dataframe.empty or any(column not in dataframe.columns for column in expected_columns):
        output_columns = ['Condition1', 'Condition2', 'Label', 'DV', 'Importance']

        if include_context:
            output_columns.insert(4, 'Context')

        return pd.DataFrame(columns=output_columns)

    cooccurrences = defaultdict(float)
    grouping_columns = ['Context', 'DV'] if include_context else ['DV']
    grouped = dataframe[dataframe['type'] == label].groupby(grouping_columns, dropna=False)

    for group_key, subset in grouped:
        if include_context:
            context, dependent_variable = group_key
        else:
            dependent_variable = group_key[0] if isinstance(group_key, tuple) else group_key
            context = None

        for _, row in subset.iterrows():
            rule = html.unescape(str(row['rule']))
            conditions = [condition.replace('(X)', '').replace('~', '').strip() for condition in rule.split('->')[0].split('&')]
            conditions = sorted(set(condition for condition in conditions if condition))
            importance = float(row['importance'])

            for first_condition, second_condition in itertools.combinations(conditions, 2):
                if include_context:
                    key = (first_condition, second_condition, label, dependent_variable, context)
                else:
                    key = (first_condition, second_condition, label, dependent_variable)

                cooccurrences[key] += importance

    if include_context:
        rows = [{'Condition1': key[0], 'Condition2': key[1], 'Label': key[2], 'DV': key[3], 'Context': key[4], 'Importance': value} for key, value in cooccurrences.items()]
    else:
        rows = [{'Condition1': key[0], 'Condition2': key[1], 'Label': key[2], 'DV': key[3], 'Importance': value} for key, value in cooccurrences.items()]

    return pd.DataFrame(rows)


def build_graph(dataframe):
    if dataframe is None or dataframe.empty:
        return nx.Graph()

    required_columns = {'Condition1', 'Condition2', 'Importance'}

    if not required_columns.issubset(dataframe.columns):
        return nx.Graph()

    edges = dataframe.groupby(['Condition1', 'Condition2'], as_index=False)['Importance'].sum()
    edges = edges[np.abs(edges['Importance']) > 0]

    if edges.empty:
        return nx.Graph()

    return nx.from_pandas_edgelist(edges, source='Condition1', target='Condition2', edge_attr='Importance')


def nx_to_igraph(graph):
    node_names = [str(node) for node in graph.nodes()]
    node_positions = {node: index for index, node in enumerate(node_names)}

    igraph_graph = ig.Graph()
    igraph_graph.add_vertices(node_names)
    igraph_graph.add_edges([(node_positions[str(first)], node_positions[str(second)]) for first, second in graph.edges()])
    igraph_graph.es['weight'] = [abs(float(graph[first][second].get('Importance', 0.0))) for first, second in graph.edges()]

    return igraph_graph


def run_leiden_partition(graph, edge_sign='positive', resolution_parameter=None, signed = False):
    if graph.number_of_nodes() == 0:
        return {}

    signed_graph = nx.Graph()
    signed_graph.add_nodes_from(str(node) for node in graph.nodes())

    for first, second, attributes in graph.edges(data=True):
        importance = float(attributes.get('Importance', 0.0))

        if edge_sign == 'positive' and importance > 0:
            signed_graph.add_edge(str(first), str(second), Importance=importance)
            
        if signed == True:
            if edge_sign == 'negative' and importance < 0:
                signed_graph.add_edge(str(first), str(second), Importance=abs(importance))
        else:
            if edge_sign == 'negative' and importance > 0:
                signed_graph.add_edge(str(first), str(second), Importance=importance)
            
       
    if signed_graph.number_of_edges() == 0:
        return {str(node): index for index, node in enumerate(signed_graph.nodes())}

    igraph_graph = nx_to_igraph(signed_graph)
    positive_weights = [weight for weight in igraph_graph.es['weight'] if weight > 0]

    if resolution_parameter is None:
        resolution_parameter = float(np.mean(positive_weights)) * 0.75 if positive_weights else 1.0

    partition = leidenalg.find_partition(igraph_graph, leidenalg.CPMVertexPartition, weights='weight', resolution_parameter=resolution_parameter)

    return {node: partition.membership[index] for index, node in enumerate(igraph_graph.vs['name'])}


def assign_node_edge_colors(graph, title_prefix='Difference', resolution_parameter=None, signed = False):
    if graph.number_of_nodes() == 0:
        return graph

    positive_partition = run_leiden_partition(graph, edge_sign='positive', resolution_parameter=resolution_parameter, signed = signed)
    negative_partition = run_leiden_partition(graph, edge_sign='negative', resolution_parameter=resolution_parameter, signed = signed)

    positive_counts = Counter(positive_partition.values())
    negative_counts = Counter(negative_partition.values())

    valid_positive = {community for community, count in positive_counts.items() if count >= 2}
    valid_negative = {community for community, count in negative_counts.items() if count >= 2}

    positive_palette = {community: colors_pos[index % len(colors_pos)] for index, community in enumerate(sorted(valid_positive))}
    negative_palette = {community: colors_neg[index % len(colors_neg)] for index, community in enumerate(sorted(valid_negative))}

    for node in graph.nodes():
        positive_community = positive_partition.get(str(node))
        negative_community = negative_partition.get(str(node))
        positive_color = positive_palette.get(positive_community)
        negative_color = negative_palette.get(negative_community)

        if title_prefix == 'Positive':
            graph.nodes[node]['fillcolor'] = positive_color if positive_color else 'silver'
        elif title_prefix == 'Negative':
            graph.nodes[node]['fillcolor'] = negative_color if negative_color else 'silver'
        elif positive_color and negative_color:
            graph.nodes[node]['fillcolor'] = (positive_color, negative_color)
        elif positive_color:
            graph.nodes[node]['fillcolor'] = positive_color
        elif negative_color:
            graph.nodes[node]['fillcolor'] = negative_color
        else:
            graph.nodes[node]['fillcolor'] = 'silver'

    for first, second in graph.edges():
        importance = float(graph[first][second].get('Importance', 0.0))

        if title_prefix == 'Positive':
            graph[first][second]['color'] = 'royalblue'
        elif title_prefix == 'Negative':
            graph[first][second]['color'] = 'coral'
        else:
            graph[first][second]['color'] = 'royalblue' if importance >= 0 else 'coral'

    graph.graph['positive_partition'] = positive_partition
    graph.graph['negative_partition'] = negative_partition

    return graph


def draw_nodes_with_fillcolor(graph, positions, axis, node_size=600):
    if graph.number_of_nodes() == 0:
        axis.axis('off')
        return

    radius = node_size ** 0.5 / 350

    for node in graph.nodes():
        fillcolor = graph.nodes[node].get('fillcolor', 'silver')
        x_coordinate, y_coordinate = positions[node]

        if isinstance(fillcolor, tuple):
            first_color, second_color = fillcolor
            axis.add_patch(Wedge((x_coordinate, y_coordinate), radius, 90, 270, facecolor=first_color, edgecolor='black', linewidth=0.25))
            axis.add_patch(Wedge((x_coordinate, y_coordinate), radius, 270, 450, facecolor=second_color, edgecolor='black', linewidth=0.25))
        else:
            axis.add_patch(Circle((x_coordinate, y_coordinate), radius, facecolor=fillcolor, edgecolor='black', linewidth=0.5))

    axis.set_xlim(min(position[0] for position in positions.values()) - radius, max(position[0] for position in positions.values()) + radius)
    axis.set_ylim(min(position[1] for position in positions.values()) - radius, max(position[1] for position in positions.values()) + radius)
    axis.set_aspect('equal')
    axis.axis('off')


def plot_rule_network(dataframe, rows, columns, dependent_variables, contexts=None, layout='spring', filename='network.pdf', title_prefix='', draw_partition=False, resolution_parameter=None):
    
    combinations = [(None, dependent_variable) for dependent_variable in dependent_variables] if contexts is None or len(contexts) == 0 else [(context, dependent_variable) for context in contexts for dependent_variable in dependent_variables]

    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 6 * rows), squeeze=False)

    for index, (context, dependent_variable) in enumerate(combinations):
        row, column = divmod(index, columns)

        if row >= rows:
            break

        axis = axes[row][column]

        if context is not None and 'Context' in dataframe.columns:
            filtered = dataframe[(dataframe['DV'] == dependent_variable) & (dataframe['Context'] == context)]
        else:
            filtered = dataframe[dataframe['DV'] == dependent_variable]

        graph = build_graph(filtered)
        title = (str(context) + ' ' if context is not None else '') + str(dependent_variable)

        if graph.number_of_nodes() == 0:
            axis.set_title(title)
            axis.axis('off')
            continue

        if draw_partition:
            graph = assign_node_edge_colors(graph, title_prefix=title_prefix, resolution_parameter=resolution_parameter)

        positions = nx.spring_layout(graph, seed=RANDOM_SEED) if layout == 'spring' else nx.circular_layout(graph, scale=2)
        edge_colors = [graph[first][second].get('color', 'gray') for first, second in graph.edges()]
        edge_widths = [max(0.5, abs(float(graph[first][second].get('Importance', 0.0))) * 8) for first, second in graph.edges()]

        nx.draw_networkx_edges(graph, positions, ax=axis, edge_color=edge_colors, width=edge_widths)

        if draw_partition:
            draw_nodes_with_fillcolor(graph, positions, axis, node_size=600)
        else:
            nx.draw_networkx_nodes(graph, positions, ax=axis, node_color='silver', node_size=600)

        nx.draw_networkx_labels(graph, positions, ax=axis)
        axis.set_title(title)
        axis.axis('off')

    for index in range(min(len(combinations), rows * columns), rows * columns):
        row, column = divmod(index, columns)
        figure.delaxes(axes[row][column])

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(figure)
    

def make_difference_networks(rule_dataframe, include_context=False):
    if rule_dataframe.empty:
        return {}, pd.DataFrame()

    positive = count_cooccurrences(rule_dataframe[rule_dataframe['type'] == 'positive'], 'positive', include_context=include_context)
    negative = count_cooccurrences(rule_dataframe[rule_dataframe['type'] == 'negative'], 'negative', include_context=include_context)

    merge_columns = ['Condition1', 'Condition2', 'DV'] + (['Context'] if include_context else [])

    positive = positive.rename(columns={'Importance': 'Positive_Importance'}).drop(columns='Label', errors='ignore')
    negative = negative.rename(columns={'Importance': 'Negative_Importance'}).drop(columns='Label', errors='ignore')

    if positive.empty and negative.empty:
        output_columns = merge_columns + ['Positive_Importance', 'Negative_Importance', 'Importance', 'Label']
        return {}, pd.DataFrame(columns=output_columns)

    difference = pd.merge(positive, negative, on=merge_columns, how='outer')
    difference['Positive_Importance'] = difference['Positive_Importance'].fillna(0.0)
    difference['Negative_Importance'] = difference['Negative_Importance'].fillna(0.0)
    difference['Importance'] = difference['Positive_Importance'] - difference['Negative_Importance']
    difference['Label'] = 'Difference'

    graph_dictionary = {}

    if include_context:
        for group_key, group_dataframe in difference.groupby(['Context', 'DV']):
            graph_dictionary[group_key] = build_graph(group_dataframe)
    else:
        for dependent_variable, group_dataframe in difference.groupby('DV'):
            graph_dictionary[dependent_variable] = build_graph(group_dataframe)

    return graph_dictionary, difference



def calculate_network_resolution(graph, multiplier=10):
    weights = [abs(float(attributes.get('Importance', 0.0))) for _, _, attributes in graph.edges(data=True) if not np.isclose(float(attributes.get('Importance', 0.0)), 0.0)]

    if not weights:
        return 1.0

    return float(np.mean(weights)) * float(multiplier)


def compare_networks_to_primary(network_rows, grouping_columns):
    comparison_rows = []
    network_lookup = {}

    for row in network_rows:
        key = tuple(row[column] for column in grouping_columns) + (row['Specification'],)
        network_lookup[key] = row

    groups = sorted(set(tuple(row[column] for column in grouping_columns) for row in network_rows))
    specifications = sorted(set(row['Specification'] for row in network_rows))

    for group in groups:
        primary_key = group + ('primary',)

        if primary_key not in network_lookup:
            continue

        primary_entry = network_lookup[primary_key]
        primary_graph = primary_entry['Graph']
        primary_resolution = calculate_network_resolution(primary_graph, primary_entry['Resolution_Multiplier'])

        for specification_name in specifications:
            comparison_key = group + (specification_name,)

            if comparison_key not in network_lookup:
                continue

            comparison_entry = network_lookup[comparison_key]
            comparison_graph = comparison_entry['Graph']
            comparison_resolution = calculate_network_resolution(comparison_graph, comparison_entry['Resolution_Multiplier'])

            community_similarity, edge_similarity, node_similarity = color_similarity_distance(
                primary_graph,
                comparison_graph,
                first_resolution=primary_resolution,
                second_resolution=comparison_resolution
            )

            row = {
                'Specification': specification_name,
                'Resolution_Multiplier': comparison_entry['Resolution_Multiplier'],
                'Community_ARI': community_similarity,
                'Signed_Edge_Jaccard': edge_similarity,
                'Node_Jaccard': node_similarity
            }

            for column, value in zip(grouping_columns, group):
                row[column] = value

            comparison_rows.append(row)

    return pd.DataFrame(comparison_rows)



def add_network_only_rows(network_rows, grouping_columns, network_specifications):
    """
    Duplicate primary networks for network-only sensitivity specifications.

    Model and ranking sensitivity analyses generate their own rule sets and
    therefore their own networks. Network-only sensitivity specifications
    instead reuse the primary network structure while varying the network
    resolution parameter used during community detection.
    """

    primary_rows = [
        row
        for row in network_rows
        if row['Specification'] == 'primary'
    ]

    additional_rows = []

    for specification in network_specifications:

        for primary_row in primary_rows:

            new_row = {
                column: primary_row[column]
                for column in grouping_columns
            }

            new_row['Specification'] = specification['SPECIFICATION']
            new_row['Graph'] = primary_row['Graph'].copy()
            new_row['Resolution_Multiplier'] = (
                specification['NETWORK_RESOLUTION_MULTIPLIER']
            )

            additional_rows.append(new_row)

    return network_rows + additional_rows



'''
Network similarity
'''

def jaccard_similarity(first_set, second_set):
    if not first_set and not second_set:
        return 1.0

    union = first_set | second_set

    if not union:
        return np.nan

    return len(first_set & second_set) / len(union)


def color_similarity_distance(first_graph, second_graph, first_resolution=None, second_resolution=None):
    first_nodes = {str(node) for node in first_graph.nodes()}
    second_nodes = {str(node) for node in second_graph.nodes()}
    shared_nodes = sorted(first_nodes & second_nodes)
    node_union = first_nodes | second_nodes
    node_jaccard = len(shared_nodes) / len(node_union) if node_union else 1.0

    if len(shared_nodes) >= 2:
        first_positive = run_leiden_partition(first_graph, edge_sign='positive', resolution_parameter=first_resolution)
        second_positive = run_leiden_partition(second_graph, edge_sign='positive', resolution_parameter=second_resolution)
        first_negative = run_leiden_partition(first_graph, edge_sign='negative', resolution_parameter=first_resolution)
        second_negative = run_leiden_partition(second_graph, edge_sign='negative', resolution_parameter=second_resolution)

        positive_ari = adjusted_rand_score([first_positive.get(node, -1) for node in shared_nodes], [second_positive.get(node, -1) for node in shared_nodes])
        negative_ari = adjusted_rand_score([first_negative.get(node, -1) for node in shared_nodes], [second_negative.get(node, -1) for node in shared_nodes])
        community_similarity = (positive_ari + negative_ari) / 2
    else:
        community_similarity = np.nan

    def signed_edge_set(graph):
        edges = set()

        for first, second, attributes in graph.edges(data=True):
            pair = tuple(sorted((str(first), str(second))))
            sign = 1 if float(attributes.get('Importance', 0.0)) >= 0 else -1
            edges.add((pair[0], pair[1], sign))

        return edges

    edge_similarity = jaccard_similarity(signed_edge_set(first_graph), signed_edge_set(second_graph))

    return community_similarity, edge_similarity, node_jaccard


def get_node_color_parts(fillcolor):
    if isinstance(fillcolor, tuple):
        return fillcolor[0], fillcolor[1]

    if fillcolor:
        return fillcolor, fillcolor

    return 'silver', 'silver'


def resolve_color(primary, secondary, positive_colors, negative_colors):
    valid_colors = list(positive_colors) + list(negative_colors) + ['silver']
    primary = primary if primary in valid_colors else None
    secondary = secondary if secondary in valid_colors else None

    if primary is None and secondary is None:
        return None

    if primary == 'silver' and secondary == 'silver':
        return 'neutral'

    if primary == 'silver':
        return secondary

    if secondary == 'silver' or secondary is None:
        return primary

    if primary != secondary:
        return 'dual'

    return primary


'''
Primary Heatmap / Association Figures
'''


def plot_difference_heatmap(difference_aggregate,dependent_variables,filename,contexts=None):

    if contexts is None:
        contexts=[None]

    fig,axes=plt.subplots(len(contexts),len(dependent_variables),figsize=(6*len(dependent_variables),6*len(contexts)))

    if len(contexts)==1 and len(dependent_variables)==1:
        axes=np.array([[axes]])
    elif len(contexts)==1:
        axes=np.array([axes])
    elif len(dependent_variables)==1:
        axes=np.array([[ax] for ax in axes])

    balance=defaultdict(float)

    for _,row in difference_aggregate.iterrows():
        balance[row['Condition1']]+=row['Importance']
        balance[row['Condition2']]+=row['Importance']

    order=pd.Series(balance).sort_values().index.tolist()

    for r,context in enumerate(contexts):

        for c,dv in enumerate(dependent_variables):

            ax=axes[r,c]

            subset=difference_aggregate[difference_aggregate['DV']==dv]

            if context is not None:
                subset=subset[subset['Context']==context]

            if subset.empty:
                ax.axis('off')
                continue

            mat=pd.DataFrame(0.0,index=order,columns=order)

            for _,row in subset.iterrows():
                mat.loc[row['Condition1'],row['Condition2']]=row['Importance']
                mat.loc[row['Condition2'],row['Condition1']]=row['Importance']

            np.fill_diagonal(mat.to_numpy(copy=True),np.nan)
            vmax=np.nanmax(np.abs(mat.values))

            sns.heatmap(mat.loc[order,order],cmap='RdBu_r',center=0,vmin=-vmax,vmax=vmax,square=True,ax=ax,cbar=(r==0 and c==len(dependent_variables)-1))

            ax.set_title(dv if context is None else f'{context} - {dv}')

    plt.tight_layout()
    plt.savefig(filename,bbox_inches='tight')
    plt.close()
    
    
def make_circular_layout(nodes,start_angle=np.pi/2):

    theta=np.linspace(0,2*np.pi,len(nodes), endpoint=False)
    theta=theta+start_angle
    pos={node:(np.cos(t),np.sin(t)) for node,t in zip(nodes,theta)}
    return pos
    
def draw_hypergraph_density(ax, df, pos, node_importance, outcome, alpha_min=0.03, alpha_max=0.25, node_scale=1000):

    imp=df['importance']

    if imp.max()==imp.min():
        norm=np.ones(len(imp))
    else:
        norm=(imp-imp.min())/(imp.max()-imp.min())

    df=df.copy()
    df['norm']=norm
    df=df.sort_values('norm')

    if outcome=="positive":
        cmap=mcolors.LinearSegmentedColormap.from_list("pos",["white","royalblue"])
    else:
        cmap=mcolors.LinearSegmentedColormap.from_list("neg",["white","darkorange"]        )

    # draw polygons
    for _,row in df.iterrows():
        lhs=row['rule'].split('->')[0]
        dps=[x.strip().replace('(X)','')for x in lhs.split('&')]

        verts=np.array([pos[dp] for dp in dps if dp in pos])

        if len(verts)<3:
            continue

        v=row['norm']
        color=list(cmap(v))
        color[-1]=alpha_min+(alpha_max-alpha_min)*v

        poly=Polygon(verts,closed=True, facecolor=tuple(color), edgecolor='none', zorder=1)
        
        ax.add_patch(poly)

    # draw nodes

    node_importance=node_importance.copy()

    if node_importance.max()>0:
        node_importance=node_importance/node_importance.max()

    for node,(x,y) in pos.items():

        size=10+node_scale*node_importance.get(node,0)
        ax.scatter(x,y,s=size,facecolor="lightgrey", edgecolor="grey", linewidth=1,zorder=10)

    # draw labels outside circle

    for node,(x,y) in pos.items():
        r=np.sqrt(x*x+y*y)

        ax.text(1.18*x/r, 1.18*y/r, node, fontsize=20, ha="center",va="center", zorder=20)

    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_xlim(-1.35,1.35)
    ax.set_ylim(-1.35,1.35)
    
    
    
    
    
    