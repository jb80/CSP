#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Wed Sep 24 10:42:00 2025

Probabilistic Soft Logic analysis with sensitivity analysis.

The script:

1. Loads threshold-specific analysis data and LASSO results.
2. Starts the PSL Java virtual machine.
3. Defines a primary PSL specification.
4. Defines model-stage, ranking-stage, and network-stage sensitivity analyses.
5. Estimates aggregate and topic-context PSL models.
6. Learns initial rule weights.
7. Filters competing positive and negative rules.
8. Re-estimates filtered PSL models.
9. Ranks configurations using alternative importance functions.
10. Retains alternative top-N rule sets.
11. Constructs condition co-occurrence networks.
12. Compares rules and networks across specifications.
13. Saves rule-retention, direction, ranking, edge, node, and community robustness.

The primary missing-probability threshold is 0.66.

Missing-probability thresholds of 0.50 and 0.75 use threshold-specific coding
scores and threshold-specific LASSO results produced by script 5.

PSL goodness-of-fit statistics are in-sample reconstruction diagnostics and
are not interpreted as external predictive performance.

@author: jbaggio
"""


'''
Packages
'''

import os
import pickle
import warnings
from pathlib import Path
from copy import deepcopy
from collections import defaultdict


import jpype
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine

#import already written functions for PSL analysis
import CSP_6_PSL_ModelBuilder as pqm


'''
Directories
'''
mainres = Path('/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis/MainResults')
pslout = Path('/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis/PSL')
lassoout = pslout / 'LASSO'
sensitivityout = pslout / 'Sensitivity'
figureout = pslout / 'Figures'
networkout = pslout / 'Networks'

for directory in [pslout, sensitivityout, figureout, networkout]:
    directory.mkdir(parents=True, exist_ok=True)


'''
Reproducibility and settings
'''

RANDOM_SEED = 7879

np.random.seed(RANDOM_SEED)
warnings.filterwarnings('ignore')

OUTCOME_COLUMNS = ['RSC', 'CFL', 'IEQ']
FEATURE_COLUMNS = ['SCB', 'RSB', 'FIT', 'PRP', 'PRT', 'MON', 'ACM', 'GRS', 'CFR', 'AUT', 'NST']

PRIMARY_MISSING_PROBABILITY_THRESHOLD = 0.66
LASSO_COEFFICIENT_TOLERANCE = 1e-4

LASSO_SELECTION_CUTOFF = 0.33
LASSO_SIGN_CUTOFF = 0 #not used as PSL handles this better. only used if LASSO is the last step (keep it in case one prefers that)

SAVE_LIVE_PSL_MODELS = True


'''
Load analysis data
'''

with open(mainres / 'Analysis_Data.p', 'rb') as input_file:
    analysis_data = pickle.load(input_file)

required_analysis_keys = ['analysis_data_by_threshold', 'dfall', 'depvars_scaled', 'indepvars_scaled']
missing_analysis_keys = [key for key in required_analysis_keys if key not in analysis_data]

if missing_analysis_keys:
    raise KeyError('Analysis_Data.p is missing keys: ' + str(missing_analysis_keys))

analysis_data_by_threshold = analysis_data['analysis_data_by_threshold']

expected_threshold_keys = ['0.50', '0.66', '0.75']
missing_threshold_keys = [key for key in expected_threshold_keys if key not in analysis_data_by_threshold]

if missing_threshold_keys:
    raise KeyError('Analysis_Data.p is missing threshold datasets: ' + str(missing_threshold_keys))


'''
Load primary LASSO results
'''

with open(lassoout / 'LASSO_All.p', 'rb') as input_file:
    lasso_all = pickle.load(input_file)
    
with open(lassoout / 'LASSO_Stability_All.p', 'rb') as input_file:
    lasso_stability_all = pickle.load(input_file)

with open(lassoout / 'LASSO_Topic.p', 'rb') as input_file:
    lasso_topic = pickle.load(input_file)

with open(lassoout / 'LASSO_Stability_Topic.p', 'rb') as input_file:
    lasso_stability_topic = pickle.load(input_file)


'''
Load threshold-specific LASSO results
'''

with open(lassoout / 'LASSO_Threshold_All.p', 'rb') as input_file:
    threshold_lasso_all = pickle.load(input_file)

with open(lassoout / 'LASSO_Threshold_Stability_All.p', 'rb') as input_file:
    threshold_lasso_stability_all = pickle.load(input_file)

with open(lassoout / 'LASSO_Threshold_Topic.p', 'rb') as input_file:
    threshold_lasso_topic = pickle.load(input_file)

with open(lassoout / 'LASSO_Threshold_Stability_Topic.p', 'rb') as input_file:
    threshold_lasso_stability_topic = pickle.load(input_file)


'''
Start Java virtual machine
'''

os.environ['JAVA_HOME'] = '/Users/jbaggio/miniconda3/envs/topmod'

library_directory = Path('/Users/jbaggio/miniconda3/envs/topmod/lib/psl-jars')

if not library_directory.exists():
    raise FileNotFoundError('PSL JAR directory does not exist: ' + str(library_directory))

classpath = [str(jar) for jar in library_directory.iterdir() if jar.suffix == '.jar']

if not classpath:
    raise FileNotFoundError('No PSL JAR files were found in: ' + str(library_directory))

if not jpype.isJVMStarted():
    jpype.startJVM(jpype.getDefaultJVMPath(), classpath=classpath)


'''
Primary specification
'''

PRIMARY_SPECIFICATION = {
    'SPECIFICATION': 'primary',
    'SENSITIVITY_STAGE': 'model',
    'MISSING_PROBABILITY_THRESHOLD': PRIMARY_MISSING_PROBABILITY_THRESHOLD,
    'LASSO_SELECTION_THRESHOLD': LASSO_SELECTION_CUTOFF,
    'LASSO_SIGN_STABILITY_THRESHOLD': LASSO_SIGN_CUTOFF,
    'USE_FULL_LASSO_MODEL': False,
    'USE_GLOBAL_EXPLORATORY': False,
    'SUBSET_DECAY': 0.50,
    'ADD_NEGATIVE_DV_PRIOR': True,
    'DV_PRIOR_WEIGHT': 0.10,
    'MAX_K': 5,
    'MIN_K': 3,
    'MIN_SUPPORT_FRAC': 0.05,
    'MIN_MEAN_BODY': 0.00,
    'ALL_IVOBS': True,
    'LASSO_W': False,
    'LASSO_DIRECTION_MODE': 'both',
    'SUBSET_INCL': False,
    'SCREENING_OPERATOR': 'lukas_avg',
    'MIN_RULE_WEIGHT': 1e-3,
    'MAX_RULES': 5000,
    'IMPORTANCE_METRIC': 'cov_cons_eff',
    'TOP_N': None,
    'NETWORK_RESOLUTION_MULTIPLIER': 1.00
}


'''
Helper Functions
'''

'''
Specification and threshold
'''

def make_specification(name, stage='model', **changes):
    specification = deepcopy(PRIMARY_SPECIFICATION)
    specification['SPECIFICATION'] = name
    specification['SENSITIVITY_STAGE'] = stage
    specification.update(changes)

    return specification


def threshold_key_from_specification(specification):
    threshold = specification.get('MISSING_PROBABILITY_THRESHOLD', PRIMARY_MISSING_PROBABILITY_THRESHOLD)

    return format(float(threshold), '.2f')


def get_analysis_data_for_specification(specification):
    threshold_key = threshold_key_from_specification(specification)

    if threshold_key not in analysis_data_by_threshold:
        raise KeyError('No analysis data found for threshold ' + threshold_key)

    threshold_data = analysis_data_by_threshold[threshold_key]
    threshold_dfall = threshold_data['dfall'].copy()
    threshold_depvars = threshold_data['depvars_scaled'].copy()
    threshold_indepvars = threshold_data['indepvars_scaled'].copy()

    required_dataframe_columns = ['txt_title', 'topname']
    missing_dataframe_columns = [column for column in required_dataframe_columns if column not in threshold_dfall.columns]

    if missing_dataframe_columns:
        raise KeyError('Threshold ' + threshold_key + ' is missing columns: ' + str(missing_dataframe_columns))

    missing_outcomes = [column for column in OUTCOME_COLUMNS if column not in threshold_depvars.columns]
    missing_features = [column for column in FEATURE_COLUMNS if column not in threshold_indepvars.columns]

    if missing_outcomes:
        raise KeyError('Threshold ' + threshold_key + ' is missing outcomes: ' + str(missing_outcomes))

    if missing_features:
        raise KeyError('Threshold ' + threshold_key + ' is missing predictors: ' + str(missing_features))

    if len(threshold_dfall) != len(threshold_depvars) or len(threshold_dfall) != len(threshold_indepvars):
        raise ValueError('Threshold ' + threshold_key + ' has inconsistent dataframe lengths.')

    titles = threshold_dfall['txt_title'].astype(str)

    if titles.duplicated().any():
        raise ValueError('Threshold ' + threshold_key + ' contains duplicated titles.')

    threshold_indepvars.index = titles
    threshold_depvars.index = titles

    return threshold_key, threshold_dfall, threshold_indepvars, threshold_depvars


'''
Topic Groups
'''

def create_topic_groups(threshold_dfall, threshold_indepvars, threshold_depvars):
    independent_topic_data = threshold_indepvars.copy()
    dependent_topic_data = threshold_depvars.copy()

    independent_topic_data['topname'] = threshold_dfall['topname'].to_numpy()
    dependent_topic_data['topname'] = threshold_dfall['topname'].to_numpy()

    return independent_topic_data.groupby('topname'), dependent_topic_data.groupby('topname')


'''
Rule-frame function
'''

def append_rule_frames(rule_frames, positive_rules, negative_rules, specification, depvar, threshold_key, context=None):
    metadata = {
        'Specification': specification['SPECIFICATION'],
        'Sensitivity_Stage': specification['SENSITIVITY_STAGE'],
        'Missing_Probability_Threshold': float(threshold_key),
        'DV': depvar,
        'Importance_Metric': specification['IMPORTANCE_METRIC'],
        'Top_N_Per_Direction': specification['TOP_N']
    }

    if context is not None:
        metadata['Context'] = context

    positive_rules = positive_rules.assign(**metadata, type='positive')
    negative_rules = negative_rules.assign(**metadata, type='negative')
    rule_frames.extend([positive_rules, negative_rules])



'''

START RUNNING SCRIPT

'''


model_specifications = [
    deepcopy(PRIMARY_SPECIFICATION),

    make_specification('lasso_full', USE_FULL_LASSO_MODEL=True),
    make_specification('min_k_2', MIN_K=2),

    make_specification('support_0025', MIN_SUPPORT_FRAC=0.025),
    make_specification('support_010', MIN_SUPPORT_FRAC=0.10),

    make_specification('lasso_weights', LASSO_W=True),

    make_specification('full_subset', SUBSET_INCL=True),
    make_specification('subset_decay_025', SUBSET_DECAY=0.25),
    make_specification('subset_decay_075', SUBSET_DECAY=0.75),

    make_specification('missing_threshold_050', MISSING_PROBABILITY_THRESHOLD=0.50),
    make_specification('missing_threshold_075', MISSING_PROBABILITY_THRESHOLD=0.75),
]

ranking_specifications = [
    make_specification('importance_coverage_effect', stage='ranking', IMPORTANCE_METRIC='coverage_effect'),
    make_specification('importance_weighted_consistency', stage='ranking', IMPORTANCE_METRIC='weighted_consistency'),
    make_specification('importance_coverage_consistency', stage='ranking', IMPORTANCE_METRIC='coverage_consistency'),
]

network_specifications = [
    make_specification('network_resolution_075', stage='network', NETWORK_RESOLUTION_MULTIPLIER=0.75),
    make_specification('network_resolution_125', stage='network', NETWORK_RESOLUTION_MULTIPLIER=1.25)
]

rule_specifications = model_specifications + ranking_specifications
all_specifications = model_specifications + ranking_specifications + network_specifications
specification_lookup = {specification['SPECIFICATION']: specification for specification in all_specifications}

if len(specification_lookup) != len(all_specifications):
    raise ValueError('Sensitivity specification names are not unique.')



'''
Run aggregate model sensitivity
'''

aggregate_results = {}
aggregate_diagnostics = []
aggregate_rule_frames = []

for specification in model_specifications:
    specification_name = specification['SPECIFICATION']
    threshold_key, specification_dfall, specification_indepvars, specification_depvars = get_analysis_data_for_specification(specification)

    print('\nAggregate model specification:', specification_name)
    print('Missing-probability threshold:', threshold_key)

    for dependent_variable in OUTCOME_COLUMNS:
        candidate_rules = pqm.get_candidate_rules(lasso_stability_all, lasso_all, dependent_variable, specification, 
            threshold_stability_dictionary=threshold_lasso_stability_all, threshold_fallback_dictionary=threshold_lasso_all, threshold_key=threshold_key, coefficient_tolerance=LASSO_COEFFICIENT_TOLERANCE)
        
        if not candidate_rules:
            print('No aggregate candidate rules:', specification_name, dependent_variable)
            continue

        result = pqm.fit_psl_model(dependent_variable, specification_indepvars, specification_depvars, candidate_rules, specification)
        aggregate_results[(specification_name, dependent_variable)] = result

        diagnostics = result['diagnostics'].copy()
        diagnostics.update({
            'Specification': specification_name,
            'Sensitivity_Stage': specification['SENSITIVITY_STAGE'],
            'Missing_Probability_Threshold': float(threshold_key),
            'DV': dependent_variable
        })

        aggregate_diagnostics.append(diagnostics)
        append_rule_frames(aggregate_rule_frames, result['positive_rules'], result['negative_rules'], specification, dependent_variable, threshold_key)


'''
Run aggregate ranking sensitivity
'''

primary_threshold_key, primary_dfall, primary_indepvars, primary_depvars = get_analysis_data_for_specification(PRIMARY_SPECIFICATION)

for specification in ranking_specifications:
    print('\nAggregate ranking specification:', specification['SPECIFICATION'])

    for dependent_variable in OUTCOME_COLUMNS:
        result_key = ('primary', dependent_variable)

        if result_key not in aggregate_results:
            continue

        positive_rules, negative_rules = pqm.rank_filtered_model(aggregate_results[result_key], dependent_variable, primary_indepvars, primary_depvars, specification)
        append_rule_frames(aggregate_rule_frames, positive_rules, negative_rules, specification, dependent_variable, primary_threshold_key)


'''
Run topic model sensitivity
'''

topic_results = {}
topic_diagnostics = []
topic_rule_frames = []

for specification in model_specifications:
    specification_name = specification['SPECIFICATION']
    threshold_key, specification_dfall, specification_indepvars, specification_depvars = get_analysis_data_for_specification(specification)
    specification_gr_indep, specification_gr_dep = create_topic_groups(specification_dfall, specification_indepvars, specification_depvars)

    print('\nTopic model specification:', specification_name)
    print('Missing-probability threshold:', threshold_key)

    for topic_name in sorted(specification_dfall['topname'].dropna().unique()):
        independent_topic = specification_gr_indep.get_group(topic_name).drop(columns='topname')
        dependent_topic = specification_gr_dep.get_group(topic_name).drop(columns='topname')

        for dependent_variable in OUTCOME_COLUMNS:
            key = (topic_name, dependent_variable)

            candidate_rules = pqm.get_candidate_rules(lasso_stability_topic, lasso_topic, key, specification, threshold_key = threshold_key,
               threshold_stability_dictionary=threshold_lasso_stability_topic, threshold_fallback_dictionary=threshold_lasso_topic, coefficient_tolerance=LASSO_COEFFICIENT_TOLERANCE)

            if not candidate_rules:
                print('No topic candidate rules:', specification_name, topic_name, dependent_variable)
                continue

            result = pqm.fit_psl_model(dependent_variable, independent_topic, dependent_topic, candidate_rules, specification)
            topic_results[(specification_name, topic_name, dependent_variable)] = result

            diagnostics = result['diagnostics'].copy()
            diagnostics.update({
                'Specification': specification_name,
                'Sensitivity_Stage': specification['SENSITIVITY_STAGE'],
                'Missing_Probability_Threshold': float(threshold_key),
                'Context': topic_name,
                'DV': dependent_variable,
                'N_Context_Observations': len(independent_topic)
            })

            topic_diagnostics.append(diagnostics)
            append_rule_frames(topic_rule_frames, result['positive_rules'], result['negative_rules'], specification, dependent_variable, threshold_key, context=topic_name)


'''
Run topic ranking sensitivity
'''

primary_gr_indep, primary_gr_dep = create_topic_groups(primary_dfall, primary_indepvars, primary_depvars)

for specification in ranking_specifications:
    print('\nTopic ranking specification:', specification['SPECIFICATION'])

    for topic_name in sorted(primary_dfall['topname'].dropna().unique()):
        independent_topic = primary_gr_indep.get_group(topic_name).drop(columns='topname')
        dependent_topic = primary_gr_dep.get_group(topic_name).drop(columns='topname')

        for dependent_variable in OUTCOME_COLUMNS:
            result_key = ('primary', topic_name, dependent_variable)

            if result_key not in topic_results:
                continue

            positive_rules, negative_rules = pqm.rank_filtered_model(topic_results[result_key], dependent_variable, independent_topic, dependent_topic, specification)
            append_rule_frames(topic_rule_frames, positive_rules, negative_rules, specification, dependent_variable, primary_threshold_key, context=topic_name)



'''
Combine and save results
'''

aggregate_diagnostics = pd.DataFrame(aggregate_diagnostics)
topic_diagnostics = pd.DataFrame(topic_diagnostics)

aggregate_rules = pd.concat(aggregate_rule_frames, ignore_index=True) if aggregate_rule_frames else pd.DataFrame()
topic_rules = pd.concat(topic_rule_frames, ignore_index=True) if topic_rule_frames else pd.DataFrame()

aggregate_diagnostics.to_csv(sensitivityout / 'PSL_Aggregate_Diagnostics.csv', index=False)
topic_diagnostics.to_csv(sensitivityout / 'PSL_Topic_Diagnostics.csv', index=False)
aggregate_rules.to_csv(sensitivityout / 'PSL_Aggregate_Rules.csv', index=False)
topic_rules.to_csv(sensitivityout / 'PSL_Topic_Rules.csv', index=False)


'''
Save serializable result summaries
'''

serializable_aggregate_results = {
    key: {
        'rule_metadata': value['rule_metadata'],
        'filtered_rule_metadata': value['filtered_rule_metadata'],
        'positive_rules': value['positive_rules'],
        'negative_rules': value['negative_rules'],
        'diagnostics': value['diagnostics']
    }
    for key, value in aggregate_results.items()
}

serializable_topic_results = {
    key: {
        'rule_metadata': value['rule_metadata'],
        'filtered_rule_metadata': value['filtered_rule_metadata'],
        'positive_rules': value['positive_rules'],
        'negative_rules': value['negative_rules'],
        'diagnostics': value['diagnostics']
    }
    for key, value in topic_results.items()
}

with open(sensitivityout / 'PSL_Aggregate_Results.p', 'wb') as output_file:
    pickle.dump(serializable_aggregate_results, output_file)

with open(sensitivityout / 'PSL_Topic_Results.p', 'wb') as output_file:
    pickle.dump(serializable_topic_results, output_file)

if SAVE_LIVE_PSL_MODELS:
    with open(sensitivityout / 'PSL_Aggregate_Live_Models.p', 'wb') as output_file:
        pickle.dump(aggregate_results, output_file)

    with open(sensitivityout / 'PSL_Topic_Live_Models.p', 'wb') as output_file:
        pickle.dump(topic_results, output_file)


'''
Rule Selection Robustness
'''

aggregate_rule_robustness = pqm.summarize_rule_robustness(aggregate_rules, ['DV'])
topic_rule_robustness = pqm.summarize_rule_robustness(topic_rules, ['Context', 'DV'])

aggregate_rule_robustness.to_csv(sensitivityout / 'PSL_Aggregate_Rule_Robustness.csv', index=False)
topic_rule_robustness.to_csv(sensitivityout / 'PSL_Topic_Rule_Robustness.csv', index=False)


'''
Rule-direction robustness
'''


aggregate_direction_robustness = pqm.summarize_direction_robustness(aggregate_rules, ['DV'])
topic_direction_robustness = pqm.summarize_direction_robustness(topic_rules, ['Context', 'DV'])

aggregate_direction_robustness.to_csv(sensitivityout / 'PSL_Aggregate_Direction_Robustness.csv', index=False)
topic_direction_robustness.to_csv(sensitivityout / 'PSL_Topic_Direction_Robustness.csv', index=False)


'''
Rule Overlap
'''


aggregate_rule_overlap = pqm.calculate_rule_set_overlap(aggregate_rules, ['DV'])
topic_rule_overlap = pqm.calculate_rule_set_overlap(topic_rules, ['Context', 'DV'])

aggregate_rule_overlap.to_csv(sensitivityout / 'PSL_Aggregate_Rule_Overlap.csv', index=False)
topic_rule_overlap.to_csv(sensitivityout / 'PSL_Topic_Rule_Overlap.csv', index=False)


'''
Importance Ranking
'''


aggregate_rank_correlations = pqm.calculate_rank_correlations(aggregate_rules, ['DV', 'type'])
topic_rank_correlations = pqm.calculate_rank_correlations(topic_rules, ['Context', 'DV', 'type'])

aggregate_rank_correlations.to_csv(sensitivityout / 'PSL_Aggregate_Rank_Correlations.csv', index=False)
topic_rank_correlations.to_csv(sensitivityout / 'PSL_Topic_Rank_Correlations.csv', index=False)


# '''
# if not consecutive = load the files (so the longer PSL rule step can be avoided)
# '''

# os.chdir(sensitivityout)

# aggregate_diagnostics = pd.read_csv ('PSL_Aggregate_Diagnostics.csv')
# topic_diagnostics = pd.read_csv('PSL_Topic_Diagnostics.csv')
# aggregate_rules = pd.read_csv('PSL_Aggregate_Rules.csv')
# topic_rules = pd.read_csv('PSL_Topic_Rules.csv')

# aggregate_rule_robustness = pd.read_csv('PSL_Aggregate_Rule_Robustness.csv')
# topic_rule_robustness = pd.read_csv('PSL_Topic_Rule_Robustness.csv')

# aggregate_direction_robustness = pd.read_csv( 'PSL_Aggregate_Direction_Robustness.csv')
# topic_direction_robustness = pd.read_csv('PSL_Topic_Direction_Robustness.csv')

# aggregate_rule_overlap = pd.read_csv('PSL_Aggregate_Rule_Overlap.csv')
# topic_rule_overlap = pd.read_csv('PSL_Topic_Rule_Overlap.csv')

# aggregate_network_robustness = pd.read_csv( 'PSL_Aggregate_Network_Robustness.csv')
# topic_network_robustness = pd.read_csv('PSL_Topic_Network_Robustness.csv')

# aggregate_rank_correlations = pd.read_csv(  'PSL_Aggregate_Rank_Correlations.csv')
# topic_rank_correlations = pd.read_csv(  'PSL_Topic_Rank_Correlations.csv')

# os.chdir(pslout)
# primary_aggregate_rules = pd.read_csv(  'Configs_All.csv')
# primary_topic_rules = pd.read_csv( 'Configs_Topic.csv')

# aggregate_difference = pd.read_csv( 'Network_Difference_All.csv')
# topic_difference = pd.read_csv(  'Network_Difference_Topic.csv')


'''
Build aggregate and topic networks
'''

aggregate_network_rows = []
topic_network_rows = []

for specification_name in sorted(aggregate_rules['Specification'].unique()):
    specification_rules = aggregate_rules[aggregate_rules['Specification'] == specification_name]
    graph_dictionary, _ = pqm.make_difference_networks(specification_rules, include_context=False)

    for dependent_variable, graph in graph_dictionary.items():
        aggregate_network_rows.append({
            'Specification': specification_name,
            'DV': dependent_variable,
            'Graph': graph,
            'Resolution_Multiplier': specification_lookup[specification_name]['NETWORK_RESOLUTION_MULTIPLIER']
        })

for specification_name in sorted(topic_rules['Specification'].unique()):
    specification_rules = topic_rules[topic_rules['Specification'] == specification_name]
    graph_dictionary, _ = pqm.make_difference_networks(specification_rules, include_context=True)

    for key, graph in graph_dictionary.items():
        context, dependent_variable = key

        topic_network_rows.append({
            'Specification': specification_name,
            'Context': context,
            'DV': dependent_variable,
            'Graph': graph,
            'Resolution_Multiplier': specification_lookup[specification_name]['NETWORK_RESOLUTION_MULTIPLIER']
        })

aggregate_network_rows = pqm.add_network_only_rows(aggregate_network_rows, ['DV'], network_specifications)
topic_network_rows = pqm.add_network_only_rows(topic_network_rows, ['Context', 'DV'], network_specifications)


'''
Compare networks to primary specification
'''

aggregate_network_robustness = pqm.compare_networks_to_primary(aggregate_network_rows, ['DV'])
topic_network_robustness = pqm.compare_networks_to_primary(topic_network_rows, ['Context', 'DV'])

aggregate_network_robustness.to_csv(sensitivityout / 'PSL_Aggregate_Network_Robustness.csv', index=False)
topic_network_robustness.to_csv(sensitivityout / 'PSL_Topic_Network_Robustness.csv', index=False)


'''
Primary-specification outputs
'''

primary_aggregate_rules = aggregate_rules[aggregate_rules['Specification'] == 'primary'].copy()
primary_topic_rules = topic_rules[topic_rules['Specification'] == 'primary'].copy()

primary_aggregate_rules.to_csv(pslout / 'Configs_All.csv', index=False)
primary_topic_rules.to_csv(pslout / 'Configs_Topic.csv', index=False)

aggregate_graphs, aggregate_difference = pqm.make_difference_networks(primary_aggregate_rules, include_context=False)
topic_graphs, topic_difference = pqm.make_difference_networks(primary_topic_rules, include_context=True)

aggregate_difference.to_csv(pslout / 'Network_Difference_All.csv', index=False)
topic_difference.to_csv(pslout / 'Network_Difference_Topic.csv', index=False)



'''
Primary network figures - Partitioning with Leiden (but networks are small.. very difficult and it would be better for bigger networks)
heatmap are more informative....
'''


if not primary_aggregate_rules.empty:
    dependent_variables = sorted(primary_aggregate_rules['DV'].unique())
    
    subset =  aggregate_rules[(aggregate_rules['Specification'] == 'primary')]

    positive_aggregate = pqm.count_cooccurrences(subset[subset['type'] == 'positive'], 'positive', include_context=False)
    negative_aggregate = pqm.count_cooccurrences(subset[subset['type'] == 'negative'], 'negative', include_context=False)
    graph_diff, difference_aggregate = pqm.make_difference_networks(subset, include_context=False)

    if dependent_variables:
        pqm.plot_rule_network(positive_aggregate, 1, len(dependent_variables), dependent_variables, layout = 'spring', filename=networkout / 'Rule_StructureNetwork_Positive.pdf', title_prefix='Positive', draw_partition=True, resolution_parameter = 1)
        pqm.plot_rule_network(negative_aggregate, 1, len(dependent_variables), dependent_variables, layout = 'spring', filename=networkout / 'Rule_StructureNetwork_Negative.pdf', title_prefix='Negative', draw_partition=True,  resolution_parameter = 1)
        pqm.plot_rule_network(difference_aggregate, 1, len(dependent_variables), dependent_variables, layout = 'spring', filename=networkout / 'Rule_StructureNetwork_Difference.pdf', title_prefix='Difference', draw_partition=True,  resolution_parameter = 1)
        pqm.plot_difference_heatmap(difference_aggregate,dependent_variables,networkout/'Rule_StructureHeatmap_Difference.pdf')


if not primary_aggregate_rules.empty:
    dependent_variables = sorted(primary_aggregate_rules['DV'].unique())
    
    subset =  aggregate_rules[(aggregate_rules['Specification'] == 'min_k_2')]

    positive_aggregate = pqm.count_cooccurrences(subset[subset['type'] == 'positive'], 'positive', include_context=False)
    negative_aggregate = pqm.count_cooccurrences(subset[subset['type'] == 'negative'], 'negative', include_context=False)
    graph_diff, difference_aggregate = pqm.make_difference_networks(subset, include_context=False)

    if dependent_variables:
        pqm.plot_rule_network(positive_aggregate, 1, len(dependent_variables), dependent_variables, layout = 'spring', filename=networkout / 'Rule_StructureNetwork_Positive_K2_.pdf', title_prefix='Positive', draw_partition=True, resolution_parameter = 1)
        pqm.plot_rule_network(negative_aggregate, 1, len(dependent_variables), dependent_variables, layout = 'spring', filename=networkout / 'Rule_StructureNetwork_Negative_K2_.pdf', title_prefix='Negative', draw_partition=True, resolution_parameter = 1)
        pqm.plot_rule_network(difference_aggregate, 1, len(dependent_variables), dependent_variables, layout = 'spring', filename=networkout / 'Rule_StructureNetwork_Difference_K2_.pdf', title_prefix='Difference', draw_partition=True, resolution_parameter = 1)
        pqm.plot_difference_heatmap(difference_aggregate,dependent_variables,networkout/'Rule_StructureHeatmap_Difference_K2.pdf')

if not primary_topic_rules.empty:
    contexts = sorted(primary_topic_rules['Context'].unique())
    topic_dependent_variables = sorted(primary_topic_rules['DV'].unique())
    
    subset =  topic_rules[(topic_rules['Specification'] == 'min_k_3')]


    positive_topic = pqm.count_cooccurrences(subset[subset['type'] == 'positive'], 'positive', include_context=True)
    negative_topic = pqm.count_cooccurrences(subset[subset['type'] == 'negative'], 'negative', include_context=True)

    number_of_topic_plots = len(contexts) * len(topic_dependent_variables)
    topic_columns = 3
    topic_rows = max(1, int(np.ceil(number_of_topic_plots / topic_columns)))

    if contexts and topic_dependent_variables:
        pqm.plot_rule_network(positive_topic, topic_rows, topic_columns, topic_dependent_variables, contexts=contexts, filename=networkout / 'Rule_Context_Network_Positive.pdf', title_prefix='Positive', draw_partition=True)
        pqm.plot_rule_network(negative_topic, topic_rows, topic_columns, topic_dependent_variables, contexts=contexts, filename=networkout / 'Rule_Context_Network_Negative.pdf', title_prefix='Negative', draw_partition=True)
        pqm.plot_rule_network(topic_difference, topic_rows, topic_columns, topic_dependent_variables, contexts=contexts, filename=networkout / 'Rule_Context_Network_Difference.pdf', title_prefix='Difference', draw_partition=True)
        pqm.plot_difference_heatmap(topic_difference,topic_dependent_variables,networkout/'Rule_Context_Heatmap_Difference.pdf',contexts=contexts)


'''
Overall DP ingredient importance and configuration to assess duality <- Main Paper
'''
subset=aggregate_rules[aggregate_rules['Specification']=='primary'].copy()
dfconfig=(subset.pivot_table(index=['DV', 'type'], columns='rule', values='importance', aggfunc='sum', fill_value=0))

records=[]

for (dv,outcome),grp in subset.groupby(['DV', 'type']):

    dp_imp=defaultdict(float)

    for _,row in grp.iterrows():

        lhs=row['rule'].split('->')[0]

        dps=[x.strip() for x in lhs.split('&')]

        for dp in dps:
            dp_imp[dp]+=row['importance']

    records.append(pd.Series(dp_imp,name=(dv,outcome)))

dfall_profile=pd.DataFrame(records).fillna(0)
dfbar = dfall_profile.T
dfbar = dfbar.rename(index=lambda x: x.replace('(X)',''))

df_pos = dfbar.xs('positive', level=1, axis=1).reindex(columns=['RSC', 'IEQ', 'CFL'])
df_neg = dfbar.xs('negative', level=1, axis=1).reindex(columns=['RSC', 'IEQ', 'CFL'])

# Configurations

gr_dv_out = (subset.groupby(['DV','type','rule'],as_index=False)['importance'].sum())

hyprules = []
for (dv,typ),grp in gr_dv_out.groupby(['DV','type']):
    grp = grp.sort_values('importance',ascending=False).copy()
    grp['cumprop'] = (grp['importance'].cumsum() / grp['importance'].sum())
    grp = grp[grp['cumprop'] <= 0.50]
    grp['DV'] = dv
    grp['type'] = typ
    hyprules.append(grp)
hyprules = pd.concat(hyprules,ignore_index=True)
                            

# 8 Panel Figure, may be too much but clear figure.
pos = pqm.make_circular_layout(FEATURE_COLUMNS)

fig = plt.figure(figsize=(20,10))
gs = fig.add_gridspec(nrows=2, ncols=4,
    width_ratios=[1,1,1,1.2])

# hypergraphs
axes = [
    fig.add_subplot(gs[0,0]),
    fig.add_subplot(gs[0,1]),
    fig.add_subplot(gs[0,2]),
    fig.add_subplot(gs[1,0]),
    fig.add_subplot(gs[1,1]),
    fig.add_subplot(gs[1,2]),
]

ax_pos = fig.add_subplot(gs[0,3])
ax_neg = fig.add_subplot(gs[1,3])

panels=[
    ('RSC','positive'),
    ('IEQ','positive'),
    ('CFL','positive'),
    ('RSC','negative'),
    ('IEQ','negative'),
    ('CFL','negative')
]

letters = ['A','B','C','D','E','F']

for ax,(letter,(dv,outcome)) in zip(axes, zip(letters,panels)):

    df=hyprules[(hyprules['DV']==dv) & (hyprules['type']==outcome)]
    node_imp=(df_pos[dv] if outcome=="positive" else df_neg[dv])
    pqm.draw_hypergraph_density(ax=ax, df=df,pos=pos, node_importance=node_imp,outcome=outcome)
    ax.set_title(f"{letter}) {dv} ({outcome})", fontsize=24) #n" f"{len(df)} rules (50% importance) note top rules that lead to cover 50% of all importance

# Positive profile
sns.heatmap(df_pos, cmap = 'Blues', ax=ax_pos, annot=True, fmt='.2f', linewidths=0.5)
ax_pos.set_title('G) Positive DP profiles', fontsize=20)
ax_pos.set_xlabel('')
ax_pos.set_ylabel('')
ax_pos.set_xticklabels(ax_pos.get_xticklabels(), fontsize=16)
ax_pos.set_yticklabels(ax_pos.get_yticklabels(), fontsize=16, rotation=0)

# Negative profile
sns.heatmap(df_neg, cmap = 'Oranges', ax=ax_neg, annot=True, fmt='.2f', linewidths=0.5)
ax_neg.set_title('H) Negative DP profiles', fontsize=20)
ax_neg.set_xlabel('')
ax_neg.set_ylabel('')
ax_neg.set_xticklabels(ax_neg.get_xticklabels(), fontsize=16)
ax_neg.set_yticklabels(ax_neg.get_yticklabels(), fontsize=16, rotation=0)

fig.savefig(networkout/"PSL_Hypergraph_Profile.pdf", bbox_inches='tight')

plt.close('all')



'''
Configuration similarity within outcome and within context <- Main Paper
'''

subset=topic_rules[topic_rules['Specification']=='primary'].copy()

dfconfig=(subset.pivot_table(index=['DV', 'Context','type'], columns='rule', values='importance', aggfunc='sum', fill_value=0))


# Configuration-level distances

records=[]

for i in range(len(dfconfig)):
    for j in range(i+1,len(dfconfig)):

        dv1, context1,outcome1=dfconfig.index[i]
        dv2, context2,outcome2=dfconfig.index[j]

        dist=cosine(dfconfig.iloc[i],dfconfig.iloc[j])

        if dv1 == dv2 and outcome1==outcome2 and context1!=context2:
            comparison="Within outcome"
        elif dv1!=dv2 and context1==context2 and outcome1==outcome2:
            comparison="Within S-E Context"
        else:
            continue

        records.append([context1,outcome1,context2,outcome2,dist,comparison])

dfdist_config=pd.DataFrame(records,columns=['Context1','Outcome1','Context2','Outcome2','Distance','Comparison'])

# DP-profile distances based on configurations using importance. similar to the network figure

records=[]

for (dv,context,outcome),grp in subset.groupby(['DV','Context','type']):

    dp_imp=defaultdict(float)

    for _,row in grp.iterrows():

        lhs=row['rule'].split('->')[0]

        dps=[x.strip() for x in lhs.split('&')]

        for dp in dps:
            dp_imp[dp]+=row['importance']

    records.append(pd.Series(dp_imp,name=(dv,context,outcome)))

dfprofile=pd.DataFrame(records).fillna(0)

records=[]

for i in range(len(dfprofile)):
    for j in range(i+1,len(dfprofile)):

        dv1, context1,outcome1=dfprofile.index[i]
        dv2, context2,outcome2=dfprofile.index[j]

        dist=cosine(dfprofile.iloc[i],dfprofile.iloc[j])

        if dv1==dv2 and outcome1==outcome2 and context1!=context2:
            comparison="Within outcome"
        elif dv1!=dv2 and context1==context2 and outcome1==outcome2:
            comparison="Within S-E Context"
        else:
            continue

        records.append([context1,outcome1,context2,outcome2,dist,comparison])

dfdist_profile=pd.DataFrame(records, columns=['Context1','Outcome1','Context2','Outcome2','Distance','Comparison'])


#Figure comparing configurations and profile ()
fig,axes=plt.subplots(1,2,figsize=(10,5),sharey=True)

# Exact configurations

sns.violinplot(data=dfdist_config, x='Comparison',  y='Distance', inner=None, cut=0, alpha=0.5, ax=axes[0])
sns.boxplot(data=dfdist_config, x='Comparison',y='Distance',width=0.5,showfliers=False,ax=axes[0])

axes[0].set_title('Configuration profiles',fontsize=16)
axes[0].set_xlabel('')
axes[0].set_ylabel('Cosine distance',fontsize=16)

# DP importance profiles
sns.violinplot(data=dfdist_profile, x='Comparison',  y='Distance', inner=None, cut=0, alpha=0.5, ax=axes[1])
sns.boxplot(data=dfdist_profile, x='Comparison',y='Distance',width=0.5,showfliers=False,ax=axes[1])

axes[0].set_title('Configuration profiles',fontsize=16)
axes[0].set_xlabel('')
axes[0].set_ylabel('Cosine distance',fontsize=16)

axes[1].set_title('DP importance profiles',fontsize=16)
axes[1].set_xlabel('')
axes[1].set_ylabel('')

for ax in axes:
    ax.tick_params(axis='x',labelsize=11)
    ax.tick_params(axis='y',labelsize=11)

plt.tight_layout()
plt.savefig(networkout/"PSL_Configuration_vs_DPProfile_Distance.pdf",bbox_inches='tight')
plt.close()


'''
Sensitivity figures
'''

if not aggregate_network_robustness.empty:
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.boxplot(data=aggregate_network_robustness, x='DV', y='Community_ARI', ax=axes[0])
    sns.boxplot(data=aggregate_network_robustness, x='DV', y='Signed_Edge_Jaccard', ax=axes[1])
    sns.boxplot(data=aggregate_network_robustness, x='DV', y='Node_Jaccard', ax=axes[2])

    axes[0].set_title('Community robustness')
    axes[1].set_title('Signed-edge robustness')
    axes[2].set_title('Node robustness')

    figure.tight_layout()
    figure.savefig(figureout / 'PSL_Aggregate_Network_Sensitivity.pdf')
    plt.close(figure)

if not aggregate_rule_robustness.empty:
    top_robust_rules = aggregate_rule_robustness.sort_values(['DV', 'retention_frequency', 'importance_median'], ascending=[True, False, False]).groupby('DV').head(20)

    figure, axes = plt.subplots(1, len(OUTCOME_COLUMNS), figsize=(18, 8), squeeze=False)

    for axis, dependent_variable in zip(axes[0], OUTCOME_COLUMNS):
        subset = top_robust_rules[top_robust_rules['DV'] == dependent_variable].sort_values('retention_frequency')
        axis.barh(subset['rule_key'], subset['retention_frequency'], color='steelblue')
        axis.set_title(dependent_variable)
        axis.set_xlim(0, 1)
        axis.set_xlabel('Retention frequency')

    figure.tight_layout()
    figure.savefig(figureout / 'PSL_Aggregate_Rule_Robustness.pdf', bbox_inches='tight')
    plt.close(figure)

if not topic_network_robustness.empty:
    topic_network_summary = topic_network_robustness.groupby(['Context', 'DV']).agg(
        Community_ARI_Median=('Community_ARI', 'median'),
        Signed_Edge_Jaccard_Median=('Signed_Edge_Jaccard', 'median'),
        Node_Jaccard_Median=('Node_Jaccard', 'median')
    ).reset_index()

    community_table = topic_network_summary.pivot(index='Context', columns='DV', values='Community_ARI_Median')
    edge_table = topic_network_summary.pivot(index='Context', columns='DV', values='Signed_Edge_Jaccard_Median')
    node_table = topic_network_summary.pivot(index='Context', columns='DV', values='Node_Jaccard_Median')

    figure, axes = plt.subplots(1, 3, figsize=(18, 8))

    sns.heatmap(community_table, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
    sns.heatmap(edge_table, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1, ax=axes[1])
    sns.heatmap(node_table, annot=True, fmt='.2f', cmap='Greens', vmin=0, vmax=1, ax=axes[2])

    axes[0].set_title('Median community ARI')
    axes[1].set_title('Median signed-edge Jaccard')
    axes[2].set_title('Median node Jaccard')

    figure.tight_layout()
    figure.savefig(figureout / 'PSL_Topic_Network_Sensitivity.pdf', bbox_inches='tight')
    plt.close(figure)


'''
Sensitivity specification table
'''

specification_records = []

for specification in all_specifications:
    specification_records.append({
        'Specification': specification['SPECIFICATION'],
        'Sensitivity_Stage': specification['SENSITIVITY_STAGE'],
        'Missing_Probability_Threshold': specification['MISSING_PROBABILITY_THRESHOLD'],
        'LASSO_Selection_Threshold': specification['LASSO_SELECTION_THRESHOLD'],
        'LASSO_Sign_Stability_Threshold': specification['LASSO_SIGN_STABILITY_THRESHOLD'],
        'Use_Full_LASSO_Model': specification['USE_FULL_LASSO_MODEL'],
        'Minimum_Rule_Order': specification['MIN_K'],
        'Maximum_Rule_Order': specification['MAX_K'],
        'Minimum_Support_Fraction': specification['MIN_SUPPORT_FRAC'],
        'Screening_Operator': specification['SCREENING_OPERATOR'],
        'LASSO_Direction_Mode': specification['LASSO_DIRECTION_MODE'],
        'LASSO_Initial_Weights': specification['LASSO_W'],
        'Subset_Inclusion': specification['SUBSET_INCL'],
        'Subset_Decay': specification['SUBSET_DECAY'],
        'Importance_Metric': specification['IMPORTANCE_METRIC'],
        'Top_N_Per_Direction': specification['TOP_N'],
        'Network_Resolution_Multiplier': specification['NETWORK_RESOLUTION_MULTIPLIER']
    })

specification_table = pd.DataFrame(specification_records)
specification_table.to_csv(sensitivityout / 'PSL_Sensitivity_Specifications.csv', index=False)


'''
Sensitivity summary
'''

sensitivity_summary = pd.DataFrame({
    'Measure': [
        'Model-stage specifications',
        'Ranking-stage specifications',
        'Network-stage specifications',
        'Aggregate PSL models',
        'Topic PSL models',
        'Aggregate retained rule-sign combinations',
        'Topic retained rule-sign combinations',
        'Aggregate robust rule-sign combinations at 75 percent',
        'Topic robust rule-sign combinations at 75 percent'
    ],
    'Value': [
        len(model_specifications),
        len(ranking_specifications),
        len(network_specifications),
        len(aggregate_results),
        len(topic_results),
        len(aggregate_rule_robustness),
        len(topic_rule_robustness),
        int((aggregate_rule_robustness['retention_frequency'] >= 0.75).sum()) if not aggregate_rule_robustness.empty else 0,
        int((topic_rule_robustness['retention_frequency'] >= 0.75).sum()) if not topic_rule_robustness.empty else 0
    ]
})

sensitivity_summary.to_csv(sensitivityout / 'PSL_Sensitivity_Summary.csv', index=False)

'''
Sensitivity Final Figures and Tables -> Supplementary
'''

# Outcome vs Context robustness across specifications

records_config = []
records_profile = []

for spec in sorted(topic_rules['Specification'].unique()):

    sub = topic_rules[topic_rules['Specification'] == spec].copy()

    if len(sub) == 0:
        continue

    # Configuration profiles

    dfconfig = sub.pivot_table(index=['DV','Context','type'], columns='rule', values='importance', aggfunc='sum', fill_value=0)

    for i in range(len(dfconfig)):
        for j in range(i+1,len(dfconfig)):

            dv1,ctx1,out1 = dfconfig.index[i]
            dv2,ctx2,out2 = dfconfig.index[j]

            d = cosine(dfconfig.iloc[i],dfconfig.iloc[j])

            if dv1==dv2 and out1==out2 and ctx1!=ctx2:
                comp='Within outcome'

            elif dv1!=dv2 and out1==out2 and ctx1==ctx2:
                comp='Within context'

            else:
                continue

            records_config.append([spec,comp,d])

    # DP profiles

    prof=[]

    for (dv,ctx,out),grp in sub.groupby(['DV','Context','type']):

        dp_imp=defaultdict(float)

        for _,row in grp.iterrows():

            lhs=row['rule'].split('->')[0]

            for dp in [x.strip() for x in lhs.split('&')]:
                dp_imp[dp]+=row['importance']

        prof.append(pd.Series(dp_imp,name=(dv,ctx,out)))

    dfprof=pd.DataFrame(prof).fillna(0)

    for i in range(len(dfprof)):
        for j in range(i+1,len(dfprof)):

            dv1,ctx1,out1=dfprof.index[i]
            dv2,ctx2,out2=dfprof.index[j]

            d=cosine(dfprof.iloc[i],dfprof.iloc[j])

            if dv1==dv2 and out1==out2 and ctx1!=ctx2:
                comp='Within outcome'

            elif dv1!=dv2 and out1==out2 and ctx1==ctx2:
                comp='Within context'

            else:
                continue

            records_profile.append([spec,comp,d])

df_sens_config = pd.DataFrame(records_config, columns=['Specification','Comparison','Distance'])
df_sens_profile = pd.DataFrame(records_profile, columns=['Specification','Comparison','Distance'])


fig,axes=plt.subplots(1,2,figsize=(14,8),sharey=True)

sns.boxplot(data=df_sens_config, x='Specification', y='Distance', hue='Comparison',ax=axes[0])
sns.boxplot(data=df_sens_profile, x='Specification', y='Distance', hue='Comparison',ax=axes[1])

axes[0].set_title('Configuration profiles')
axes[1].set_title('DP profiles')

for ax in axes:
    ax.set_xlabel('')                     # remove x-axis label
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)

    ax.tick_params(axis='x',rotation=90, labelsize=16)

    ax.tick_params(axis='y',labelsize=16)

    leg = ax.get_legend()
    if leg is not None:
        leg.set_title(leg.get_title().get_text(), prop={'size':16})
        for txt in leg.get_texts():
            txt.set_fontsize(16)

plt.tight_layout()
plt.savefig(figureout/'PSL_Outcome_Context_Robustness.pdf',bbox_inches='tight')
plt.close()


#outcome vs context similarity robustness

summary=[]

for spec,grp in df_sens_config.groupby('Specification'):

    med = grp.groupby('Comparison')['Distance'].median()

    summary.append({
        'Specification':spec,
        'Profile':'Configuration',
        'Within_Outcome_Median':med['Within outcome'],
        'Within_Context_Median':med['Within context'],
        'Outcome_Lower':med['Within outcome']
                         < med['Within context']
    })

for spec,grp in df_sens_profile.groupby('Specification'):

    med = grp.groupby('Comparison')['Distance'].median()

    summary.append({
        'Specification':spec,
        'Profile':'DP_Profile',
        'Within_Outcome_Median':med['Within outcome'],
        'Within_Context_Median':med['Within context'],
        'Outcome_Lower':med['Within outcome']
                         < med['Within context']
    })

summary=pd.DataFrame(summary)

summary.to_csv(sensitivityout/'PSL_Outcome_Context_Robustness.csv', index=False)

#rule robustness
rule_summary = (aggregate_rule_robustness.groupby('DV').agg(
        N_Rules=('rule_key','count'),
        Median_Retention=('retention_frequency','median'),
        Mean_Retention=('retention_frequency','mean'),
        Robust_50=('retention_frequency',
                   lambda x:(x>=0.5).sum()),
        Robust_75=('retention_frequency',
                   lambda x:(x>=0.75).sum())
    ).reset_index())

rule_summary.to_csv(sensitivityout/'PSL_Rule_Robustness_Summary.csv', index=False)

#rank robustness
rank_summary = (aggregate_rank_correlations.groupby(['DV','type']).agg(Median_Spearman=('Spearman','median'),
        Min_Spearman=('Spearman','min'),
        Max_Spearman=('Spearman','max')
    ).reset_index())

rank_summary.to_csv(sensitivityout/'PSL_Rank_Robustness_Summary.csv', index=False)


#hypergraph robustness
# Primary hypergraph rules

primary = aggregate_rules[aggregate_rules['Specification']=='primary'].copy()

gr = (primary.groupby(['DV','type','rule'],as_index=False)['importance'].sum())

hyprules_primary=[]

for (dv,typ),grp in gr.groupby(['DV','type']):

    grp=grp.sort_values('importance',ascending=False).copy()

    grp['cumprop'] = (grp['importance'].cumsum() / grp['importance'].sum())

    grp=grp[grp['cumprop']<=0.50].copy()

    grp['DV']=dv
    grp['type']=typ

    hyprules_primary.append(grp)

hyprules_primary=pd.concat(hyprules_primary,ignore_index=True)


# Hypergraph rule retention

records=[]

for spec in sorted(aggregate_rules['Specification'].unique()):

    sub=aggregate_rules[aggregate_rules['Specification']==spec].copy()
    gr=(sub.groupby(['DV','type','rule'],as_index=False)['importance'].sum())

    for (dv,typ),grp in gr.groupby(['DV','type']):

        grp=grp.sort_values('importance',ascending=False).copy()
        grp['cumprop']=(grp['importance'].cumsum() / grp['importance'].sum())
        grp=grp[grp['cumprop']<=0.50]
        primary_rules=set(hyprules_primary[(hyprules_primary['DV']==dv) & (hyprules_primary['type']==typ)]['rule'])
        alt_rules=set(grp['rule'])
        jaccard=(len(primary_rules & alt_rules) / len(primary_rules | alt_rules) if len(primary_rules | alt_rules)>0 else np.nan)
        records.append([spec, dv, typ, len(primary_rules), len(alt_rules), jaccard])

hypergraph_robustness=pd.DataFrame(records, columns=['Specification','DV','type','N_Primary','N_Alternative','Rule_Jaccard'])

#Hypergraph rule overlap
plt.figure(figsize=(9,6))

ax = sns.boxplot(data=hypergraph_robustness,x='DV',y='Rule_Jaccard',hue='type')

ax.set_xlabel('')
ax.set_ylabel('Jaccard similarity', fontsize=16)

ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

leg = ax.get_legend()
leg.set_title('type', prop={'size':16})

for txt in leg.get_texts():
    txt.set_fontsize(16)

plt.title('Hypergraph-rule robustness', fontsize=18)
plt.tight_layout()
plt.savefig(figureout/'PSL_Hypergraph_Rule_Robustness.pdf',bbox_inches='tight')

# Hypergraph DP-profile robustness

records=[]
primary_profiles={}

for (dv,typ),grp in hyprules_primary.groupby(['DV','type']):

    imp=defaultdict(float)

    for _,row in grp.iterrows():
        lhs=row['rule'].split('->')[0]
        dps=[x.strip() for x in lhs.split('&')]
        
        for dp in dps:
            imp[dp]+=row['importance']

    primary_profiles[(dv,typ)] = pd.Series(imp)
    
#Compare vs alternatives
profile_records=[]

for spec in sorted(aggregate_rules['Specification'].unique()):

    sub=aggregate_rules[aggregate_rules['Specification']==spec].copy()
    gr=(sub.groupby(['DV','type','rule'],as_index=False)['importance'].sum())

    for (dv,typ),grp in gr.groupby(['DV','type']):
        grp=grp.sort_values('importance',ascending=False).copy()
        grp['cumprop']=(grp['importance'].cumsum() / grp['importance'].sum())
        grp=grp[grp['cumprop']<=0.50]
        imp=defaultdict(float)

        for _,row in grp.iterrows():
            lhs=row['rule'].split('->')[0]
            dps=[x.strip() for x in lhs.split('&')]

            for dp in dps:
                imp[dp]+=row['importance']

        current=pd.Series(imp)
        base=primary_profiles[(dv,typ)]
        idx=base.index.union(current.index)
        base=base.reindex(idx,fill_value=0)
        current=current.reindex(idx,fill_value=0)

        corr=base.corr(current,method='spearman')

        profile_records.append([spec,dv,typ,corr])

hypergraph_profile_robustness=pd.DataFrame(
    profile_records,columns=['Specification', 'DV', 'type', 'Profile_Spearman'])

#Figure
plt.figure(figsize=(10,5))

ax = sns.boxplot(data=hypergraph_profile_robustness, x='DV', y='Profile_Spearman', hue='type')

plt.axhline(0.8,ls='--',c='red',alpha=0.5)

ax.set_xlabel('')
ax.set_ylabel('Spearman correlation', fontsize=16)

ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

leg = ax.get_legend()
leg.set_title('type', prop={'size':16})

for txt in leg.get_texts():
    txt.set_fontsize(16)

plt.title('DP-profile robustness from top-50% hypergraph rules',fontsize=18)

plt.tight_layout()
plt.savefig(figureout/'PSL_Hypergraph_Profile_Robustness.pdf',bbox_inches='tight')

plt.close()


'''
Final summary
'''

print('\nPSL sensitivity summary:')
print(sensitivity_summary)

print('\nPrimary aggregate diagnostics:')

if not aggregate_diagnostics.empty:
    print(aggregate_diagnostics[aggregate_diagnostics['Specification'] == 'primary'])
else:
    print('No aggregate diagnostics were produced.')

print('\nPrimary topic diagnostics:')

if not topic_diagnostics.empty:
    print(topic_diagnostics[topic_diagnostics['Specification'] == 'primary'])
else:
    print('No topic diagnostics were produced.')

print('\nAggregate rules:')
print(len(aggregate_rules))

print('\nTopic rules:')
print(len(topic_rules))

print('\nDone')
