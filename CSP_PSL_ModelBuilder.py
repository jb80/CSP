#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 09:52:06 2025

@author: jbaggio
"""

#routines for Probabilistic soft logic
from pslpython.model import Model
from pslpython.predicate import Predicate
from pslpython.rule import Rule
from pslpython.partition import Partition
from sklearn.metrics import mean_squared_error
from sklearn.metrics import adjusted_rand_score


#usual suspects
import numpy as np
import itertools
import html
from collections import Counter
import pandas as pd
from collections import defaultdict


# network visualization
import networkx as nx
import igraph as ig
import leidenalg
from matplotlib.patches import Wedge, Circle
import matplotlib.pyplot as plt

rndstate = 7879




# Helper functions for PSL

#functions to make names  of rulesfrom lasso and PSL compatible 
def sanitize_name(name):
    return name.replace("(", "").replace(")", "").replace(":", "_").replace(" ", "_")

def split_term(term):
    # Handles both "_AND_" and space-separated terms
    if "_AND_" in term:
        return term.replace("(", "").replace(")", "").split("_AND_")
    else:
        return term.replace("(", "").replace(")", "").split()
    

#get lower ranked terms from lasso rules (i.e. if rule is A and B and C -> DV then it will include A and B -> DV, C and B -> DV, A and C -> DV)
def get_lower_order_terms(term):
    parts = split_term(term)
    return set("_AND_".join(sorted(sub)) for r in range(1, len(parts)) for sub in itertools.combinations(parts, r))

#lukas t-norm for assessing "consistency" of rules
def lukas_and_orig(V):
    return np.maximum(0, np.sum(V, axis=1) - V.shape[1] + 1)

#lukas t-norm with average to reduce penalty from increased number of factors
def lukas_and(V):
    n = V.shape[1]
    mean_vals = np.mean(V, axis=1)
    threshold = (n - 1) / n
    return np.maximum(0, mean_vals - threshold)

#add rules with meta data
def add_rule_with_meta(model, name_map, dv, body_parts, head_is_pos, weight, squared, origin, added_rules, rule_meta_list):
    body = " & ".join(f"{name_map[p]}(X)" for p in body_parts)
    head = f"{name_map[dv]}(X)" if head_is_pos else f"~{name_map[dv]}(X)"
    rule_str = f"{body} -> {head}"
    
    if rule_str in added_rules:
        return False

    model.add_rule(Rule(rule_str, weight=float(weight), squared=bool(squared)))
    added_rules.add(rule_str)
    rule_meta_list.append({
        'rule': rule_str,
        'origin': origin,
        'sign': +1 if head_is_pos else -1,
        'parts': tuple(body_parts),
        'k': len(body_parts),
        'init_weight': float(weight),
        'squared': bool(squared),
    })
    return True

# Modular components for building PSL model

#create predicates
def create_predicates(model, atomic_tokens, name_map):
    for var in sorted(atomic_tokens):
        model.add_predicate(Predicate(name_map[var], 1))

# create exploratory rules if we want them.
def generate_exploratory_rules(model, indepvars, depvar, exploratory_terms, name_map, config, added_rules, rule_meta_list, min_coef):
    for term in sorted(exploratory_terms):
        parts = sorted(split_term(term))
        if not parts:
            continue
        cols = [c for c in parts if c in indepvars.columns]
        V = indepvars[cols].astype(float).values
        body = lukas_and(V)
        support_frac = float(np.mean(body > 0))
        mean_body = float(body.mean())
        if support_frac < config['MIN_SUPPORT_FRAC'] or mean_body < config['MIN_MEAN_BODY']:
            continue
        add_rule_with_meta(model, name_map, depvar, parts, True, min_coef, False, 'global', added_rules, rule_meta_list)
        add_rule_with_meta(model, name_map, depvar, parts, False, min_coef, False, 'global', added_rules, rule_meta_list)

#creates rules from lasso derived interactions
def generate_lasso_rules(model, indepvars, depvar, rule_list, name_map, config, added_rules, rule_meta_list, min_coef):
    rule_size_counter = Counter()
    for term, coef in rule_list or []:
        parts = sorted(split_term(term))
        if not parts:
            continue
        if config['MAX_K'] is not None and len(parts) > config['MAX_K']:
            continue
        seed_sizes = [k for k in range(config['MIN_K'], min(len(parts), config['MAX_K']) + 1)]
        for L in seed_sizes:
            for seed in itertools.combinations(parts, L):
                cols = [c for c in seed if c in indepvars.columns]
                if len(cols) != len(seed):
                    continue
                V = indepvars[cols].astype(float).values
                body = lukas_and(V)
                support_frac = float(np.mean(body > 0))
                mean_body = float(body.mean())
                if support_frac < config['MIN_SUPPORT_FRAC'] or mean_body < config['MIN_MEAN_BODY']:
                    continue
                base_w = round(abs(coef), 6) if config['LASSO_W'] else 1
                add_rule_with_meta(model, name_map, depvar, seed, True, base_w, False, 'lasso', added_rules, rule_meta_list)
                add_rule_with_meta(model, name_map, depvar, seed, False, base_w, False, 'lasso', added_rules, rule_meta_list)
                rule_size_counter[len(seed)] += 1
                if config['SUBSET_INCL'] and L >= (config['MIN_K'] + 1):
                    for r in range(L - 1, config['MIN_K'] - 1, -1):
                        if config['MAX_K'] is not None and r > config['MAX_K']:
                            continue
                        for subset in itertools.combinations(seed, r):
                            cols2 = [c for c in subset if c in indepvars.columns]
                            if len(cols2) != len(subset):
                                continue
                            V2 = indepvars[cols2].astype(float).values
                            body2 = lukas_and(V2)
                            support_frac2 = float(np.mean(body2 > 0))
                            mean_body2 = float(body2.mean())
                            if support_frac2 < config['MIN_SUPPORT_FRAC'] or mean_body2 < config['MIN_MEAN_BODY']:
                                continue
                            subset_w = max(min_coef, round(base_w * (config['SUBSET_DECAY'] ** (L - r)), 6))
                            add_rule_with_meta(model, name_map, depvar, subset, True, subset_w, False, 'subset', added_rules, rule_meta_list)
                            add_rule_with_meta(model, name_map, depvar, subset, False, subset_w, False, 'subset', added_rules, rule_meta_list)
                            rule_size_counter[len(subset)] += 1

#filter rules keeping only high weight if rule can affect both DV and ~DV. If equal weight disregard the rule.
def filter_rule_meta_by_weight(rule_list, model):
    """
    Filters rule metadata to retain only the direction (positive or negative)
    with the higher learned weight for each rule body. Excludes rules with equal weights as not considered important.

    Parameters:
        rule_list (list): List of rule metadata dictionaries.
        model (pslpython.model.Model): The PSL model after weight learning.

    Returns:
        list: Filtered rule metadata list.
    """
    # Step 1: Get learned weights from model
    learned_weights = {}
    for rule in model.get_rules():
        rule_str = str(rule)
        if ':' in rule_str:
            rule_str = rule_str.split(':', 1)[1].strip()
        rule_str = rule_str.strip()  # 🔄 Removed html.unescape
        learned_weights[rule_str] = rule.weight()

    # Step 2: Group rules by body
    rule_groups = {}
    for meta in rule_list:
        rule_str = str(meta['rule']).strip()  # 🔄 Removed html.unescape
        weight = learned_weights.get(rule_str, None)
        if weight is None:
            continue
        if '->' in rule_str:  # ✅ Changed from '-&gt;' to '->'
            body, head = rule_str.split('->')
            body = body.strip()
            head = head.strip()
            direction = 'positive' if not head.startswith('~') else 'negative'
            rule_groups.setdefault(body, {})[direction] = (meta, weight)  # ✅ Safe dictionary assignment

    # Step 3: Filter rules
    filtered_meta = []
    for body, directions in rule_groups.items():
        pos = directions.get('positive')
        neg = directions.get('negative')

        if pos and neg:
            if pos[1] > neg[1]:
                filtered_meta.append(pos[0])
            elif neg[1] > pos[1]:
                filtered_meta.append(neg[0])
            # Skip if equal weight
        elif pos:
            filtered_meta.append(pos[0])
        elif neg:
            filtered_meta.append(neg[0])

    return filtered_meta


#add observation, targets and truth (the data)
def inject_data(model, indepvars, depvar, depvars, name_map, config):
    pred_map = model.get_predicates()
    for var in indepvars.columns:
        if var in name_map and name_map[var] in pred_map:
            pred = pred_map[name_map[var]]
            if config['ALL_IVOBS']:
                rows = [[str(entity), float(value)] for entity, value in indepvars[var].items()]
            else:
                rows = [[str(entity), float(value)] for entity, value in indepvars[var].items() if value != 0.5]
            if rows:
                pred.add_data(Partition.OBSERVATIONS, rows)

    dv_pred = pred_map[name_map[depvar]]
    all_entities = depvars[depvar].index
    dv_pred.add_data(Partition.TARGETS, [[str(e)] for e in all_entities])
    truth_rows = [[str(e), float(v)] for e, v in depvars[depvar].items() if v != 0.5]
    if truth_rows:
        dv_pred.add_data(Partition.TRUTH, truth_rows)

# Refactored build_psl_model function
def build_psl_model(*, depvar, indepvars, depvars, rule_list, filtered, config):
    selected_features = set(var for var, _ in rule_list) if rule_list else set()
    valid_terms = set()
    
    if config['USE_GLOBAL_EXPLORATORY']:
        for term in selected_features:
            valid_terms.update(get_lower_order_terms(term))
    exploratory_terms = (valid_terms - selected_features) if config['USE_GLOBAL_EXPLORATORY'] else set()
    atomic_tokens = set(indepvars.columns)
    atomic_tokens.add(depvar)
    
    for term in selected_features.union(exploratory_terms):
        for p in split_term(term):
            atomic_tokens.add(p)
    name_map = {name: sanitize_name(name) for name in atomic_tokens}
    model = Model(name=f'psl_model_{name_map[depvar]}')
    create_predicates(model, atomic_tokens, name_map)
    
    if config['ADD_NEGATIVE_DV_PRIOR']:
        model.add_rule(Rule(f"~{name_map[depvar]}(X)", weight=float(config['DV_PRIOR_WEIGHT']), squared=True))
    min_coef = round(min(abs(coef) for _, coef in rule_list), 6) if config['LASSO_W'] and rule_list else 1
    added_rules = set()
    rule_meta_list = []
    
    if config['USE_GLOBAL_EXPLORATORY']:
        generate_exploratory_rules(model, indepvars, depvar, exploratory_terms, name_map, config, added_rules, rule_meta_list, min_coef)
    
    if filtered:
        for body_str, sign in rule_list:
            rule_head = f"{name_map[depvar]}(X)" if sign > 0 else f"~{name_map[depvar]}(X)"
            rule_str = " & ".join([f"{name_map[p]}(X)" for p in body_str.split()]) + f" -> {rule_head}"
            if rule_str in added_rules:
                continue
            model.add_rule(Rule(rule_str, weight=1.0, squared=False))  # Always positive weight
            added_rules.add(rule_str)
            rule_meta_list.append({
                'rule': rule_str,
                'origin': 'filtered',
                'sign': sign,
                'parts': tuple(body_str.split()),
                'k': len(body_str.split()),
                'init_weight': 1.0,
                'squared': False,
            })
    else:
        # Original logic for LASSO terms
        generate_lasso_rules(model, indepvars, depvar, rule_list, name_map, config, added_rules, rule_meta_list, min_coef)
    
    inject_data(model, indepvars, depvar, depvars, name_map, config)
    
    return model, rule_meta_list, name_map


def gof_psl (model, depvar, inferences):
    pred_obj, pred_df = next(iter(inferences.items()))
    pred_df = pred_df.copy()
    val_col = 'truth' if 'truth' in pred_df.columns else pred_df.columns[-1]
    ent_col = [c for c in pred_df.columns if c != val_col][0]
    preds_df = pred_df.rename(columns={ent_col: 'entity', val_col: 'y_pred'})
    preds_df['entity'] = preds_df['entity'].astype(str)

    truth_df = (depvar.rename('y_true').reset_index().rename(columns={'index':'entity'}))
    truth_df = truth_df[truth_df['y_true'] != 0.5]
    truth_df['entity'] = truth_df['entity'].astype(str)

    eval_df = truth_df.merge(preds_df, on='entity', how='inner')
    y_true = eval_df['y_true'].astype(float).to_numpy()
    y_pred = eval_df['y_pred'].astype(float).to_numpy()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse  = mean_squared_error(y_true, y_pred)
    return rmse, mse, eval_df



#extract rules qca style
def extract_and_rank_rules(rulelist = None, model = None, inferred_res = None, indeps = None, deps = None, single_dep = None, top_n=None,
                           direction='both', importance_metric='coverage_consistency', use_inferred = False):
    """
    Extracts and ranks PSL rules based on learned weights and evaluates coverage and consistency.

    Parameters:
        rulelist (list)          : Metadata for each rule including rule string, origin, sign, etc.
        model (pslpython.model.Model) : The PSL model after weight learning.
        indeps (pd.DataFrame)         : DataFrame of independent variables.
        deps (pd.DataFrame)           : DataFrame of dependent variables.
        single_dep (str)              : Name of the dependent variable.
        top_n (int, optional)         : If specified, returns only the top N rules by importance.
        direction (str)               : 'positive', 'negative', or 'both' — filters rules by their sign.
        importance_metric (str)       : One of 'coverage', 'consistency', 'coverage_consistency' or 'coverage_consistency_effect'.
        use_inferred                  : whether we use inferred values or not.

    Returns:
        pd.DataFrame: Ranked rules with metrics.
    """

    # Get learned weights from model
    learned_weights = {}
    for rule in model.get_rules():
        rule_str = str(rule)
        if ':' in rule_str:
            rule_str = rule_str.split(':', 1)[1].strip()
        rule_str = html.unescape(rule_str)
        weight = rule.weight()
        learned_weights[rule_str] = weight

    # Prepare DV values
    if use_inferred:
        for predicate, df in inferred_res.items():
            dv_values = pd.DataFrame(df)
            dv_values = dv_values.set_index(dv_values[0].astype(int))
        
        valid_mask = dv_values.notna()
    else:
        dv_values = deps[single_dep].astype(float)
        valid_mask = dv_values != 0.5

    dv_values = dv_values[valid_mask]
    dv_baseline = dv_values['truth'].mean()

    ranked_rules = []

    for meta in rulelist:
        rule_str = html.unescape(str(meta['rule']).strip())
        parts = meta['parts']
        origin = meta['origin']
        sign = meta['sign']
        k = meta['k']
        squared = meta['squared']
        init_weight = meta['init_weight']
        weight = learned_weights.get(rule_str, None)

        # Skip rule if weight is not found
        if weight is None:
            continue

        # Filter by direction
        if direction == 'positive' and sign != 1:
            continue
        if direction == 'negative' and sign != -1:
            continue

        # Compute rule body satisfaction using Lukasiewicz AND
        cols = [p for p in parts if p in indeps.columns]
        if not cols:
            continue
        V = indeps.loc[dv_values.index, cols].astype(float).values
        body_satisfaction = np.maximum(0, np.sum(V, axis=1) - len(cols) + 1)

        # Coverage: fraction of cases where body > 0
        coverage = float(np.mean(body_satisfaction > 0))

        # Consistency: average DV value where body > 0
        if np.any(body_satisfaction > 0):
            consistency = float(dv_values['truth'][body_satisfaction > 0].mean())
        else:
            consistency = np.nan
        
        # Effect: difference from baseline
        effect = consistency - dv_baseline if not np.isnan(consistency) else 0
        
        # Importance metric
        importance = {
            'coverage': coverage,
            'consistency': consistency if not np.isnan(consistency) else 0,
            'coverage_consistency': coverage * consistency if not np.isnan(consistency) else 0,
            'coverage_consistency_effect' : coverage * (effect + consistency) if not np.isnan(consistency) else 0,
        }.get(importance_metric, coverage * consistency if not np.isnan(consistency) else 0)

        ranked_rules.append({
            'rule': rule_str,
            'weight': weight,
            'origin': origin,
            'sign': sign,
            'k': k,
            'squared': squared,
            'init_weight': init_weight,
            'coverage': coverage,
            'consistency': consistency,
            'effect' : effect,
            'importance': importance
        })

    # Convert to DataFrame and sort
    df = pd.DataFrame(ranked_rules)
    if not df.empty and 'importance' in df.columns:
        df = df.sort_values(by='importance', ascending=False)

    if top_n is not None:
        df = df.head(top_n)

    return df.reset_index(drop=True)

# Helper functions for visualizing rule structures 

# Function to clean rule and extract conditions
# Clean rule labels
def clean_rule(rule):
    rule = rule.replace('(X)', '')
    rule = rule.split('->')[0].strip()
    return rule


# Function to weight co-occurrences and return a DataFrame
def count_cooccurrences(df, label, include_context=False):
    cooccur = defaultdict(float)

    # Determine grouping columns
    group_cols = ['DV']
    if include_context:
        group_cols = ['Context', 'DV']

    grouped = df[df['type'] == label].groupby(group_cols)

    for group_keys, subset in grouped:
        for _, row in subset.iterrows():
            rule = html.unescape(row['rule'])
            importance = row['importance']
            lhs = rule.split('->')[0].replace('(X)', '').strip()
            conditions = [cond.strip() for cond in lhs.split('&')]
            for i in range(len(conditions)):
                for j in range(i + 1, len(conditions)):
                    pair = tuple(sorted([conditions[i], conditions[j]]))
                    if include_context:
                        context, dv = group_keys
                        cooccur[(pair[0], pair[1], label, dv, context)] += importance
                    else:
                        dv = group_keys[0]
                        cooccur[(pair[0], pair[1], label, dv)] += importance

    # Convert to DataFrame
    if include_context:
        df_result = pd.DataFrame([
            {'Condition1': k[0], 'Condition2': k[1], 'Label': k[2], 'DV': k[3], 'Context': k[4], 'Importance': round(v, 5)}
            for k, v in cooccur.items()
        ])
    else:
        df_result = pd.DataFrame([
            {'Condition1': k[0], 'Condition2': k[1], 'Label': k[2], 'DV': k[3], 'Importance': round(v, 5)}
            for k, v in cooccur.items()
        ])

    return df_result

# Network plotting:
colors_pos = {
    0: 'aliceblue',
    1: 'aqua',
    2: 'aquamarine',
    3: 'blue',
    4: 'blueviolet',
    5: 'cadetblue',
    6: 'cornflowerblue',
    7: 'cyan',
    8: 'darkblue',
    9: 'darkcyan'
}


colors_neg = {
    0: 'darkgoldenrod',
    1: 'darkorange',
    2: 'gold',
    3: 'goldenrod',
    4: 'greenyellow',
    5: 'lightgoldenrodyellow',
    6: 'lightyellow',
    7: 'orange',
    8: 'orangered',
    9: 'palegoldenrod'
}

    
def nx_to_igraph(G):
    ig_graph = ig.Graph()
    ig_graph.add_vertices(list(G.nodes()))
    ig_graph.add_edges(list(G.edges()))
    ig_graph.es['weight'] = [G[u][v]['Importance'] for u, v in G.edges()]
    return ig_graph

def run_leiden_partition(G, invert=False, title_prefix = ''):
    ig_graph = nx_to_igraph(G)
    if title_prefix == 'Difference':
        weights = [-w if invert else w for w in ig_graph.es['weight']]
    else:
        weights = [w for w in ig_graph.es['weight']]
        
    positives = [w for w in weights if w > 0]
    res = np.mean(positives) if positives else 1.0
    ig_graph.es['weight'] = weights 
    partition = leidenalg.find_partition(ig_graph, leidenalg.CPMVertexPartition, weights='weight', resolution_parameter = res)
    membership = partition.membership
    return {node: membership[i] for i, node in enumerate(ig_graph.vs['name'])}


def assign_color(G, pos_partition, neg_partition, pos_colors, neg_colors):
    """
    Assigns Graphviz-style fillcolor attributes to nodes in a NetworkX graph G.
    - If a node belongs to both pos and neg communities: 'color1;0.5:color2'
    - If only one: that color
    - If none: 'silver'
    """
    for node in G.nodes():
        node_str = str(node)  # Ensure consistent key format
        pos_group = pos_partition.get(node_str)
        neg_group = neg_partition.get(node_str)

        if pos_group is not None and neg_group is not None:
            color1 = pos_colors.get(pos_group, 'silver')
            color2 = neg_colors.get(neg_group, 'silver')
            fillcolor = f"{color1};0.5:{color2}"
        elif pos_group is not None:
            fillcolor = pos_colors.get(pos_group, 'silver')
        elif neg_group is not None:
            fillcolor = neg_colors.get(neg_group, 'silver')
        else:
            fillcolor = 'silver'

        G.nodes[node]['fillcolor'] = fillcolor
        
def assign_color_one(G, pos_partition, neg_partition, pos_colors=None, neg_colors=None):
    """
    Assigns fillcolor attributes to nodes in a NetworkX graph G.
    - If a node belongs to both pos and neg communities: (color1, color2)
    - If only one: that color
    - If none: 'silver'
    """
    for node in G.nodes():
        node_str = str(node)
        pos_group = pos_partition.get(node_str)
        neg_group = neg_partition.get(node_str)

        has_pos = pos_group is not None and pos_colors is not None
        has_neg = neg_group is not None and neg_colors is not None

        if has_pos and has_neg:
            color1 = pos_colors.get(pos_group, 'silver')
            color2 = neg_colors.get(neg_group, 'silver')
            G.nodes[node]['fillcolor'] = (color1, color2)
        elif has_pos:
            G.nodes[node]['fillcolor'] = pos_colors.get(pos_group, 'silver')
        elif has_neg:
            G.nodes[node]['fillcolor'] = neg_colors.get(neg_group, 'silver')
        else:
            G.nodes[node]['fillcolor'] = 'silver'


def draw_nodes_with_fillcolor(G, pos, ax, node_size=600):
    radius = (node_size ** 0.5) / 350  # Adjust radius to match networkx node size visually
    for node in G.nodes():
        fillcolor = G.nodes[node].get('fillcolor', 'silver')
        x, y = pos[node]

        if ';0.5:' in fillcolor:
            color1, color2 = fillcolor.split(';0.5:')
            # If one of the colors is silver, use the other color for the whole node
            if color1 == 'silver' and color2 != 'silver':
                circle = Circle((x, y), radius, facecolor=color2, edgecolor='black', lw=0)
                ax.add_patch(circle)
            elif color2 == 'silver' and color1 != 'silver':
                circle = Circle((x, y), radius, facecolor=color1, edgecolor='black', lw=0)
                ax.add_patch(circle)
            else:
                wedge1 = Wedge((x, y), radius, 90, 270, facecolor=color1, edgecolor='black', lw=0)
                wedge2 = Wedge((x, y), radius, 270, 90, facecolor=color2, edgecolor='black', lw=0)
                ax.add_patch(wedge1)
                ax.add_patch(wedge2)
        else:
            circle = Circle((x, y), radius, facecolor=fillcolor, edgecolor='black', lw=0.5)
            ax.add_patch(circle)

    ax.set_xlim(min(x for x, y in pos.values()) - radius, max(x for x, y in pos.values()) + radius)
    ax.set_ylim(min(y for x, y in pos.values()) - radius, max(y for x, y in pos.values()) + radius)
    ax.set_aspect('equal')
    ax.axis('off')

def plot_rule_network(df, rows, cols,  dvs, ctx=None, layout='spring', filename='network.pdf', title_prefix='', draw_partition = False):
    # Create list of all (context, DV) combinations
    if ctx is None or (hasattr(ctx, '__len__') and len(ctx) == 0):
        combinations = [(None, dv) for dv in dvs]
    else:
        combinations = [(context, dv) for context in ctx for dv in dvs]
    print(combinations)
    # Create subplots
    n_plots = len(combinations)
    n_cols = cols
    n_rows = rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)

    for idx, (context, dv) in enumerate(combinations):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        # Filter DataFrame based on DV and Context
        if context is not None and 'Context' in df.columns:
            df_filtered = df[(df['DV'] == dv) & (df['Context'] == context)]
        else:
            df_filtered = df[df['DV'] == dv]

        G = nx.from_pandas_edgelist(df_filtered, source='Condition1', target='Condition2',
                                    edge_attr='Importance')

        # Edge coloring logic and weights
        if title_prefix == 'Difference':
            edge_colors = ['royalblue' if G[u][v]['Importance'] > 0 else 'coral' for u, v in G.edges()]
        else:
            edge_colors = ['royalblue' if title_prefix == 'Positive' else 'coral'] * len(G.edges())

        edge_thickness = [abs(G[u][v]['Importance'] * 8) for u, v in G.edges()]
        
        #layout
        pos = nx.spring_layout(G, seed=rndstate) if layout == 'spring' else nx.circular_layout(G, scale=2)
        
        if draw_partition == True:
                ww
                # Convert to iGraph and run Leiden
                pos_partition = run_leiden_partition(G, invert=False, title_prefix = title_prefix)
                neg_partition = run_leiden_partition(G, invert=True, title_prefix = title_prefix)


                # Count occurrences of each community
                pos_counts = Counter(pos_partition.values())
                neg_counts = Counter(neg_partition.values())
        
                # Filter communities with at least two nodes
                valid_pos_communities = {comm for comm, count in pos_counts.items() if count >= 2}
                valid_neg_communities = {comm for comm, count in neg_counts.items() if count >= 2}

                # Assign green tones to positive communities
                
                pos_color_indices = list(colors_pos.keys())
                neg_color_indices = list(colors_neg.keys())
                
                if title_prefix == 'Positive': 
                    pos_colors = {
                        comm: colors_pos[pos_color_indices[i % len(pos_color_indices)]]
                        for i, comm in enumerate(sorted(valid_pos_communities))
                        }
                    neg_colors = None

                elif title_prefix == 'Negative':
                    neg_colors = {
                        comm: colors_neg[neg_color_indices[i % len(neg_color_indices)]]
                        for i, comm in enumerate(sorted(valid_neg_communities))
                        }
                    pos_colors = None

                else: #i.e. when it does the Difference
                    pos_colors = {
                        comm: colors_pos[pos_color_indices[i % len(pos_color_indices)]]
                        for i, comm in enumerate(sorted(valid_pos_communities))
                        }
                    neg_colors = {
                        comm: colors_neg[neg_color_indices[i % len(neg_color_indices)]]
                        for i, comm in enumerate(sorted(valid_neg_communities))
                        }
                    
                # Assign colors
                if title_prefix == 'Difference':
                    assign_color(G, pos_partition, neg_partition, pos_colors, neg_colors)
                else:
                    assign_color_one(G, pos_partition, neg_partition, pos_colors, neg_colors)


                # Draw network
                nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_thickness)
                draw_nodes_with_fillcolor(G, pos, ax, node_size=600)
        
        else:
            nx.draw(G, pos, with_labels=False, 
                    edge_color=edge_colors,
                    width = edge_thickness,
                    node_color = 'silver',
                    node_size=600, 
                    ax=ax
                    )
        nx.draw_networkx_labels(G, pos, ax=ax)

        # Title
        ctx_label = f"{context}" if context is not None else ""
        ax.set_title(f"{ctx_label} {dv}")
        ax.axis('off')

    # Hide unused subplots
    for idx in range(n_plots, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        fig.delaxes(axes[row][col])

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()


def assign_node_edge_colors(G, title_prefix):
    """
    Assigns node fillcolors and edge colors to the graph G based on community partitions and edge importance.
    Stores the fillcolor as node attribute and color as edge attribute for later use in plotting or similarity metrics.
    """
    # Run Leiden partitions
    pos_partition = run_leiden_partition(G, invert=False, title_prefix=title_prefix)
    neg_partition = run_leiden_partition(G, invert=True, title_prefix=title_prefix)

    # Count occurrences of each community
    pos_counts = Counter(pos_partition.values())
    neg_counts = Counter(neg_partition.values())

    # Filter communities with at least two nodes
    valid_pos_communities = {comm for comm, count in pos_counts.items() if count >= 2}
    valid_neg_communities = {comm for comm, count in neg_counts.items() if count >= 2}

    # Assign colors to communities
    pos_color_indices = list(colors_pos.keys())
    neg_color_indices = list(colors_neg.keys())

    if title_prefix == 'Positive':
        pos_colors = {
            comm: colors_pos[pos_color_indices[i % len(pos_color_indices)]]
            for i, comm in enumerate(sorted(valid_pos_communities))
        }
        neg_colors = None
    elif title_prefix == 'Negative':
        neg_colors = {
            comm: colors_neg[neg_color_indices[i % len(neg_color_indices)]]
            for i, comm in enumerate(sorted(valid_neg_communities))
        }
        pos_colors = None
    else:  # Difference
        pos_colors = {
            comm: colors_pos[pos_color_indices[i % len(pos_color_indices)]]
            for i, comm in enumerate(sorted(valid_pos_communities))
        }
        neg_colors = {
            comm: colors_neg[neg_color_indices[i % len(neg_color_indices)]]
            for i, comm in enumerate(sorted(valid_neg_communities))
        }

    # Assign node fillcolors using assign_color_one
    assign_color(G, pos_partition, neg_partition, pos_colors, neg_colors)

    # Assign edge colors
    for u, v in G.edges():
        importance = G[u][v].get('Importance', 0)
        if title_prefix == 'Difference':
            G[u][v]['color'] = 'royalblue' if importance > 0 else 'coral'
        else:
            G[u][v]['color'] = 'royalblue' if title_prefix == 'Positive' else 'coral'

    return G



def plot_rule_network_assigned_colors (df, rows, cols, dvs, ctx=None, layout='spring', filename='network.pdf', title_prefix='', draw_partition=True):
    # Create list of all (context, DV) combinations
    if ctx is None or (hasattr(ctx, '__len__') and len(ctx) == 0):
        combinations = [(None, dv) for dv in dvs]
    else:
        combinations = [(context, dv) for context in ctx for dv in dvs]

    # Create subplots
    n_plots = len(combinations)
    n_cols = cols
    n_rows = rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)

    for idx, (context, dv) in enumerate(combinations):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        # Filter DataFrame based on DV and Context
        if context is not None and 'Context' in df.columns:
            df_filtered = df[(df['DV'] == dv) & (df['Context'] == context)]
        else:
            df_filtered = df[df['DV'] == dv]

        G = nx.from_pandas_edgelist(df_filtered, source='Condition1', target='Condition2', edge_attr='Importance')

        # Assign node and edge colors using external function
        if draw_partition and len(G.nodes) > 0:
            assign_node_edge_colors(G, title_prefix)

        # Edge thickness
        edge_thickness = [abs(G[u][v]['Importance'] * 8) for u, v in G.edges()]

        # Layout
        pos = nx.spring_layout(G, seed=42) if layout == 'spring' else nx.circular_layout(G, scale=2)

        # Draw edges
        edge_colors = [G[u][v].get('color', 'gray') for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_thickness)

        # Draw nodes
        draw_nodes_with_fillcolor(G, pos, ax, node_size=600)
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax)

        # Title
        ctx_label = f"{context}" if context is not None else ""
        ax.set_title(f"{ctx_label} {dv}")
        ax.axis('off')

    # Hide unused subplots
    for idx in range(n_plots, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        fig.delaxes(axes[row][col])

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()
    

def color_similarity_distance(G1, G2, nodes_union):
    # Run Leiden partitions
    G1_pos_partition = run_leiden_partition(G1, invert=False, title_prefix='Difference')
    G1_neg_partition = run_leiden_partition(G1, invert=True, title_prefix='Difference')
    G2_pos_partition = run_leiden_partition(G2, invert=False, title_prefix='Difference')
    G2_neg_partition = run_leiden_partition(G2, invert=True, title_prefix='Difference')

    # Align partitions using the unified node set
    def get_partition_labels(partition, nodes_union):
        return [partition.get(node, -1) for node in nodes_union]

    G1_pos_labels = get_partition_labels(G1_pos_partition, nodes_union)
    G2_pos_labels = get_partition_labels(G2_pos_partition, nodes_union)
    G1_neg_labels = get_partition_labels(G1_neg_partition, nodes_union)
    G2_neg_labels = get_partition_labels(G2_neg_partition, nodes_union)

    node_ari_pos = adjusted_rand_score(G1_pos_labels, G2_pos_labels)
    node_ari_neg = adjusted_rand_score(G1_neg_labels, G2_neg_labels)

    # Edge color similarity using Jaccard
    def jaccard_similarity(set1, set2):
        if not set1 and not set2:
            return 1.0  # Both empty → perfect similarity
        return len(set1 & set2) / len(set1 | set2)

    edges_union = set(G1.edges).union(set(G2.edges))

    # Build sets of (u, v, color) for each graph
    def get_color_edge_set(G, edges_union):
        color_edges = set()
        for u, v in edges_union:
            if G.has_edge(u, v):
                color = G[u][v].get('color', 'none')
                color_edges.add((u, v, color))
        return color_edges

    G1_color_edges = get_color_edge_set(G1, edges_union)
    G2_color_edges = get_color_edge_set(G2, edges_union)

    edge_jaccard = jaccard_similarity(G1_color_edges, G2_color_edges)

    # Combine node and edge similarity
    similarity_mod = (node_ari_pos + node_ari_neg) / 2
    similarity_edge = edge_jaccard

    return similarity_mod, similarity_edge

def build_graph(df):
        return nx.from_pandas_edgelist(df, source='Condition1', target='Condition2', edge_attr='Importance')

def get_partition_labels(partition, nodes_union):    
    return [partition.get(node, -1) for node in nodes_union]

def resolve_color(primary, secondary, colors_pos_list, colors_neg_list):
    # Only consider positive/negative colors or silver
    valid_primary = primary if primary in colors_pos_list + colors_neg_list + ['silver'] else 'ignore'
    valid_secondary = secondary if secondary in colors_pos_list + colors_neg_list + ['silver'] else 'ignore'
    
    if valid_primary == 'ignore' and valid_secondary == 'ignore':
        return None  # skip if both are irrelevant
    
    if valid_primary == 'silver' and valid_secondary == 'silver':
        return 'neutral'
    elif valid_primary == 'silver' and valid_secondary != 'silver':
        return valid_secondary
    elif valid_secondary == 'silver' and valid_primary != 'silver':
        return valid_primary
    elif valid_primary != valid_secondary:  # both non-silver and different
        return 'dual'
    else:  # both non-silver and same
        return valid_primary
