#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 10:42:00 2025

@author: jbaggio
"""

#usual suspect plus java calls and for data loading
import jpype
import os
import pickle
import pandas as pd
import numpy as np
from collections import Counter


from sklearn.cluster import KMeans


#routines for lasso regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

# ad-hoc psl model builder routine
from PSL_QCA_ModelBuilder import build_psl_model, filter_rule_meta_by_weight, gof_psl, extract_and_rank_rules
import PSL_QCA_ModelBuilder as pqm

#to graph results
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


rndstate = 7879

#relevant directories (files and output)
mainres = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis/MainResults'
#where to download results:
pslout = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis/PSL'

os.chdir(mainres)
dfpreds = pickle.load(open('predicted_codes.p', 'rb'))

dftops = pd.read_csv('TopicsMain.csv')
dftops = dftops.drop(['Unnamed: 0'], axis = 1)

topvals = list(set(dftops['topic']))
count_topics = dftops['topic'].value_counts()
#eliminate string columns
dftopclus = dftops.drop (['txt_title','topic'], axis = 1)
#dftopclus = dftopclus.where(dftopclus> 0.2, other=0)

#from the graph check elbow point
kmeans = KMeans(n_clusters = 7, init='k-means++', n_init = 'auto', random_state = 76)
kmeans.fit(dftopclus)
dftops['kmeans'] = kmeans.predict(dftopclus)
dftops['kmeans'].value_counts()

#we now combine kmeans and DBSCAN clusters with the main topic to finalize clustering. We do this visually
dftops['maintop'] = dftops['kmeans']


#now make sure you have one column per value of probs
dfprobs = dfpreds.loc[:, dfpreds.columns.str.endswith('probs')]
dfprobs = dfprobs.copy()
#for case study the argmax always coincide with probability > 0.5
dfprobs['CaseStudy'] = dfpreds['CaseStudy_p']
#now only take into account coded case studies
dfprobs_cases = dfprobs[dfprobs['CaseStudy'] == 1]


"""
#now create three dataframes with values only for presence and absence (missing = 1- (pres + abs)) 
DO not take the caseStudy prob into account as we are using argmax (CaseStudy_p) 
and the dataframe is already filtered for only values where CaseStudy_p == 1
"""

#only presence and absence, we avoid the missing part, as not relevant here (after all prob_miss = prob_abs + prob_pres)
dfpreds_all = pd.DataFrame()
for col in dfprobs_cases.columns:
    if col!='CaseStudy_probs':
        colname = col.replace('_probs', '')
        if col != 'CaseStudy':
            new_col = colname + '_pres'
            dfpreds_all[new_col] = dfprobs_cases[col].str[1]
            new_col = colname + '_abs'
            dfpreds_all[new_col] = dfprobs_cases[col].str[0]
            new_col = colname + '_mis'
            dfpreds_all[new_col] = dfprobs_cases[col].str[2]
dfpreds_all = dfpreds_all.reset_index(drop = True)



#now give three letter (5 with presence absence) dictionaries name for columns, better for figures
      
depcols = {'rescond': 'RSC', 'conflict':'CFL', 'inequality':'IEQ' }

indepcols = {'resbound': 'RSB','socbound'	: 'SCB', 'fit' : 'FIT', 'prop' : 'PRP', 'participation' : 'PRT', 
             'monitor' : 'MON', 'accmonitor' : 'ACM', 'gradsanc' : 'GRS', 'confres'	: 'CFR', 
             'autonomy' : 'AUT', 'nested' : 'NST'}

#avoid cat variable issue with pandas warning for now
import warnings
warnings.filterwarnings('ignore')


'''
now assess difference in presence/absence as this is important information, 
and set values to no difference in case of high probability of it
being classfied as missing > 0.66
'''

dfdiff = pd.DataFrame()
 # Iterate through the columns and calculate the differences
for col in dfpreds_all.columns:
    if col.endswith('_pres'):
        base_col = col[:-5]  # Remove the suffix '_pres'
        if base_col == 'conflict' or base_col == 'inequality':
            pres_col = dfpreds_all[base_col +'_pres']
            abs_col = dfpreds_all[base_col + '_abs']
            mis_col = dfpreds_all[base_col + '_mis']

            # Calculate the difference and set to 0 if mis > 0.66
            dfdiff[base_col] = np.where(mis_col > 0.66, 0, abs_col - pres_col)
            print(base_col + ' Absence is good')
        else:
            pres_col = dfpreds_all[base_col +'_pres']
            abs_col = dfpreds_all[base_col + '_abs']
            mis_col = dfpreds_all[base_col + '_mis']

            # Calculate the difference and set to 0 if mis > 0.66
            dfdiff[base_col] = np.where(mis_col > 0.66, 0, pres_col - abs_col)
            print(base_col + ' Presence is good')
            

dfdiff = dfdiff.rename(columns=depcols)
dfdiff = dfdiff.rename(columns=indepcols)

#drop resource change as not relevant for the analysis
dfdiff = dfdiff = dfdiff.drop(['reschange'], axis=1)


dfall = pd.concat([dfdiff, dftops], axis=1)
dfall.to_csv('ATopic_Prob.csv')
#check the clustering (maintop with kmeans clusters very well and see the names - as topic combines multiple topics, write them in topnames
#if random seed in kmeans is not changed, the results should be the same and produce the following topics
topnames = ['Land Rights', 'Forestry', 'Fishery', 'Livestock', 'Water', 'Development', 'Biodiversity']
labels = {i: topnames[i] for i in range(len(topnames))}
dfall['topname'] = dfall['maintop'].map(labels)

#divide dataframe for outcome and features.
depvars = dfdiff[['RSC', 'CFL', 'IEQ']].copy()
indepvars = dfdiff[['SCB', 'RSB', 'FIT', 'PRP', 'PRT', 'MON', 'ACM', 'GRS', 'CFR', 'AUT', 'NST']].copy()

depvars_orig = depvars.copy
indepvars_orig = indepvars.copy

depvars = (depvars + 1) / 2
indepvars = (indepvars + 1) / 2

indepvars_top = indepvars
indepvars_top['topname'] = dfall['topname']
depvars_top = depvars
depvars_top['topname'] = dfall['topname']
gr_indep = indepvars_top.groupby(indepvars_top['topname'])
gr_dep = depvars_top.groupby(depvars_top['topname'])


"""

Check simple spearman correlations and significance for individual IV on outcome

"""
os.chdir(pslout)

# Calculate correlations
# Assuming depvars and indepvars are already defined DataFrames
combined_df = pd.concat([depvars, indepvars], axis=1)
combined_df = combined_df.drop('topname', axis=1)

# Get column names
depcols = list(depvars.columns)
depcols.remove('topname')
indepcols = list(indepvars.columns)
indepcols.remove('topname')

# Initialize matrices for correlation and p-values
correlation_matrix = pd.DataFrame(index=indepcols, columns=depcols)
pvalue_matrix = pd.DataFrame(index=indepcols, columns=depcols)

# Calculate Spearman correlation and p-values
for indep in indepcols:
    for dep in depcols:
        corr, pval = spearmanr(combined_df[indep], combined_df[dep])
        correlation_matrix.loc[indep, dep] = corr
        pvalue_matrix.loc[indep, dep] = pval

# Convert to numeric
spearman_mat = correlation_matrix.astype(float)
pval_mat = pvalue_matrix.astype(float)

# Create annotations with significance stars
def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

annot_mat = spearman_mat.round(2).astype(str)
for i in spearman_mat.index:
    for j in spearman_mat.columns:
        annot_mat.loc[i, j] += significance_stars(pval_mat.loc[i, j])

# Plot heatmap using seaborn
plt.figure(figsize=(10, 6))
sns.heatmap(spearman_mat.astype(float), annot=annot_mat, fmt='', cmap='coolwarm_r', center=0)
plt.tight_layout()
plt.savefig('Spearman_heatmap.pdf')
plt.show()


# Function to compute Spearman correlation and p-values with significance stars
def compute_spearman_with_significance(dep_df, indep_df, depcols, indepcols):
    corr_matrix = pd.DataFrame(index=indepcols, columns = depcols)
    pval_matrix = pd.DataFrame(index=indepcols, columns = depcols)
    annot_matrix = pd.DataFrame(index=indepcols, columns = depcols)

    for dep_col in depcols:
        for indep_col in indepcols:
            corr, pval = spearmanr(indep_df[indep_col], dep_df[dep_col], nan_policy='omit')
            corr_matrix.loc[indep_col, dep_col] = corr
            pval_matrix.loc[indep_col, dep_col] = pval
            if pval < 0.001:
                star = '***'
            elif pval < 0.01:
                star = '**'
            elif pval < 0.05:
                star = '*'
            else:
                star = ''
            annot_matrix.loc[indep_col, dep_col] = f"{corr:.2f}{star}"

    return corr_matrix.astype(float), annot_matrix


def plot_grouped_heatmaps(grdep, grindep, depcols, indepcols):
    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)

    for i, (group, df) in enumerate(grdep):
        row = i // ncols
        col = i % ncols
        print(i)
        print(row)
        print(col)
        dep_df = grdep.get_group(group)
        indep_df = grindep.get_group(group)
        corr_matrix, annot_matrix = compute_spearman_with_significance(indep_df, dep_df, indepcols, depcols)

        sns.heatmap(corr_matrix, annot=annot_matrix, fmt='', cmap='coolwarm_r', center=0,
                    annot_kws={"size": 10}, ax=axes[row, col], cbar=True,
                    xticklabels=corr_matrix.columns, yticklabels=corr_matrix.index)
        axes[row, col].set_title(f"{group}")

    # Hide unused subplots if any
    for j in range(i + 1, nrows * ncols):
        row = j // ncols
        col = j % ncols
        fig.delaxes(axes[row][col])

    plt.tight_layout()
    plt.savefig("Spearman_Grouped_heatmap.pdf")
    plt.show()


plot_grouped_heatmaps(gr_indep, gr_dep, indepcols, depcols)



"""

Lasso regression for parsimonious combination of DP that affect the outcome. 
We use up to 8 DP in combination to assess configurations of DP and their effect. We use standard values for LASSO (default hyperparameters)

"""


os.chdir(pslout)

lasso_dict = {}
mse_dict = {}

for dep in depvars:
    print (dep)
    
    ydep = depvars[dep]
    
    poly = PolynomialFeatures(degree=8, interaction_only=True, include_bias=False)
    indep_poly = poly.fit_transform(indepvars)
    feature_names = poly.get_feature_names_out(indepvars.columns)
    lasso = LassoCV(cv = 10, random_state = rndstate)
    lasso.fit(indep_poly, ydep)
    
    selected = np.abs(lasso.coef_) > 1e-4
    selected_features = feature_names[selected]
    selected_coefficients = lasso.coef_[selected]
    
    y_pred = lasso.predict(indep_poly)
    mse = mean_squared_error(ydep, y_pred)
    
    lasso_dict[dep] = list(zip(selected_features, selected_coefficients))
    mse_dict[dep] = mse
    
pickle.dump(lasso_dict, open('dict_lasso.p', 'wb'))
df_mse = pd.DataFrame(mse_dict, index = [0])
df_mse.to_csv('MSE.csv') 

    

        
topic_lasso_dict = {}
topic_mse_dict = {}

for topic, indep_topic in gr_indep:  
    indep_topic = indep_topic.drop('topname', axis = 1)
    dep_topic = gr_dep.get_group(topic)
    for dep in dep_topic.columns.drop('topname'):
        print (topic)
        print(dep)
        
        ydep = dep_topic[dep]
        
        poly = PolynomialFeatures(degree=5, interaction_only=True, include_bias=False)
        indep_poly = poly.fit_transform(indep_topic)
        feature_names = poly.get_feature_names_out(indep_topic.columns)
        lasso = LassoCV(cv = 10, random_state = rndstate)
        lasso.fit(indep_poly, ydep)
        
        selected = np.abs(lasso.coef_) > 1e-4
        selected_features = feature_names[selected]
        selected_coefficients = lasso.coef_[selected]
        
        y_pred = lasso.predict(indep_poly)
        mse = mean_squared_error(ydep, y_pred)

        topic_lasso_dict[topic, dep] = list(zip(selected_features, selected_coefficients))
        topic_mse_dict[topic, dep] = mse

pickle.dump(topic_lasso_dict, open('topic_lasso.p', 'wb'))
df_msetopic = pd.DataFrame.from_dict(topic_mse_dict, orient='index', columns=['MSE'])
df_msetopic.index = pd.MultiIndex.from_tuples(df_msetopic.index, names=['Topic', 'Variable'])
df_msetopic.to_csv('Topic_mse.csv')


"""
PROBABILISTIC SOFT LOGIC MODELLING

"""

# Load the dictionary from the pickle file
lasso_dict = pickle.load(open('dict_lasso.p', 'rb'))
topic_lasso_dict = pickle.load(open('topic_lasso.p', 'rb'))


#start java virtual machine
os.environ["JAVA_HOME"] = "/Users/jbaggio/miniconda3/envs/topmod"


lib_dir = "/Users/jbaggio/miniconda3/envs/topmod/lib/psl-jars"
classpath = [os.path.join(lib_dir, jar) for jar in os.listdir(lib_dir) if jar.endswith(".jar")]
jpype.startJVM(jpype.getDefaultJVMPath(), classpath=classpath)



# Configuration parameters
config = {
    'USE_GLOBAL_EXPLORATORY': False,
    'SUBSET_DECAY': 0.50,
    'ADD_NEGATIVE_DV_PRIOR': True,
    'DV_PRIOR_WEIGHT': 0.1,
    'MAX_K': 11,
    'MIN_K': 3,
    'MIN_SUPPORT_FRAC': 0.005,
    'MIN_MEAN_BODY': 0.00,
    'LEARN_WEIGHTS': True,
    'ALL_IVOBS': True,
    'LASSO_W': False,
    'SUBSET_INCL': True
}


# Storage dictionaries
#original all rules model
model_dict = {}
infer_dict = {}
rule_dict = {}
gof_dict = {}

#filtered rules model
filter_model_dict = {}
filter_infer_dict = {}
filter_rule_dict = {}
filter_gof_dict = {}

# Step-by-step PSL analysis
for dv, coef_list in lasso_dict.items():
    print(f"[INFO] Processing DV: {dv}")

    # Step 1: Build initial model with all candidate rules
    model, rule_meta_list, name_map = build_psl_model(
        depvar=dv,
        indepvars=indepvars,
        depvars=depvars,
        rule_list=coef_list,
        filtered=False,
        config=config
        )
    # Step 2: Learn rule weights
    model.learn()
    model_infer = model.infer()

    # Step 3: Filter rules based on learned weights
    filtered_rules = filter_rule_meta_by_weight(rule_meta_list, model)

    # Step 4: Rebuild model with filtered rules
    filtered_rule_list = [(" ".join(r['parts']), r['sign']) for r in filtered_rules]   
   
    model_filtered, rule_meta_filtered, name_map_filtered = build_psl_model(
        depvar=dv,
        indepvars=indepvars,
        depvars=depvars,
        rule_list=filtered_rule_list,
        filtered=True,
        config=config
        )
    
    # Step 5: Re-learn weights
    model_filtered.learn()

    # Step 6: Run inference
    filter_infer = model_filtered.infer()

    mse, rmse, eval_df = gof_psl(model, depvars[dv], model_infer)
    gof_orig = f"[{dv}] N={len(eval_df)} | MSE={mse:.4f} | RMSE={rmse:.4f}"

    fmse, frmse, feval_df = gof_psl (model_filtered, depvars[dv], filter_infer)
    gof_filter = f"[{dv}] N={len(feval_df)} | MSE={fmse:.4f} | RMSE={frmse:.4f}"

   
    # Store results
    model_dict[dv] = model
    rule_dict[dv] = rule_meta_list
    infer_dict[dv] = model_infer
    gof_dict[dv] = gof_orig
    
    filter_model_dict[dv] = model_filtered
    filter_rule_dict[dv] = rule_meta_filtered
    filter_infer_dict[dv] = filter_infer
    filter_gof_dict[dv] = gof_filter

    
  
    
"""

TOPIC BASED PSL

"""  

# Storage dictionaries
#original all rules model
topic_model_dict = {}
topic_infer_dict = {}
topic_rule_dict = {}
topic_gof_dict = {}

#filtered rules model
topic_filter_model_dict = {}
topic_filter_infer_dict = {}
topic_filter_rule_dict = {}
topic_filter_gof_dict = {}


# Step-by-step PSL analysis
for dtv, coef_list in topic_lasso_dict.items():
    print(f"[INFO] Processing DV: {dv}")
    dv = dtv[1]
    topic = dtv[0]
    # Step 1: Build initial model with all candidate rules
    model, rule_meta_list, name_map = build_psl_model(
        depvar=dv,
        indepvars= gr_indep.get_group(topic),
        depvars= gr_dep.get_group(topic),
        rule_list=coef_list,
        filtered=False,
        config=config
        )
    # Step 2: Learn rule weights
    model.learn()
    model_infer = model.infer()

    # Step 3: Filter rules based on learned weights
    filtered_rules = filter_rule_meta_by_weight(rule_meta_list, model)

    # Step 4: Rebuild model with filtered rules
    filtered_rule_list = [(" ".join(r['parts']), r['sign']) for r in filtered_rules]   
   
    model_filtered, rule_meta_filtered, name_map_filtered = build_psl_model(
        depvar=dv,
        indepvars= gr_indep.get_group(topic),
        depvars=gr_dep.get_group(topic),
        rule_list=filtered_rule_list,
        filtered=True,
        config=config
        )
    
    # Step 5: Re-learn weights
    model_filtered.learn()

    # Step 6: Run inference
    filter_infer = model_filtered.infer()

    mse, rmse, eval_df = gof_psl(model, gr_dep.get_group(topic)[dv], model_infer)
    gof_orig = f"[{dv}] N={len(eval_df)} | MSE={mse:.4f} | RMSE={rmse:.4f}"

    fmse, frmse, feval_df = gof_psl (model_filtered, gr_dep.get_group(topic)[dv], filter_infer)
    gof_filter = f"[{dv}] N={len(feval_df)} | MSE={fmse:.4f} | RMSE={frmse:.4f}"

   
    # Store results
    topic_model_dict[dtv] = model
    topic_rule_dict[dtv] = rule_meta_list
    topic_infer_dict[dtv] = model_infer
    topic_gof_dict[dtv] = gof_orig
    
    topic_filter_model_dict[dtv] = model_filtered
    topic_filter_rule_dict[dtv] = rule_meta_filtered
    topic_filter_infer_dict[dtv] = filter_infer
    topic_filter_gof_dict[dtv] = gof_filter


dfgof = pd.DataFrame.from_dict(gof_dict, orient = 'index')    
dfgof_filter = pd.DataFrame.from_dict(filter_gof_dict, orient = 'index')

dfgof_topic = pd.DataFrame.from_dict(topic_gof_dict, orient = 'index')    
dfgof_topic_filter = pd.DataFrame.from_dict(topic_filter_gof_dict, orient = 'index')

dfgof.to_csv('Gof_PSL.csv')
dfgof_filter.to_csv('Gof_PSL_Filter.csv')

dfgof_topic.to_csv('Gof_Topic_PSL.csv')
dfgof_topic_filter.to_csv('Gof_Topic_PSL_Filter.csv')


pickle.dump(model_dict, open('filter_model_dict.p', 'wb'))
pickle.dump(topic_model_dict, open('topic_filter_model_dict.p', 'wb'))

    
"""
QCA type results: check original vs filtered rules model and then choose the most appropriate one for rule extraction
    
Importance here is determined by coverage * (consistency + effect). Effect is calcualted as consistency - baseline, so 
the importance weighs consistency/effect more than coverage as some rule may be very effective but only in a handful of cases. 


"""

filter_model_dict = pickle.load(open('filter_model_dict.p', 'rb'))
topic_filter_model_dict = pickle.load(open('topic_filter_model_dict.p', 'rb'))


#now select the most important N rules that have highest consistency and coverage and push the outcome either twards positive or negative.
pos_rules = {}
neg_rules = {}

# Top 10 rules that increase DV, ranked by coverage × consistency
for key in filter_model_dict:

    model_temp = filter_model_dict[key]
    rule_meta = filter_rule_dict[key]
    infer_dv = infer_dict[key]
    rules = model_temp.get_rules()
    
    p_rules = extract_and_rank_rules(rulelist = rule_meta, model = model_temp, indeps = indepvars, deps = depvars, single_dep = key,
                       top_n=50, direction='positive', importance_metric='coverage_consistency_effect', use_inferred = True, inferred_res = infer_dv)
    n_rules = extract_and_rank_rules(rulelist = rule_meta, model = model_temp, indeps = indepvars, deps = depvars, single_dep = key,
                       top_n=50, direction='negative', importance_metric='coverage_consistency_effect', use_inferred = True, inferred_res = infer_dv)
    
    #store in dictionary
    pos_rules[key] = p_rules
    neg_rules[key] = n_rules
            
   
#now select the most important 20 rules that have highest consistency,. coverage  and effect, and push the outcome either twards positive or negative.
topic_pos_rules = {}
topic_neg_rules = {}

# Top 10 rules that increase DV, ranked by coverage × consistency
for ktdv in topic_filter_model_dict:
    key = ktdv[1]
    topic = ktdv[0]
    model_temp = topic_filter_model_dict[ktdv]
    rule_meta = topic_filter_rule_dict[ktdv]
    infer_topdv = topic_infer_dict[ktdv]
    rules = model_temp.get_rules()
    indep_top = gr_indep.get_group(topic)
    dep_top = gr_dep.get_group(topic)
    print(str(ktdv) + ': Indep N = ' + str(len(indep_top)) + ', Dep N = ' + str(len(dep_top)))
    
    p_rules = extract_and_rank_rules(rulelist = rule_meta, model = model_temp, indeps = indep_top, deps = dep_top, single_dep = key,
                           top_n=50, direction='positive', importance_metric='coverage_consistency_effect', use_inferred = True, inferred_res = infer_topdv)
    n_rules = extract_and_rank_rules(rulelist = rule_meta, model = model_temp, indeps = gr_indep.get_group(topic), deps = dep_top, single_dep = key,
                           top_n=50, direction='negative', importance_metric='coverage_consistency_effect', use_inferred = True, inferred_res = infer_topdv)
    
    #store data in dict
    topic_pos_rules[ktdv] = p_rules
    topic_neg_rules[ktdv] = n_rules
    


# Convert pos_rules and neg_rules dictionaries to DataFrames
df_pos_rules = pd.concat([df.assign(DV=dv) for dv, df in pos_rules.items()], ignore_index=True)
df_neg_rules = pd.concat([df.assign(DV=dv) for dv, df in neg_rules.items()], ignore_index=True)

# Convert topic_pos_rules and topic_neg_rules dictionaries to DataFrames
df_topic_pos_rules = pd.concat([df.assign(Context=ctx, DV=dv) for (ctx, dv), df in topic_pos_rules.items()], ignore_index=True)
df_topic_neg_rules = pd.concat([df.assign(Context=ctx, DV=dv) for (ctx, dv), df in topic_neg_rules.items()], ignore_index=True)



''' 

START VISUALIZATION 

'''
os.chdir(pslout)

# Add rule type
df_pos_rules["type"] = "positive"
df_neg_rules["type"] = "negative"
df_topic_pos_rules['type'] = 'positive'
df_topic_neg_rules ['type'] = 'negative'

# Merge dataframes
dfrules = pd.concat([df_pos_rules, df_neg_rules], ignore_index=True)
dfrules['clean_rule'] = dfrules['rule'].apply(pqm.clean_rule)

dfrules_topic = pd.concat([df_topic_pos_rules, df_topic_neg_rules], ignore_index = True)
dfrules_topic['clean_rule'] = dfrules_topic['rule'].apply(pqm.clean_rule)

# Save Dataframes to CSV
dfrules.to_csv('Configs_All.csv')
dfrules_topic.to_csv('Configs_Topic.csv')

dfrules = pd.read_csv('Configs_All.csv')
dfrules_topic = pd.read_csv('Configs_Topic.csv')


#Calculate co-occurrence weighted by importance for rule networks


# Separate positive and negative rules
df_pos = dfrules[dfrules['type'] == 'positive']
df_neg = dfrules[dfrules['type'] == 'negative'] 

dfctx_pos = dfrules_topic[dfrules_topic['type'] == 'positive']
dfctx_neg = dfrules_topic[dfrules_topic['type'] == 'negative'] 

# Count co-occurrences

pos_cooccur = pqm.count_cooccurrences(df_pos, 'positive', include_context = False)
neg_cooccur = pqm.count_cooccurrences(df_neg, 'negative', include_context = False)

# Rename importance columns for clarity
pos_df = pos_cooccur.rename(columns={'Importance': 'Positive_Importance'})
neg_df = neg_cooccur.rename(columns={'Importance': 'Negative_Importance'})

# Merge on shared keys
posneg_df = pd.merge(
    pos_df,
    neg_df,
    on=['Condition1', 'Condition2', 'DV'],
    how='outer'  # ensures all pairs are included
)

# Fill missing values with 0 and add label for difference
posneg_df['Label'] = 'Difference'
posneg_df['Positive_Importance'] = posneg_df['Positive_Importance'].fillna(0)
posneg_df['Negative_Importance'] = posneg_df['Negative_Importance'].fillna(0)

# Compute the difference
posneg_df['Importance'] = posneg_df['Positive_Importance'] - posneg_df['Negative_Importance']

#no do the same bu with context
ctx_pos_cooccur = pqm.count_cooccurrences(dfctx_pos, 'positive', include_context = True)
ctx_neg_cooccur = pqm.count_cooccurrences(dfctx_neg, 'negative', include_context = True)

# Rename importance columns for clarity
ctx_pos_df = ctx_pos_cooccur.rename(columns={'Importance': 'Positive_Importance'})
ctx_neg_df = ctx_neg_cooccur.rename(columns={'Importance': 'Negative_Importance'})

# Merge on shared keys
ctx_posneg_df = pd.merge(
    ctx_pos_df,
    ctx_neg_df,
    on=['Condition1', 'Condition2', 'Context', 'DV'],
    how='outer'  # ensures all pairs are included
)

# Fill missing values with 0 amd add labels for difference:
ctx_posneg_df['Label'] = 'Difference'
ctx_posneg_df['Positive_Importance'] = ctx_posneg_df['Positive_Importance'].fillna(0)
ctx_posneg_df['Negative_Importance'] = ctx_posneg_df['Negative_Importance'].fillna(0)

# Compute the difference
ctx_posneg_df['Importance'] = ctx_posneg_df['Positive_Importance'] - ctx_posneg_df['Negative_Importance']



'''

All topics combined

'''

# Scatter plot: consistency vs coverage per DV
dvs = dfrules["DV"].unique()
ctx = dfrules_topic['Context'].unique()

# Define fixed colors for rule types
color_map = {
    'positive': 'royalblue',
    'negative': 'coral'
}

fig, axes = plt.subplots(1, len(dvs), figsize=(6 * len(dvs), 5), squeeze=False)

for i, dv in enumerate(dvs):
    ax = axes[0, i]
    subset = dfrules[dfrules["DV"] == dv]
    for rule_type in subset["type"].unique():
        sub = subset[subset["type"] == rule_type]
        ax.scatter(sub["coverage"], sub["consistency"], label=rule_type, alpha=0.7, color=color_map[rule_type])
    ax.set_title(f"{dv}")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Consistency")
    ax.legend()

plt.tight_layout()
plt.savefig('Rule_consistencyCoverage.pdf')
plt.close()


# Rule structure network

pqm.plot_rule_network(pos_cooccur, 1, 3, dvs, ctx = None, layout='spring', filename='Rule_StructureNetwork_Positive.pdf', title_prefix='Positive')
pqm.plot_rule_network(neg_cooccur, 1, 3, dvs, ctx = None, layout='spring', filename='Rule_StructureNetwork_Negative.pdf', title_prefix='Negative')
pqm.plot_rule_network(posneg_df, 1, 3, dvs, ctx = None, layout='spring', filename='Rule_StructureNetwork_Diff.pdf', title_prefix='Difference')

pqm.plot_rule_network(pos_cooccur, 1, 3, dvs, ctx = None, layout='spring', filename='Part_Rule_StructureNetwork_Positive.pdf', title_prefix='Positive', draw_partition = True)
pqm.plot_rule_network(neg_cooccur, 1, 3, dvs, ctx = None, layout='spring', filename='Part_Rule_StructureNetwork_Negative.pdf', title_prefix='Negative', draw_partition = True)
pqm.plot_rule_network(posneg_df, 1, 3, dvs, ctx = None, layout='spring', filename='Part_Rule_StructureNetwork_Diff.pdf', title_prefix='Difference', draw_partition = True)

 

'''
Subplots per topic

'''

# Create subplots: rows = contexts, columns = dvs
fig, axes = plt.subplots(len(ctx), len(dvs), figsize=(3 * len(dvs), 3 * len(ctx)), squeeze=False)

for i, context in enumerate(ctx):
    for j, dv in enumerate(dvs):
        ax = axes[i, j]
        subset = dfrules_topic[(dfrules_topic["DV"] == dv) & (dfrules_topic["Context"] == context)]
        print (dv + ': ' + context + ': ' + str(len(subset)))
        for rule_type in ['positive', 'negative']:  # Ensure both types are checked
            sub = subset[subset["type"] == rule_type]
            if not sub.empty:
                ax.scatter(sub["coverage"], sub["consistency"], label=rule_type, alpha=0.7, color=color_map[rule_type])
        ax.set_title(f"{context}, {dv}")
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Consistency")
        ax.legend()

plt.tight_layout()
plt.savefig('Rule_Context_ConsistencyCoverage.pdf')
plt.close()


# Rule structure network
    
pqm.plot_rule_network(ctx_pos_cooccur, 4, 6, dvs, ctx=ctx, layout='spring', filename='Rule_Context_StructureNetwork_Positive.pdf', title_prefix='Positive')
pqm.plot_rule_network(ctx_neg_cooccur, 4, 6, dvs, ctx=ctx, layout='spring', filename='Rule_Context_StructureNetwork_Negative.pdf', title_prefix='Negative')
pqm.plot_rule_network(ctx_posneg_df, 4, 6, dvs, ctx = ctx, layout='spring', filename='Rule_Context_StructureNetwork_Diff.pdf', title_prefix='Difference')

pqm.plot_rule_network(ctx_pos_cooccur, 4, 6, dvs, ctx=ctx, layout='spring', filename='Part_Rule_Context_StructureNetwork_Positive.pdf', title_prefix='Positive', draw_partition = True)
pqm.plot_rule_network(ctx_neg_cooccur, 4, 6, dvs, ctx=ctx, layout='spring', filename='Part_Rule_Context_StructureNetwork_Negative.pdf', title_prefix='Negative', draw_partition = True)
pqm.plot_rule_network(ctx_posneg_df, 4, 6, dvs, ctx = ctx, layout='spring', filename='Part_Rule_Context_StructureNetwork_Diff.pdf', title_prefix='Difference', draw_partition = True)



"""Calculate network similarity based on color similarity"""

#assess similarity between matrices
all_nodes = sorted(set(posneg_df['Condition1']).union(set(posneg_df['Condition2'])))

#Do with edit distance with highest penalty for insertion/deletion of nodes, second highest for sign-flips.
similarity = np.zeros((len(dvs), len(dvs)))
similarity_edge = np.zeros((len(dvs), len(dvs)))

#Edit for all contexts together per DV
i = 0
for dv in dvs:
    df_filtered_i = posneg_df[posneg_df['DV'] == dv]
    G_i =pqm.build_graph(df_filtered_i)
    G_i = pqm.assign_node_edge_colors(G_i, 'Difference')
    

    j = 0
    for dv2 in dvs:
        df_filtered_j = posneg_df[posneg_df['DV'] == dv2]
        G_j = pqm.build_graph(df_filtered_j)
        G_j = pqm.assign_node_edge_colors(G_j, 'Difference')

    
        # Compute custom edit distance
        sim, sim_edge = pqm.color_similarity_distance(G_i, G_j, all_nodes)
        similarity[i, j] = sim
        similarity_edge [i, j] = sim_edge

        j += 1
    i += 1
    
# Create and plot heatmap looking at similarities not distances
df_sim = pd.DataFrame(similarity, index=dvs, columns=dvs)
df_sim_edge = pd.DataFrame(similarity_edge, index=dvs, columns=dvs)

sns.heatmap(df_sim, cmap="coolwarm_r", annot=True, fmt='.3f')
plt.savefig('Similarity_All_modules.pdf')
plt.close()

sns.heatmap(df_sim_edge, cmap="coolwarm_r", annot=True, fmt='.3f')
plt.savefig('Similarity_All_edges.pdf')
plt.close()


# Create all (Topic, DV) combinations
labels = [(topic, dv) for topic in ctx for dv in dvs]
combs = len(labels)
# Initialize distance matrices
similarity_top = np.zeros((combs, combs))
similarity_top_edge = np.zeros((combs, combs))

# Compute distances
for i, (topic_i, dv_i) in enumerate(labels):
    df_i = ctx_posneg_df[(ctx_posneg_df['DV'] == dv_i) & (ctx_posneg_df['Context'] == topic_i)]
    Gt_i = pqm.build_graph(df_i)
    
    if len(Gt_i.nodes) == 0:
        for j in range(len(labels)):
            similarity_top[i, j] = -1
        continue
    Gt_i = pqm.assign_node_edge_colors(Gt_i, 'Difference')

    for j, (topic_j, dv_j) in enumerate(labels):
        df_j = ctx_posneg_df[(ctx_posneg_df['DV'] == dv_j) & (ctx_posneg_df['Context'] == topic_j)]
        Gt_j = pqm.build_graph(df_j)

        if len(Gt_j.nodes) == 0:
            similarity_top[i, j] = -1
            continue

        Gt_j = pqm.assign_node_edge_colors(Gt_j, 'Difference')        
        sim_mod, sim_edge = pqm.color_similarity_distance(Gt_i, Gt_j, all_nodes)
        similarity_top[i, j] = sim_mod
        similarity_top_edge[i,j] = sim_edge
        
# Create MultiIndex DataFrames
index = pd.MultiIndex.from_tuples(labels, names=["Topic", "DV"])
df_sim_top = pd.DataFrame(similarity_top, index=index, columns=index)
df_sim_top_edge = pd.DataFrame(similarity_top_edge, index=index, columns=index)


# Sort by total distance from each row
df_sim_top['Total'] = df_sim_top.sum(axis=1)
df_sim_top_sorted = df_sim_top.sort_values(by='Total').drop(columns='Total')
df_sim_top_sorted = df_sim_top_sorted[df_sim_top_sorted.index]
#eliminate the Forestry-CFL as it is empty (no rules found)
df_sim_top_sorted = df_sim_top_sorted.drop(index=df_sim_top_sorted.index[0], columns=df_sim_top_sorted.columns[0])

# Sort by total distance from each row
df_sim_top_edge['Total'] = df_sim_top_edge.sum(axis=1)
df_sim_top_edge_sorted = df_sim_top_edge.sort_values(by='Total').drop(columns='Total')
df_sim_top_edge_sorted = df_sim_top_edge_sorted[df_sim_top_edge_sorted.index]
#eliminate the Forestry-CFL as it is empty (no rules found)
df_sim_top_edge_sorted = df_sim_top_edge_sorted.drop(index=df_sim_top_edge_sorted.index[0], columns=df_sim_top_edge_sorted.columns[0])


# Plot heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(df_sim_top_sorted, cmap="coolwarm_r", annot=True,  vmax = 0.5, fmt=".2f")
plt.tight_layout()
plt.savefig("Similarity_CTX_modules.pdf")
plt.close()



# Plot heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(df_sim_top_edge_sorted, cmap="coolwarm_r", annot=True,  vmax = 0.5, fmt=".2f")
plt.tight_layout()
plt.savefig("Similarity_CTX_edges.pdf")
plt.close()

# Define color mappings for topics and categories
group_colors = {
    'Biodiversity': 'lime', 'Development': 'black', 'Fishery': 'cyan',
    'Forestry': 'yellowgreen', 'Land Rights': 'orange', 'Livestock': 'sienna', 'Water': 'blue'
}
dep_colors = {'RSC': 'darkgreen', 'CFL': 'gray', 'IEQ': 'magenta'}

# Create lists of colors for rows and columns
dv_colors    = df_sim_top_sorted.index.get_level_values(1).map(dep_colors)
topic_colors = df_sim_top_sorted.index.get_level_values(0).map(group_colors)

plt.figure()
g = sns.clustermap(df_sim_top_sorted, cmap="coolwarm_r", annot=True,  vmax = 0.5, fmt=".2f", figsize=(15, 12),  metric="euclidean", method ="average", row_colors=[dv_colors, topic_colors])
g.ax_col_dendrogram.set_visible(False)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=24)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=24)
g.cax.tick_params(labelsize=20)
pos = g.cax.get_position()
g.cax.set_position([0.92, -0.09, 0.05, 0.2])  
g.ax_heatmap.set_xlabel("")  
g.ax_heatmap.set_ylabel("")
g.savefig("Similarity_ClusterCTX_modules.pdf")
plt.show()


# Create lists of colors for rows and columns
dv_colors    = df_sim_top_edge_sorted.index.get_level_values(1).map(dep_colors)
topic_colors = df_sim_top_edge_sorted.index.get_level_values(0).map(group_colors)

plt.figure()
g2 = sns.clustermap(df_sim_top_edge_sorted, cmap="coolwarm_r", annot=True, vmax = 0.5, fmt=".2f", figsize=(15, 12), metric="euclidean", method ="average", row_colors=[dv_colors, topic_colors])
g2.ax_col_dendrogram.set_visible(False)
g2.ax_heatmap.set_xticklabels(g2.ax_heatmap.get_xticklabels(), fontsize=24)
g2.ax_heatmap.set_yticklabels(g2.ax_heatmap.get_yticklabels(), fontsize=24)
g2.cax.tick_params(labelsize=20)
pos = g2.cax.get_position()
g2.cax.set_position([0.92, -0.09, 0.05, 0.2])  
g2.ax_heatmap.set_xlabel("")  
g2.ax_heatmap.set_ylabel("")
g2.savefig("Similarity_ClusterCTX_edges.pdf")
plt.show()



g_dict ={}

#now do a dataframe for easier writing
for i, (topic, dv) in enumerate(labels):
    df_temp = ctx_posneg_df[(ctx_posneg_df['DV'] == dv) & (ctx_posneg_df['Context'] == topic)]
    G_temp = pqm.build_graph(df_temp)
    if len(G_temp.nodes)> 0:
        g_part = pqm.assign_node_edge_colors(G_temp, "Difference") 
        g_dict[topic, dv] = g_part


role_posneg = []

for (context, dv), G in g_dict.items():
    nodes = G.nodes(data=True)
    
    for node, attrs in nodes:
        fill = attrs.get('fillcolor', '')
        if fill:
            parts = fill.split(';')
            primary = parts[0]
            secondary = parts[-1].split(':')[-1]
            
            role_posneg.append({
                'Context': context,
                'DV': dv,
                'Node': node,
                'PrimaryColor': primary,
                'SecondaryColor': secondary
            })

# Convert to DataFrame
color_df = pd.DataFrame(role_posneg)

#color list as defined in PSL_QCA_ModelBuilder

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


# Classify polarity based on FinalColor
colors_pos_list = list(colors_pos.values())
colors_neg_list = list(colors_neg.values())


positive_colors = set(colors_pos.values())
negative_colors = set(colors_neg.values())

def classify_color(color):
    if color in positive_colors:
        return 'positive'
    elif color in negative_colors:
        return 'negative'
    elif color == 'neutral':
        return 'singleton'
    else:
        return 'dual'

color_df['FinalColor'] = color_df.apply(lambda row: pqm.resolve_color(row['PrimaryColor'], row['SecondaryColor'], colors_pos_list, colors_neg_list), axis=1)

color_df['Polarity'] = color_df['FinalColor'].apply(classify_color)


df_dv_summary = pd.pivot_table(data = color_df, index = ['DV', 'Node'], columns = 'Polarity', values = 'FinalColor', aggfunc = 'count')
#percentages of pos, neg, neutral or dual ,taking into account that Forestry CFL is an empty network (no rule devised leading to either increase or decrease)
df_dv_percentage = df_dv_summary / (len(ctx) - 1)

df_ctx_summary = pd.pivot_table(data = color_df, index = ['Context', 'Node'], columns = 'Polarity', values = 'FinalColor', aggfunc = 'count')
#percentages of pos, neg, neutral or dual 
df_ctx_percentage = df_ctx_summary / (len(dv))

