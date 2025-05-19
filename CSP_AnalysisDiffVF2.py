#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:12:10 2022

@author: jbaggio
"""

#packages to change directory and load datta from Meta_DataLoad_andPrep.py
import os
#import pickle to save files in case kernel gets stuck or too long to do
import pickle

#usual suspects
import pandas as pd
import numpy as np

#for figures
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

#for clustering
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


#import for boosted regression xgboost, explain features shap 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import root_mean_squared_error

import xgboost as xgb
import optuna
import shap 


#objective function for optuna for xgboost, regression with logistic link as our dep var is [0,1] bounded.
#Note we use 10 cross validation to balance over and underfitting given the size of the dataset -2031 obs.
def objective(n_trials):
    dtrain = xgb.DMatrix(data = trainfeat, label = trainlab)
    #dval = xgb.DMatrix(data = testfeat, label = testlab)
    
    params = {
        'tree_method' : n_trials.suggest_categorical('tree_method', ['exact']),
        'n_estimators' : n_trials.suggest_int('n_estimators', 10, 100, step = 10),
        'objective': n_trials.suggest_categorical('objective', ['reg:logistic']),
        'booster': n_trials.suggest_categorical('booster', ['gbtree']),
        'eta': n_trials.suggest_float('eta', 1e-4, 0.1, log=True),
        'lambda': n_trials.suggest_float('lambda', 1e-3, 100.0, log=True),
        'alpha': n_trials.suggest_float('alpha', 1e-3, 100.0, log=True),
        'random_state': rndstate  # Set seed for reproducibility
    }
    if params['booster'] == 'gbtree':
        params['subsample'] = n_trials.suggest_categorical('subsample', [0.25, 0.5, 0.75, 1.0])
        params['colsample_bytree'] = n_trials.suggest_float('colsample_bytree', 0.2, 1)
        params['min_child_weight'] = n_trials.suggest_float('min_child_weight', 1.0, 50.0)
        params['max_depth'] = n_trials.suggest_int('max_depth', 2, 6, step=1) 
        params['gamma'] = n_trials.suggest_float('gamma', 1e-2, 1.0, log=True)
        

    cv_results = xgb.cv(
        params,
        dtrain,
        nfold = 10,
        early_stopping_rounds=50,
        as_pandas=True, 
        metrics="rmse"
    )
    return cv_results['test-rmse-mean'].min()


#objective function for optuna for xgboost for different topics, slightly different as there are more overfitting issues,
# note lower cv also - 5 - as data per topic vary between approx 200 and approx 400 observations
def objective_topic(n_trials):
    dtrain = xgb.DMatrix(data = trainfeat, label = trainlab)
    #dval = xgb.DMatrix(data = testfeat, label = testlab)
        
    params = {
        'tree_method' : n_trials.suggest_categorical('tree_method', ['exact']),
        'n_estimators' : n_trials.suggest_int('n_estimators', 10, 1000, step = 50),
        'objective': n_trials.suggest_categorical('objective', ['reg:logistic']),
        'booster': n_trials.suggest_categorical('booster', ['gbtree']),
        'eta': n_trials.suggest_float('eta', 1e-4, 0.1, log=True),
        'lambda': n_trials.suggest_float('lambda', 1e-2, 200.0, log=True),
        'alpha': n_trials.suggest_float('alpha', 1e-2, 200.0, log=True),
        'random_state': rndstate  # Set seed for reproducibility
    }
    if params['booster'] == 'gbtree':
        params['subsample'] = n_trials.suggest_categorical('subsample', [0.25, 0.5, 0.75, 1.0])
        params['colsample_bytree'] = n_trials.suggest_float('colsample_bytree', 0.2, 1)
        params['min_child_weight'] = n_trials.suggest_float('min_child_weight', 1.0, 10.0)
        params['max_depth'] = n_trials.suggest_int('max_depth', 2, 6, step=1) 
        params['gamma'] = n_trials.suggest_float('gamma', 1e-2, 1.0, log=True)

  
    cv_results = xgb.cv(
        params,
        dtrain,
        nfold=5,
        early_stopping_rounds=50,
        as_pandas=True, 
        metrics="rmse"
    )
    return cv_results['test-rmse-mean'].min()


#relevant directories (files and output)
mainres = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis/MainResults'
#where to download results:
shapout = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis/Shapley'

os.chdir(mainres)
dfpreds = pickle.load(open('predicted_codes.p', 'rb'))

dftops = pd.read_csv('TopicsMain.csv')
dftops = dftops.drop(['Unnamed: 0'], axis = 1)

topvals = list(set(dftops['topic']))
count_topics = dftops['topic'].value_counts()
#eliminate string columns
dftopclus = dftops.drop (['txt_title','topic'], axis = 1)
#dftopclus = dftopclus.where(dftopclus> 0.2, other=0)

#cluster topics:
neigh = NearestNeighbors(n_neighbors=2  )
nbrs = neigh.fit(dftopclus)
distances, indices = nbrs.kneighbors(dftopclus)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances)
plt.title('K-distance Graph',fontsize=20)
plt.grid(color='k', linestyle='--', which='both')
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show() 
    
#from visual inspection check elbow point
ep = 0.125
clustering = DBSCAN(eps=ep, min_samples=32).fit(dftopclus)
dftops['clusters'] = clustering.labels_
dftops['clusters'].value_counts()

SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_clusters = cluster, init='k-means++', n_init = 'auto', random_state=76)
    kmeans.fit(dftopclus)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('ClusterInertia.pdf', bbox_inches = 'tight')

#from the graph check elbow point
kmeans = KMeans(n_clusters = 7, init='k-means++', n_init = 'auto', random_state = 76)
kmeans.fit(dftopclus)
dftops['kmeans'] = kmeans.predict(dftopclus)
dftops['kmeans'].value_counts()

#we now combine kmeans and DBSCAN clusters with the main topic to finalize clustering. We do this visually
dftops['maintop'] = dftops['kmeans']
dftops['maintop'] = dftops['maintop'].mask((dftops['clusters']== 7), 7)


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

            # Calculate the difference and set to NaN if mis > 0.5
            dfdiff[base_col] = np.where(mis_col > 0.66, 0, abs_col - pres_col)
            print(base_col + ' Absence is good')
        else:
            pres_col = dfpreds_all[base_col +'_pres']
            abs_col = dfpreds_all[base_col + '_abs']
            mis_col = dfpreds_all[base_col + '_mis']

            # Calculate the difference and set to NaN if mis > 0.5
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


'''
Use XGBoost with Shapley values and take advantage of the shapley values additive property for group influence on the outcome
Use tree_path_dependencies for shapley values given potential non-independency between features 
as showcased by spearman, pearson and mutual info

'''

os.chdir(shapout)

#fix random seed for reproducibilty and devise the number of models to increase shapley reliability
rndstate = 8081
np.random.seed(rndstate)
nmodels = 35 #number of models to run on best_params on different train-test split to assess SHAP variance given 2031 observations

#rescale variables in the 0,1 space, in this case 0 = 100% probability of being absent and 1 = 100% probability of being present, 0.5 is the 
#"threshold" and implies that a value has 0 probability of being present or absent. My values are centered around zero, so the following works:
depvars_scaled = (depvars + 1) / 2
indepvars_scaled = (indepvars + 1) / 2

#dictionaries to store results:

xgboptuna = {}
xgbmodels = {}
shapexpldict = {}
shapdict = {}
shapdict_std = {}
shap_effects = {}
shapinterdict = {}
shapinterdict_std = {}
shapexpldict = {}
diagnostic = {}

for dep in depvars_scaled:
    print(dep)
    depvar_single = depvars_scaled[dep]
    trainfeat, testfeat, trainlab, testlab = train_test_split(indepvars_scaled, depvar_single, test_size=0.25, random_state = np.random.randint(5000))

    #Optuna trainig 
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=500)
    xgboptuna[dep] = study
    best = study.best_trial
    parbest = best.params
    
    #Now train multiple models on the best hyperparameters
    #list we need for the models
    models = []
    rmse_train, rmse_test = [], []
    cvscores_mean, cvscores_std = [], []
    shap_values_list, base_values_list = [], []
    shap_interact_list = []
    
    xgbmodels[dep] = {}
    for i in range (nmodels):
        print('Starting Model for Aggregate Results')
        print(i)
        #train and test split variations for better generalizability
        trainfeat, testfeat, trainlab, testlab = train_test_split(indepvars_scaled, depvar_single, test_size=0.25, random_state = np.random.randint(5000))
        
        model = xgb.XGBRegressor(**parbest)
        model.fit(trainfeat, trainlab)
        models.append(model)
        xgbmodels[dep][i] = model
        # Performmance evaluation
        pred_test = model.predict(testfeat)
        pred_train = model.predict(trainfeat)

        cv_scores = cross_val_score(model, trainfeat, trainlab, cv=10, scoring='neg_root_mean_squared_error')
        cv_scores = -cv_scores  # Convert to positive RMSE values
            
        rtrain = root_mean_squared_error(trainlab, pred_train)
        rtest = root_mean_squared_error(testlab, pred_test)
        
        rmse_train.append(rtrain)
        rmse_test.append(rtest)
        cvscores_mean.append(np.mean(cv_scores))
        cvscores_std.append(np.std(cv_scores))
    
        #extract shapley values
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        explanation = explainer(indepvars_scaled)

        shap_values = explainer.shap_values(indepvars_scaled)  # Shape: (num_samples, num_features)
        shap_values_list.append(shap_values)
        base_values_list.append(explanation.base_values)  # Base values
        
        # Extract SHAP interaction values
        shap_interact_list.append(explainer.shap_interaction_values(indepvars_scaled))


    #Compute aggregated diagnostic
    diagnostic[dep] = {
        'Train_RMSE_Mean' : np.mean(rmse_train), 'Test_RMSE_Mean' : np.mean(rmse_test), 
        'Train_RMSE_StDev' : np.std(rmse_train), 'Test_RMSE_StDev' : np.std(rmse_test),
        'RMSE_CV_Mean' : np.mean(cvscores_mean), 'RSMSE_CV_StDev' : np.mean(cvscores_std),
        }
    
    # Compute aggregated shap values
    shap_mean_values = np.mean(np.array(shap_values_list), axis=0)
    shap_std_values = np.std(np.array(shap_values_list), axis=0)
    base_mean_values = np.mean(np.array(base_values_list), axis=0)

    #Check the sum of features that we will then use later on on the mean shapley values
    shap_values_array = np.array(shap_values_list)  # Shape: (N_models, N_samples, N_features)
    shap_sum_values = np.sum(shap_values_array, axis=1)  # Shape: (N_models, N_features)


    #Create an Aggregated SHAP Explainer
    aggregated_explainer = shap.Explanation(
        values = shap_mean_values,
        base_values = base_mean_values,
        data = indepvars_scaled.to_numpy(),
        feature_names = list(indepvars_scaled.columns)
    )

    # Store the aggregated explainer
    shapexpldict[dep] = aggregated_explainer
   
    #store values and standard deviation of shapley values for easy use
    shapdict[dep] = shap_mean_values
    shapdict_std[dep] = shap_std_values
    shap_effects[dep] = shap_sum_values
    
    #store interaction shapley values in case 
    shap_interact_array = np.array(shap_interact_list)  # Shape: (n_models, n_samples, n_features, n_features)
    shapinterdict[dep] = np.mean(shap_interact_array, axis=0)
    shapinterdict_std[dep] = np.std(shap_interact_array, axis=0)

#save cross validation and rmse test train
dfdiagnostic = pd.DataFrame.from_dict(diagnostic, orient='index')
dfdiagnostic.to_csv('Diagnostic_All.csv')


#check general effect of variables per outcome using mean, median and standard deviation of sum of shap for the 10 models
    
#standard dev of the sum of the shap for the 10 models
shap_std = {}
shap_median = {}
shap_mean = {}
for key in shap_effects:
    sh_vals = np.array(shap_effects[key])
    shap_std[key] = np.std(sh_vals, axis=0)
    shap_median[key] = np.median(sh_vals, axis=0)
    shap_mean[key] = np.mean(sh_vals, axis=0)

dfshap_std = pd.DataFrame.from_dict(shap_std).T
dfshap_std.columns = indepvars.columns
dfshap_std.to_csv('Shap_All_StDev.csv')

dfshap_median = pd.DataFrame.from_dict(shap_median).T
dfshap_median.columns = indepvars.columns
dfshap_median.to_csv('Shap_All_Median.csv')

dfshap_mean = pd.DataFrame.from_dict(shap_mean).T
dfshap_mean.columns = indepvars.columns
dfshap_mean.to_csv('Shap_All_Mean.csv') 

os.chdir(shapout + '/All_Together')

# Plot each SHAP distributions
long_df = []

for key in shap_effects:  # Loop through each dependent variable (DV)
    sh_vals = np.array(shap_effects[key])  
    df = pd.DataFrame(sh_vals, columns=indepvars.columns)  # Convert to DataFrame
    df['DV'] = key  # Add DV name as a column
    long_df.append(df)

# Concatenate all DFs
long_df = pd.concat(long_df, ignore_index=True)

# Melt DataFrame to long format
melted_df = long_df.melt(id_vars=['DV'], var_name='Feature', value_name='SHAP Value')

# Plot boxplot with hue for DVs
plt.figure(figsize=(14, 7))
sns.boxplot(data=melted_df, x='Feature', y='SHAP Value', hue='DV', showmeans=True, 
            meanprops={"marker":"+", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":"5"})

plt.xticks(rotation=45)
plt.ylabel('SHAP Value', fontsize = 24)
plt.xlabel('Feature', fontsize = 24)
plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20) 
plt.grid(axis = 'y', linestyle='--', alpha=0.7)
plt.legend(title = 'Outcome',  title_fontsize=22, fontsize = 18, ncol=3)  # Legend for DVs
plt.savefig('Boxplot_All_DVs.pdf')
plt.show()

for key in shap_effects:
    print(key)
    sh_vals = np.array(shap_effects[key])
    df = pd.DataFrame.from_dict(sh_vals)
    df.columns = indepvars.columns
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, showmeans=True, meanprops={"marker":"+", "markerfacecolor":"black", 
                            "markeredgecolor":"black", "markersize":"5"})
    plt.xticks(rotation=45)
    plt.title(f"SHAP Value Distribution for {key}")
    plt.ylabel("SHAP Value")
    plt.xlabel("Feature")
    plt.grid(axis ='y', linestyle='--', alpha=0.7)
    plt.savefig('Boxplot_' + key + '.pdf')
    plt.show()


# Create a custom color map based on RdYlBu_r with white at the center
colors = [(0.0, "blue"), (0.45, "lightblue"), (0.499, "white"), (0.501, "white"), (0.55, "coral"), (1.0, "red")]
cmapshap = LinearSegmentedColormap.from_list("custom_RdYlBu", colors)

#sort dfshap to follow same order than for Topics
dfshap_effects = dfshap_mean.copy()
dfshap_effects = dfshap_effects.sort_index(ascending=False)

#Create figure and heatmap
fig, ax = plt.subplots(figsize=(10, 8))
# Base heatmap (magnitude only)
sns.heatmap(dfshap_effects, cmap=cmapshap, annot=True, fmt=".2f", cbar=True, ax=ax, linewidths=0.5, center = 0, vmin = -10, vmax = 10)
ytick_labels = dfshap_effects.index.get_level_values(0)
xtick_labels = dfshap_effects.columns
ax.set_yticklabels(ytick_labels, fontsize=18)
ax.set_xticklabels(xtick_labels, fontsize=18, rotation = 90)
cbar = ax.collections[0].colorbar

fig.savefig('AShapEffects_All.pdf')
plt.show()

#clustermap 
plt.figure()
g = sns.clustermap(dfshap_effects, cmap=cmapshap, annot=True, fmt=".2f", cbar=True, row_cluster = False, linewidths=0.5, center = 0, vmin = -10, vmax = 10, metric='cosine')
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=18, rotation=90)  # X-axis labels
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=18)  # Y-axis labels
plt.savefig('AShapEffects_All_Cluster.pdf', bbox_inches = 'tight')
plt.close()

# Figure for sum of effects (additive shapley to leverage)
n_rows = len(dfshap_effects.index)
fig, axes = plt.subplots(1, 3, figsize=(30, 10))
for i, row in enumerate(dfshap_effects.index):
    # Get the row data (shap values for a given outcome)
    row_data = dfshap_effects.loc[row]
    sorted_row_data = row_data.sort_values(ascending=False)
    sns.barplot(x=sorted_row_data.index, y=sorted_row_data.values, color='skyblue', ax=axes[i])
    axes[i].set_title(f"SHAP Values for {row} (Sorted by Net Positive Effect)")
    axes[i].set_xlabel('Features')
    axes[i].set_ylabel('Shapley Value')
    axes[i].tick_params(axis='x', rotation=45)  # Rotate the feature names for better readability
plt.tight_layout()
fig.savefig('Pos_Neg_OverallEffects.pdf')

#Check original shap values as violinplots and scatter plots
for dep in shapexpldict:
    print(dep)
    #explain the dataset with shapley values
    shap_values = shapexpldict[dep]
    #scatter plot of feature/outcome vor different shap values and actual raw value of the feature
    scatter = shap.plots.scatter(shap_values, ylabel="SHAP Value", show=False, title = dep)
    plt.savefig('Scatter_' + dep + '.pdf')
    plt.close()

for key in shapdict:
    shap_values = shapexpldict[key]
    plt.figure()
    shap.plots.violin(shap_values, max_display = 11, show=False)
    plt.title(key)
    plt.savefig('Violin_' + key + '.pdf', bbox_inches='tight')
    
    
for key in shapdict:
    shap_values = shapexpldict[key]
    fig, ax = plt.subplots(figsize=(10, 8))
    clust = shap.utils.hclust(indepvars_scaled, depvars_scaled[dep], linkage="single")
    shap.plots.bar(shap_values, clustering=clust, clustering_cutoff=1, ax=ax)
    fig.suptitle(key)
    fig.savefig('BarClust' + key + '.pdf', bbox_inches='tight')

    
'''
Use XGBOost and Shapley values for each main topic
Considering dependencies in the independend variables (checked with pearson, spearman and mutual info), 
it is best to calculate shapley with the tree_path_dependent instead of intervention. 
'''
os.chdir(shapout)

nmodels = 15 #change the number of models run for assessing feature importances given the size of the dataset (between 200 and 400 observations)

indepvars_topic = indepvars_scaled.copy()
indepvars_topic['topname'] = dfall['topname']
depvars_topic = depvars_scaled.copy()
depvars_topic['topname'] = dfall['topname']

gr_indep = indepvars_topic.groupby(indepvars_topic['topname'])
gr_dep = depvars_topic.groupby(depvars_topic['topname'])


xgboptuna_topic = {}
xgbmodel_topic = {}
shapexpldict_topic = {}
shap_effects_topic = {}
shapdict_topic = {}
shapdict_topic_std = {}
shapinterdict_topic = {}
shapinteract_topic_std = {}
diagnostic_topic = {}

# Iterate over each group (topic)
for topic, indep_topic in gr_indep:  
    dep_topic = gr_dep.get_group(topic)
    
    for dep in dep_topic.columns.drop('topname'):  # Drop 'topname' column to keep only dependent variables
        print(topic + ' ' + dep)
        depvar_single = dep_topic[dep]
        
        #create initial train test for otuna
        trainfeat, testfeat, trainlab, testlab = train_test_split(indep_topic.drop(columns=['topname']), dep_topic[dep], test_size=0.25, random_state = np.random.randint(5000))

        # Optuna optimization for hyperparameters
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_topic, n_trials=500)
        xgboptuna_topic[(topic, dep)] = study
        parbest = study.best_trial.params
        
        # Now train multiple models on the best hyperparameters
        models = []
        rmse_train, rmse_test = [], []
        cvscores_mean, cvscores_std = [], []
        shap_values_list, base_values_list = [], []
        shap_interact_list = []
        
        xgbmodel_topic[(topic, dep)] = {}
        
        for i in range(nmodels):  # Train multiple models
            print('Starting model run: ' + str(i))
            
            # Train and test split variations for better generalizability
            trainfeat, testfeat, trainlab, testlab = train_test_split(indep_topic.drop(columns=['topname']), dep_topic[dep], test_size=0.25, random_state=np.random.randint(5000))
            
            model = xgb.XGBRegressor(**parbest)
            model.fit(trainfeat, trainlab)
            models.append(model)
            xgbmodel_topic[(topic, dep)][i] = model
            
            # Performance evaluation
            pred_test = model.predict(testfeat)
            pred_train = model.predict(trainfeat)
            
            cv_scores = cross_val_score(model, trainfeat, trainlab, cv=10, scoring='neg_root_mean_squared_error')
            cv_scores = -cv_scores  # Convert to positive RMSE values
            
            rtrain = root_mean_squared_error(trainlab, pred_train)
            rtest = root_mean_squared_error(testlab, pred_test)
            
            rmse_train.append(rtrain)
            rmse_test.append(rtest)
            cvscores_mean.append(np.mean(cv_scores))
            cvscores_std.append(np.std(cv_scores))
        
            # Extract SHAP values
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            explanation = explainer(trainfeat)  # Explanation on the training data
            
            shap_values = explainer.shap_values(trainfeat)
            shap_values_list.append(shap_values)
            base_values_list.append(explanation.base_values)
            
            # Extract SHAP interaction values
            shap_interact_list.append(explainer.shap_interaction_values(trainfeat))
        
        # Compute aggregated diagnostic metrics
        diagnostic_topic[(topic, dep)] = {
            'Train_RMSE_Mean': np.mean(rmse_train),
            'Test_RMSE_Mean': np.mean(rmse_test),
            'Train_RMSE_StDev': np.std(rmse_train),
            'Test_RMSE_StDev': np.std(rmse_test),
            'RMSE_CV_Mean': np.mean(cvscores_mean),
            'RMSE_CV_StDev': np.mean(cvscores_std),
        }
        
        # Compute aggregated SHAP values
        shap_mean_values = np.mean(np.array(shap_values_list), axis=0)
        shap_std_values = np.std(np.array(shap_values_list), axis=0)
        base_mean_values = np.mean(np.array(base_values_list), axis=0)
        
        # Sum SHAP values for further use
        shap_values_array = np.array(shap_values_list)  # Shape: (N_models, N_samples, N_features)
        shap_sum_values = np.sum(shap_values_array, axis=1)  # Shape: (N_models, N_features)
        
        # Create an aggregated SHAP explainer
        aggregated_explainer = shap.Explanation(
            values=shap_mean_values,
            base_values=base_mean_values,
            data=trainfeat.to_numpy(),
            feature_names=list(trainfeat.columns)
        )
        
        # Store the aggregated SHAP explainer
        shapexpldict_topic[(topic, dep)] = aggregated_explainer
        
        # Store SHAP values, standard deviations, and effects
        shapdict_topic[(topic, dep)] = shap_mean_values
        shapdict_topic_std[(topic, dep)] = shap_std_values
        shap_effects_topic[(topic, dep)] = shap_sum_values
        
        # Store SHAP interaction values
        shap_interact_array = np.array(shap_interact_list)  # Shape: (n_models, n_samples, n_features, n_features)
        shapinterdict_topic[(topic, dep)] = np.mean(shap_interact_array, axis=0)
        shapinteract_topic_std[(topic, dep)] = np.std(shap_interact_array, axis=0)


#save cross validation and rmse test train
dfdiagnostic_topic = pd.DataFrame.from_dict(diagnostic_topic, orient='index')
dfdiagnostic_topic.to_csv('Diagnostic_Topic.csv')

#standard dev of the sum of the shap for the 10 models
shap_std_topic = {}
shap_median_topic = {}
shap_mean_topic = {}
for key in shap_effects_topic:
    sh_vals = np.array(shap_effects_topic[key])
    shap_std_topic[key] = np.std(sh_vals, axis=0)
    shap_median_topic[key] = np.median(sh_vals, axis=0)
    shap_mean_topic[key] = np.mean(sh_vals, axis=0)

dfshap_std_topic = pd.DataFrame.from_dict(shap_std_topic).T
dfshap_std_topic.columns = indepvars.columns
dfshap_std_topic.to_csv('Shap_Topic_StDev.csv')

dfshap_median_topic = pd.DataFrame.from_dict(shap_median_topic).T
dfshap_median_topic.columns = indepvars.columns
dfshap_median_topic.to_csv('Shap_Topic_Median.csv')

dfshap_mean_topic = pd.DataFrame.from_dict(shap_mean_topic).T
dfshap_mean_topic.columns = indepvars.columns
dfshap_mean_topic.to_csv('Shap_Topic_Mean.csv') 

os.chdir(shapout + '/Per_Topic')

#Plot shap distribution per topic
long_df = []

for (topic), sh_vals in shap_effects_topic.items():  
    sh_vals = np.array(sh_vals)  # Convert SHAP values to array
    df = pd.DataFrame(sh_vals, columns=indepvars.columns)  # Convert to DataFrame
    df['DV'] = topic[1]  # Store the dependent variable
    df['Topic'] = topic[0]  # Store the topic
    long_df.append(df)

# Concatenate all DFs
long_df = pd.concat(long_df, ignore_index=True)

# Melt DataFrame to long format
melted_df = long_df.melt(id_vars=['DV', 'Topic'], var_name='Feature', value_name='SHAP Value')

# Get unique topics
topics = melted_df['Topic'].unique()

# Loop over each topic and create a separate plot
for topic in topics:
    plt.figure(figsize=(14, 7))

    # Filter data for the current topic
    topic_df = melted_df[melted_df['Topic'] == topic]

    # Boxplot with hue for DVs
    sns.boxplot(data=topic_df, x='Feature', y='SHAP Value', hue='DV', showmeans=True, 
                meanprops={"marker":"+", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":"5"})

    plt.xticks(rotation=45, fontsize=20)  # Increase x-tick label size
    plt.yticks(fontsize=20)  # Increase y-tick label size
    plt.title(f'{topic}', fontsize=26)  # Dynamic title per topic
    plt.ylabel('SHAP Value', fontsize=24)
    plt.xlabel('Feature', fontsize=24)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Outcome', title_fontsize=22, fontsize=18, ncol=3)

    plt.savefig(f'Boxplot_{topic}.pdf')
    plt.show()

# Plot each SHAP distribution per key
for key in shap_effects_topic:
    print(key)
    sh_vals = np.array(shap_effects_topic[key])
    df = pd.DataFrame.from_dict(sh_vals)
    df.columns = indepvars.columns
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, showmeans=True, meanprops={"marker":"+", "markerfacecolor":"black", 
                            "markeredgecolor":"black", "markersize":"5"})
    plt.xticks(rotation=45, fontsize = 20)
    plt.title(f"{key}")
    plt.ylabel("SHAP Value")
    plt.xlabel("Feature")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('BoxplotTopic_' + str(key) + '.pdf')
    plt.show()


#distribution are not symmetrical, best to use median
#sort dfshap to follow same order than for Topics
dfshap_effects_topic = dfshap_mean_topic.copy()
dfshap_effects_topic = dfshap_effects_topic.sort_index(ascending=False)


#change order so that all resource, all conflict and all ieq are together and then per topic
dfshap_effects_topic.index = pd.MultiIndex.from_tuples(dfshap_effects_topic.index, names=['Topic', 'Category'])
dfshap_effects_topic = dfshap_effects_topic.sort_index(level=1, ascending=False)

#Create figure and heatmap
fig, ax = plt.subplots(figsize=(10, 8))
# Base heatmap (magnitude only)
sns.heatmap(dfshap_effects_topic, cmap=cmapshap, annot=True, fmt=".2f", cbar=True, ax=ax, linewidths=0.5, center = 0, vmin = -2, vmax = 2)
# Add color bar with distinct labels
cbar = ax.collections[0].colorbar
cbar.set_ticks([-2, -1.5, -1, -0.5,  0, 0.5,  1, 1.5, 2])
cbar.set_ticklabels(['<-2', '-1.5','-1', '-0.5', '0', '0.5', '1', '1.5', '>2'])

# Add horizontal lines to divide by category
categories = dfshap_effects_topic.index.get_level_values(1).unique()
for category in categories:
    idx = dfshap_effects_topic.index.get_level_values(1).tolist().index(category)
    ax.axhline(idx, color='black', linewidth=5)

# Customize x-tick labels to show only the first level of the multi-index
ytick_labels = dfshap_effects_topic.index.get_level_values(0)
xtick_labels = dfshap_effects_topic.columns
ax.set_yticklabels(ytick_labels, fontsize=18)
ax.set_xticklabels(xtick_labels, fontsize=18, rotation = 90)

# Add centered labels for the second level of the multi-index
second_level_labels = dfshap_effects_topic.index.get_level_values(1).unique()
for label in second_level_labels:
    idx = dfshap_effects_topic.index.get_level_values(1).tolist().index(label)
    ax.text(-3.5, idx + len(dfshap_effects_topic.index) / (2 * len(second_level_labels)), label,  fontsize = 24, ha='center', va='center', rotation=90)

ax.set_ylabel('')
# Title and show plot
fig.savefig('AShapEffects_Topic.pdf', bbox_inches = 'tight')
plt.show()


# Define color mappings for topics and categories
group_colors = {
    'Biodiversity': 'lime', 'Development': 'black', 'Fishery': 'cyan',
    'Forestry': 'yellowgreen', 'Land Rights': 'orange', 'Livestock': 'sienna', 'Water': 'blue'
}
dep_colors = {'RSC': 'darkgreen', 'CFL': 'gray', 'IEQ': 'magenta'}

# Create lists of colors for rows and columns
dv_colors    = dfshap_effects_topic.index.get_level_values(1).map(dep_colors)
topic_colors = dfshap_effects_topic.index.get_level_values(0).map(group_colors)

#clustermap
plt.figure()
g = sns.clustermap(dfshap_effects_topic, cmap=cmapshap, annot=True, fmt=".2f", col_cluster = False, row_cluster = True, linewidths=0.5, center = 0, vmin = -2, vmax = 2, row_colors=[dv_colors, topic_colors], metric='cosine')
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=18, rotation=90)  # X-axis labels
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=18)  # Y-axis labels
plt.savefig('AShapEffects_Topic_Cluster.pdf', bbox_inches = 'tight')
plt.close()


gr_shap_eff_top = dfshap_effects_topic.groupby(['Topic'])

for group, df in gr_shap_eff_top:
    df = df.droplevel('Topic')
    constant_columns = df.columns[df.nunique() == 1]
    print(group)
    print(constant_columns)
    df = df.drop(constant_columns, axis=1)
    g = sns.clustermap(df, cmap=cmapshap, annot=True, fmt=".2f", row_cluster = False,  linewidths=0.5, center = 0, vmin = -2, vmax = 2, metric='cosine')
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=20, rotation=90)  # X-axis labels
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=20)  # Y-axis labels

    g.ax_heatmap.set_ylabel(group[0], fontsize=24, labelpad=5)
    g.ax_heatmap.yaxis.set_label_position("left")  # Move label to the left
    plt.savefig(f"Cluster_{group}_Cos.pdf", bbox_inches="tight")
    plt.close()
    

#Pos and Neg overall feature effects
fig, axes = plt.subplots(7, 3, figsize=(30, 50), sharex=True)
axes = axes.reshape(7, 3)
# Ensure consistent order of features (columns)
feature_order = dfshap_effects_topic.columns.tolist()

# Iterate over groups and dependent variables
for row_idx, (group, sub_df) in enumerate(dfshap_effects_topic.groupby(level=0)):  # Iterate over groups
    for col_idx, (dv, row_data) in enumerate(sub_df.iterrows()):  # Iterate over dependent variables
        ax = axes[row_idx, col_idx]

        # Ensure consistent column order
        row_data = row_data[feature_order]

        # Barplot
        sns.barplot(x=row_data.index, y=row_data.values, color='skyblue', ax=ax)
        ax.set_title(f"{group} - {dv}")
        ax.set_xlabel('')
        ax.set_ylabel('SHAP Value')
        ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
fig.savefig('Pos_Neg_OverallEffects_Topics.pdf')
plt.show()


#Shapley scatter and violin plots
for dep in shapexpldict_topic:
    #explain the dataset with shapley values
    shap_values = shapexpldict_topic[dep]
    #scatter plot of feature/outcome vor different shap values and actual raw value of the feature
    scatter = shap.plots.scatter(shap_values, ylabel="SHAP Value", cmap='RdBu', show=False)
    plt.title(dep , color = "black")
    plt.savefig('ScatterTopic_' + str(dep) + '.pdf')
    plt.close()

#FViolin Plots    
for key in shapdict_topic:
    shap_temp = shapdict_topic[key]
    temp_expl = shapexpldict_topic[key]
    plt.figure()
    shap.plots.violin(temp_expl, max_display = 11, show=False)
    plt.title(key)
    plt.savefig('ViolinTopic_' + str(key) +'.pdf', bbox_inches='tight')
    plt.close()
    

for key in shapdict_topic:
    print(key[0])
    temp_expl = shapexpldict_topic[key]
    temp_indep = gr_indep.get_group(key[0])
    temp_indep = temp_indep.drop('topname', axis = 1)
    temp_dep = gr_dep.get_group(key[0])
    dep_sing = temp_dep[key[1]]
    fig, ax = plt.subplots(figsize=(10, 8))
    clust = shap.utils.hclust(temp_indep, dep_sing, linkage="single")
    shap.plots.bar(temp_expl, clustering=clust, clustering_cutoff=1,max_display=11)
    fig.suptitle(key)
    fig.savefig('BarClust_' + str(key[0]) + '_' + str(key[1]) + '.pdf', bbox_inches='tight')



'''

Mutual Information pairwise, feat - feeat and freat - out
'''
os.chdir(shapout + '/ZZZMutual_Info')
 
# Initialize an empty dictionary to store the mutual information scores
infodict = {}

for dep in depvars.columns:
    # Compute mutual information scores
    mi_scores = mutual_info_regression(X=indepvars, y=depvars[dep])
    
    # Store results in a dictionary with feature names as keys
    infodict[dep] = dict(zip(indepvars.columns, mi_scores))
dfinfo = pd.DataFrame.from_dict(infodict)

mi_scores_indep = pd.DataFrame(index=indepvars.columns, columns=indepvars.columns)
for v1 in indepvars.columns:
    for v2 in indepvars.columns:
        if v1 != v2:
            mi_scores_indep.loc[v1, v2] = mutual_info_regression(indepvars[[v1]], indepvars[v2])[0]
        else:
            mi_scores_indep.loc[v1,v2] = 0

finfo = sns.heatmap(dfinfo, cmap = 'coolwarm')
figure = finfo.get_figure()    
figure.savefig('InfoDeps_heatmap.pdf', bbox_inches='tight')

mi_scores_indep = mi_scores_indep.astype(float)
finfo2 = sns.heatmap(mi_scores_indep, cmap = 'coolwarm')
figure2 = finfo2.get_figure()    
figure2.savefig('InfoIndeps_heatmap.pdf', bbox_inches='tight')


'''
Mutual information for design principles per Topic
'''

for group, dfind in gr_indep:
    dfdep = gr_dep.get_group(group)
    for dep in dfdep.drop(columns=['topname']).columns:
        print(dep)
        # Compute mutual information scores
        mi_scores = mutual_info_regression(X=dfind.drop(columns=['topname']), y=dfdep[dep])
        
        # Store results in a dictionary with feature names as keys
        infodict[dep] = dict(zip(dfind.drop(columns=['topname']).columns, mi_scores))
    dfinfo = pd.DataFrame.from_dict(infodict)
    plt.figure()
    sns.heatmap(dfinfo, cmap = 'coolwarm')
    plt.title('MI for ' + str(group))
    plt.savefig('Info_Dep_' + str(group) + '.pdf')


dfmi_indep = {}
for group, dfind in gr_indep:
    mi_scores_indep = pd.DataFrame(index=indepvars.columns, columns=indepvars.columns)
    for v1 in dfind.drop(columns = ['topname']).columns:
        for v2 in dfind.drop(columns=['topname']).columns:
            if v1 != v2:
                mi_scores_indep.loc[v1, v2] = mutual_info_regression(dfind[[v1]], dfind[v2])[0]
            else:
                mi_scores_indep.loc[v1,v2] = 0
    
    mi_scores_indep = mi_scores_indep.astype(float)
    dfmi_indep[group] = mi_scores_indep
    plt.figure()
    sns.heatmap(mi_scores_indep, cmap = 'coolwarm')
    plt.title('MI for ' + str(group))
    plt.savefig('Info_InDep_' + str(group) + '.pdf')

        