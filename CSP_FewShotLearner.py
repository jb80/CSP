#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:42:45 2022

@author: jbaggio

Build an embedding model to work with a few shot learner to classify papers
#make sure you are using the csp environment

"""

#packages to change directory and load datta from Meta_DataLoad_andPrep.py
import os
#import pickle to save files in case kernel gets stuck or too long to do
import pickle

#usual suspects
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#for few shot learner
import pyarrow as pa
from datasets import Dataset
from sentence_transformers import losses
from setfit import SetFitModel, SetFitTrainer

#for text summarization, given the lenght of abstract/results/discussion and conclusion sections
#we have issues with embeddings. Summarization allows to convey key information. We use a model pretrained on scientific
#articles (albeit trained on pubmed)
import torch
if torch.backends.mps.is_available():
    if torch.backends.mps.is_built():
        device = 'mps'
    else:
        device = 'cpu'
else:
    device = 'cpu'
torch.device(device)


#to calculate weights for unbalanced classes and cross entropy loss
from sklearn.utils.class_weight import compute_class_weight
#check confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef


    

"""
Workflow:
1)  get documents and the training set, 
2) train embeddings on text.
3) train Few-Shot encoder on the manually labelled dataset per variable via set fit
4) save the trained models in dictionaries and pickle dump them
    
"""
#first set the folders where files are and where dataset should be stored and figures and output saved to
traintext = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/DataCoded/Articles/TextForTraining'
cleandata = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Data'
mainres = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis/MainResults'
modelfiles = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis/Models'
figout = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis/Figures'

#first load the extracted text
os.chdir(cleandata)
dfcases = pickle.load(open ('extracted_df.p', 'rb'))
dfcases_train = pickle.load(open('extracted_df_train.p', 'rb'))
dfcases_train = dfcases_train.dropna(subset = 'text') #remove nan text



#load training/test data labels
os.chdir(traintext)

trainset = pd.read_excel('matching.xlsx', sheet_name ='trainset')
allpapers = pd.read_excel('matching.xlsx', sheet_name='result')

trainset = pd.merge(left=dfcases_train, right= trainset, on='txt_title' )

#now that we have the training dataframe, check how many paper per clas we have
#start by checking frequency of labelled cases 
colnames = ['CaseStudy','rescond', 'reschange', 
       'conflict', 'inequality', 'socbound',
        'resbound', 'fit', 'prop',
       'participation',  'monitor',
       'accmonitor',  'gradsanc', 'confres',  'autonomy', 
       'nested']

os.chdir(cleandata)
tabulate = {}     
for col in colnames:
    trainclass = trainset.copy()
    trainclass['IsCase'] = trainclass['CaseStudy']
    if col != 'CaseStudy':
        trainclass = trainclass[trainclass['IsCase'] == 1]
    trainclass = trainclass.drop(columns = 'IsCase') 
    trainclass = trainclass.dropna(subset = 'text')
    trainclass = trainclass.replace(99,2)
    trainclass = trainclass.fillna(2)
    tabulate[col] = trainclass[col].value_counts()
tabulate = pd.DataFrame.from_dict(tabulate)
os.chdir(figout)
tabulate.to_csv('TrainSet_Values.csv')

"""
#now train a few shot learner via setfit (huggingface, using sentence based bert for embeddings)
#first load the embedding model using Allenai longformer, the model will need to be trained, and we do this for each column separately: 

We can use different pre-trained model, the choice is based on whether pre-training was done to assess document similarity/classification or sentence similarity.
Further, different models allow for different length, given the length of the documents it is best to use a model that allows for more than 512 tokens.


Allenai model scico is based on (limited to 4096 tokens - document relation (coreference)):
    
    @inproceedings{cattan2021scico,
    title={SciCo: Hierarchical Cross-Document Coreference for Scientific Concepts},
    author={Arie Cattan and Sophie Johnson and Daniel S Weld and Ido Dagan and Iz Beltagy and Doug Downey and Tom Hope},
    booktitle={3rd Conference on Automated Knowledge Base Construction},
    year={2021},
    url={https://openreview.net/forum?id=OFLbgUP04nC}
    }
    

Both allenai and kiddo are based on longformer:
@article{Beltagy2020Longformer,
  title={Longformer: The Long-Document Transformer},
  author={Iz Beltagy and Matthew E. Peters and Arman Cohan},
  journal={arXiv:2004.05150},
  year={2020},
}
  
Few shot learner is based on:
    
  doi = {10.48550/ARXIV.2209.11055},
  url = {https://arxiv.org/abs/2209.11055},
  author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Efficient Few-Shot Learning Without Prompts},
  publisher = {arXiv},
  year = {2022}, 
  
For hyperparameter optimization we used optuna
@inproceedings{optuna_2019,
    title={Optuna: A Next-generation Hyperparameter Optimization Framework},
    author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
    booktitle={Proceedings of the 25th {ACM} {SIGKDD} International Conference on Knowledge Discovery and Data Mining},
    year={2019}
}
  
 """
 
#initialize functions for the model
def model_init(params):
        params = params or {}
        max_iter = params.get("max_iter", 1000)    # this and the next parameter are to use if using sklearn wrapper
        solver = params.get("solver", "liblinear")
        params = {
            "head_params": {
            #"out_features": nclasses
            "max_iter": max_iter,
            "solver": solver,
            }
        }
        return SetFitModel.from_pretrained("allenai/longformer-scico",
                                           #use_differentiable_head = True,
                                           device = 'mps',
                                           **params)
    
#Use cosine similarity loss.
def hp_space(trial):  # Training parameters
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 2e-4, log=True),
            "num_epochs": trial.suggest_categorical("num_epochs", [1]),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
            "num_iterations": trial.suggest_int("num_iterations", 10, 50, step = 5),
            "seed": trial.suggest_categorical("seed", [1221, 13123]),
            "l2_weight": trial.suggest_categorical("l2_weight", [0.00, 0.01, 0.1]),
            "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"])
            
        }

#store results, trainer trained in fsdict, metric test score and evaluation data (test data)
trainerdict = {}
modeldict = {}
testmetric = {}
test_datadict = {}
optrun = {}

#add column to identify case studies but with a different name to avoid issues with renaming it as label
os.chdir(modelfiles)
trainset['IsCase'] = trainset['CaseStudy']



colnames = ['CaseStudy','rescond', 'reschange', 
         'conflict', 'inequality', 'socbound',
        'resbound', 'fit', 'prop',
       'participation',  'monitor','accmonitor',  
       'gradsanc', 'confres',  'autonomy', 'nested']

for col in colnames:
    #name file to save the model class
    filename = col + '_pretrained'
    #rename dfdata as the dataframe for a specific variable and rename the label column label
    dftrain = trainset[['text', 'IsCase', col]]
    #only assess if Case Study == 1
    if col != 'CaseStudy':
        dftrain = dftrain[dftrain['IsCase'] == 1]
    dftrain = dftrain.drop(columns = 'IsCase')       
    dftrain = dftrain.dropna(subset ='text')
    #change missing coded and missing per se as missing values
    dftrain = dftrain.rename(columns={ dftrain.columns[1]: 'label'})
    dftrain = dftrain.replace(99, 2) #missing coded
    dftrain = dftrain.fillna(2) #missing as not coded
    #create arrow dataset to use with setfit
    hg_dataset = Dataset(pa.Table.from_pandas(dftrain))
    origclass = len(set(hg_dataset['label']))
    # define only the training/test dataset:
    datadict = hg_dataset.train_test_split(test_size=0.50, seed=13123)
    #first check the number of classes in the training set, they should be at least 2 or 3 depending on missing values
    #if there is only one class in the training set redo the split this time with random seed
    nclasses = len(set(datadict['train']['label']))
    nclasstest = len(set(datadict['test']['label']))
    while (nclasses < origclass) or nclasstest < origclass:
        print('Re-Splitting')
        datadict = hg_dataset.train_test_split(test_size=0.5)
        nclasses = len(set(datadict['train']['label']))
        nclasstest = len(set(datadict['test']['label']))
    #make a copy, in case it is needed
    training = datadict['train']
    evaldata = datadict['test']
    #compute class weights as we have highly imbalanced dataset
    wclass = compute_class_weight(class_weight = 'balanced',
                                  classes = np.unique(training['label']),
                                  y = training['label'])
    torchweight = torch.FloatTensor(wclass)
      
    print(col)
    print ('labels in training set = ' + str(set(training['label'])))
    print ('labels in test set = ' + str(set(datadict['test']['label'])))
    print ('samples per class in training set = ' + str(pd.Series(training['label']).value_counts()))
    print ('samples per class in test set = ' + str(pd.Series(evaldata['label']).value_counts()))
    
    test_datadict[col] = pd.DataFrame(evaldata)
   
    """
    Now: 
    Create trainer with hyperparameter search. 
    Create function for hyperparameter search and model initalization,using the pretrained model based on 
    the functions model_init and hp_space

    """
    trainer = SetFitTrainer(
        model_init = model_init,
        train_dataset = training,
        eval_dataset = evaldata,
        loss_class= losses.CosineSimilarityLoss,
        warmup_proportion = 0.05,
        metric = "f1",
        column_mapping={"text": "text", "label": "label"} # Map dataset columns to text/label expected by trainer
        )
    best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=25)
    optrun[col] = best_run
    #train and evaluate best run hyperparameters.
    trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
    trainer.train()

    print(col)
    trainerdict[col] = trainer
    modeldict [col] = trainer.model
    testmetric[col] = trainer.evaluate()
    print(trainer.model.model_head)
    print(testmetric[col])
    #save the pretrained model
    trainer.model._save_pretrained(save_directory = filename)
pickle.dump(testmetric, open('F1_Score.p', 'wb'))
pickle.dump(test_datadict, open('testdata.p', 'wb'))

#assess confusion matries and calculate precision, recall and accuracy on trainset data all.
dfcheck_cs =  trainset.dropna(subset ='text')
dfpreds_cs = pd.DataFrame(index = dfcheck_cs.txt_title)
#create dataframe for only casestudies
dfcheck = dfcheck_cs[dfcheck_cs['IsCase'] == 1]
dfpreds = pd.DataFrame(index = dfcheck.txt_title)

for key in modeldict:
    mod_temp = modeldict[key]
    print(key)
    if key == 'CaseStudy':
        preds = (mod_temp.predict(list(dfcheck_cs['text'])))
        preds = preds.tolist()
        probs = mod_temp.predict_proba(list(dfcheck_cs['text']))
        probs = probs.tolist()
        colname = key + '_p'
        colprob = key +'_probs'
        dfpreds_cs[colname] = preds
        dfpreds_cs[colprob] = probs
        print('CaseStudy-Colname')
    else:
        preds = (mod_temp.predict(list(dfcheck['text'])))
        preds = preds.tolist()
        probs = mod_temp.predict_proba(list(dfcheck['text']))
        probs = probs.tolist()
        colname = key + '_p'
        colprob = key + '_probs'
        print('colname')
        dfpreds[colname] = preds
        dfpreds[colprob] = probs
        
        
#merge the original coding and the predicted one on both train and test for confusion matrices
os.chdir(figout)

trainconf = trainset.copy()
trainconf = trainconf.replace(99,2)
trainconf = trainconf.fillna(2)
dfconfusion = pd.merge(left = trainconf, right = dfpreds, on ='txt_title')
dfconfusion_cs = pd.merge(left = trainconf, right = dfpreds_cs, on = 'txt_title')
dfconfusion_cs.to_csv('resultsCS.csv')
dfconfusion.to_csv('resultsAll.csv')

precision = {}
recall = {}
f1 = {}
mcc = {}
accuracy = {}
mcc = {}
#check confusion matrix and evaluation metrics using macro, and then using mcc that is balanced, as we are worried about minority classes 
#first let's check by predicting the whole dataset (train and test together)
for col in colnames:
    print(col)
    pname = col + '_p'
    if col == 'CaseStudy':
        vals = dfconfusion_cs[col]
        preds = dfconfusion_cs[pname]
        print(len(vals))
    else:
        vals = dfconfusion[col]
        preds = dfconfusion[pname]
        print(len(vals))
    recall[col] = recall_score(vals, preds, average = 'macro')
    precision[col] = precision_score(vals, preds, average = 'macro')
    f1[col] = f1_score(vals, preds, average = 'macro')
    accuracy[col] = accuracy_score(vals, preds)
    mcc[col] = matthews_corrcoef(vals, preds)
    cmat = confusion_matrix(vals, preds)
    disp = ConfusionMatrixDisplay(cmat)
    disp.title = col
    disp.plot()
    figname = 'ConfusionMatrix_' + col + '.pdf'
    plt.savefig(figname)
    plt.close()
dfeval = pd.DataFrame({'accuracy':pd.Series(accuracy),'recall':pd.Series(recall), 'precision':pd.Series(precision),'f1':pd.Series(f1), 'mcc':pd.Series(mcc)})
dfeval.to_csv('EvalAll.csv')

"""
#now do evaluation only on test set, using the seed we used before or by using the saved test data
"""
    
test_precision = {}
test_recall = {}
test_f1 = {}
test_mcc = {}
test_accuracy = {}
for col in colnames:
    print(col)
    pname = col + '_p'
    if col == 'CaseStudy':
        testdf_cs = pd.DataFrame(test_datadict[col])
        dfconf_cs = pd.merge(left = testdf_cs, right = dfconfusion_cs, left_on = 'text', right_on ='text')
        vals = dfconf_cs[col]
        preds = dfconf_cs[pname]
        print(col + '_OnlyCases')
    else:
        testdf = pd.DataFrame(test_datadict[col])
        dfconf = pd.merge(left = testdf, right = dfconfusion, left_on = 'text', right_on ='text') 
        vals = dfconf[col]
        preds = dfconf[pname]
    test_recall[col] = recall_score(vals, preds, average = 'macro')
    test_precision[col] = precision_score(vals, preds, average = 'macro')
    test_f1[col] = f1_score(vals, preds, average = 'macro')
    test_accuracy[col] = accuracy_score(vals, preds)
    test_mcc[col] = matthews_corrcoef(vals, preds)
    cmat = confusion_matrix(vals, preds)
    disp = ConfusionMatrixDisplay(cmat)
    disp.title = col
    disp.plot()
    figname = 'Test_ConfusionMatrix_' + col + '.pdf'
    plt.savefig(figname)
    plt.close()
test_dfeval = pd.DataFrame({'accuracy':pd.Series(test_accuracy),'recall':pd.Series(test_recall), 'precision':pd.Series(test_precision),'f1':pd.Series(test_f1), 'mcc':pd.Series(test_mcc)})
test_dfeval.to_csv('testEval.csv')


#import the pre-trained models or use directly modeldict, so this bit of code is optional.
os.chdir(modelfiles)
modlist = os.listdir()
modeldict = {}
for item in modlist:
    if item == '.DS_Store':
        pass
    else:
        key_name = item.split('_', 1)[0]
        modeldict[key_name] = SetFitModel._from_pretrained(item)

#import main datafiles, all papers full text, papers in trainingset with and without nan values
os.chdir(cleandata)
dfcases_unseen = pickle.load(open ('extracted_df.p', 'rb'))
dfcases_train = pickle.load(open('extracted_df_train.p', 'rb'))
dfcases = pd.concat([dfcases_unseen, dfcases_train],axis=0, ignore_index=True)
dfcases = dfcases.drop_duplicates(subset='text')
dfcases = dfcases.dropna(subset='text')
dfcases = dfcases.reset_index()

#predict labels for unseen data (dfcases) as well as for labelled data (dfcases_train) without nan values
colnames = ['CaseStudy', 'rescond', 'reschange', 
       'conflict', 'inequality', 'socbound',
        'resbound', 'fit', 'prop',
       'participation',  'monitor',
       'accmonitor',  'gradsanc', 'confres',  'autonomy', 
       'nested']
#now do predicted on full data
#import main datafiles, all papers full text, papers in trainingset with and without nan values
os.chdir(cleandata)
dfcases_unseen = pickle.load(open ('extracted_df.p', 'rb'))
dfcases_train = pickle.load(open('extracted_df_train.p', 'rb'))
dfcases = pd.concat([dfcases_unseen, dfcases_train],axis=0, ignore_index=True)
dfcases = dfcases.drop_duplicates(subset='text')
dfcases = dfcases.dropna(subset='text')
dfcases = dfcases.reset_index()

#predict labels for unseen data (dfcases) as well as for labelled data (dfcases_train) without nan values
colnames = ['CaseStudy', 'rescond', 'reschange', 
       'conflict', 'inequality', 'socbound',
        'resbound', 'fit', 'prop',
       'participation',  'monitor',
       'accmonitor',  'gradsanc', 'confres',  'autonomy', 
       'nested']

dict_preds = {} #where to store the predicted probabilities and the argmax results for the coding
for col in colnames:
    print(col)
    #get the fine-tuned model
    trained_model = modeldict[col]   
    #predict labels:
    preds = trained_model.predict(list(dfcases['text']))
    preds = preds.tolist()
    probs = trained_model.predict_proba(list(dfcases['text']))
    probs = probs.tolist()
    colname = col + '_p'
    colprob = col +'_probs'
    dict_preds[colname] = preds
    dict_preds[colprob] = probs
dfpreds = pd.DataFrame.from_dict(dict_preds)
dfpreds = pd.concat([dfcases, dfpreds], axis=1)

#only use casestudy coded from few_shot
dfpreds_cases = dfpreds[dfpreds['CaseStudy_p'] == 1]

os.chdir(mainres)
pickle.dump(dict_preds, open('dict_predicted_codes.p', 'wb'))
pickle.dump(dfpreds, open('predicted_codes.p', 'wb'))
pickle.dump(dfpreds_cases, open('predicted_codes_cases.p', 'wb')) #this is used for geolocations and topic modelling
