#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Revised few-shot learner with:

1. Longformer-SciCo embeddings.
2. BatchAllTripletLoss.
3. Balanced logistic-regression classification head.
4. Stratified nested Optuna optimization.
5. Out-of-fold evaluation without training-data leakage.
6. Hard manually coded labels.
7. Model probabilities retained for uncertainty-aware analyses.
8. Hard CaseStudy gate using a semantic AnyCaseStudy classifier followed by a rule-based single-versus-multiple case counter.
9. Existing pickle filenames preserved.

This script does not perform PDF extraction, OCR or GROBID processing.

Required inputs:
    extracted_df.p
    extracted_df_train.p
    matching.xlsx

Legacy outputs preserved:
    F1_Score.p
    testdata.p
    dict_predicted_codes.p
    predicted_codes.p
    predicted_codes_cases.p

@author: jbaggio
"""

# Set unsupported MPS operations to fall back to CPU
import os

print('TDull Version FewShot Training')

SELECT_DEVICE = 'cuda'  # auto, cpu, mps, cuda

if SELECT_DEVICE == 'cpu':
    os.environ['ACCELERATE_USE_CPU'] = 'true'
    os.environ.pop('PYTORCH_ENABLE_MPS_FALLBACK', None)

elif SELECT_DEVICE == 'mps':
    os.environ.pop('ACCELERATE_USE_CPU', None)
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    #os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

elif SELECT_DEVICE in ['cuda', 'auto']:
    os.environ.pop('ACCELERATE_USE_CPU', None)
    os.environ.pop('PYTORCH_ENABLE_MPS_FALLBACK', None)

else:
    raise ValueError("SELECT_DEVICE must be 'auto', 'cpu', 'mps', or 'cuda'.")
    
 

# File operations
import pickle
import random
import shutil
import gc
import copy
import re
from pathlib import Path

# Usual suspects
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Hugging Face datasets and SetFit
import pyarrow as pa
from datasets import Dataset
from sentence_transformers import losses
from setfit import SetFitModel, SetFitTrainer

# Hyperparameter optimization
import optuna

# PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

'''
Device; check if torch can use GPU 
'''
def select_device(requested):
    if requested == 'auto':
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    if requested == 'mps':
        if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
            raise RuntimeError('MPS requested but unavailable.')
        return 'mps'

    elif requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable.')
        return 'cuda'

    elif requested == 'cpu':
        return 'cpu'

    raise ValueError(...)

  


device = select_device(SELECT_DEVICE)

print('PyTorch version:', torch.__version__)
print('MPS built:', torch.backends.mps.is_built())
print('MPS available:', torch.backends.mps.is_available())
print('CUDA available:', torch.cuda.is_available())
print('Selected device:', device)


print('PyTorch version:', torch.__version__)
print('MPS built:', torch.backends.mps.is_built())
print('MPS available:', torch.backends.mps.is_available())
print('Selected device:', device)

#make sure memory is cledar
objects_to_delete = [
    'trainer',
    'model',
    'temporary_model',
    'any_trainer',
    'any_model',
    'count_trainer',
    'count_model',
]

for object_name in objects_to_delete:
    if object_name in globals():
        del globals()[object_name]

gc.collect()

if torch.backends.mps.is_available():
    torch.mps.empty_cache()


# Cross-validation and evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss


'''
Load data from Path

'''

#PROJECT_ROOT = Path(os.environ.get('CSP_PROJECT_ROOT', '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject')).resolve()

#PROJECT_ROOT = Path('~/FewShotProject').expanduser()

#traintext = PROJECT_ROOT / ('DataCoded/Articles/TextForTraining')
#cleandata = PROJECT_ROOT / ('NLPSynth/Data')
#mainres = PROJECT_ROOT / ('NLPSynth/Analysis/MainResults')
#modelfiles = PROJECT_ROOT / ('NLPSynth/Analysis/Models')
#figout = PROJECT_ROOT / ('NLPSynth/Analysis/Figures')

PROJECT_ROOT = Path.home() / 'FewShotProject'

traintext = PROJECT_ROOT / 'Labels'
cleandata = PROJECT_ROOT / 'Data'

mainres = PROJECT_ROOT / 'Results/MainResults'
modelfiles = PROJECT_ROOT / 'Results/Models'
figout = PROJECT_ROOT / 'Results/Figures'

for output_path in [mainres, modelfiles, figout]:
    output_path.mkdir(parents=True, exist_ok=True)
    
    
'''
Hard coded paths in case


traintext = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/DataCoded/Articles/TextForTraining'
cleandata = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Data'
mainres = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis/MainResults'
modelfiles = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis/Models'
figout = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis/Figures'

os.makedirs(mainres, exist_ok=True)
os.makedirs(modelfiles, exist_ok=True)
os.makedirs(figout, exist_ok=True)

'''



'''
Variables
'''

colnames = ['CaseStudy','rescond', 'conflict','inequality','socbound','resbound',
            'fit','prop','participation','monitor','accmonitor','gradsanc','confres','autonomy','nested']

colnocase = ['rescond', 'conflict','inequality','socbound','resbound',
            'fit','prop','participation','monitor','accmonitor','gradsanc','confres','autonomy','nested']


'''
Model settings
'''

BASE_MODEL = 'allenai/longformer-scico'
MAX_SEQ_LENGTH = 4096

RANDOM_SEED = 13123

# Outer folds are used to calculate unbiased performance.
N_OUTER_FOLDS = 5

# Inner folds are used by Optuna within each outer training fold.
N_INNER_FOLDS = 3

# Start with a small number on the M1.
# Increase after confirming that the complete script runs correctly.
N_OPTUNA_TRIALS_OUTER = 30
N_OPTUNA_TRIALS_FINAL = 50

#Setup up Optuna random trials
OPTUNA_STARTUP_TRIALS = 10
OPTUNA_MULTIVARIATE = True

# Default values are used if Optuna cannot run for a variable.
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_NUM_EPOCHS = 1
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_ITERATIONS = 20
DEFAULT_HEAD_C = 1.0
DEFAULT_HEAD_SOLVER = 'lbfgs'

HEAD_MAX_ITER = 2000
WARMUP_PROPORTION = 0.05

# Triplet loss is the intended embedding loss. If false it defaults to contrastive learning and cosine loss.
USE_TRIPLET_LOSS = True

# Predict Case Study with case_tetxt that is extracted via GROBID and focuses on case study related section of a paper r just the selected text.
CASESTUDY_TEXT_COLUMN = 'case_text'

#Type of Casestudy selection (semantic, semantic_counter)
CASESTUDY_METHOD = 'semantic_counter'


# Other variables use the GROBID-selected text.
OTHER_TEXT_COLUMN = 'text'

# Hard code used to identify a single case-study paper.
CASESTUDY_POSITIVE_LABEL = 1

#Batch size for triplet batch learning
TRIPLET_BATCH_LIST = [8]

# Codes treated as missing human labels, here missing means unclear or insufficient information to make a determination.
MISSING_CLASS = 2
MISSING_LABEL_CODES = [99]
CONVERT_99 = True #code 
CONVERT_NaN = False


# Prediction batches for the full corpus.
PREDICTION_BATCH_SIZE = 8

# Use of annotated text
ANNOTATED_VARIABLES = {
    'conflict', 		#better class 0 recall, differentiating between 0 and 1 is the key more so than 2
    'rescond', 			#much better class 0 recall and overall improvement model finally learned absence
    'monitor', 			#annotation improves presence/absence distinction
    'confres', 			#all classes improved with annotation
    'socbound', 		#another clear increase with annotation in F1macro and MCC
    'prop',				#slight improvement in F1macro and MCC
}

#fit, inequality, nested, gradsanc, resbound, accmonitor, participation and autonomy 
#require overall contextual evidence and annotation hurt the embeddings or does not improve it in the test configuration
#participation is really borderline.

ANNOTATION_PREFIX = 'CODING EVIDENCE: '
MIN_ANNOTATION_CHARS = 10

ANNOTATION_COLUMNS = {
    'rescond':'txt_rescond',
    'conflict':'txt_conflict',
    'inequality':'txt_inequality',
    'socbound':'txt_socbound',
    'resbound':'txt_resbound',
    'fit':'txt_fit',
    'prop':'txt_prop',
    'participation':'txt_participation',
    'monitor':'txt_monitor',
    'accmonitor':'txt_accmonitor',
    'gradsanc':'txt_gradsanc',
    'confres':'txt_confres',
    'autonomy':'txt_autonomy',
    'nested':'txt_nested'
}


# Do not rerun completed outer folds when a saved temporary result exists.
REUSE_SAVED_OOF_FOLDS = True

RUN_NAME = (CASESTUDY_METHOD + '_OF' + str(N_OUTER_FOLDS) + '_IF' + str(N_INNER_FOLDS) + '_OT' + str(N_OPTUNA_TRIALS_OUTER)
    + '_FT' + str(N_OPTUNA_TRIALS_FINAL) + '_ST' + str(OPTUNA_STARTUP_TRIALS) + '_L' + str(MAX_SEQ_LENGTH) + '_triplet' + str(int(USE_TRIPLET_LOSS))
    + '_batch-' + '-'.join(str(value) for value in TRIPLET_BATCH_LIST))

foldres = os.path.join(modelfiles, 'OOF_Temporary', RUN_NAME)

checkpoint_dir = os.path.join(mainres, 'variable_checkpoints', RUN_NAME)

os.makedirs(foldres, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

print('RUN NAME:', RUN_NAME)



'''
Reproducibility
'''

def set_all_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
set_all_seeds(RANDOM_SEED)
    



'''
Functions
'''

# Clear Python and accelerator memory
def clear_memory():
    gc.collect()
    if device == 'mps' and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()


# Convert tensors, lists or arrays into NumPy arrays
def to_numpy(values):
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


# Return default hyperparameters
def default_hyperparameters():
    return {'learning_rate': DEFAULT_LEARNING_RATE, 
            'num_epochs': DEFAULT_NUM_EPOCHS, 
            'batch_size': DEFAULT_BATCH_SIZE, 
            'num_iterations': DEFAULT_NUM_ITERATIONS, 
            'C': DEFAULT_HEAD_C, 
            'solver': DEFAULT_HEAD_SOLVER
            }


# Report duplicated paper titles; this is taken care of in the CSP_PdfToCleanText_VF script but double checking it is a good idea.
def report_duplicate_titles(dataframe, dataframe_name):
    duplicates = dataframe[dataframe['txt_title'].duplicated(keep=False)]['txt_title']
    if len(duplicates) > 0:
        print('\n')
        print('WARNING: DUPLICATED TITLES IN ' + dataframe_name)
        print(duplicates.value_counts().head(30))


# Return the selected embedding loss. If there are enough cases per class (at least 10, then do tirplet, if not do cosine)
def get_loss_class(training_df=None):

    if not USE_TRIPLET_LOSS:
        return losses.CosineSimilarityLoss

    if training_df is not None:

        try:
            validate_triplet_training_data(training_df, 'Triplet validation')
            return losses.BatchAllTripletLoss

        except ValueError:

            print('Triplet validation failed. Switching to cosine similarity loss.')
            return losses.CosineSimilarityLoss

    return losses.BatchAllTripletLoss


# Initialize the SetFit model
def model_init(params=None):
    params = params or default_hyperparameters()
    head_params = {
        'max_iter': HEAD_MAX_ITER,
        'solver': params.get('solver', DEFAULT_HEAD_SOLVER),
        'C': float(params.get('C', DEFAULT_HEAD_C)),
        'class_weight': 'balanced',
        'random_state': RANDOM_SEED
    }
    model = SetFitModel.from_pretrained(
        BASE_MODEL,
        device=device,
        head_params=head_params
    )
    model.model_body.max_seq_length = MAX_SEQ_LENGTH
    return model

# Inspect the effective model input length
def inspect_model_length():
    temporary_model = model_init()
    print('SetFit maximum sequence length:', temporary_model.model_body.max_seq_length)
    print('Tokenizer model maximum length:', temporary_model.model_body.tokenizer.model_max_length)

    if device == 'mps':
        try:
            test_tensor = torch.ones((100,100), device='mps')
            test_result = test_tensor @ test_tensor
            print('MPS tensor test:', test_result.device)
            del test_tensor
            del test_result
        except Exception as e:
            print('MPS tensor test failed:', e)

    del temporary_model
    clear_memory()


# Create a Hugging Face dataset
def make_huggingface_dataset(dfdata):
    dataset_df = dfdata[['model_text','label']].copy()
    dataset_df = dataset_df.rename(columns={'model_text': 'text'})
    dataset_df = dataset_df.reset_index(drop=True)
    return Dataset(pa.Table.from_pandas(dataset_df, preserve_index=False))
     

'''
Case Study selection specific function

'''

'''
Case Study selection specific functions
'''

# Counter patterns for explicit numbers and multiple-case language
CASE_COUNTER_PATTERNS = {
    'numbered_case': r'\bcase\s+(?:[0-9]+|[a-z])\b',
    'ordinal_case': (r'\b(first|second|third|fourth|fifth|sixth)\s+case\b'),
    'cross_case': r'\bcross[- ]case\b','multiple_case': (r'\b(multiple|several|various|numerous)\s+cases?\b'),
    'comparative_case': (r'\bcomparative\s+'r'(?:case\s+study|case\s+studies|case\s+analysis|cases)\b'),
    'matched_cases': r'\bmatched\s+cases?\b'
}

CASE_COUNTER_NUMBER_PATTERNS = [
    (r'\b(one|1)\s+(case study|case)\b', 1),
    (r'\b(two|2)\s+(case studies|cases)\b', 2),
    (r'\b(three|3)\s+(case studies|cases)\b', 3),
    (r'\b(four|4)\s+(case studies|cases)\b', 4),
    (r'\b(five|5)\s+(case studies|cases)\b', 5),
    (r'\b(six|6)\s+(case studies|cases)\b', 6)
]




# Estimate the number of cases from explicit counts, labels and comparative language
def estimate_case_count(text):
    text = str(text).lower()
    explicit_counts = []

    for pattern, count in CASE_COUNTER_NUMBER_PATTERNS:
        if re.search(pattern, text):
            explicit_counts.append(count)

    if explicit_counts:
        return max(explicit_counts)

    numbered_cases = set(re.findall(CASE_COUNTER_PATTERNS['numbered_case'], text))
    ordinal_cases = set(re.findall(CASE_COUNTER_PATTERNS['ordinal_case'], text))
    multiple_evidence = any (re.search(CASE_COUNTER_PATTERNS[key], text)
        for key in [
            'cross_case',
            'multiple_case',
            'comparative_case',
            'matched_cases'
        ]
    )

    if len(numbered_cases) >= 2:
        return len(numbered_cases)

    elif len(ordinal_cases) >= 2:
        return len(ordinal_cases)

    elif multiple_evidence:
        return 2

    else:
    	return 1

#clean data for case study
def clean_text_column(series):
    return (
        series
        .fillna('')
        .astype(str)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

# Prepare hierarchical CaseStudy datasets
def prepare_hierarchical_case_data(trainset):
    """
    original_df:
        0 = no case study
        1 = single case
        2 = multiple cases

    any_case_df:
        0 = no case study
        1 = any case study

    case_count_df:
        0 = single case
        1 = multiple cases
    """

    original_df = trainset[['txt_title',CASESTUDY_TEXT_COLUMN,'CaseStudy']].copy()
    original_df = original_df.rename(columns={CASESTUDY_TEXT_COLUMN:'model_text','CaseStudy':'original_label'})
    original_df = original_df.dropna(subset=['txt_title', 'original_label'])
    original_df['model_text'] = clean_text_column(original_df['model_text'])
    original_df = original_df[original_df['model_text'].str.len() > 0].copy()
    original_df['original_label'] = original_df['original_label'].astype(int)
    original_df = original_df.drop_duplicates(subset=['txt_title'],keep='first')
    original_df = original_df.reset_index(drop=True)
    original_df['row_id'] = np.arange(len(original_df))

    observed_classes = set(original_df['original_label'].unique())

    if not {0,1,2}.issubset(observed_classes):
        raise ValueError('Hierarchical CaseStudy training requires classes 0, 1 and 2. Observed classes: ' + str(sorted(observed_classes)))

    any_case_df = original_df[['row_id','txt_title','model_text']].copy()
    any_case_df['label'] = (original_df['original_label'] != 0).astype(int)
    any_case_df = any_case_df.reset_index(drop=True)

    case_count_df = original_df[original_df['original_label'].isin([1,2])][
        ['row_id','txt_title','model_text','original_label']
    ].copy()

    case_count_df['label'] = (case_count_df['original_label'] == 2).astype(int)
    case_count_df = case_count_df.drop(columns='original_label')
    case_count_df = case_count_df.reset_index(drop=True)

    if any_case_df['label'].nunique() < 2:
        raise ValueError('AnyCaseStudy has fewer than two classes.')

    if case_count_df['label'].nunique() < 2:
        raise ValueError('CaseCount has fewer than two classes.')

    print('\n')
    print('ORIGINAL CASESTUDY COUNTS')
    print(original_df['original_label'].value_counts().sort_index())

    print('\n')
    print('ANY CASE COUNTS')
    print(any_case_df['label'].value_counts().sort_index())

    print('\n')
    print('CASE COUNT COUNTS')
    print(case_count_df['label'].value_counts().sort_index())

    return original_df,any_case_df,case_count_df


#return counter probabilities
def fit_counter_probability_map(training_df):
    true_cases = training_df[
        training_df['original_label'].isin([1, 2])
    ].copy()

    if len(true_cases) == 0:
        raise ValueError('No labeled case papers available for counter fitting.')

    true_multiple = (true_cases['original_label'] == 2).astype(int).to_numpy()
    rule_multiple = np.asarray([int(estimate_case_count(text) > 1) for text in true_cases['model_text']], dtype=int)
    
    probability_map = {}

    for rule_value in [0, 1]:
        mask = rule_multiple == rule_value

        probability_map[rule_value] = float(
            (true_multiple[mask].sum() + 1)
            / (mask.sum() + 2)
        )

    return probability_map
    
# Return the fitted probability for one class
def get_positive_probability(model,texts,positive_class=1):
    probabilities = to_numpy(model.predict_proba(texts))
    classes = get_model_classes(model)
    matching_position = np.where(classes == positive_class)[0]

    if len(matching_position) != 1:
        raise ValueError('Could not find class ' + str(positive_class) + ' in fitted classes ' + str(classes.tolist()))

    return probabilities[:,int(matching_position[0])]


# Predict AnyCaseStudy semantically and single/multiple using the counter
def create_case_counter_predictions(any_model, texts, counter_probability_map, prediction_batch_size=PREDICTION_BATCH_SIZE):
    
    any_hard = []
    any_probability = []

    if len(texts) == 0:
        return {
            'CaseStudy_p': [],
            'CaseStudy_probs': [],
            'CaseStudy_classes': [],
            'CaseStudy_prob_0': [],
            'CaseStudy_prob_1': [],
            'CaseStudy_prob_2': [],
            'AnyCaseStudy_p': [],
            'AnyCaseStudy_prob_1': [],
            'EstimatedCaseCount': [],
            'CaseCount_p': [],
            'CaseCount_prob_multiple': []
        }

    for start in range(0, len(texts), prediction_batch_size):
        end = min(start + prediction_batch_size, len(texts))
        text_batch = texts[start:end]

        batch_hard = to_numpy(any_model.predict(text_batch)).reshape(-1).astype(int)
        batch_probability = get_positive_probability(any_model, text_batch, positive_class=1)
        any_hard.extend(batch_hard.tolist())
        any_probability.extend(batch_probability.tolist())
        clear_memory()

    any_hard = np.asarray(any_hard, dtype=int)
    any_probability = np.asarray(any_probability, dtype=float)

    estimated_counts = np.asarray([estimate_case_count(text) for text in texts], dtype=int)

    count_hard = (estimated_counts > 1).astype(int)

    count_probability = np.asarray([counter_probability_map[int(value)] for value in count_hard], dtype=float)

    final_hard = np.zeros(len(texts), dtype=int)
    predicted_case = any_hard == 1
    final_hard[predicted_case] = 1 + count_hard[predicted_case]

    final_estimated_counts = estimated_counts.copy()
    final_estimated_counts[~predicted_case] = 0

    final_probabilities = np.column_stack([1.0 - any_probability, any_probability * (1.0 - count_probability), any_probability * count_probability])

    final_probabilities = np.clip(final_probabilities, 1e-12, 1.0)

    final_probabilities /= final_probabilities.sum(axis=1, keepdims=True)

    return {
        'CaseStudy_p': final_hard.tolist(),
        'CaseStudy_probs': final_probabilities.tolist(),
        'CaseStudy_classes': [[0, 1, 2]] * len(texts),
        'CaseStudy_prob_0': final_probabilities[:, 0].tolist(),
        'CaseStudy_prob_1': final_probabilities[:, 1].tolist(),
        'CaseStudy_prob_2': final_probabilities[:, 2].tolist(),
        'AnyCaseStudy_p': any_hard.tolist(),
        'AnyCaseStudy_prob_1': any_probability.tolist(),
        'EstimatedCaseCount': final_estimated_counts.tolist(),
        'CaseCount_p': count_hard.tolist(),
        'CaseCount_prob_multiple': count_probability.tolist()
    }


# Create leakage-safe CaseStudy OOF predictions using SetFit AnyCaseStudy plus counter
def create_case_counter_oof_predictions(original_df):
    labels = original_df['original_label'].to_numpy()
    outer_folds = determine_number_of_folds(labels,N_OUTER_FOLDS)
    outer_splitter = StratifiedKFold(n_splits=outer_folds,shuffle=True,random_state=RANDOM_SEED)

    final_predictions = np.full(len(original_df),np.nan)
    final_probabilities = np.full((len(original_df),3),np.nan)
    any_predictions = np.full(len(original_df),np.nan)
    any_probabilities = np.full(len(original_df),np.nan)
    estimated_case_counts = np.full(len(original_df),np.nan)
    count_predictions = np.full(len(original_df),np.nan)
    count_probabilities = np.full(len(original_df),np.nan)
    fold_parameter_records = []

    print('\n')
    print('=' * 80)
    print('CASE COUNTER OUT-OF-FOLD EVALUATION')
    print('OBSERVATIONS:',len(original_df))
    print('OUTER FOLDS:',outer_folds)
    print('CLASS COUNTS:',original_df['original_label'].value_counts().sort_index().to_dict())
    print('=' * 80)

    for outer_fold_number,split_indices in enumerate(outer_splitter.split(original_df['model_text'],labels),start=1):
        outer_train_indices,outer_test_indices = split_indices
        outer_train = original_df.iloc[outer_train_indices].copy()
        outer_test = original_df.iloc[outer_test_indices].copy()

        temporary_file = os.path.join(foldres,'CaseStudy_counter_' + CASESTUDY_TEXT_COLUMN + '_L' + str(MAX_SEQ_LENGTH) + '_OF' + str(N_OUTER_FOLDS) + '_IF' + str(N_INNER_FOLDS) + '_OT' + str(N_OPTUNA_TRIALS_OUTER) + '_fold_' + str(outer_fold_number) + '.p')

        print('\n')
        print('OUTER FOLD:',outer_fold_number)
        print('TRAIN COUNTS:',outer_train['original_label'].value_counts().sort_index().to_dict())
        print('TEST COUNTS:',outer_test['original_label'].value_counts().sort_index().to_dict())

        if REUSE_SAVED_OOF_FOLDS and os.path.exists(temporary_file):
            print('LOADING SAVED COUNTER OUTER FOLD:',temporary_file)
            saved_fold = pickle.load(open(temporary_file,'rb'))

            if saved_fold.get('test_indices') != outer_test_indices.tolist():
                raise ValueError('Saved counter fold indices do not match the current fold. Delete the saved fold and rerun.')

            if saved_fold.get('test_titles') != outer_test['txt_title'].tolist():
                raise ValueError('Saved counter fold titles do not match the current fold. Delete the saved fold and rerun.')
            
            if saved_fold.get('run_name') != RUN_NAME:
                raise ValueError('Saved counter fold belongs to another run.')

            fold_final_predictions = np.asarray(saved_fold['final_predictions'])
            fold_final_probabilities = np.asarray(saved_fold['final_probabilities'])
            fold_any_predictions = np.asarray(saved_fold['any_predictions'])
            fold_any_probabilities = np.asarray(saved_fold['any_probabilities'])
            fold_estimated_counts = np.asarray(saved_fold['estimated_counts'])
            fold_count_predictions = np.asarray(saved_fold['count_predictions'])
            fold_count_probabilities = np.asarray(saved_fold['count_probabilities'])
            best_any_params = saved_fold['best_any_params']
            best_any_score = saved_fold['best_any_score']

        else:
            outer_train_any = outer_train[['row_id','txt_title','model_text']].copy()
            outer_train_any['label'] = (outer_train['original_label'] != 0).astype(int).to_numpy()

            best_any_params,best_any_score = optimize_hyperparameters(outer_train_any,'AnyCaseStudy',N_OPTUNA_TRIALS_OUTER,'AnyCaseStudy_counter_outer_' + str(outer_fold_number))
            any_trainer = train_setfit_model(outer_train_any,best_any_params)
            any_model = any_trainer.model
            outer_test_texts = outer_test['model_text'].tolist()

            fold_any_predictions = to_numpy(any_model.predict(outer_test_texts)).reshape(-1).astype(int)
            fold_any_probabilities = get_positive_probability(any_model,outer_test_texts,positive_class=1)

            fold_estimated_counts = np.asarray([estimate_case_count(text) for text in outer_test_texts],dtype=int)
            fold_count_predictions = (fold_estimated_counts > 1).astype(int)

            # These are heuristic counter scores, not calibrated learned probabilities.
            counter_probability_map = fit_counter_probability_map(outer_train)
            fold_count_probabilities = np.asarray([counter_probability_map[int(value)] for value in fold_count_predictions],dtype=float)
            fold_final_predictions = np.zeros(len(outer_test),dtype=int)
            predicted_case = fold_any_predictions == 1
            fold_final_predictions[predicted_case] = 1 + fold_count_predictions[predicted_case]

            fold_estimated_counts = fold_estimated_counts.copy()
            fold_estimated_counts[~predicted_case] = 0

            fold_final_probabilities = np.column_stack([
                1.0 - fold_any_probabilities,
                fold_any_probabilities * (1.0 - fold_count_probabilities),
                fold_any_probabilities * fold_count_probabilities
            ])

            fold_final_probabilities = np.clip(fold_final_probabilities,1e-12,1.0)
            fold_final_probabilities = fold_final_probabilities / fold_final_probabilities.sum(axis=1,keepdims=True)

            saved_fold = {
            	'counter_probability_map': counter_probability_map,
				'run_name': RUN_NAME,
                'final_predictions':fold_final_predictions,
                'final_probabilities':fold_final_probabilities,
                'any_predictions':fold_any_predictions,
                'any_probabilities':fold_any_probabilities,
                'estimated_counts':fold_estimated_counts,
                'count_predictions':fold_count_predictions,
                'count_probabilities':fold_count_probabilities,
                'best_any_params':best_any_params,
                'best_any_score':best_any_score,
                'test_indices':outer_test_indices.tolist(),
                'test_titles':outer_test['txt_title'].tolist()
            }

            pickle.dump(saved_fold,open(temporary_file,'wb'))

            del any_trainer
            del any_model
            del outer_train_any
            clear_memory()

        final_predictions[outer_test_indices] = fold_final_predictions
        final_probabilities[outer_test_indices,:] = fold_final_probabilities
        any_predictions[outer_test_indices] = fold_any_predictions
        any_probabilities[outer_test_indices] = fold_any_probabilities
        estimated_case_counts[outer_test_indices] = fold_estimated_counts
        count_predictions[outer_test_indices] = fold_count_predictions
        count_probabilities[outer_test_indices] = fold_count_probabilities

        fold_parameter_records.append({
            'variable':'CaseStudy',
            'outer_fold':outer_fold_number,
            'any_case_best_inner_f1':best_any_score,
            'any_case_best_params':best_any_params,
            'case_count_method':'counter'
        })

        del outer_train
        del outer_test
        clear_memory()

    if np.isnan(final_predictions).any():
        raise ValueError('Some counter CaseStudy OOF predictions are missing.')

    if np.isnan(final_probabilities).any():
        raise ValueError('Some counter CaseStudy OOF probabilities are missing.')

    if np.isnan(any_predictions).any():
        raise ValueError('Some AnyCaseStudy OOF predictions are missing.')

    if np.isnan(estimated_case_counts).any():
        raise ValueError('Some estimated case counts are missing.')

    if np.isnan(count_predictions).any():
        raise ValueError('Some counter CaseCount predictions are missing.')

    prediction_df = original_df[['row_id','txt_title','original_label']].copy()
    prediction_df = prediction_df.rename(columns={'original_label':'label'})
    prediction_df['CaseStudy_p'] = final_predictions.astype(int)
    prediction_df['CaseStudy_probs'] = final_probabilities.tolist()
    prediction_df['CaseStudy_classes'] = [[0,1,2]] * len(prediction_df)
    prediction_df['CaseStudy_prob_0'] = final_probabilities[:,0]
    prediction_df['CaseStudy_prob_1'] = final_probabilities[:,1]
    prediction_df['CaseStudy_prob_2'] = final_probabilities[:,2]
    prediction_df['AnyCaseStudy_p'] = any_predictions.astype(int)
    prediction_df['AnyCaseStudy_prob_1'] = any_probabilities
    prediction_df['EstimatedCaseCount'] = estimated_case_counts.astype(int)
    prediction_df['CaseCount_p'] = count_predictions.astype(int)
    prediction_df['CaseCount_prob_multiple'] = count_probabilities

    return prediction_df,np.asarray([0,1,2]),final_probabilities,final_predictions.astype(int),fold_parameter_records


# Create leakage-safe hierarchical CaseStudy OOF predictions using semantic models
def create_hierarchical_case_oof_predictions(original_df):
    labels = original_df['original_label'].to_numpy()
    outer_folds = determine_number_of_folds(labels,N_OUTER_FOLDS)
    outer_splitter = StratifiedKFold(n_splits=outer_folds,shuffle=True,random_state=RANDOM_SEED)

    final_predictions = np.full(len(original_df),np.nan)
    final_probabilities = np.full((len(original_df),3),np.nan)
    any_predictions = np.full(len(original_df),np.nan)
    any_probabilities = np.full(len(original_df),np.nan)
    count_predictions = np.full(len(original_df),np.nan)
    count_probabilities = np.full(len(original_df),np.nan)
    fold_parameter_records = []

    print('\n')
    print('=' * 80)
    print('HIERARCHICAL CASESTUDY OUT-OF-FOLD EVALUATION')
    print('OBSERVATIONS:',len(original_df))
    print('OUTER FOLDS:',outer_folds)
    print('CLASS COUNTS:',original_df['original_label'].value_counts().sort_index().to_dict())
    print('=' * 80)

    for outer_fold_number,split_indices in enumerate(outer_splitter.split(original_df['model_text'],labels),start=1):
        outer_train_indices,outer_test_indices = split_indices
        outer_train = original_df.iloc[outer_train_indices].copy()
        outer_test = original_df.iloc[outer_test_indices].copy()

        temporary_file = os.path.join(foldres,'CaseStudy_hierarchical_' + CASESTUDY_TEXT_COLUMN + '_L' + str(MAX_SEQ_LENGTH) + '_OF' + str(N_OUTER_FOLDS) + '_IF' + str(N_INNER_FOLDS) + '_OT' + str(N_OPTUNA_TRIALS_OUTER) + '_fold_' + str(outer_fold_number) + '.p')

        print('\n')
        print('OUTER FOLD:',outer_fold_number)
        print('TRAIN COUNTS:',outer_train['original_label'].value_counts().sort_index().to_dict())
        print('TEST COUNTS:',outer_test['original_label'].value_counts().sort_index().to_dict())

        if REUSE_SAVED_OOF_FOLDS and os.path.exists(temporary_file):
            print('LOADING SAVED HIERARCHICAL OUTER FOLD:',temporary_file)
            saved_fold = pickle.load(open(temporary_file,'rb'))
            
            if saved_fold.get('run_name') != RUN_NAME:
                raise ValueError('Saved hierarchical fold belongs to another run.')

            if saved_fold.get('test_indices') != outer_test_indices.tolist():
                raise ValueError('Saved hierarchical fold indices do not match the current fold. Delete the saved fold and rerun.')

            if saved_fold.get('test_titles') != outer_test['txt_title'].tolist():
                raise ValueError('Saved hierarchical fold titles do not match the current fold. Delete the saved fold and rerun.')

            
            fold_final_predictions = np.asarray(saved_fold['final_predictions'])
            fold_final_probabilities = np.asarray(saved_fold['final_probabilities'])
            fold_any_predictions = np.asarray(saved_fold['any_predictions'])
            fold_any_probabilities = np.asarray(saved_fold['any_probabilities'])
            fold_count_predictions = np.asarray(saved_fold['count_predictions'])
            fold_count_probabilities = np.asarray(saved_fold['count_probabilities'])
            best_any_params = saved_fold['best_any_params']
            best_any_score = saved_fold['best_any_score']
            best_count_params = saved_fold['best_count_params']
            best_count_score = saved_fold['best_count_score']

        else:
            outer_train_any = outer_train[['row_id','txt_title','model_text']].copy()
            outer_train_any['label'] = (outer_train['original_label'] != 0).astype(int).to_numpy()

            best_any_params,best_any_score = optimize_hyperparameters(outer_train_any,'AnyCaseStudy',N_OPTUNA_TRIALS_OUTER,'AnyCaseStudy_outer_' + str(outer_fold_number))
            any_trainer = train_setfit_model(outer_train_any,best_any_params)
            any_model = any_trainer.model
            outer_test_texts = outer_test['model_text'].tolist()

            fold_any_predictions = to_numpy(any_model.predict(outer_test_texts)).reshape(-1).astype(int)
            fold_any_probabilities = get_positive_probability(any_model,outer_test_texts,positive_class=1)

            outer_train_count = outer_train[outer_train['original_label'].isin([1,2])][
                ['row_id','txt_title','model_text','original_label']
            ].copy()

            outer_train_count['label'] = (outer_train_count['original_label'] == 2).astype(int)
            outer_train_count = outer_train_count.drop(columns='original_label')

            best_count_params,best_count_score = optimize_hyperparameters(outer_train_count,'CaseCount',N_OPTUNA_TRIALS_OUTER,'CaseCount_outer_' + str(outer_fold_number))
            count_trainer = train_setfit_model(outer_train_count,best_count_params)
            count_model = count_trainer.model

            fold_count_predictions = to_numpy(count_model.predict(outer_test_texts)).reshape(-1).astype(int)
            fold_count_probabilities = get_positive_probability(count_model,outer_test_texts,positive_class=1)

            fold_final_predictions = np.zeros(len(outer_test),dtype=int)
            predicted_case = fold_any_predictions == 1
            fold_final_predictions[predicted_case] = 1 + fold_count_predictions[predicted_case]

            fold_final_probabilities = np.column_stack([
                1.0 - fold_any_probabilities,
                fold_any_probabilities * (1.0 - fold_count_probabilities),
                fold_any_probabilities * fold_count_probabilities
            ])

            fold_final_probabilities = np.clip(fold_final_probabilities,1e-12,1.0)
            fold_final_probabilities = fold_final_probabilities / fold_final_probabilities.sum(axis=1,keepdims=True)

            saved_fold = {
                'final_predictions':fold_final_predictions,
                'final_probabilities':fold_final_probabilities,
                'any_predictions':fold_any_predictions,
                'any_probabilities':fold_any_probabilities,
                'count_predictions':fold_count_predictions,
                'count_probabilities':fold_count_probabilities,
                'best_any_params':best_any_params,
                'best_any_score':best_any_score,
                'best_count_params':best_count_params,
                'best_count_score':best_count_score,
                'test_indices':outer_test_indices.tolist(),
                'test_titles':outer_test['txt_title'].tolist(),
                'run_name': RUN_NAME
            }

            pickle.dump(saved_fold,open(temporary_file,'wb'))

            del any_trainer
            del any_model
            del count_trainer
            del count_model
            del outer_train_any
            del outer_train_count
            clear_memory()

        final_predictions[outer_test_indices] = fold_final_predictions
        final_probabilities[outer_test_indices,:] = fold_final_probabilities
        any_predictions[outer_test_indices] = fold_any_predictions
        any_probabilities[outer_test_indices] = fold_any_probabilities
        count_predictions[outer_test_indices] = fold_count_predictions
        count_probabilities[outer_test_indices] = fold_count_probabilities

        fold_parameter_records.append({
            'variable':'CaseStudy',
            'outer_fold':outer_fold_number,
            'any_case_best_inner_f1':best_any_score,
            'any_case_best_params':best_any_params,
            'case_count_best_inner_f1':best_count_score,
            'case_count_best_params':best_count_params
        })

        del outer_train
        del outer_test
        clear_memory()

    if np.isnan(final_predictions).any():
        raise ValueError('Some hierarchical CaseStudy OOF predictions are missing.')

    if np.isnan(final_probabilities).any():
        raise ValueError('Some hierarchical CaseStudy OOF probabilities are missing.')

    if np.isnan(any_predictions).any():
        raise ValueError('Some AnyCaseStudy OOF predictions are missing.')

    if np.isnan(count_predictions).any():
        raise ValueError('Some CaseCount OOF predictions are missing.')

    prediction_df = original_df[['row_id','txt_title','original_label']].copy()
    prediction_df = prediction_df.rename(columns={'original_label':'label'})
    prediction_df['CaseStudy_p'] = final_predictions.astype(int)
    prediction_df['CaseStudy_probs'] = final_probabilities.tolist()
    prediction_df['CaseStudy_classes'] = [[0,1,2]] * len(prediction_df)
    prediction_df['CaseStudy_prob_0'] = final_probabilities[:,0]
    prediction_df['CaseStudy_prob_1'] = final_probabilities[:,1]
    prediction_df['CaseStudy_prob_2'] = final_probabilities[:,2]
    prediction_df['AnyCaseStudy_p'] = any_predictions.astype(int)
    prediction_df['AnyCaseStudy_prob_1'] = any_probabilities
    prediction_df['CaseCount_p'] = count_predictions.astype(int)
    prediction_df['CaseCount_prob_multiple'] = count_probabilities

    return prediction_df,np.asarray([0,1,2]),final_probabilities,final_predictions.astype(int),fold_parameter_records


# Predict hierarchical CaseStudy classes using two semantic models
def predict_hierarchical_case_batched(any_model,count_model,texts,prediction_batch_size=PREDICTION_BATCH_SIZE):
    any_hard = []
    any_probability = []
    count_hard = []
    count_probability = []

    if len(texts) == 0:
        return {
            'CaseStudy_p':[],
            'CaseStudy_probs':[],
            'CaseStudy_classes':[],
            'CaseStudy_prob_0':[],
            'CaseStudy_prob_1':[],
            'CaseStudy_prob_2':[],
            'AnyCaseStudy_p':[],
            'AnyCaseStudy_prob_1':[],
            'CaseCount_p':[],
            'CaseCount_prob_multiple':[]
        }

    for start_position in range(0,len(texts),prediction_batch_size):
        end_position = min(start_position + prediction_batch_size,len(texts))
        text_batch = texts[start_position:end_position]

        batch_any_hard = to_numpy(any_model.predict(text_batch)).reshape(-1).astype(int)
        batch_any_probability = get_positive_probability(any_model,text_batch,positive_class=1)
        batch_count_hard = to_numpy(count_model.predict(text_batch)).reshape(-1).astype(int)
        batch_count_probability = get_positive_probability(count_model,text_batch,positive_class=1)

        any_hard.extend(batch_any_hard.tolist())
        any_probability.extend(batch_any_probability.tolist())
        count_hard.extend(batch_count_hard.tolist())
        count_probability.extend(batch_count_probability.tolist())

        clear_memory()

    any_hard = np.asarray(any_hard,dtype=int)
    any_probability = np.asarray(any_probability,dtype=float)
    count_hard = np.asarray(count_hard,dtype=int)
    count_probability = np.asarray(count_probability,dtype=float)

    final_hard = np.zeros(len(texts),dtype=int)
    predicted_case = any_hard == 1
    final_hard[predicted_case] = 1 + count_hard[predicted_case]

    final_probabilities = np.column_stack([
        1.0 - any_probability,
        any_probability * (1.0 - count_probability),
        any_probability * count_probability
    ])

    final_probabilities = np.clip(final_probabilities,1e-12,1.0)
    final_probabilities = final_probabilities / final_probabilities.sum(axis=1,keepdims=True)

    return {
        'CaseStudy_p':final_hard.tolist(),
        'CaseStudy_probs':final_probabilities.tolist(),
        'CaseStudy_classes':[[0,1,2]] * len(texts),
        'CaseStudy_prob_0':final_probabilities[:,0].tolist(),
        'CaseStudy_prob_1':final_probabilities[:,1].tolist(),
        'CaseStudy_prob_2':final_probabilities[:,2].tolist(),
        'AnyCaseStudy_p':any_hard.tolist(),
        'AnyCaseStudy_prob_1':any_probability.tolist(),
        'CaseCount_p':count_hard.tolist(),
        'CaseCount_prob_multiple':count_probability.tolist()
    }
    
#case study components metrics
def evaluate_case_components(prediction_df):
    y_true = prediction_df['label'].to_numpy()
    y_pred = prediction_df['CaseStudy_p'].to_numpy()

    any_true = (y_true != 0).astype(int)
    any_pred = (y_pred != 0).astype(int)

    true_case_mask = y_true != 0

    count_true = (y_true[true_case_mask] == 2).astype(int)

    count_pred = (y_pred[true_case_mask] == 2).astype(int)

    return {
        'three_class_macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'any_case_macro_f1': f1_score(any_true, any_pred, average='macro', zero_division=0),
        'one_multiple_macro_f1': f1_score( count_true, count_pred, average='macro', zero_division=0),
        'any_case_mcc': matthews_corrcoef(any_true, any_pred),
        'one_multiple_mcc': matthews_corrcoef(count_true, count_pred)
   	}


'''
end specific case study selection functions
'''



# Prepare human-coded data for one variable 
def prepare_label_data(trainset, col):

    if col == 'CaseStudy':
        dftrain = trainset[['txt_title', CASESTUDY_TEXT_COLUMN, 'CaseStudy']].copy()
        dftrain = dftrain.rename(columns={CASESTUDY_TEXT_COLUMN:'model_text','CaseStudy':'label'})

    else:
        annotation_column = ANNOTATION_COLUMNS.get(col)
        use_annotations = col in ANNOTATED_VARIABLES
        if use_annotations:
            if annotation_column is None:
                raise ValueError('No annotation column was specified for ' + col)

            if annotation_column not in trainset.columns:
                raise ValueError('Annotation column not found for ' + col + ': ' + annotation_column)

            dftrain = trainset[['txt_title', OTHER_TEXT_COLUMN, 'CaseStudy', col, annotation_column]].copy()
            dftrain = dftrain.rename(columns={OTHER_TEXT_COLUMN:'model_text', col:'label',annotation_column:'annotation_text'})

        else:
            dftrain = trainset[[ 'txt_title', OTHER_TEXT_COLUMN, 'CaseStudy', col]].copy()
            dftrain = dftrain.rename(columns={OTHER_TEXT_COLUMN:'model_text', col:'label'})
            dftrain['annotation_text'] = ''

        dftrain = dftrain[dftrain['CaseStudy']== CASESTUDY_POSITIVE_LABEL].copy()
        dftrain = dftrain.drop(columns='CaseStudy')
        
    n_code_99 = int(dftrain['label'].isin(MISSING_LABEL_CODES).sum())
    n_nan = int(dftrain['label'].isna().sum())

    if CONVERT_99:
        dftrain['label'] = dftrain['label'].replace(MISSING_LABEL_CODES,MISSING_CLASS)
        print(col + ': converted ' + str(n_code_99) + ' code-99 labels to class ' + str(MISSING_CLASS))
    else:
        dftrain['label'] = dftrain['label'].replace(MISSING_LABEL_CODES,np.nan)
        print(col + ': excluded ' + str(n_code_99) + ' code-99 labels')

    if CONVERT_NaN:
        dftrain['label'] = dftrain['label'].fillna(MISSING_CLASS)
        print(col + ': converted ' + str(n_nan) + ' NaN labels to class ' + str(MISSING_CLASS))
    else:
        print(col + ': excluded ' + str(n_nan) + ' NaN labels')

    dftrain = dftrain.dropna(subset=['txt_title', 'label'])
    dftrain['model_text'] = clean_text_column(dftrain['model_text'])
    if 'annotation_text' in dftrain.columns:
    	dftrain['annotation_text'] = clean_text_column(dftrain['annotation_text'])
    else:
    	dftrain['annotation_text'] = ''
    
    dftrain = dftrain[dftrain['model_text'].str.len() > 0].copy()
    dftrain['label'] = dftrain['label'].astype(int)
    dftrain = dftrain.drop_duplicates(subset=['txt_title'],keep='first')
    dftrain = dftrain.reset_index(drop=True)
    dftrain['row_id'] = np.arange(len(dftrain))

    if dftrain['label'].nunique() < 2:
        raise ValueError(col + ' has fewer than two manually coded classes.')

    print(col + ' final class counts: ' + str(dftrain['label'].value_counts().sort_index().to_dict()))

    return dftrain

#Triplet validation
def validate_triplet_training_data(df, context):
    counts = df['label'].value_counts()

    if counts.min() < 2:
        print(context + ' has fewer than two training examples in a class: ' + str(counts.to_dict()))
        return False

    return True
    
# Construct embedding-training examples
# Construct embedding-training examples
def prepare_embedding_training_data(training_df):
    embedding_df = training_df.copy()


    if 'annotation_text' not in embedding_df.columns:
        return embedding_df
    if embedding_df['annotation_text'].fillna('').str.len().max()==0:
    	return embedding_df

    annotation_rows = embedding_df[embedding_df['label'].isin([0,1]) & 
    embedding_df['annotation_text'].fillna('').str.len().ge(MIN_ANNOTATION_CHARS)].copy()

    if len(annotation_rows) == 0:
        print('ANNOTATION EMBEDDING TRAINING: no usable annotations')
        return embedding_df

    annotation_rows['model_text'] = (ANNOTATION_PREFIX + annotation_rows['annotation_text'].astype(str))

    annotation_rows['txt_title'] = (annotation_rows['txt_title'].astype(str) + '__annotation')

    annotation_rows['row_id'] = np.arange(len(embedding_df), len(embedding_df) + len(annotation_rows))

    embedding_df = pd.concat([embedding_df,annotation_rows], ignore_index=True)

    print('ANNOTATION EMBEDDING TRAINING: articles=' + str(len(training_df)) + ', annotations=' +
        str(len(annotation_rows)) + ', embedding examples=' + str(len(embedding_df)))

    return embedding_df

#refit logistic-head using embeddings and annotations
def refit_classification_head_on_articles(model, article_df):
    article_texts = article_df['model_text'].astype(str).tolist()
    article_labels = article_df['label'].astype(int).to_numpy()

    article_embeddings = model.model_body.encode(article_texts, batch_size=PREDICTION_BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)

    model.model_head.fit(article_embeddings, article_labels)

    print('REFITTED LOGISTIC HEAD ON ARTICLE EMBEDDINGS: ' + str(len(article_labels)) +
        ' articles, classes=' + str(sorted(np.unique(article_labels).tolist())))

    return model

# Train a SetFit model using supplied hyperparameters
def train_setfit_model(training_df, params=None):
    params = (
        copy.deepcopy(params)
        if params is not None
        else default_hyperparameters()
    )

    params.setdefault('solver',DEFAULT_HEAD_SOLVER)
    params.setdefault('num_iterations',DEFAULT_NUM_ITERATIONS)

    embedding_training_df = prepare_embedding_training_data(training_df)

    loss_class = get_loss_class(embedding_training_df)

    embedding_training_dataset = make_huggingface_dataset(embedding_training_df)

    model = model_init(params)

    trainer = SetFitTrainer(
        model=model,
        train_dataset=embedding_training_dataset,
        loss_class=loss_class,
        learning_rate=float(params['learning_rate']),
        num_epochs=int(params['num_epochs']),
        batch_size=int(params['batch_size']),
        num_iterations=int(params['num_iterations']),
        warmup_proportion=WARMUP_PROPORTION,
        column_mapping={
            'text':'text',
            'label':'label'
        }
    )

    trainer.train()

    trainer.model = refit_classification_head_on_articles(trainer.model, training_df)

    return trainer


# Get the fitted classification-head class order
def get_model_classes(model):
    if hasattr(model.model_head, 'classes_'):
        return np.asarray(model.model_head.classes_)
    raise ValueError('The fitted SetFit classification head has no classes_ attribute.')



# Define the Optuna search space
def suggest_hyperparameters(trial):
    if USE_TRIPLET_LOSS:
        batch_size = trial.suggest_categorical('batch_size', TRIPLET_BATCH_LIST)
        num_iterations = DEFAULT_NUM_ITERATIONS
    else:
        batch_size = trial.suggest_categorical('batch_size', [2, 4, 8])
        num_iterations = trial.suggest_categorical('num_iterations', [10, 20, 30, 40])

    return {
        'learning_rate': trial.suggest_float(
            'learning_rate', 5e-6, 8e-5, log=True
        ),
        'num_epochs': trial.suggest_categorical(
            'num_epochs', [1, 2]
        ),
        'batch_size': batch_size,
        'num_iterations': num_iterations,
        'C': trial.suggest_float(
            'C', 0.03, 10.0, log=True
        ),
        'solver': 'lbfgs'
    }

#assess consensus for outer-fold winners
def consensus_params(fold_records):
    p = [r['best_params'] for r in fold_records]
    mode = lambda k: int(pd.Series([x[k] for x in p]).mode().iloc[0])
    logmed = lambda k: float(np.exp(np.median([np.log(x[k]) for x in p])))
    return {'learning_rate': logmed('learning_rate'), 'C': logmed('C'),
                 'num_epochs': mode('num_epochs'), 'batch_size': mode('batch_size'),
                 'num_iterations': mode('num_iterations'), 'solver': 'lbfgs'}


# Determine the maximum valid number of stratified folds
def determine_number_of_folds(labels, requested_folds):
    class_counts = pd.Series(labels).value_counts()
    minimum_class_count = int(class_counts.min())
    n_splits = min(requested_folds, minimum_class_count)

    if n_splits < 2:
        raise ValueError('At least two coded observations are required in every class. Class counts: ' + str(class_counts.to_dict()))

    return n_splits


# Tune hyperparameters using cross-validation within the supplied training data
def optimize_hyperparameters(dftrain, col, n_trials, optimization_name):
    
    labels = dftrain['label'].to_numpy()
    class_counts = pd.Series(labels).value_counts()
    minimum_class_count = int(class_counts.min())

    if minimum_class_count < 2:
        raise ValueError('Too few observations for Optuna: ' + col + ' ' + str(class_counts.to_dict()))

    inner_folds = min(N_INNER_FOLDS, minimum_class_count)

    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_SEED)

    splits = list(splitter.split(dftrain['model_text'], labels))
       
    print('\nOPTUNA:', optimization_name)
    print('INNER FOLDS:', inner_folds)
    print('TRIALS:', n_trials)
    print('CLASS COUNTS:', class_counts.sort_index().to_dict())

    def objective(trial):
        params = suggest_hyperparameters(trial)
        fold_scores = []

        for inner_fold_number, split_indices in enumerate(splits, start=1):
        
            train_indices, validation_indices = split_indices
            inner_train = dftrain.iloc[train_indices].copy()

            inner_validation = dftrain.iloc[validation_indices].copy()

            trainer = None

            try:
                set_all_seeds(RANDOM_SEED + inner_fold_number)

                validate_triplet_training_data(inner_train, optimization_name + ' inner fold ' + str(inner_fold_number))

                trainer = train_setfit_model(inner_train, params)

                predictions = to_numpy(trainer.model.predict(inner_validation['model_text'].tolist())).reshape(-1).astype(int)

                fold_scores.append(f1_score(inner_validation['label'].to_numpy(), predictions, average='macro', zero_division=0))
 
            except RuntimeError as error:
                print("\nRUNTIME ERROR:")
                print(repr(error))
                print(error)
                
                error_text = str(error).lower()

                if ('out of memory' in error_text or 'mps backend out of memory' in error_text or 'cuda out of memory' in error_text):
                    clear_memory()
                    raise optuna.TrialPruned() from error

                raise

            finally:
                if trainer is not None:
                    del trainer

                del inner_train
                del inner_validation
                clear_memory()

        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED, 
    	n_startup_trials=min(OPTUNA_STARTUP_TRIALS, n_trials), multivariate=OPTUNA_MULTIVARIATE)

    study = optuna.create_study(direction='maximize', sampler=sampler, study_name=(RUN_NAME + '_' + optimization_name))

    study.optimize (objective, n_trials=n_trials, gc_after_trial=True)

    completed_trials = [trial for trial in study.trials if (trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None)]

    if len(completed_trials) == 0:
        raise RuntimeError('No valid Optuna trial completed for: ' + col)

    best_params = copy.deepcopy(study.best_trial.params)

    best_params['solver'] = 'lbfgs'
    best_params.setdefault('num_iterations', DEFAULT_NUM_ITERATIONS)
    best_score = float(study.best_value)

    print('BEST PARAMETERS:', best_params)
    print('BEST INNER-CV MACRO F1:', best_score)
    print('COMPLETED TRIALS:', len(completed_trials))

    return best_params, best_score


# Calculate multiclass Brier score
def multiclass_brier_score(y_true, probabilities, classes):
    class_to_position = {class_value: position for position, class_value in enumerate(classes)}
    y_onehot = np.zeros((len(y_true), len(classes)), dtype=float)

    for row_position, class_value in enumerate(y_true):
        if class_value in class_to_position:
            y_onehot[row_position, class_to_position[class_value]] = 1.0

    return np.mean(np.sum((probabilities - y_onehot) ** 2, axis=1))


# Create leakage-safe nested out-of-fold predictions
def create_nested_oof_predictions(dftrain, col):
    labels = dftrain['label'].to_numpy()
    all_classes = np.sort(dftrain['label'].unique())
    outer_folds = determine_number_of_folds(labels, N_OUTER_FOLDS)
    outer_splitter = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=RANDOM_SEED)

    oof_predictions = np.full(len(dftrain), np.nan)
    oof_probabilities = np.full((len(dftrain), len(all_classes)), np.nan)
    fold_parameter_records = []

    print('\n')
    print('=' * 80)
    print('NESTED OUT-OF-FOLD EVALUATION:', col)
    print('OBSERVATIONS:', len(dftrain))
    print('OUTER FOLDS:', outer_folds)
    print('CLASS COUNTS:', dftrain['label'].value_counts().sort_index().to_dict())
    print('=' * 80)

    for outer_fold_number, split_indices in enumerate(outer_splitter.split(dftrain['model_text'], labels), start=1):
        outer_train_indices, outer_test_indices = split_indices
        outer_train = dftrain.iloc[outer_train_indices].copy()
        outer_test = dftrain.iloc[outer_test_indices].copy()
        temporary_file = os.path.join(foldres, col + '_' + RUN_NAME + '_fold_' + str(outer_fold_number) + '.p')
        
        print('\n')
        print('OUTER FOLD:', outer_fold_number)
        print('TRAIN COUNTS:', outer_train['label'].value_counts().sort_index().to_dict())
        print('TEST COUNTS:', outer_test['label'].value_counts().sort_index().to_dict())

        if REUSE_SAVED_OOF_FOLDS and os.path.exists(temporary_file):
            print('LOADING SAVED OUTER FOLD:', temporary_file)
            saved_fold = pickle.load(open(temporary_file, 'rb'))
            if saved_fold.get('run_name') != RUN_NAME:
                raise ValueError('Saved substantive fold belongs to another run.')
            
            if saved_fold.get('test_indices') != outer_test_indices.tolist():
                raise ValueError('Saved substantive fold indices do not match.')

            if saved_fold.get('test_titles') != outer_test['txt_title'].tolist():
                raise ValueError('Saved substantive fold titles do not match.')
            
            fold_predictions = np.asarray(saved_fold['predictions'])
            fold_probabilities = np.asarray(saved_fold['probabilities'])
            fold_classes = np.asarray(saved_fold['classes'])
            best_params = saved_fold['best_params']
            best_inner_score = saved_fold['best_inner_score']
            
        else:
            optimization_name = col + '_outer_' + str(outer_fold_number)
            best_params, best_inner_score = optimize_hyperparameters(outer_train, col, N_OPTUNA_TRIALS_OUTER, optimization_name)
            trainer = train_setfit_model(outer_train, best_params)
            fold_model = trainer.model
            fold_predictions = to_numpy(fold_model.predict(outer_test['model_text'].tolist())).reshape(-1)
            fold_probabilities = to_numpy(fold_model.predict_proba(outer_test['model_text'].tolist()))
            fold_classes = get_model_classes(fold_model)

            saved_fold = {
            	'predictions': fold_predictions, 
            	'probabilities': fold_probabilities, 
            	'classes': fold_classes.tolist(), 
            	'best_params': best_params, 
            	'best_inner_score': best_inner_score, 
            	'test_indices': outer_test_indices.tolist(),
            	'test_titles': outer_test['txt_title'].tolist(),
				'run_name': RUN_NAME
			}

            pickle.dump(saved_fold, open(temporary_file, 'wb'))

            del trainer
            del fold_model
            clear_memory()

        oof_predictions[outer_test_indices] = fold_predictions

        for fold_class_position, class_value in enumerate(fold_classes):
            matching_position = np.where(all_classes == class_value)[0]

            if len(matching_position) != 1:
                raise ValueError('Could not match class ' + str(class_value) + ' for ' + col)

            global_class_position = int(matching_position[0])
            oof_probabilities[outer_test_indices, global_class_position] = fold_probabilities[:, fold_class_position]

        fold_parameter_records.append({'variable': col, 'outer_fold': outer_fold_number, 'best_inner_f1': best_inner_score, 'best_params': best_params})

        del outer_train
        del outer_test
        clear_memory()

    if np.isnan(oof_predictions).any():
        raise ValueError('Some out-of-fold hard predictions are missing for ' + col)

    if np.isnan(oof_probabilities).any():
        raise ValueError('Some out-of-fold probabilities are missing for ' + col)

    oof_predictions = oof_predictions.astype(int)

    prediction_df = dftrain[['row_id','txt_title','label']].copy()
    prediction_df[col + '_p'] = oof_predictions
    prediction_df[col + '_probs'] = oof_probabilities.tolist()
    prediction_df[col + '_classes'] = [all_classes.tolist()] * len(prediction_df)

    for class_position, class_value in enumerate(all_classes):
        probability_column = col + '_prob_' + str(class_value)
        prediction_df[probability_column] = oof_probabilities[:, class_position]

    return prediction_df, all_classes, oof_probabilities, oof_predictions, fold_parameter_records


# Evaluate nested out-of-fold predictions
def evaluate_oof_predictions(prediction_df, classes, probabilities, predictions, col):
    y_true = prediction_df['label'].to_numpy()

    metrics = {
        'accuracy': accuracy_score(y_true, predictions),
        'balanced_accuracy': balanced_accuracy_score(y_true, predictions),
        'recall': recall_score(y_true, predictions, average='macro', zero_division=0),
        'precision': precision_score(y_true, predictions, average='macro', zero_division=0),
        'f1': f1_score(y_true, predictions, average='macro', zero_division=0),
        'mcc': matthews_corrcoef(y_true, predictions),
        'log_loss': log_loss(y_true, probabilities, labels=classes),
        'brier': multiclass_brier_score(y_true, probabilities, classes),
        'n_observations': len(y_true),
        'n_classes': len(classes)
    }

    print('\n')
    print('OUT-OF-FOLD METRICS:', col)
    print(metrics)

    print('\n')
    print('TRUE LABEL DISTRIBUTION')
    print(pd.Series(y_true).value_counts().sort_index())

    print('\n')
    print('PREDICTED LABEL DISTRIBUTION')
    print(pd.Series(predictions).value_counts().sort_index())

    print('\n')
    print('CLASSIFICATION REPORT')
    print(classification_report(y_true, predictions, labels=classes, digits=3, zero_division=0))

    cmat = confusion_matrix(y_true, predictions, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=classes)
    disp.plot()
    plt.title(col + ' nested out-of-fold confusion matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(figout, 'OOF_ConfusionMatrix_' + col + '.pdf'))
    plt.close()

    report = classification_report(y_true, predictions, labels=classes, digits=6, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(figout, 'OOF_ClassificationReport_' + col + '.csv'))

    return metrics


# Train final production model
def train_final_model(dftrain, col, params):
    print('\n')
    print('=' * 80)
    print('TRAINING FINAL PRODUCTION MODEL:', col)
    print('OBSERVATIONS:', len(dftrain))
    print('CLASS COUNTS:', dftrain['label'].value_counts().sort_index().to_dict())
    print('FINAL PARAMETERS:', params)
    print('=' * 80)

    trainer = train_setfit_model(dftrain, params)
    final_model = trainer.model
    save_directory = os.path.join(modelfiles, col + '_pretrained')

    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)

    final_model.save_pretrained(save_directory)

    return trainer, final_model


# Predict one variable in manageable batches
def predict_variable_batched(model, texts, col, prediction_batch_size=PREDICTION_BATCH_SIZE):
    classes = get_model_classes(model)
    all_hard_predictions = []
    all_probabilities = []

    if len(texts) == 0:
        output = {}
        output[col + '_p'] = []
        output[col + '_probs'] = []
        output[col + '_classes'] = []

        for class_value in classes:
            output[col + '_prob_' + str(class_value)] = []

        return output

    for start_position in range(0, len(texts), prediction_batch_size):
        end_position = min(start_position + prediction_batch_size, len(texts))
        text_batch = texts[start_position:end_position]
        batch_hard_predictions = to_numpy(model.predict(text_batch)).reshape(-1)
        batch_probabilities = to_numpy(model.predict_proba(text_batch))

        all_hard_predictions.extend(batch_hard_predictions.astype(int).tolist())
        all_probabilities.append(batch_probabilities)

        clear_memory()

    probabilities = np.vstack(all_probabilities)

    output = {}
    output[col + '_p'] = all_hard_predictions
    output[col + '_probs'] = probabilities.tolist()
    output[col + '_classes'] = [classes.tolist()] * len(texts)

    for class_position, class_value in enumerate(classes):
        output[col + '_prob_' + str(class_value)] = probabilities[:, class_position].tolist()

    return output


''' 
to restore from checkpoint

'''
def restore_substantive_checkpoint(col, checkpoint_file):
    checkpoint = pickle.load(open(checkpoint_file, 'rb'))

    if checkpoint.get('run_name') != RUN_NAME:
        raise ValueError('Checkpoint belongs to another run: ' + col)

    testmetric[col] = checkpoint['testmetric']
    test_datadict[col] = checkpoint['test_datadict']
    class_order_dict[col] = checkpoint['class_order']
    outer_optrun[col] = checkpoint['outer_optrun']
    optrun[col] = checkpoint['optrun']
    final_params_dict[col] = checkpoint['final_params']

    model_path = os.path.join(modelfiles, col + '_pretrained')

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    loaded_model = SetFitModel.from_pretrained(model_path, device=device)

    loaded_model.model_body.max_seq_length = (MAX_SEQ_LENGTH )
    modeldict[col] = loaded_model


'''

Start actual script

'''

'''
Inspect model and accelerator
'''

inspect_model_length()


'''
Load text produced by the separate CSP_PdfToCleanText_VF script
'''

os.chdir(cleandata)

dfcases_unseen = pickle.load(open('extracted_df.p', 'rb'))
dfcases_train = pickle.load(open('extracted_df_train.p', 'rb'))

required_columns = ['txt_title','origtext','text', 'case_text']

for required_column in required_columns:
    if required_column not in dfcases_unseen.columns:
        raise ValueError(required_column + ' is missing from extracted_df.p')

    if required_column not in dfcases_train.columns:
        raise ValueError(required_column + ' is missing from extracted_df_train.p')

print('\n')
print('UNSEEN PAPERS:', len(dfcases_unseen))
print('TRAINING-TEXT PAPERS:', len(dfcases_train))

report_duplicate_titles(dfcases_unseen, 'extracted_df.p')
report_duplicate_titles(dfcases_train, 'extracted_df_train.p')


'''

Load manually coded labels

'''

os.chdir(traintext)

manual_trainset = pd.read_excel('matching.xlsx', sheet_name='trainset')
allpapers = pd.read_excel('matching.xlsx', sheet_name='result')

report_duplicate_titles(manual_trainset, 'matching.xlsx trainset')

dfcases_train = dfcases_train.drop_duplicates(subset=['txt_title'], keep='first')
manual_trainset = manual_trainset.drop_duplicates(subset=['txt_title'], keep='first')

trainset = pd.merge(left=dfcases_train, right=manual_trainset, on='txt_title', how='inner', validate='one_to_one')
trainset_orig = trainset.copy()

#make sure you label multiple cases as those are not taken into account in the final analysis
trainset = trainset.rename(columns={"CaseStudy": "CasesPresent"})
trainset["CaseStudy"] = 0
trainset.loc[trainset["CasesPresent"] == 1, "CaseStudy"] = 1
trainset.loc[(trainset["CasesPresent"] == 1) & (trainset["NumCase"] > 1), "CaseStudy"] = 2


print('\n')
print('MANUALLY CODED ROWS:', len(manual_trainset))
print('MATCHED MANUALLY CODED PAPERS:', len(trainset_orig))
print('UNMATCHED MANUALLY CODED PAPERS:', len(manual_trainset) - len(trainset_orig))


if len(trainset) == 0:
    raise ValueError('No manually coded titles matched extracted_df_train.p.')


'''

Tabulate manually coded classes

'''

tabulate = {}

for col in colnames:
    prepared_data = prepare_label_data(trainset, col)
    tabulate[col] = prepared_data['label'].value_counts().sort_index()

tabulate = pd.DataFrame.from_dict(tabulate)

os.chdir(figout)

tabulate.to_csv('TrainSet_Values.csv')

print('\n')
print('MANUALLY CODED CLASS COUNTS')
print(tabulate)


'''
Nested out-of-fold evaluation

Optuna uses only each outer training fold.
The corresponding outer test fold remains untouched until prediction.
'''
#columns with substantive variables, no hierarchical selection only semantic
cols_vars = ['rescond', 'conflict','inequality','socbound','resbound',
            'fit','prop','participation','monitor','accmonitor','gradsanc','confres','autonomy','nested']

print('\nANNOTATION COVERAGE')

for col in cols_vars:
    annotation_column = ANNOTATION_COLUMNS.get(col)

    if annotation_column not in trainset.columns:
        print(col + ': COLUMN NOT FOUND')
        continue

    audit_df = trainset[trainset['CaseStudy'] == CASESTUDY_POSITIVE_LABEL][[col,annotation_column]].copy()

    audit_df['has_annotation'] = (audit_df[annotation_column].fillna('').astype(str).str.strip().str.len().ge(MIN_ANNOTATION_CHARS))

    print('\n' + col)
    print(pd.crosstab(audit_df[col], audit_df['has_annotation'], margins=True))


trainerdict = {}
modeldict = {}
testmetric = {}
test_datadict = {}
optrun = {}
outer_optrun = {}
class_order_dict = {}
final_params_dict = {}



'''
PHASE 1
Case Study Detection
'''

for col in ['CaseStudy']:

    checkpoint_file = os.path.join(checkpoint_dir, col + '_results.pkl')

    print('\n')
    print('#' * 80)
    print('VARIABLE:', col)
    print('#' * 80)

    original_case_df, any_case_df, case_count_df = (prepare_hierarchical_case_data(trainset))

    if CASESTUDY_METHOD == 'semantic_counter':

        print('\nUSING SEMANTIC GATE  PLUS RULE-BASED CASE COUNTER')
        

        oof_df, class_order, oof_probabilities, oof_predictions, fold_parameter_records = create_case_counter_oof_predictions (original_case_df)

    elif CASESTUDY_METHOD == 'semantic':

        print('\nUSING TWO SEMANTIC MODELS')
        
        oof_df, class_order, oof_probabilities, oof_predictions, fold_parameter_records = create_hierarchical_case_oof_predictions (original_case_df)


    else:
        raise ValueError( "CASESTUDY_METHOD must be 'semantic_counter' or 'semantic'.")

    test_datadict[col] = oof_df.copy()
    class_order_dict[col] = class_order.tolist()
    outer_optrun[col] = fold_parameter_records

    testmetric[col] = evaluate_oof_predictions(oof_df, class_order, oof_probabilities, oof_predictions, col)

    testmetric[col].update (evaluate_case_components (oof_df))

    final_any_params, final_any_score = (optimize_hyperparameters (any_case_df, 'AnyCaseStudy', N_OPTUNA_TRIALS_FINAL,'AnyCaseStudy_final'))
    
    any_trainer, any_model = train_final_model(any_case_df,'AnyCaseStudy', final_any_params)

    if CASESTUDY_METHOD == 'semantic':

        final_count_params, final_count_score = (optimize_hyperparameters(case_count_df, 'CaseCount', N_OPTUNA_TRIALS_FINAL, 'CaseCount_final'))

        count_trainer, count_model = train_final_model(case_count_df, 'CaseCount', final_count_params)
        
        trainerdict[col] = {
            'AnyCaseStudy': any_trainer,
            'CaseCount': count_trainer
        }

        modeldict[col] = {
            'AnyCaseStudy': any_model,
            'CaseCount': count_model
        }

        final_params_dict[col] = {
            'method': CASESTUDY_METHOD,
            'AnyCaseStudy': final_any_params,
            'AnyCaseStudy_best_inner_f1': final_any_score,
            'CaseCount': final_count_params,
            'CaseCount_best_inner_f1': final_count_score
        }

    else:

        final_counter_probability_map = (fit_counter_probability_map (original_case_df))
           
        trainerdict[col] = {
            'AnyCaseStudy': any_trainer
        }

        modeldict[col] = {
            'AnyCaseStudy': any_model,
            'CaseCounter': estimate_case_count,
            'CounterProbabilityMap': final_counter_probability_map
        }

        final_params_dict[col] = {
            'method': CASESTUDY_METHOD,
            'AnyCaseStudy': final_any_params,
            'AnyCaseStudy_best_inner_f1': final_any_score,
            'CounterProbabilityMap': final_counter_probability_map
        }

    optrun[col] = {
        'best_params': final_params_dict[col],
        'best_inner_f1': final_any_score
    }

    checkpoint_data = {
        'run_name': RUN_NAME,
        'variable': col,
        'testmetric': testmetric[col],
        'test_datadict': test_datadict[col],
        'class_order': class_order_dict[col],
        'outer_optrun': outer_optrun[col],
        'optrun': optrun[col],
        'final_params': final_params_dict[col]
    }

    pickle.dump(
        checkpoint_data,
        open(checkpoint_file, 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL
    )

    clear_memory()


'''
 PHASE 2
 SES Variables
 Train only on single-case papers
'''

for col in cols_vars:

    checkpoint_file = os.path.join(checkpoint_dir, f'{col}_results.pkl' )

    if os.path.exists(checkpoint_file):

        print('\n')
        print('SKIPPING COMPLETED VARIABLE:',col)
        
        restore_substantive_checkpoint(col, checkpoint_file)

        continue

    print('\n')
    print('#' * 80)
    print('VARIABLE:',col)
    print('#' * 80)

    # prepare_label_data already keeps:
    # CaseStudy == 1

    dftrain = prepare_label_data(trainset, col)

    oof_df,class_order,oof_probabilities,oof_predictions,fold_parameter_records = (create_nested_oof_predictions(dftrain, col))

    test_datadict[col] = oof_df.copy()

    class_order_dict[col] = class_order.tolist()

    outer_optrun[col] = fold_parameter_records

    testmetric[col] = evaluate_oof_predictions(oof_df, class_order, oof_probabilities, oof_predictions, col)

    final_params,final_inner_score = (optimize_hyperparameters (dftrain, col, N_OPTUNA_TRIALS_FINAL, col + '_final'))


    trainer,final_model = train_final_model(dftrain, col, final_params)

    trainerdict[col] = trainer

    modeldict[col] = final_model

    final_params_dict[col] = final_params

    optrun[col] = {
        'best_params':final_params,
        'best_inner_f1':final_inner_score
    }

    checkpoint_data = {
        'run_name' : RUN_NAME,
        'variable':col,
        'testmetric':testmetric[col],
        'test_datadict':test_datadict[col],
        'class_order':class_order_dict[col],
        'outer_optrun':outer_optrun[col],
        'optrun':optrun[col],
        'final_params':final_params_dict[col]
    }

    pickle.dump(checkpoint_data, open(checkpoint_file,'wb'))

    clear_memory()

'''
Save legacy model and evaluation objects
'''

os.chdir(modelfiles)

pickle.dump(testmetric, open('F1_Score.p', 'wb'))
pickle.dump(test_datadict, open('testdata.p', 'wb'))
pickle.dump(optrun, open('optrun.p', 'wb'))
pickle.dump(outer_optrun, open('outer_optrun.p', 'wb'))
pickle.dump(final_params_dict, open('final_hyperparameters.p', 'wb'))
pickle.dump(class_order_dict, open('class_order.p', 'wb'))


'''
Save out-of-fold evaluation tables
'''

os.chdir(figout)
completed_columns = [col for col in colnames if col in test_datadict]

dfeval = pd.DataFrame.from_dict(testmetric, orient='index')

dfeval.to_csv('OOF_Evaluation.csv')

# Compatibility names retained.
# These now contain leakage-safe out-of-fold metrics.
dfeval.to_csv('EvalAll.csv')

all_oof_rows = []

for col in completed_columns:
    current_oof = test_datadict[col].copy()
    current_oof['variable'] = col
    all_oof_rows.append(current_oof)

if len(all_oof_rows) > 0:
    oof_predictions_long = pd.concat(all_oof_rows,axis=0,ignore_index=True)
    pickle.dump(oof_predictions_long,open('OOF_Predictions.p','wb'))
    oof_predictions_long.to_csv('OOF_Predictions.csv',index=False)


'''
Create legacy-style manually coded prediction tables

These use out-of-fold predictions rather than in-sample predictions.
'''

if 'CaseStudy' in test_datadict:
    case_oof = test_datadict['CaseStudy'].copy()
    case_oof = case_oof.rename(columns={'label': 'CaseStudy'})
    dfconfusion_cs = pd.merge(left=trainset,right=case_oof.drop(columns=['row_id']),on=['txt_title','CaseStudy'],how='inner',validate='one_to_one')
    dfconfusion_cs.to_csv('resultsCS.csv',index=False)

dfconfusion = trainset.copy()

for col in completed_columns:
    if col == 'CaseStudy':
        continue

    current_oof = test_datadict[col].copy()
    current_oof = current_oof.drop(columns=['row_id','label'])
    dfconfusion = pd.merge(left=dfconfusion,right=current_oof,on='txt_title',how='left',validate='one_to_one')

dfconfusion.to_csv('resultsAll.csv',index=False)


'''
Create wide out-of-fold probability table

This is the preferred file for analyses involving manually coded papers because
every prediction was produced by a model that did not train on that paper.
'''

oof_wide = None

for col in completed_columns:
    current = test_datadict[col].copy()
    keep_columns = ['txt_title','label',col + '_p']

    for current_column in current.columns:
        if '_prob_' in current_column:
            keep_columns.append(current_column)

    keep_columns = list(dict.fromkeys(keep_columns))
    current = current[keep_columns]
    current = current.rename(columns={'label': col + '_human'})

    if oof_wide is None:
        oof_wide = current
    else:
        oof_wide = pd.merge(left=oof_wide,right=current,on='txt_title',how='outer',validate='one_to_one')

if oof_wide is not None:
    pickle.dump(oof_wide,open('OOF_Predictions_Wide.p','wb'))
    oof_wide.to_csv('OOF_Predictions_Wide.csv',index=False)



'''
Create complete corpus for production prediction
'''

os.chdir(cleandata)

dfcases_unseen = pickle.load(open('extracted_df.p', 'rb'))
dfcases_train = pickle.load(open('extracted_df_train.p', 'rb'))

dfcases = pd.concat([dfcases_unseen, dfcases_train],axis=0,ignore_index=True)

# Preserve one article per title.
dfcases = dfcases.drop_duplicates(subset=['txt_title'],keep='first')

# Use selected GROBID text when available and full article text as fallback.
dfcases['origtext'] = clean_text_column(dfcases['origtext'])
dfcases['text'] = dfcases['text'].where (dfcases['text'].notna(), dfcases['origtext'])
dfcases['text'] = clean_text_column (dfcases['text'])

dfcases[CASESTUDY_TEXT_COLUMN] = clean_text_column (dfcases[CASESTUDY_TEXT_COLUMN])
dfcases = dfcases[dfcases[CASESTUDY_TEXT_COLUMN].str.len() > 0].copy()

dfcases = dfcases.reset_index(drop=True)

print('\n')
print('TOTAL UNIQUE PAPERS FOR PRODUCTION PREDICTION:', len(dfcases))



'''
Predict CaseStudy for every paper
'''

if 'CaseStudy' not in modeldict:

    any_case_model = SetFitModel.from_pretrained(os.path.join( modelfiles, 'AnyCaseStudy_pretrained'), device=device)
    
    any_case_model.model_body.max_seq_length = (MAX_SEQ_LENGTH)

    modeldict['CaseStudy'] = {'AnyCaseStudy': any_case_model}

    if CASESTUDY_METHOD == 'semantic_counter':

        if 'CaseStudy' not in final_params_dict:
            final_parameters_path = os.path.join (modelfiles,'final_hyperparameters.p')

            if not os.path.exists(final_parameters_path):
                raise FileNotFoundError(final_parameters_path)

            final_params_dict = pickle.load (open (final_parameters_path, 'rb'))

        probability_map = final_params_dict['CaseStudy']['CounterProbabilityMap']

        modeldict['CaseStudy']['CounterProbabilityMap'] = probability_map

    elif CASESTUDY_METHOD == 'semantic':

        case_count_model = (SetFitModel.from_pretrained(os.path.join(modelfiles,'CaseCount_pretrained'), device=device))
        case_count_model.model_body.max_seq_length = (MAX_SEQ_LENGTH)

        modeldict['CaseStudy']['CaseCount'] = case_count_model

    else:
        raise ValueError('Invalid CASESTUDY_METHOD.')


dict_preds = {}

case_models = modeldict['CaseStudy']
any_case_model = case_models['AnyCaseStudy']

case_texts = dfcases[CASESTUDY_TEXT_COLUMN].tolist()

if CASESTUDY_METHOD == 'semantic_counter':

    case_predictions = create_case_counter_predictions( any_case_model, case_texts, case_models['CounterProbabilityMap'])

else:

    case_predictions = predict_hierarchical_case_batched(any_case_model, case_models['CaseCount'], case_texts)


for output_column, output_values in case_predictions.items():
    dict_preds[output_column] = output_values
    dfcases[output_column] = output_values

case_mask = dfcases['CaseStudy_p'] == CASESTUDY_POSITIVE_LABEL
case_indices = dfcases.index[case_mask]
case_texts_for_other_models = dfcases.loc[case_indices,OTHER_TEXT_COLUMN].tolist()

print('\n')
print('TOTAL PAPERS:',len(dfcases))
print('PREDICTED NO-CASE PAPERS:',int((dfcases['CaseStudy_p'] == 0).sum()))
print('PREDICTED SINGLE-CASE PAPERS:',int((dfcases['CaseStudy_p'] == 1).sum()))
print('PREDICTED MULTIPLE-CASE PAPERS:',int((dfcases['CaseStudy_p'] == 2).sum()))
print('PAPERS PASSED TO SUBSTANTIVE MODELS:',int(case_mask.sum()))


'''
Predict substantive codes only for hard-predicted case-study papers

The CaseStudy decision remains a hard gate, as in the previous workflow.
Model probabilities remain available for uncertainty-aware downstream analyses.

'''

available_substantive_models = [col for col in colnocase if col in modeldict]
for col in available_substantive_models:
    print('\n')
    print('=' * 80)
    print('PREDICTING:',col)
    print('CASE-STUDY PAPERS:',len(case_indices))
    print('=' * 80)
    
    trained_model = modeldict[col]

    variable_predictions = predict_variable_batched(trained_model, case_texts_for_other_models, col)

    for output_column, output_values in variable_predictions.items():
        complete_values = [np.nan] * len(dfcases)

        if output_column.endswith('_classes'):
            if len(output_values) > 0:
                class_value = output_values[0]
            else:
                class_value = class_order_dict[col]

            for dataframe_index in case_indices:
                complete_values[dataframe_index] = class_value
        else:
            for local_position, dataframe_index in enumerate(case_indices):
                complete_values[dataframe_index] = output_values[local_position]

        dict_preds[output_column] = complete_values
        dfcases[output_column] = complete_values

    clear_memory()


'''
Create final prediction dataframes
'''

dfpreds = dfcases.copy()

# Hard CaseStudy gate preserved.
dfpreds_cases = dfpreds[dfpreds['CaseStudy_p'] == CASESTUDY_POSITIVE_LABEL].copy()

case_probability_column = 'CaseStudy_prob_' + str(CASESTUDY_POSITIVE_LABEL)

if case_probability_column in dfpreds.columns:
    dfpreds['case_probability'] = dfpreds[case_probability_column]
    dfpreds_cases['case_probability'] = dfpreds_cases[case_probability_column]


'''
Save output
'''

os.chdir(mainres)

pickle.dump(dict_preds, open('dict_predicted_codes.p', 'wb'))
pickle.dump(dfpreds, open('predicted_codes.p', 'wb'))
pickle.dump(dfpreds_cases, open('predicted_codes_cases.p', 'wb'))


'''
Save CSV copies
'''

dfpreds.to_csv('predicted_codes.csv', index=False)
dfpreds_cases.to_csv('predicted_codes_cases.csv', index=False)


'''
Save compact probability-only files

Every class probability occupies a separate numeric column.
'''

probability_columns = ['txt_title']

for column_name in dfpreds.columns:
    if '_prob_' in column_name:
        probability_columns.append(column_name)

if 'case_probability' in dfpreds.columns:
    probability_columns.append('case_probability')

dfprobabilities = dfpreds[probability_columns].copy()

pickle.dump(dfprobabilities, open('predicted_probabilities.p', 'wb'))
dfprobabilities.to_csv('predicted_probabilities.csv', index=False)


'''
Final diagnostics
'''

print('\n')
print('=' * 80)
print('NESTED OUT-OF-FOLD EVALUATION')
print('=' * 80)
print(dfeval)

print('\n')
print('=' * 80)
print('CASE-STUDY HARD-PREDICTION DISTRIBUTION')
print('=' * 80)
print(dfpreds['CaseStudy_p'].value_counts(dropna=False).sort_index())

if case_probability_column in dfpreds.columns:
    print('\n')
    print('=' * 80)
    print('CASE-STUDY PROBABILITY DISTRIBUTION')
    print('=' * 80)
    print(dfpreds[case_probability_column].describe())

print('\n')
print('=' * 80)
print('LEGACY FILES SAVED')
print('=' * 80)
print(os.path.join(modelfiles, 'F1_Score.p'))
print(os.path.join(modelfiles, 'testdata.p'))
print(os.path.join(modelfiles, 'optrun.p'))
print(os.path.join(mainres, 'dict_predicted_codes.p'))
print(os.path.join(mainres, 'predicted_codes.p'))
print(os.path.join(mainres, 'predicted_codes_cases.p'))

print('\n')
print('=' * 80)
print('ADDITIONAL FILES SAVED')
print('=' * 80)
print(os.path.join(modelfiles, 'outer_optrun.p'))
print(os.path.join(modelfiles, 'final_hyperparameters.p'))
print(os.path.join(modelfiles, 'class_order.p'))
print(os.path.join(figout, 'OOF_Evaluation.csv'))
print(os.path.join(figout, 'OOF_Predictions.p'))
print(os.path.join(figout, 'OOF_Predictions.csv'))
print(os.path.join(figout, 'OOF_Predictions_Wide.p'))
print(os.path.join(figout, 'OOF_Predictions_Wide.csv'))
print(os.path.join(mainres, 'predicted_probabilities.p'))
print(os.path.join(mainres, 'predicted_probabilities.csv'))

print('\n')
print('=' * 80)
print('DONE')
print('=' * 80)
