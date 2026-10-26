#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script extracts main topics for papers classified as case studies by the
CSP_FewShot learner. Topic discovery is performed using Embedded Topic
Modelling.

Input:
    predicted_codes_cases.p

Principal outputs:
    fittingETM.p
    EvalETM_Full.pdf
    TopicsEtm.csv
    TopicsEtm_csv
    Doc_Topic_etm_df5.csv
    cases_prob_Topics.csv
    TopicsMain.csv
    CSP_wordcloud_Full.pdf
    CSP_topwords_Full.pdf
    CSP_TfIdfClustermap_Full.jpg
    CSP_TfIdfClustermap_Full.pdf

Important:
    The topic names must be checked against TopicsEtm.csv after every new run.
    Topic ordering may change when the corpus or preprocessing changes.

@author: jbaggio
"""

# File and system operations
import os
import pickle
import re
import random

# Dataset and numerical operations
import pandas as pd
import numpy as np

# NLP and topic modeling
import gensim
import nltk
from embedded_topic_model.utils import preprocessing
from embedded_topic_model.models.etm import ETM
from nltk import FreqDist
from nltk.corpus import stopwords as nltk_stopwords

# Figures
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from wordcloud import STOPWORDS

# TF-IDF and clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


'''
Settings
'''

mainres = '/CommonsSynthProject/NLPSynth/Analysis/MainResults'
output = '/CommonsSynthProject/NLPSynth/Analysis'
figout = '/CommonsSynthProject/NLPSynth/Analysis/Figures/Topics'

os.makedirs(mainres,exist_ok=True)
os.makedirs(output,exist_ok=True)
os.makedirs(figout,exist_ok=True)

RANDOM_SEED = 2019

BIGRAM_MIN_COUNT = 5
BIGRAM_THRESHOLD = 15

TRIGRAM_MIN_COUNT = 5
TRIGRAM_THRESHOLD = 10

ETM_MIN_DF = 5
ETM_MAX_DF = 0.95
ETM_EPOCHS = 300

TOPICS_RANGE = list(range(2,20,2))

#thresholds for dominant topic
SINGLE_TOPIC_THRESHOLD = 0.50
TOPIC_DIFFERENCE_THRESHOLD = 0.1

TFIDF_MAX_DF = 0.95
TFIDF_MIN_DF = 5
TFIDF_MAX_FEATURES = 500

# A full clustermap with thousands of papers is difficult to read and can use
# substantial memory. A reproducible sample is used above this size.
CLUSTERMAP_MAX_DOCUMENTS = 500


'''
Reproducibility
'''

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


'''
NLTK resources
'''

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords',quiet=True)


'''
Functions
'''

# Tokenize one document at a time
def stw(text):
    for document in text:
        yield gensim.utils.simple_preprocess(str(document),deacc=True,min_len=2)


# Identity tokenizer for already-tokenized TF-IDF input
def identityTok(text):
    return text


# Remove stopwords from tokenized documents
def remstop(text,stopword_set):
    return [[word for word in document if word not in stopword_set] for document in text]


# Expand contractions using an escaped regular expression
def expand_contractions(text,contractions_dict,regexpcontractions):
    def replace(match):
        matched_text = match.group(0).lower()
        return contractions_dict.get(matched_text,match.group(0))

    return regexpcontractions.sub(replace,str(text))


# Assign a dominant or combined topic label
def determine_topic(row, topic_columns):

    topic_scores = row[topic_columns].astype(float).sort_values(ascending=False)

    top1 = topic_scores.iloc[0]
    top2 = topic_scores.iloc[1]
    
    if top1 >= SINGLE_TOPIC_THRESHOLD:
        return topic_scores.index[0]

    elif (top1 - top2) >= TOPIC_DIFFERENCE_THRESHOLD:
        return topic_scores.index[0]

    return "Mixed"


# Retrieve document-topic distributions while allowing for minor ETM API differences
def get_document_topic_distribution(model, train_data):
    try:
        distribution = model.get_document_topic_dist()
    except TypeError:
        distribution = model.get_document_topic_dist(train_data)

    try:
        return distribution.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(distribution)


''' 
Patch for count_vectorizer in sklearn. This is necessary as embedded_topic_modelling and newer sklearn versions are not fully compatible.
ETM expects that count vectorizer contains stopwords, but the newer version (or the version i am using, does not produce that, 
so we force the creation of the stop_words_)
'''

from sklearn.feature_extraction.text import CountVectorizer

_original_fit_transform = CountVectorizer.fit_transform

def patched_fit_transform(self, raw_documents, y=None):
    result = _original_fit_transform(self, raw_documents, y)

    if not hasattr(self, "stop_words_"):
        self.stop_words_ = set()

    return result

CountVectorizer.fit_transform = patched_fit_transform


'''
Contractions

Academic papers contain relatively few colloquial contractions. This compact
dictionary covers the common forms while avoiding the malformed and ambiguous
entries in the previous dictionary.
'''

contractions = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how're": "how are",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "mayn't": "may not",
    "may've": "may have",
    "mightn't": "might not",
    "might've": "might have",
    "mustn't": "must not",
    "must've": "must have",
    "needn't": "need not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there'd": "there would",
    "there'll": "there will",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'd": "what did",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "where'd": "where did",
    "where'll": "where will",
    "where're": "where are",
    "where's": "where is",
    "which'd": "which would",
    "which'll": "which will",
    "which're": "which are",
    "which's": "which is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "why'd": "why did",
    "why're": "why are",
    "why's": "why is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "y'all": "you all",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}


'''
Corpus-specific stopwords
'''

specificwords = [
    'commons',
    'common_pool_resources',
    'common_pool_resource',
    'common_pool',
    'environmental_management',
    'community_based_management',
    'community_based',
    'cbnrm',
    'et_al',
    'et',
    'community',
    'communities',
    'resources',
    'resource',
    'management',
    'resource_management',
    'area',
    'data',
    'table',
    'figure',
    'fig',
    'abstract',
    'result',
    'results',
    'discussion',
    'conclusion',
    'conclusions',
    'copyediting',
    'typesetting',
    'proof',
    'century',
    'recent_work',
    'thus',
    'new',
    'scholars_argue',
    'recent_research',
    'literature',
    'review',
    'published',
    'local',
    'using',
    'notof',
    'one',
    'research',
    'project',
    'notand',
    'may',
    'well',
    'time',
    'two',
    'used',
    'number',
    'study',
    'use',
    'many',
    'based',
    'often',
    'example',
    'first',
    'notin',
    'even',
    'part',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'due',
    'others',
    'rather',
    'within',
    'made',
    'case_study'
    'people',
    'government',
    'process',
    'support',
    'important',
    'different',
    'members',
    'group',
    'groups',
    'state',
    'activities',
    'level',
    'found',
    'among',
    'system',
    'systems',
    'approach',
    'analysis'
]

specificwords = {word.lower() for word in specificwords}

stopword_set = set(nltk_stopwords.words('english'))
stopword_set.update(nltk_stopwords.words('spanish'))
stopword_set.update({word.lower() for word in STOPWORDS})
stopword_set.update(specificwords)


'''
Load predicted case-study papers
'''

os.chdir(mainres)

dfcases = pickle.load(open('predicted_codes_cases.p','rb'))

required_columns = ['txt_title','text','origtext']

for required_column in required_columns:
    if required_column not in dfcases.columns:
        raise ValueError(required_column + ' is missing from predicted_codes_cases.p')

print('Original predicted case-study rows:',len(dfcases))

dfcases = dfcases.drop_duplicates(subset=['txt_title'],keep='first').copy()
dfcases['text'] = dfcases['text'].fillna(dfcases['origtext'])
dfcases['origtext'] = dfcases['origtext'].fillna('').astype(str)
dfcases['text'] = dfcases['text'].fillna('').astype(str)

empty_text_mask = dfcases['text'].str.strip().str.len() == 0
dfcases.loc[empty_text_mask,'text'] = dfcases.loc[empty_text_mask,'origtext']

still_empty_mask = dfcases['text'].str.strip().str.len() == 0

if still_empty_mask.any():
    excluded_empty_input = dfcases.loc[still_empty_mask,['txt_title','text','origtext']].copy()
    excluded_empty_input.to_csv(os.path.join(figout,'Excluded_Empty_Input_Documents.csv'),index=False)
    dfcases = dfcases.loc[~still_empty_mask].copy()

dfcases = dfcases.reset_index(drop=True)

print('Unique usable predicted case-study papers:',len(dfcases))


'''
Text cleaning
'''

contractions = {key.lower(): value for key,value in contractions.items()}
contraction_patterns = sorted(contractions.keys(),key=len,reverse=True)
regexpcontractions = re.compile('(' + '|'.join(re.escape(key) for key in contraction_patterns) + ')',flags=re.I)

dfcases['abclean'] = dfcases['text'].astype(str)
dfcases['abclean'] = dfcases['abclean'].str.replace('\u00ad','',regex=False)
dfcases['abclean'] = dfcases['abclean'].str.replace('ﬁ','fi',regex=False)
dfcases['abclean'] = dfcases['abclean'].str.replace('ﬂ','fl',regex=False)
dfcases['abclean'] = dfcases['abclean'].str.replace('Ô¨Å','fi',regex=False)
dfcases['abclean'] = dfcases['abclean'].str.lower()
dfcases['abclean'] = dfcases['abclean'].map(lambda text: expand_contractions(text,contractions,regexpcontractions))
dfcases['abclean'] = dfcases['abclean'].map(lambda text: re.sub(r'_+|[^\w-]+',' ',text))
dfcases['abclean'] = dfcases['abclean'].map(lambda text: re.sub(r'\s+',' ',text).strip())


'''
Tokenization and stopword removal
'''

cleanabst = dfcases['abclean'].tolist()
textclean = list(stw(cleanabst))
textstop = remstop(textclean,stopword_set)


'''
Bigram and trigram modeling

The trigram model is trained on the bigram-transformed corpus.
The final trigram transformation is applied to the bigram-transformed documents.
'''

conwords = gensim.models.phrases.ENGLISH_CONNECTOR_WORDS

bigram = gensim.models.Phrases(textstop,min_count=BIGRAM_MIN_COUNT,threshold=BIGRAM_THRESHOLD,connector_words=conwords)
bimodel = gensim.models.phrases.Phraser(bigram)

bigram_text = [bimodel[document] for document in textstop]

trigram = gensim.models.Phrases(bigram_text,min_count=TRIGRAM_MIN_COUNT,threshold=TRIGRAM_THRESHOLD,connector_words=conwords)
trimodel = gensim.models.phrases.Phraser(trigram)

textetm = [trimodel[document] for document in bigram_text]

# Remove stopwords again after phrase construction.
textetm = remstop(textetm,stopword_set)


'''
Remove documents that have no remaining topic-model tokens

This filtering is applied to dfcases and textetm together so that document-topic
rows remain correctly aligned with article titles.
'''

document_token_counts = np.asarray([len(document) for document in textetm])
valid_document_mask = document_token_counts > 0

print('Documents empty after preprocessing:',int((~valid_document_mask).sum()))

if (~valid_document_mask).any():
    excluded_after_cleaning = dfcases.loc[~valid_document_mask,['txt_title','text','origtext']].copy()
    excluded_after_cleaning['tokens_after_cleaning'] = document_token_counts[~valid_document_mask]
    excluded_after_cleaning.to_csv(os.path.join(figout,'Excluded_After_Topic_Cleaning.csv'),index=False)

dfcases = dfcases.loc[valid_document_mask].reset_index(drop=True)
textetm = [document for document,valid in zip(textetm,valid_document_mask) if valid]

dfcases['topic_token_count'] = [len(document) for document in textetm]

print('Documents retained for topic modeling:',len(dfcases))
print(dfcases['topic_token_count'].describe())


'''
Rebuild one string per document for ETM preprocessing
'''

lststop = [' '.join(document) for document in textetm]


'''
Embedded Topic Modeling

The ETM represents words and topics in a shared embedding space.
The full corpus is used to train the unsupervised topic model.
'''

os.chdir(figout)

voclem,train,test = preprocessing.create_etm_datasets(lststop, min_df=ETM_MIN_DF, max_df=ETM_MAX_DF,train_size=1.0)

print('ETM vocabulary size:',len(voclem))
print('Topic counts evaluated:',TOPICS_RANGE)


'''
Create and fit ETM models
'''

instances = []

for number_of_topics in TOPICS_RANGE:
    etm_instance = ETM(voclem,num_topics=number_of_topics,epochs=ETM_EPOCHS,seed=RANDOM_SEED,debug_mode=False,train_embeddings=True)
    instances.append(etm_instance)

fittingall = []

for model_number,instance in enumerate(instances,start=1):
    print('Model ' + str(model_number) + ' of ' + str(len(instances)))
    print('Number of topics: ' + str(TOPICS_RANGE[model_number - 1]))

    instance.fit(train)
    fittingall.append(instance)

pickle.dump(fittingall,open('fittingETM.p','wb'))


'''
Assess topic coherence, diversity and combined goodness of fit
'''

coher = []
divers = []
gof = []

#give weights, given that papers may use similar vocabulary we want to make sure that coherence is the most important attribute here.
# we give preference to coherence as internal coherence is more important than diversity that decreases with increased number of topics.
# here interpretation of the topic is key, not prediction. 
c_w = 0.8
d_w = 0.2 

for instance in fittingall:
    coh = instance.get_topic_coherence()
    div = instance.get_topic_diversity()
    coher.append(coh)
    divers.append(div)

coher = np.array(coher)
divers = np.array(divers)
gof = coher * c_w + divers * d_w


evalemball = pd.DataFrame({'Coherence': coher,'Diversity': divers,'GoF': gof,'Topics': TOPICS_RANGE})

evalemball.to_csv('EvalETM_Full.csv',index=False)

evalfig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5),sharey=False,sharex=False)

sns.lineplot(data=evalemball,x='Topics',y='Coherence',marker='o',ax=ax1)
sns.lineplot(data=evalemball,x='Topics',y='Diversity',marker='o',ax=ax2)
sns.lineplot(data=evalemball,x='Topics',y='GoF',marker='o',ax=ax3)

ax1.set_title('Topic Coherence')
ax2.set_title('Topic Diversity')
ax3.set_title('GoF')

evalfig.tight_layout()
evalfig.savefig('EvalETM_Full.pdf')
plt.close(evalfig)


'''
Select the ETM model

The historical analysis used eight topics. 
'''

SELECTED_NUM_TOPICS = 6


if SELECTED_NUM_TOPICS not in TOPICS_RANGE:
    raise ValueError('SELECTED_NUM_TOPICS is not present in TOPICS_RANGE.')

selected_index = TOPICS_RANGE.index(SELECTED_NUM_TOPICS)
selected_model = fittingall[selected_index]

print('Selected number of topics:',SELECTED_NUM_TOPICS)


'''
Extract and save topic terms
'''

topics = selected_model.get_topics(20)
dftopics = pd.DataFrame(topics).T

# Correct filename.
dftopics.to_csv('Topics6Etm.csv',index=False)



'''
Assign manual names to selected topics

Inspect TopicsEtm.csv before accepting this topic order after every retraining.
'''

# These names must be checked against TopicsEtm.csv after model fitting.
TOPIC_NAMES = ['Biodiversity Conservation','Forestry and Livelihoods','Institutions and Governance','Water Governance','Fisheries','Land Rights']

if len(TOPIC_NAMES) != SELECTED_NUM_TOPICS:
    raise ValueError('The number of manual topic names does not match SELECTED_NUM_TOPICS.')


'''
Calculate document-topic distributions
'''

distrdoc = get_document_topic_distribution(selected_model,train)
ddoc = pd.DataFrame(distrdoc)

if ddoc.shape[1] != SELECTED_NUM_TOPICS:
    raise ValueError('The number of document-topic columns does not match SELECTED_NUM_TOPICS.')

if len(ddoc) != len(dfcases):
    raise ValueError('Document-topic rows do not match the number of retained documents.')

ddoc.columns = TOPIC_NAMES
ddoc.to_csv('Doc_Topic_etm.csv',index=False)


'''
Plot topic-distribution histograms
'''

histogram_axes = ddoc.hist(figsize=(14,10))
plt.tight_layout()
plt.savefig('Tdistra.pdf',bbox_inches='tight')
plt.close()


'''
Merge topic probabilities with article information
'''

dfcases = pd.concat([dfcases.reset_index(drop=True),ddoc.reset_index(drop=True)],axis=1)

dfcases.to_csv('cases_prob_Topics.csv',index=False)

# Preserve a pickle containing all article data and topic probabilities.
pickle.dump(dfcases,open('cases_prob_Topics.p','wb'))


'''
Assign dominant or combined topic names to documents
'''

dftops = dfcases[['txt_title'] + TOPIC_NAMES].copy()

dftops['topic'] = dftops.apply(lambda row: determine_topic(row,TOPIC_NAMES),axis=1)
dftops['argmax'] = dftops[TOPIC_NAMES].idxmax(axis=1)

dftops.to_csv('TopicsMain.csv',index=False)

pickle.dump(dftops,open('TopicsMain.p','wb'))


'''
Word cloud
'''

tokall = []

for document in textetm:
    tokall.extend(document)

if len(tokall) == 0:
    raise ValueError('No tokens remain for word-frequency analysis.')

fdist = FreqDist(tokall)

wordcloud = WordCloud(background_color='white',max_words=100,contour_width=3,contour_color='black',collocations=False,colormap='tab10')

cloud = wordcloud.generate_from_frequencies(fdist)

fig,ax = plt.subplots(1,1,figsize=(10,10),sharey=True,sharex=True)
ax.imshow(cloud)
ax.set_title('Most Frequent Words')
ax.axis('off')
fig.tight_layout()
fig.savefig('CSP_wordcloud_Full.pdf')
plt.close(fig)


'''
Most common words
'''

number_of_words_saved = 100
number_of_words_plotted = 30

k_words = fdist.most_common(number_of_words_saved)
topkw = k_words[:number_of_words_plotted]

dftop = pd.DataFrame(topkw,columns=['words','freq'])
dftop['frac'] = dftop['freq'] / len(tokall)

wordim = dftop.copy()
wordfreq = k_words

dftop.to_csv('CSP_topwords_Full.csv',index=False)
pickle.dump(wordfreq,open('CSP_wordfreq_Full.p','wb'))

fbar,ax = plt.subplots(1,1,figsize=(20,10),sharey=True,sharex=True)

sns.barplot(data=dftop,x='words',y='frac',color='steelblue',ax=ax)

ax.set_title('30 Most Common Words',fontsize=30,color='black')
ax.tick_params(axis='x',which='major',labelsize=20,rotation=90)
ax.set_ylabel('Fraction of Words',color='black',fontsize=30)
ax.set_xlabel('Words',color='black',fontsize=30)

plt.tight_layout()
fbar.savefig('CSP_topwords_Full.pdf')
plt.close(fbar)


'''
TF-IDF and cosine similarity

The documents are already represented as token lists, so identityTok is used
as both tokenizer and preprocessor.
'''

tfidveclem = TfidfVectorizer(analyzer='word',tokenizer=identityTok,preprocessor=identityTok,token_pattern=None,max_df=TFIDF_MAX_DF,max_features=TFIDF_MAX_FEATURES,min_df=TFIDF_MIN_DF,stop_words=None,use_idf=True)

tfidmatlem = tfidveclem.fit_transform(textetm)

simillem = cosine_similarity(tfidmatlem)

# Exclude self-similarity values from average document similarity.
if len(simillem) > 1:
    upper_triangle = simillem[np.triu_indices_from(simillem,k=1)]
    avgsimillem = float(upper_triangle.mean())
else:
    avgsimillem = np.nan

avgcslem = avgsimillem

print('Average pairwise cosine similarity:',avgcslem)

with open('Average_Cosine_Similarity.txt','w') as similarity_file:
    similarity_file.write(str(avgcslem))


'''
TF-IDF hierarchical clustering

A random but reproducible sample is used when the corpus is larger than
CLUSTERMAP_MAX_DOCUMENTS. The complete document-topic output remains unchanged.
'''

if len(dfcases) > CLUSTERMAP_MAX_DOCUMENTS:
    random_generator = np.random.RandomState(RANDOM_SEED)
    selected_positions = np.sort(random_generator.choice(np.arange(len(dfcases)),size=CLUSTERMAP_MAX_DOCUMENTS,replace=False))
else:
    selected_positions = np.arange(len(dfcases))

sampled_titles = dfcases.iloc[selected_positions]['txt_title'].tolist()
sampled_similarity = simillem[np.ix_(selected_positions,selected_positions)]

sampled_distance = 1 - sampled_similarity
np.fill_diagonal(sampled_distance,0)
sampled_distance = np.clip(sampled_distance,0,1)

clusdf = pd.DataFrame(sampled_distance,index=sampled_titles,columns=sampled_titles)

if len(clusdf) > 1:
    cosdist = squareform(sampled_distance,checks=False)
    linklem = linkage(cosdist,method='average')

    sns.set(font_scale=0.6)

    fclus = sns.clustermap(clusdf,cmap='RdYlBu',figsize=(20,30),row_linkage=linklem,col_linkage=linklem,xticklabels=False,yticklabels=False)

    fclus.savefig('CSP_TfIdfClustermap_Full.jpg',dpi=300)
    fclus.savefig('CSP_TfIdfClustermap_Full.pdf')
    plt.close(fclus.fig)

pd.DataFrame({'txt_title': sampled_titles}).to_csv('CSP_TfIdfClustermap_Documents.csv',index=False)


'''
Save preprocessing objects for reproducibility
'''

preprocessing_objects = {
    'random_seed': RANDOM_SEED,
    'bigram_min_count': BIGRAM_MIN_COUNT,
    'bigram_threshold': BIGRAM_THRESHOLD,
    'trigram_min_count': TRIGRAM_MIN_COUNT,
    'trigram_threshold': TRIGRAM_THRESHOLD,
    'etm_min_df': ETM_MIN_DF,
    'etm_max_df': ETM_MAX_DF,
    'etm_epochs': ETM_EPOCHS,
    'topics_range': TOPICS_RANGE,
    'selected_num_topics': SELECTED_NUM_TOPICS,
    'topic_names': TOPIC_NAMES,
    'stopwords': sorted(stopword_set),
    'bigram_model': bimodel,
    'trigram_model': trimodel,
    'vocabulary': voclem
}

pickle.dump(preprocessing_objects,open('ETM_Preprocessing_Objects.p','wb'))


'''
Final summary
'''

print('\n')
print('=' * 80)
print('TOPIC MODELING SUMMARY')
print('=' * 80)
print('Predicted case-study papers used:',len(dfcases))
print('Vocabulary size:',len(voclem))
print('Selected number of topics:',SELECTED_NUM_TOPICS)
print('Average pairwise TF-IDF cosine similarity:',avgcslem)
print('\n')
print('Topic-model evaluation:')
print(evalemball)
print('\n')
print('Dominant and combined topic counts:')
print(dftops['topic'].value_counts(dropna=False))
print('\n')
print('=' * 80)
print('DONE')
print('=' * 80)



