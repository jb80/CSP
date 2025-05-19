#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:28:58 2023

This script extracts main topics for all the full texts that are considered case studies predicted via the CSP_FewShot
Learner script. It does so via embedded topic modelling

@author: jbaggio
"""

#to merge all files (starting always from scratch)
#packages to change directory and load datta from Meta_DataLoad_andPrep.py
import os
#to select all files in a directory
import glob
#import pickle to save files in case kernel gets stuck or too long to do
import pickle
#import for iterating
import itertools

#usual suspects
import pandas as pd
import numpy as np
from collections import defaultdict

#packages for figures and graphs
import seaborn as sns

#Import for natural language processing and topic modelling
import re
#import gensim
import gensim
#import nltk and nltk sub-models
import nltk
#now create a list of english words
nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')
#import spacy and load spacy pre-trained model on englis core web for lemmatization. 
import spacy
sp = spacy.load('en_core_web_trf')
sp.max_length = 1500000 
#import embedded topic modelling. Embedding is basically a wrapper for gensimWord2Vec.
#preprocessing allows to define vocabulary, training set (here set at 100%), and min/max frequency to filter words
from embedded_topic_model.utils import preprocessing
#ETM performs the actual topic modelling on embeddings (see reference Dieng et al. 2019)
from embedded_topic_model.models.etm import ETM


#import specific NLP routines for figure and data analysis
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from nltk import FreqDist

#define functions for NLP and Topic Modelling
#function that tokenizes a text
def stw(text):
    for sentence in text:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True, min_len=2))  # deacc=True removes punctuations

#same as stopwords, but here we limit to eliminate words that are abbreviation of little meaning in the text (3 letterwords or less)
def keepeng (text, corpkeep):
    #return [word for word in text if word not in stopwords]
   return [[word for word in doc if word  in corpkeep] for doc in text]

#for term frequency inverse document frequency (TFIDF)
def identityTok(text):
   return text

# Regular expression for finding and expanding contractions
def expand_contractions(text,contractions_dict, regexpcontractions):
    def replace(match):
        return contractions_dict[match.group(0)]
    return regexpcontractions.sub(replace, text)


#here we eliminate common 2 and 3 words from the list we want to avoid in the text
def remcommons (wordlist, select):
    return [word for word in wordlist if word not in select]

#same as stopwords, but here we limit to eliminate words that are abbreviation of little meaning in the text 
# including mostly < 3 letter words
def remnosense (text, nos):
    #return [word for word in text if word not in stopwords]
   return [[word for word in doc if word not in nos]for doc in text]

#remove stopwords from wordcloud and frequency, but not for ETM or TFIDF.
def remstop(text):
   return [[word for word in doc if word not in stopwords]for doc in text]

#functions to lemmatize, one does it word for word, the other one creates an sp object for the whole document (represented by a single string)
def lemmatize_plus_word(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN']): 
    tout = []
    for word in text:
        doc = sp(" ".join(word)) 
        tout.append([token.lemma_ for token in doc if token.pos_  in allowed_postags])
    return tout

def lemmatize_plus(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'PROPN']): 
    tout = []
    doc = sp(text) 
    tout.append([token.lemma_ for token in doc if token.pos_  in allowed_postags])
    return tout

# Define a custom function to determine the topic
    
def determine_topic(row, threshold=0.2, thresingle=0.5, threshdiff=0.25):
    topics_single = []
    topics_multi = {}
    total_score = 0
    
    for column in row.index[1:]:
        if row[column] >= thresingle:
            topics_single.append(column)
            total_score += row[column]
        elif row[column] >= threshold:
            topics_multi[column] = row[column]
            total_score += row[column]
    
    sorted_topics_multi = sorted(topics_multi.items(), key=lambda x: x[1], reverse=True)
    if len(topics_single) < 1 and len(topics_multi) < 1:
        return "Mixed"
    if len(topics_single) > 0:
        return topics_single[0]
    if len(sorted_topics_multi) > 1:
        if sorted_topics_multi[0][1] - sorted_topics_multi[1][1] > threshdiff:
            return sorted_topics_multi[0][0]
    if len(sorted_topics_multi) == 1 and len(topics_single) < 1:
        return sorted_topics_multi[0][0]
    else:
        return " & ".join([topic[0] for topic in sorted_topics_multi])



#file and save result locations
mainres = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis/MainResults'
output = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis'
figout = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis/Figures/Topics'

#contractions for the english language including colloquials and archaics.
contractions = {"a'ight ":"alright", "ain't ":"is not", "amn't":"am not", "n' ":"and", "arencha ":"aren’t you ", 
                "aren't":"are not", "‘bout ":"about", "can't":"cannot", "cap’n ":"captain", "cause ":"because", 
                "’cept ":"except", "could've":"could have", "couldn't":"could not", "couldn't've":"could not have", 
                "cuppa":"cup of", "daren't":"dare not ", "daresn't":"dare not", "dasn't":"dare not", "didn't":"did not", 
                "doesn't":"does not", "don't":"do not", "dunno ":"do not know", "d'ye ":"do you", "d'ya ":"do you", 
                "e'en ":"even", "e'er ":"ever", "em ":"them", "everybody's":"everybody is", "everyone's":"everyone is", 
                "finna ":"fixing to", "fo’c’sle ":"forecastle", "’gainst ":"against", "g'day ":"good day", 
                "gimme ":"give me", "giv'n ":"given", "gi'z ":"give us", "gonna ":"going to", "gon't ":"go not ", 
                "gotta ":"got to", "hadn't":"had not", "had've":"had have", "hasn't":"has not", "haven't":"have not", 
                "he'd":"he had ", "he'll":"he shall ", "helluva ":"hell of a", "he's":"he has ", "here's":"here is", 
                "how'd ":"how did ", "howdy ":"how do you do ", "how'll":"how will", "how're":"how are", 
                "how's":"how has ", "i'd":"i had ", "i'd've":"i would have", "i'd'nt":"i would not",
                "i'd'nt've":"i would not have", "if'n ":"if and when", "i'll":"i shall ", 
                "i'm":"i am", "imma ":"i am about to", "i'm'o ":"i am going to", "innit ":"it is not", 
                "ion ":" i do not", "i've":"i have", "isn't":"is not", "it'd":"it would", "it'll":"it shall ", 
                "it's":"it has ", "Idunno ":"I do not know", "kinda ":"kind of", "let's":"let us", 
                "loven't ":"love not ", "ma'am (formal)":"madam", "mayn't":"may not", "may've":"may have", 
                "methinks ":"I think", "mightn't":"might not", "might've":"might have", "mustn't":"must not", 
                "mustn't've":"must not have", "must've":"must have", "‘neath ":"beneath", "needn't":"need not", 
                "nal ":"and all", "ne'er ":"never", "o'clock":"of the clock", "o'er":"over", "ol'":"old", 
                "ought've":"ought have", "oughtn't":"ought not", "oughtn't've":"ought not have", "‘round":"around", 
                "'s":"is", "shalln't":"shall not ", "shan't":"shall not", "she'd":"she had ", "she'll":"she shall ", 
                "she's":"she has ", "should've":"should have", "shouldn't":"should not", "shouldn't've ":"should not have", 
                "somebody's":"somebody has ", "someone's":"someone has ", "something's":"something has ", 
                "so're ":"so are ", "so’s ":"so is ", "so’ve ":"so have", "that'll":"that shall ", "that're ":"that are",
                "that's":"that has ", "that'd":"that would ", "there'd":"there had ", "there'll":"there shall ", 
                "there're":"there are", "there's":"there has ", "these're":"these are", "these've":"these have", 
                "they'd":"they had ", "they'll":"they shall ", "they're":"they are ", "they've":"they have", 
                "this's":"this has ", "those're ":"those are", "those've ":"those have", "thout ":"without", 
                "’til ":"until", "tis ":"it is", "to've ":"to have", "twas ":"it was", "tween ":"between", 
                "twere ":"it were", "w'all":"we all", "w'at":"we at", "wanna":"want to", "wasn't":"was not",
                "we'd":"we had ", "we'd've":"we would have", "we'll":"we shall ", "we're":"we are", "we've":"we have", 
                "weren't":"were not", "whatcha":"what are you", "what'd":"what did", "what'll":"what shall ", 
                "what're":"what are", "what's":"what has ", "what've":"what have", "when's":"when has ", 
                "where'd":"where did", "where'll":"where shall ", "where're":"where are", "where's":"where has ", 
                "where've":"where have", "which'd":"which had ", "which'll":"which shall ", "which're":"which are", 
                "which's":"which has ", "which've":"which have", "who'd":"who would ", "who'd've":"who would have", 
                "who'll":"who shall ", "who're":"who are", "who's":"who has ", "who've":"who have", "why'd":"why did", 
                "why're":"why are", "why's":"why has ", "willn't":"will not", "won't":"will not", "wonnot":"will not", 
                "would've":"would have", "wouldn't":"would not", "wouldn't've":"would not have", 
                "y'ain't":"you are not ", "y'all":"you all ", "y'all'd've":"you all would have", 
                "y'all'd'n't've":"you all would not have", "y'all're":"you all are", "y'all'ren't":"you all are not", 
                "y'at ":"you at", "yes’m":"yes madam", "y'know":"you know", "yessir":"yes sir", "you'd":"you had ",
                "you'll":"you shall ", "you're":"you are", "you've":"you have", "when'd":"when did", "willn't":"will not"}

#these are words that do not add to topics and are fairly common in the corpus (we also include terms that are included in the article search string)
specificwords = ['commons','common_pool_resources', 'common_pool_resource', 'environmental_management', 'community_based_management', 'CBNRM', 'et_al', 'et', 'community', 
                 'resources', 'resource', 'management', 'resource_management', 'area', 'data','table', 'figure', 'fig', 'community_based', 'abstract', 
                 'result', 'results', 'discussion', 'conclusion', 'conclusions', 'copyediting', 'typesetting', 'proof', 'th', 'century',
                 'recent_work' , 'thus','new', 'scholars_argue', 'recent_research', 'literature', 'review', 'published', 'local', 'using', 
                 'notof', 'one', 'research', 'project', 'notand', 'may', 'well', 'time', 'two', 'used', 'number', 'study', 'study', 'use', 'many', 'based', 'often','example', 'first', 'notin', 'even', 'part', 'three', 'one','four','five','six','seven', 'eight', 'due',
                 'others' ,'rather' , 'within', 'made', 'case_study']

#Generate stopwords list from nltk 
stopwords = nltk.corpus.stopwords.words('english')
stopspan = nltk.corpus.stopwords.words('spanish')
stopwords.extend(stopspan)
from wordcloud import STOPWORDS
stopwords.extend(list(STOPWORDS))
#now remove duplicates in the list.
stopwords = list(set(stopwords))

os.chdir(mainres)
dfcases = pickle.load(open('predicted_codes_cases.p', 'rb'))
#case 300 has issues with cleaning (no words left after cleaning)
dfcases.iloc[300, dfcases.columns.get_loc('text')] = dfcases.iloc[300, dfcases.columns.get_loc('origtext')]

#start cleaning
#expand contractions
regexpcontractions=re.compile('(%s)' % '|'.join(contractions.keys()))
dfcases['abclean']=dfcases['text'].map(lambda x:expand_contractions(x, contractions, regexpcontractions))
#Eliminate non alphanumeric characters and lower case all, change the character for fi due to tesseract
dfcases['abclean'] = dfcases['abclean'].map(lambda x: re.sub(r'_+|[^\w-]+', ' ', x))
dfcases['abclean'] = dfcases['abclean'].map(lambda x: x.lower())
dfcases['abclean'] = dfcases['abclean'].map(lambda x: re.sub(r'_Ô¨Å', 'fi', x))
dfcases['abclean'] = dfcases['abclean'].map(lambda x: re.sub(r'ﬁ', 'fi', x)) 

cleanabst = dfcases.abclean.values.tolist()
textclean = list(stw(cleanabst))
textstop = remstop(textclean)

#Build the bigram and trigram models. Here we tuned the threshold differently for bigram and trigrams.
conwords = gensim.models.phrases.ENGLISH_CONNECTOR_WORDS
bigram = gensim.models.Phrases(textstop, min_count=5, threshold=15, connector_words = conwords)
bimodel = gensim.models.phrases.Phraser(bigram)
trigram = gensim.models.Phrases(bimodel[textstop], min_count=5, threshold=10, connector_words = conwords)
trimodel = gensim.models.phrases.Phraser(trigram)
#make bigrams and trigrams based on the bigram/trigram model defined above
textetm = [bimodel[doc] for doc in textstop]
textetm = [trimodel[doc] for doc in textstop]

stopwords.extend(list(specificwords))
textetm = remstop(textetm)

# #we lemmatize here to reduce vocabulary but also to better have a sense of the embedeed topics
# textetm = lemmatize_plus(textetm)

#create text for topic modelling rebuilding one string per document. For word2vec we remove stopwords
lststop = []
for i in textetm:
    doc = i
    joined = " ".join(doc)
    lststop.append(joined) 


   
"""
EMBEDDED TOPIC MODELLING
here we base this on Dieng et al. 2019, Topic modeling in embedding spaces, arXiv
In a nutshell we perform topic modelling via LDA leveraging embeddings (word2vec skip gram in this case).
ETM leverages embedding given a document rather than a window for words and seems to work better.

Coherence for ETM is calculated as the average NPMI. we assume where there is no possiblity to calculate NPMI the NPMI = 0 (independent)

"""

os.chdir(figout)

#make sure same random seed, just in case
import random
rseed = 2019 #original seed is 2019
random.seed(rseed) 

# Preprocessing the dataset, generate the vocabulary based on pre-processed text (not lemmatized). 
#Note that we eliminate infrequent words (that appear in less than 1% of the corpus documents, and frequent words, that appear more than in 95% of the corpus documents)
voclem, train, test = preprocessing.create_etm_datasets(
                        lststop, 
                        min_df=5, 
                        max_df=0.95, 
                        train_size=1, 
                        )

#define topic range 
topics_range = range(2, 20, 2)
# Training ETM instances training embeddings together with ETM parameters. All parameters not shown here are kept as default (starting rho = 300 for example)
instances = []
for k in topics_range :
    etm_instance = ETM(
                voclem,
                num_topics= k,
                epochs=300,
                seed = rseed,
                debug_mode=False, # may be good, the first time, to see the log
                train_embeddings=True, 
                )
    instances.append(etm_instance)

#fit different etm instances with different number of topics to the data
fittingall = []
num = 1
for inst in instances:
    print ('complete % = ' + str(num / len(instances) * 100))
    fittingall.append(inst.fit(train))
    num +=1
    
pickle.dump(fittingall, open('fittingETM.p', 'wb'))

#assess goodness of fit of the models as product of diversity and coherence, also assess coherence and diversity separately
#here coherence is the Mutual information, not U-Mass, and diversity is diversity of words in each topic.
gof = []
coher = []
divers = []
for inst in instances:
    coher.append(inst.get_topic_coherence())
    divers.append(inst.get_topic_diversity())
    val = inst.get_topic_coherence() * inst.get_topic_diversity()
    gof.append(val)

evalemb = pd.DataFrame([coher, divers, gof]).T
evalemb.columns = ['Coherence', 'Diversity','GoF']
tops = np.array(topics_range).tolist()
evalemb['Topics'] = tops
evalemball = evalemb.copy()

"""
ETM All results (evaluation, topic words and topic distribution) with embeddings calculated with topics and embeddings from longformer legal
"""

evalfig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (15, 5), sharey=False, sharex=False)
sns.lineplot(x = evalemball.Topics, y = evalemball.Coherence, ax=ax1)
sns.lineplot(x = evalemball.Topics, y = evalemball.Diversity, ax=ax2)
sns.lineplot(x = evalemball.Topics, y = evalemball.GoF, ax=ax3)
evalfig.tight_layout()
evalfig.savefig('EvalETM_Full.pdf')

"""
now assess results and select fitting based on coherence and diversity metrics
elicit topics from the best coherence/diversity metrics. Here we choose the first coherence peak, and check for diversity  of topics to be > 0.5 
We can also look at the highest peak in GoF (coherence * diversity)
"""

#overall better fit (first peak on gof)
topics = fittingall[3].get_topics(20)
dftopics = pd.DataFrame(topics).T
dftopics.to_csv('TopicsEtm_csv')

#now give, manually, names to topics for columns based on dftopics
#Development,Land Rights, Forestry, Biodiversity, Institutions, Pastoralism, Water, Fisheries 
topnames = ['Development', 'Land Rights', 'Forestry', 'Biodiversity', 'Institutions', 'Livestock', 'Water', 'Fisheries']

#assign topics to documents in corpus based on previous evaluation
distrdoc = ETM.get_document_topic_dist(fittingall[3])
ddoc = pd.DataFrame(np.array(distrdoc))
ddoc.columns = topnames
ddoc.to_csv('Doc_Topic_etm_df5.csv')
ddoc.hist()
plt.savefig('Tdistra.pdf', bbox_inches='tight')

#merge topic names and probs with dfcases

dfcases = dfcases.reset_index()
dfcases = pd.concat([dfcases, ddoc], axis=1)
dfcases.to_csv('cases_prob_Topics.csv')   

dftops = dfcases.copy()
dftops = dftops[['txt_title', 'Development', 'Land Rights', 'Forestry', 'Biodiversity', 'Institutions', 'Livestock', 'Water', 'Fisheries']]

dftops['topic'] = dftops.apply(determine_topic, axis=1)
dftops.to_csv('TopicsMain.csv')   

    
"""
Start Figures
"""
os.chdir(figout)

wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='black',
                      collocations=False, colormap='tab10')
tokall = []
for w in textetm:
    tokall.extend(w)

fdist = FreqDist(tokall)
cloud = wordcloud.generate_from_frequencies(fdist)

fig, ax = plt.subplots(1,1, figsize = (10, 10), sharey=True, sharex=True)
plt.imshow(cloud) 
plt.title('Most Frequent Words')
plt.axis ('off')
fig.savefig('CSP_wordcloud_Full.pdf')


"""
Most common words graphs (basically a bar graph of the 30 most common words showing up in wordcloud)
"""

# Now assess the most common 30 words
#get top 20 words and assess frequencies
k   = 100 #overall words we want in th list... the first N words
ttk = 30 # top N we want to put in the histogram
k_words = fdist.most_common(k)
topkw = k_words[0:ttk]
dftop = pd.DataFrame(topkw, columns=['words','freq'])
dftop['frac']= dftop['freq'] / len(tokall)    
wordim = dftop
#store the most common 100 words per time-period
wordfreq = k_words

#Figure for top words 
fbar, ax = plt.subplots(1,1, figsize = (20, 10), sharey=True, sharex=True)
sns.barplot(x = dftop.words, y = dftop.frac, palette='Blues_r')
plt.title('30 Most Common Words', fontsize=30, color ='black')
plt.xticks(rotation=90, ha='center')
plt.tick_params(axis='x', which='major', labelsize=28)
plt.ylabel('Fraction of Words', color = 'black', fontsize = 36)
plt.xlabel('Words', color = 'black', fontsize = 36)
plt.tight_layout()
fbar.savefig('CSP_topwords_Full.pdf')

"""
TF-IDF on lemmatize words

General parameters
Use vectorization of tokens to assess cosine similarity between papers based on word tokenization.
define vectorizer parameters
max_df = frequency of word affter which it carries little meaning as it is almost always present
min_df = frequency of word in documents to be considered, same as voclem for comparative purposes.
i use the same stopwords already used
"""

tfidveclem  = TfidfVectorizer(analyzer='word',tokenizer=identityTok, preprocessor=identityTok,
                              token_pattern=None, max_df=0.95, max_features=500, min_df=5, stop_words=None, use_idf=True)


tfidmatlem  = tfidveclem.fit_transform(list(textetm))
#Cosine similarity 
simillem  = cosine_similarity(tfidmatlem)
#Calculate average similarity 
avgsimillem  = np.mean(simillem)
#Store average similarity scores
avgcslem = avgsimillem
#Clustering of papers based on TF-IDF on stemmed and lemmatized matrix.
#first convert similarity to distance:
distlem  = 1 - simillem
np.fill_diagonal(distlem, 0)
#floating point inaccuracies give negative distances so we clip the distance matrix
distlem = np.clip(distlem, 0,1)

#Now convert to dataframe so we can add counties names on i and j for plotting
clusdf = pd.DataFrame(distlem, index = dfcases.index, columns = dfcases.index)

#put data in the right format for linkage function precomputed from cosine similarity
cosdist = squareform(distlem)
linklem = linkage(cosdist, 'average')

#Figure for tfidf clustermap based on cosine distance
sns.set(font_scale=1)
fclus = sns.clustermap(clusdf, cmap="RdYlBu", figsize=(20,30), row_linkage=linklem, col_linkage=linklem)
fclus.savefig('CSP_TfIdfClustermap_Full.jpg')
fclus.savefig('CSP_TfIdfClustermap_Full.pdf')                
              

"""

#lemmatize for figures if needed (not used just to check in the past potential big differences, not many found)
docnum = 0    
textlem_temp = []
for text_temp in lststop:
    docnum += 1
    print(docnum)
    textlem_temp.append(lemmatize_plus(text_temp))
#change from list of list of list to list of lists
textlem = []       
for item in textlem_temp:
    textlem.append(list(itertools.chain(*item)))

pickle.dump(textlem, open('textlem.p', 'wb'))


lststop_lem = []
for i in textlem:
    doc = i
    joined = " ".join(doc)
    lststop_lem.append(joined) 


    
#wordcloud with lemmatized text keeping only postags as defined in the lemmatize function (i.e. noun, or noun, adv, adj, verb etc.)
wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='black',
                      collocations=False, colormap='tab10')
lemmas =[]
for w in textlem:
    lemmas.extend(w)

lfdist = FreqDist(lemmas)
lcloud = wordcloud.generate_from_frequencies(lfdist)

fig, ax = plt.subplots(1,1, figsize = (10, 10), sharey=True, sharex=True)
plt.imshow(lcloud) 
plt.title('Most Frequent Words')
plt.axis ('off')
fig.savefig('CSP_wordcloud_Lemmas.pdf')

# Now assess the most common 30 words but on lemmatized as per lemmatized function used
#get top 20 words and assess frequencies
k   = 100 #overall words we want in th list... the first N words
ttk = 30 # top N we want to put in the histogram
k_words = lfdist.most_common(k)
topkw = k_words[0:ttk]
dftop = pd.DataFrame(topkw, columns=['words','freq'])
dftop['frac']= dftop['freq'] / len(tokall)    
wordim = dftop
#store the most common 100 words per time-period
wordfreq = k_words

#Figure for top words 
fbar, ax = plt.subplots(1,1, figsize = (20, 10), sharey=True, sharex=True)
sns.barplot(x = dftop.words, y = dftop.frac, palette='Blues_r')
plt.title('30 Most Common Words', fontsize=30, color ='black')
plt.xticks(rotation=90, ha='center')
plt.tick_params(axis='x', which='major', labelsize=28)
plt.ylabel('Fraction of Words', color = 'black', fontsize = 36)
plt.xlabel('Lemmas', color = 'black', fontsize = 36)
plt.tight_layout()
fbar.savefig('CSP_topwords_Lemmas.pdf')

#now do the same for lemmatized words
tfidmatlem  = tfidveclem.fit_transform(textlem)
#Cosine similarity 
simillem  = cosine_similarity(tfidmatlem)
#Calculate average similarity 
avgsimillem  = np.mean(simillem)
#Store average similarity scores
avgcslem = avgsimillem
#Clustering of papers based on TF-IDF on stemmed and lemmatized matrix.
#first convert similarity to distance:
distlem  = 1 - simillem
np.fill_diagonal(distlem, 0)
#floating point inaccuracies may give negative distances so we clip the distance matrix
distlem = np.clip(distlem, 0,1)
#Now convert to dataframe so we can add counties names on i and j for plotting
clusdf = pd.DataFrame(distlem, index = dfcases.index, columns = dfcases.index)

#put data in the right format for linkage function precomputed from cosine similarity
cosdist = squareform(distlem)
linklem = linkage(cosdist, 'average')

#Figure for tfidf clustermap based on cosine distance
sns.set(font_scale=1)
fclus = sns.clustermap(clusdf, cmap="RdYlBu", figsize=(20,30), row_linkage=linklem, col_linkage=linklem)
fclus.savefig('CSP_TfIdfClustermap_Lemmas.pdf')  
fclus.savefig('CSP_TfIdfClustermap_Lemmas.jpg')  

"""

