#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:42:45 2022

@author: jbaggio

Find locations in papers extraced and cleaned via CSP_ToCleanText and that result in being a case study based on CSP_FewShotLearner

"""

#packages to change directory and load datta from Meta_DataLoad_andPrep.py
import os
#import pickle to save files in case kernel gets stuck or too long to do
import pickle
#defaultdict
from collections import defaultdict
from collections import Counter


#usual suspects
import pandas as pd
import numpy as np

#for geolocation and maps
import geopy
from geopy.extra.rate_limiter import RateLimiter


#packages for figures and graphs
import plotly.express as px
import folium
from folium.plugins import FastMarkerCluster


#for NLP, location, few shot encoder and other utilities
import spacy
sp = spacy.load('en_core_web_trf')
sp.max_length = 3000000



#first set the folders where files are and where dataset should be stored and figures and output saved to
mainres = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis/MainResults'
output = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis'
figout = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Analysis/Figures/CaseLocations'

#load the dataframe with papers, text, prob values and topic probabilities
os.chdir(mainres)
dfcases = pd.read_csv('cases_prob_Topics.csv')

#now check per main topic - here by main topic we mean papers whose probability of a topic is > 0.5
dftops = pd.read_csv('TopicsMain.csv')


"""
now find study locations: via spacy trf, that seem more accurate and does not divide location names compared to other transformers, 
better at least for the purpose of this work, so we use spacy
"""

#do with spacy trf, that is actually better than the finetuned model for this task.
spacloc = defaultdict(dict)
for d in range (0, len(dfcases)):
    print ( str(np.round(d / len(dfcases) * 100, 2)) + '%')
    title = dfcases.txt_title[d]
    tit_toadd = title.replace('.pdf', ' ')
    templist = []
    text = tit_toadd + ' ' + dfcases['text'][d]
    text = text.replace('_',' ')
    # use spacy to extract the entities
    doc = sp(text)
    for ent in doc.ents:    
        # check if entity is equal 'LOC' or 'GPE'
        if ent.label_ in ['LOC', 'GPE']:
            templist.append((ent.text, ent.label_))
    spacloc[title] = templist

#create dictionary of dataframes
cleanloc_all = defaultdict(dict)
for key in spacloc:
    #do one dataframe per case
    tdf = pd.DataFrame(spacloc[key], columns =['location', 'tag'])
    #eliminate some common words not indicating location like province, region, digits etc..
    #tdf = tdf.loc[tdf['tag'] == 'GPE']
    tdf['loc'] = tdf['location'].str.lower()
    tdf['loc'] = tdf['location'].str.replace('the', '')
    tdf['loc'] = tdf['location'].str.replace('province', '')
    tdf['loc'] = tdf['location'].str.replace('region', '')
    tdf['loc'] = tdf['location'].str.replace('\n', '')
    tdf['loc'] = tdf['location'].str.replace('[^a-zA-Z]', ' ', regex = True)
    tdf['loc'] = tdf['location'].str.replace(' s', '')
    tdf['loc'] = tdf['location'].str.replace(' u s ', 'usa')
    tdf['loc'] = tdf['location'].str.replace(' us ', 'usa', regex=True)

    tdf = tdf.drop_duplicates(subset = 'location', keep = 'first')
    tdf = tdf.drop(['tag'], axis=1)
    cleanloc_all[key] = tdf

#save and reload
pickle.dump(cleanloc_all, open('Temp_Location.p', 'wb'))
cleanloc_all = pickle.load(open('Temp_Location.p', 'rb'))

"""
#now we calculate locations of cases studies. Here we limit the location to the first 4 (i.e. case studies that include multiple countries are allowed, but only the first four are taken into account for the actual location maps. 
While there is a limitation, we empirically determined the threshold to avoid over-representation due to comparative tables, that often appear once the case study area is already mentioned). 
"""
cleanloc = {}
noloc = {}
for key in cleanloc_all:
    temp_df = cleanloc_all[key]
    firstloc = 'none'
    secloc = 'none'
    if temp_df.empty:
        firstloc = 'none'
    else:
        firstloc = temp_df.iloc[0]['location']
        if len(temp_df) > 1:
            secloc = temp_df.iloc[1]['location']
        if len(temp_df) > 2:
            thirdloc = temp_df.iloc[2]['location']
        if len(temp_df) > 3:
            fourthloc = temp_df.iloc[3]['location']
    cleanloc[key] = [firstloc, secloc, thirdloc, fourthloc]


pickle.dump(cleanloc, open('Temp_Location_Clean.p', 'wb'))
cleanloc = pickle.load(open('Temp_Location_Clean.p', 'rb'))

locator = geopy.geocoders.Nominatim(user_agent='jj81')
geocode = RateLimiter(locator.geocode, min_delay_seconds=1)

#now get geo places for locations, some may not be possible to locate, but that is ok.
idloc = 0
for key in cleanloc:
    print ( str(np.round(idloc / len(cleanloc) * 100, 2)) + '%')
    tdf = pd.DataFrame(cleanloc[key], columns=['location'])
    if 'places' in tdf:
        pass
    else:
        tdf['places'] = tdf['location'].apply(geocode, timeout = None, language='english')
        cleanloc[key] = tdf       
    idloc += 1

#save as it is long to compute the geocodes
pickle.dump(cleanloc, open('AllDocs_LocationsEN_Spacy_4loc.p', 'wb'))

#to load from saved files
cleanloc  = pickle.load(open('AllDocs_LocationsEN_Spacy_4loc.p', 'rb'))

ctcoords = defaultdict(dict)
for key in cleanloc:
    tdf3 = cleanloc[key] 
    if len (tdf3) > 0:
        tdf3 = tdf3.dropna(subset = 'places')
        if len(tdf3) > 0:
            tdf3['ctr'] = [tdf3['places'][idx][0].split(',')[-1] for idx in tdf3.index]
            tdf3['ctr'] =  tdf3['ctr'].str.strip()
            tdf3 = tdf3.drop_duplicates(subset = ['ctr'], keep = 'first')
    ctcoords[key] = tdf3

# #issues with adding a key that is an empty dictionary, to check
# ctcoords.pop('places')

#now use ge places to get coordinates 
coords = defaultdict(dict)
for key in ctcoords:
    tdf2 = ctcoords[key]
    if not tdf2.empty:        
        tdf2['coordinates'] = tdf2['places'].apply(lambda loc: tuple(loc.point) if loc else (None, None, None))
        tdf2[['latitude', 'longitude', 'altitude']] = pd.DataFrame(tdf2['coordinates'].tolist(), index=tdf2.index)
        tdf2.latitude.isnull().sum()
        tdf2 = tdf2[pd.notnull(tdf2['latitude'])]
    coords[key] = tdf2
    

#check countries table and number of locations per country in english (can not find another way to translate country names)
geodict = {}

locator = geopy.geocoders.Nominatim(user_agent='jb9')
id1 = 0
for key in ctcoords:
    print (key)
    print(id1)
    ct = ctcoords[key].places
    #so we can restart in case of internet issues with the locator prt.
    done = geodict.keys()
    if key in done:
        id1 +=1
        pass
    else:
        geolist = []
        for idx in ct.index:
            ct1 = locator.reverse(ct[idx][1], language = 'en')
            geolist.append(ct1)
        geodict[key] = geolist
        id1 += 1
pickle.dump(geodict, open('LocationsDict.p', 'wb'))


listctr = []
noloc = []
for key in geodict:
    listplace = geodict[key]
    templist = []
    if len(listplace) == 0:
        noloc.append(key)
    else:
        for pl in listplace:
            if pl is None:
                pass
            else:
                #if the two locations are referring to the same country, add only one time.
                filter_work = (pl[0].split(','))
                templist.append(filter_work[-1])
        setlist = set(templist)
        if len (setlist) > 0:
            templist = list(setlist)
            #to check check append and you should get the number of entries = 2030 - len(noloc)
        listctr.extend(templist)
        
#check no-loc and make sure you do not need to append more 
"""
#there may be papers where nominatim can not find the location, we do this by hand, we figure out which papers do not have locations that are possible to geocode and then find them manually
#via google maps and information available on the internet. We then put them in a list and add them to listctr. 
"""

# noloclist_rev = ['manual list for documents not assessed via nominatim']

# i = 0
# for item in noloclist_rev:
#    listctr.append(item)
#    i+=1

#make sure there is no space at the beginning
listctr = [c.strip() for c in listctr]
#count countries
countctr= [Counter(listctr)]
ctrcount = sum(countctr, Counter())

dfctr = pd.DataFrame.from_dict(ctrcount, orient='index').reset_index()
dfctr.columns = ['country','count']
dfctr = dfctr.sort_values(by='count')
dfctr['Ctr N in Articles'] = pd.cut(dfctr['count'], bins = [1, 5, 10,  20, 50, 75, 100, 200, np.inf], right=False) 
dfctr['label'] = dfctr['country'] + ': Exact N = ' + dfctr['count'].astype(str)

dfctr.to_csv('CountryCounts.csv')
pickle.dump(geodict, open('CountryLocations.p', 'wb'))

os.chdir(figout)
#for figure purposes, remove empty keys (that is, those keys for which we could not geolocate anything)
#first generate a list of dataframes
figcoords = []
for key in coords:
    figcoords.append(coords[key])            
#then concatenate the list (append) to generate a single dataframe.
dfcoords = pd.concat(figcoords)

#check map to see where the cases are, on average, located (caveat: this is not hand coded, so mistakes are possible)
folium_map = folium.Map(location=[59.338315,18.089960],
            zoom_start=2,
            tiles= 'CartoDB positron')
FastMarkerCluster(data=list(zip(dfcoords['latitude'].values, dfcoords['longitude'].values))).add_to(folium_map)
folium.LayerControl().add_to(folium_map)
folium_map.save('AllLocsCtr.html')


fig = px.choropleth(data_frame = dfctr, locations='country',
                    locationmode = 'country names',
                    color='Ctr N in Articles', 
                    hover_name='label', # column to add to hover information,
                    color_discrete_sequence=px.colors.qualitative.Prism)
fig.write_html('CountriesLocs.html')
fig.write_image('CountriesLocs.pdf') 

#filter out topic / topic combinations with less than 20 papers (some topic
topvals = list(set(dftops['topic']))
count_topics = dftops['topic'].value_counts()

tokeep = count_topics[count_topics > 50]
red_topvals = tokeep.index

geotopdict = {}
nolocctr = []
for item in red_topvals:
    geo_temp = geodict.copy()
    temp_list = list((dftops[dftops['topic'] == item].txt_title))
    filter_out = set(geo_temp.keys()) - set(temp_list)
    for unwanted_key in filter_out:
        del geo_temp[unwanted_key]
    toplist = []
    for key in geo_temp:
        listplace = geo_temp[key]
        templist = []
        if len(listplace) == 0:
            nolocctr.append(key)
        else:
            for pl in listplace:
                if pl is None:
                    pass
                else:
                    # If the two locations are referring to the same country, add only one time.
                    filter_work = (pl[0].split(','))
                    templist.append(filter_work[-1])
            templist = set(templist)
            toplist.extend(templist)
    # Only add dataframes for graphs for those topics that have at least N locations
    if len(toplist) > 1:
        # Make sure there is no space at the beginning
        toplist = [c.strip() for c in toplist]
        countctr = [Counter(toplist)]
        ctrcount = sum(countctr, Counter())

        dfctr = pd.DataFrame.from_dict(ctrcount, orient='index').reset_index()
        dfctr.columns = ['country','count']
        dfctr = dfctr.sort_values(by='count')
        dfctr['label'] = dfctr['country'] + ': Exact N = ' + dfctr['count'].astype(str)
        geotopdict[item] = dfctr

for key in geotopdict:
    print(key)
    dfctr_top = geotopdict[key]
    dfctr_top['Ctr N in Articles'] = pd.cut(dfctr_top['count'],  bins = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, np.inf], right=False) 
    fig = px.choropleth(data_frame = dfctr_top, locations='country',
                        title = key,
                        locationmode = 'country names',
                        color='Ctr N in Articles', 
                        hover_name='label', # column to add to hover information,
                        color_continuous_scale=px.colors.qualitative.Prism)
    figname = 'CountriesLocs_' + key + '.html'
    os.chdir(figout+'/CtrHTML')
    fig.write_html(figname)
    os.chdir(figout)
    fig.write_image(figname + '.pdf')
