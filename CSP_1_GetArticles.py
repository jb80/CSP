#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:16:58 2022

This is the entry script for the following paper: A blueprint for Success: leveraging NLP to assess Ostrom Institutional Design Principles

Create a list from WOS and Scopus with dois for downloading full text articles

@author: jbaggio
"""
#for preparing the files
import os
import pandas as pd

#first get directories where paper titles and metadata are stored
fildir = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/DataCoded/Searching'

#gef files for titles and papers based on the following search on wos (<=2022) and Scopus (<=2019), the operators here match the WoS, scopus has equivalents
#TS = "common pool resource*" OR TS = "community based natural resource* manage*" OR TS = "community based enviro* manage*" OR TS = "community based resource* manage*"

os.chdir(fildir)
#scopus file is already well formatted #scopus search up to 2020
scopus = pd.read_csv('matched_corpus.csv')
scopustit = scopus.Title
scopusdoi = scopus.DOI

#wos is not, is exported as annotated output, wos search up to 2022
wos = pd.read_csv('paper_corpus.csv', sep = ',')
wostit = wos.Title
wosdoi = pd.read_csv('papers_dois.csv').DOI
        
#titles in trainingset
trainset = pd.read_excel('training2.xlsx', sheet_name = 'Sheet1')
traintit = trainset.Title

#now check for duplicate titles including the training set (no doi for the training set)
titall = pd.concat([scopustit, wostit, traintit])
titall = titall.str.lower()
titall = titall.str.replace('[^a-zA-Z]', ' ', regex=True)
titall = titall.str.replace('  ', ' ')
#remove duplicates (if any)
titall = titall.drop_duplicates().astype(str)
titall = list(titall)

#now merge and check doi and remove duplicates
doiall = pd.concat([scopusdoi, wosdoi])
doiall = doiall.drop_duplicates().astype(str)
listdoi = list(doiall)
doiall.to_csv('doilist.csv') #returns a list with 4430 dois that is all papers in scopus and WoS with doi minus duplicates

#once we have the doi list we download pdf articles manually, leading to 3384 articles downloaded:





