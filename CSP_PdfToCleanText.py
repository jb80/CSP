#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:16:58 2022

This script takes full text in pdf and parses them in machine readable text either via tesseract or via pyMuPDF

Further it cleans the pdf files by eliminating section starting with the following headings:['keywords','introduction', 'methods',  'methodology', 'references', 'bibliography', 'literature cited', 'acknowledgements', 'funding']
Hence the resulting text for each article should be composed by abstract, results, discussion and conclusions

@author: jbaggio
"""

#usual suspects
import os
import glob
import pickle
import re
from collections import defaultdict

#pdf to text packages
from PIL import Image 
import pytesseract 
from pdf2image import convert_from_path
import fitz


#usual suspects
import pandas as pd
import numpy as np

#function to decompose flags for font_properties for pymupdf (fitz)
def flags_decomposer(flags):
    """Make font flags human readable."""
    l = []
    if flags & 2 ** 0:
        l.append("superscript")
    if flags & 2 ** 1:
        l.append("italic")
    if flags & 2 ** 2:
        l.append("serifed")
    else:
        l.append("sans")
    if flags & 2 ** 3:
        l.append("monospaced")
    else:
        l.append("proportional")
    if flags & 2 ** 4:
        l.append("bold")
    return ", ".join(l)



#and where we want to download the full text
dwdir = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/DataCoded/Articles/PDF'
trainfiles = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/DataCoded/Articles/TextForTraining/train_pdf'
tempfile = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/DataCoded/Articles/TempImage'
ocrfiles = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/DataCoded/Articles/OCR'
cleandata = '/Users/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Data'

os.chdir(dwdir)
#get all pdf files
pdfs = glob.glob('*.pdf')


#where tesseract actually is located on the machine
pytesseract.pytesseract.tesseract_cmd = r'/Users/jbaggio/opt/anaconda3/bin/tesseract'


#using pymupdf, to get font properties where possible (bold and sizes), if pdf are scanned, then use pytesseract to extract text
mudict = defaultdict(dict)
tesseractlist = []
lentext = []
i = 0
for file in pdfs:
    print (i / len(pdfs))
    doc = fitz.open(file)
    filetxt = []
    fileprop = []
    pagetext = []
    for page in doc:
        temppage = page.get_text()
        temppage = temppage.replace('-\n', '')
        pagetext.append(temppage)
        blocks = page.get_text("dict", flags = 1)['blocks']
        for b in blocks:  # iterate through the text blocks
            for l in b["lines"]:  # iterate through the text lines
                font_properties = defaultdict(dict)
                for s in l["spans"]:  # iterate through the text spans
                    font_properties['font'] =  s["font"], # font name
                    font_properties['flags'] = flags_decomposer(s["flags"]),  # readable font flags
                    font_properties['size'] = s["size"],  # font size
                    filetxt.append(s['text'])
                    fileprop.append(font_properties)
    muocrtext = ' '.join(pagetext)
    #if the text is really short (or empty) or there are a lot of characters that can not be interpreted than try tesseract)
    if len(muocrtext) < 10000 or muocrtext.count(chr(65533)) > 1000:
        print('Trying Tesseract')
        print(file)
        tesseractlist.append(file)
        muocrlist = []
        try:
            images = convert_from_path(pdf_path=file, dpi = 500)
            for count, img in enumerate(images):
                 img_name = f"page_{count}.png"  
                 img.save(img_name, "png")
                 png_files = [f for f in os.listdir(".") if f.endswith(".png")]
            for png_file in png_files:
                extracted_text = pytesseract.image_to_string(Image.open(png_file))
                extracted_text = extracted_text.replace('-\n', '')     
                muocrlist.append(extracted_text)
                os.remove(png_file)
            muocrtx = ' '.join(muocrlist)
            mudict[file]['text'] = muocrtx
           
        except:
            print('except working')
            muocrtx = 'not able to render'
            mudict[file]['text'] = muocrtx
    else:
        mudict[file]['text'] = muocrtext      
        mudict[file]['splittext'] = filetxt
        mudict[file]['fonts'] = fileprop
        lentext.append(len(mudict[file]['text']))

    i += 1

os.chdir(ocrfiles)
pickle.dump(mudict, open('mudict.p', 'wb'))



#now do the same procedure for the training pdf
#using pymupdf, to get font properties where possible (bold and sizes), if pdf are scanned, then use pytesseract to extract text
os.chdir(trainfiles)
trainers = glob.glob('*.pdf')

traindict = defaultdict(dict)
tesseractlisttrain = []
lentraintext = []
i = 0
for file in trainers:
    print (i / len(trainers))
    doc = fitz.open(file)
    filetxt = []
    fileprop = []
    pagetext = []
    for page in doc:
        temppage = page.get_text()
        temppage = temppage.replace('-\n', '')
        pagetext.append(temppage)
        blocks = page.get_text("dict", flags = 1)['blocks']
        for b in blocks:  # iterate through the text blocks
            for l in b["lines"]:  # iterate through the text lines
                font_properties = defaultdict(dict)
                for s in l["spans"]:  # iterate through the text spans
                    font_properties['font'] =  s["font"], # font name
                    font_properties['flags'] = flags_decomposer(s["flags"]),  # readable font flags
                    font_properties['size'] = s["size"],  # font size
                    filetxt.append(s['text'])
                    fileprop.append(font_properties)
    muocrtext = ' '.join(pagetext)
    #if the text is really short (or empty) or there are a lot of characters that can not be interpreted than try tesseract)
    if len(muocrtext) < 10000 or muocrtext.count(chr(65533)) > 1000:
        print('Trying Tesseract')
        print(file)
        tesseractlisttrain.append(file)
        muocrlist = []
        try:
            images = convert_from_path(pdf_path=file, dpi = 500)
            for count, img in enumerate(images):
                 img_name = f"page_{count}.png"  
                 img.save(img_name, "png")
                 png_files = [f for f in os.listdir(".") if f.endswith(".png")]
            for png_file in png_files:
                extracted_text = pytesseract.image_to_string(Image.open(png_file))
                extracted_text = extracted_text.replace('-\n', '')     
                muocrlist.append(extracted_text)
                os.remove(png_file)
            muocrtx = ' '.join(muocrlist)
            traindict[file]['text'] = muocrtx
           
        except:
            muocrtx = 'not able to render'
            traindict[file]['text'] = muocrtx
    else:
        traindict[file]['text'] = muocrtext      
        traindict[file]['splittext'] = filetxt
        traindict[file]['fonts'] = fileprop
        lentraintext.append(len(traindict[file]['text']))

    i += 1

os.chdir(ocrfiles)
pickle.dump(traindict, open('traindict.p', 'wb'))


    
"""
Now from the OCR/pdftotext clean it up. Specifically eliminate introduction and methods, eliminate phrases in parenthesis 
and reference/aknowledgment sections. 

This is because introductions, methods and case descritpion do not provide the information we are looking for.
Further in parenthesis there is often references in text (we are looking at academic articles).
We leave all other sections as they may contain useful information for the embeddings and 
the few shot learner (see NLP_FullText_FewShots_Hyper_VMulti.py)


first find headers (here we have to make the assumptions that section headers are in between empty lines).
If a paper does not have sections with the headers, it will not eliminate text. This may be an issue but most papers 
will have introduction and methods/methodology sections. 
"""
#list of common headers of interesting sections

initlist =['abstract', 'a b s t r a c t', 'case study', 'result', 'discussion','conclusion']
nolist =['keywords','introduction', 'methods',  'methodology', 'references', 'bibliography', 'literature cited', 'acknowledgements', 'funding']
endlist = ['acknowledgements']
#first start with the following regular expression

dftext = pd.DataFrame(index = mudict.keys())
dftext['txt_title'] = pdfs
dftext['origtext'] = [mudict[key]['text'] for key in mudict]
#clean donwloaded by, text in pranethesis and tabulations
dftext['origtext'] = dftext['origtext'].str.replace('Donwloaded by', '', regex=True)
dftext['origtext'] = dftext['origtext'].str.replace('r\([^)]*\)', '', regex=True)
dftext['origtext'] = dftext['origtext'].str.replace('r\t', ' ', regex=True)
dftext['origtext'] = dftext['origtext'].str.replace(r'\[[^)]*\]', '', regex=True)

dfcases = dftext.copy()

headings = defaultdict(dict)
keeping = defaultdict(dict)
for ndoc in range(0, len(dfcases)):
    docheads = []
    keeptext = []
    idx = dfcases.index[ndoc]
    kt = 0
    end_paper = 0
    #find the regular expression above in the text
    dfcell = dfcases.iloc[ndoc]['origtext'].split('\n')
    for item in dfcell:
        if end_paper == 1:
            break
        text = item.lower()
        if any (tit in text for tit in initlist):
            docheads.append(text)
            kt = 1
            #the following bit of code states that if there is a section that is interesting than keep the text afterwards, but only alphanumeric characters
        if kt == 1:
            text =  re.sub(r'_+|[^\w-][\W]+', ' ', text)              
            keeptext.append(text)
        if any(tit in text for tit in nolist):
            if len(text) < 30:
                if (text.isdigit() == False and len(text) > 5):
                    kt = 0
                #finally when aknowledgments happen, that is usually at the end of the paper and we should skip the rest
        if any(tit in text for tit in endlist):
            end_paper = 1 
    pattern = '[0-9]'
    docheads = [re.sub(pattern, '', i) for i in docheads]
    docheads = [i for i in docheads if i]
    docheads = list(set(docheads))
    headings[idx] = docheads
    keeping[idx] = keeptext

#now merge multiple lists into one for each key
txkeep = defaultdict(dict)
for key in keeping:
    doc = keeping[key]
    if len(doc) < 1:
        txkeep[key] = np.nan
    else:
        for i in doc:
            joined = " ".join(doc)
        txkeep[key] = joined

os.chdir(cleandata)
#create dataframe for headings
dfheads = pd.DataFrame.from_dict(headings, orient='index')
pickle.dump(dfheads, open('headers.p', 'wb'))
pickle.dump(txkeep, open('text.p', 'wb'))

#now merge keept text with the dfcases dataframe, and then delete duplicates and where text is empty (no easily discernable result/discussion/conclusion)
dfcases = dfcases.assign(text = txkeep.values()) 
dfcases = dfcases.drop_duplicates(subset=['origtext'], keep='first')

pickle.dump(dfcases, open('extracted_df.p', 'wb'))
dfextract = dfcases[['txt_title', 'text']]
dfextract.to_csv('extracts.csv')


# now extract text from the training set


dftrain = pd.DataFrame(index = traindict.keys())
dftrain['txt_title'] = trainers
dftrain['origtext'] = [traindict[key]['text'] for key in traindict]
dftrain['origtext'] = dftrain['origtext'].str.replace('Donwloaded by', '', regex=True)
dftrain['origtext'] = dftrain['origtext'].str.replace('r\([^)]*\)', '', regex=True)
dftrain['origtext'] = dftrain['origtext'].str.replace('r\t', ' ', regex=True)
dftrain['origtext'] = dftrain['origtext'].str.replace(r'\[[^)]*\]', '', regex=True)

dfcases_train = dftrain.copy()

headings_train = defaultdict(dict)
keeping_train = defaultdict(dict)
for ndoc in range(0, len(dfcases_train)):
    docheads = []
    keeptext = []
    idx = dfcases_train.index[ndoc]
    kt = 0
    end_paper = 0
    #find the regular expression above in the text
    dfcell = dfcases_train.iloc[ndoc]['origtext'].split('\n')
    for item in dfcell:
        if end_paper == 1:
            break
        text = item.lower()
        if any (tit in text for tit in initlist):
            docheads.append(text)
            kt = 1
            #the following bit of code states that if there is a section that is interesting than keep the text afterwards, but only alphanumeric characters
        if kt == 1:
            text =  re.sub(r'_+|[^\w-][\W]+', ' ', text)              
            keeptext.append(text)
        if any(tit in text for tit in nolist):
            if len(text) < 30:
                if (text.isdigit() == False and len(text) > 5):
                    kt = 0
                #finally when aknowledgments happen, that is usually at the end of the paper and we should skip the rest
        if any(tit in text for tit in endlist):
            end_paper = 1 
    pattern = '[0-9]'
    docheads = [re.sub(pattern, '', i) for i in docheads]
    docheads = [i for i in docheads if i]
    docheads = list(set(docheads))
    headings_train[idx] = docheads
    keeping_train[idx] = keeptext

#now merge multiple lists into one for each key
txkeep_train = defaultdict(dict)
for key in keeping_train:
    doc = keeping_train[key]
    if len(doc) < 1:
        txkeep_train[key] = np.nan
    else:
        for i in doc:
            joined = " ".join(doc)
        txkeep_train[key] = joined

os.chdir(cleandata)
#create dataframe for headings
dfheads_train = pd.DataFrame.from_dict(headings_train, orient='index')
pickle.dump(dfheads_train, open('headers_train.p', 'wb'))
pickle.dump(txkeep_train, open('text_train.p', 'wb'))

#now merge keept text with the dfcases dataframe, and then delete duplicates and where text is empty (no easily discernable result/discussion/conclusion)
dfcases_train = dfcases_train.assign(text = txkeep_train.values()) 
dfcases_train = dfcases_train.drop_duplicates(subset=['origtext'], keep='first')

pickle.dump(dfcases_train, open('extracted_df_train.p', 'wb'))
dfextract_train = dfcases[['txt_title', 'text']]
dfextract_train.to_csv('extracts_train.csv')

