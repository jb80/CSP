#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For every PDF the script does the following:

  1. Send the original PDF to GROBID.
       Success: use GROBID section structure.
  2. Otherwise extract text with PyMuPDF.
  3. If PyMuPDF text is unusable, OCR with Tesseract and write a searchable PDF.
     Otherwise, write a cleaned/repaired PDF with PyMuPDF.
  4. Send the repaired or OCR PDF to GROBID again.
       Success: use GROBID section structure.
  5. If GROBID still fails, retain the full extracted text.

The script produces three text representations:

  origtext:
      Complete GROBID body or complete fallback text.

  text:
      Abstract, case-study sections, results, discussion and conclusions.
      Intended for substantive-variable coding.

  case_text:
      Abstract plus case-selection, research-design, study-area, study-site,
      empirical-setting and related sections. Relevant paragraphs are also
      selected from generically titled Methods sections.
      Intended for no-case versus case and single-case versus multiple-case
      classification.

GROBID must be running before this script starts:
    ./gradlew run

@author: jbaggio
"""



import os
import glob
import pickle
import re
import time
import requests
from collections import defaultdict

# PDF extraction and cleaning
import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from transformers import AutoTokenizer


# Dataset and operations
import pandas as pd
import numpy as np




'''
Settings
'''

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
GROBID_ALIVE_URL = "http://localhost:8070/api/isalive"

MIN_BODY_CHARS = 1000    # GROBID output with a shorter body counts as a failure
MIN_TEXT_CHARS = 10000   # PyMuPDF text shorter than this -> treat as scanned, OCR it
MAX_BAD_CHARS = 1000     # more than this many U+FFFD characters -> garbled, OCR it
OCR_DPI = 300

# Headings are matched on whole words, after removing numbering ("3.1", "iii.")
# and un-spacing headings such as "A B S T R A C T".
# A KEEP heading switches keeping on, a STOP heading switches it off, and any
# other heading inherits the current state (so "Policy implications" after
# "Discussion" is kept).
KEEP_PATTERNS = [
    r'abstract',
    r'case stud(?:y|ies)',
    r'results?',
    r'discussion',
    r'conclusions?',
]

CASE_SECTION_PATTERNS = [
    r'cases?',
    r'the cases?',
    r'case stud(?:y|ies)',
    r'case selection',
    r'selection of cases?',
    r'case descriptions?',
    r'description of (?:the )?cases?',
    r'background to (?:the )?cases?',
    r'research design',
    r'study design',
    r'comparative design',
    r'comparative case stud(?:y|ies)',
    r'empirical setting',
    r'empirical context',
    r'research context',
    r'study setting',
    r'study region',
    r'study regions',
    r'study areas?',
    r'study sites?',
    r'study locations?',
    r'research sites?',
    r'field sites?',
    r'site descriptions?',
    r'geographical setting',
    r'geographic setting',
    r'site selection',
    r'study communities',
    r'setting and participants',
]

CASE_PARAGRAPH_PATTERNS = [
    r'\bcase stud(?:y|ies)\b',
    r'\bcase selection\b',
    r'\b(single|multiple|comparative)\s+cases?\b',
    r'\b(first|second|third)\s+case\b',
    r'\bcase\s+(?:1|2|3|a|b|c)\b',
    r'\b(two|three|four|several|multiple)\s+'
    r'(?:cases?|sites?|areas?|communities|countries|regions)\b',
    r'\b(study|research|field)\s+sites?\b',
    r'\bstudy areas?\b',
    r'\b(across|between|both|each)\s+(?:the\s+)?cases?\b',
    r'\bcompar(?:e|ed|ing|ison|ative)\b.{0,80}'
    r'\b(cases?|sites?|areas?|communities|countries|regions)\b',
    r'\b(selected|chosen|sampled)\b.{0,100}'
    r'\b(cases?|sites?|communities|countries|regions)\b',
    r'\brespectively\b',
    r'\bcross[- ]case\b',
    r'\bwithin[- ]case\b',
]


STOP_PATTERNS = [
    r'keywords?',
    r'introduction',
    r'methods?',
    r'methodology',
    r'materials? and methods?',
    r'references',
    r'bibliography',
    r'literature cited',
    r'acknowledge?ments?',
    r'funding',
]

KEEP_RE = re.compile(r'\b(?:' + '|'.join(KEEP_PATTERNS) + r')\b')
CASE_SECTION_RE = re.compile(r'\b(?:' + '|'.join(CASE_SECTION_PATTERNS) + r')\b', re.I)
CASE_PARAGRAPH_RE = re.compile('|'.join(CASE_PARAGRAPH_PATTERNS), re.I)
STOP_RE = re.compile(r'\b(?:' + '|'.join(STOP_PATTERNS) + r')\b')


'''
Functions
'''

# function to decompose flags for font_properties for pymupdf (fitz)
def flags_decomposer(flags):
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


# function to clean some potential issues with the text and flags
def clean_text(text, remove_parentheses=True, remove_brackets=True):
    if pd.isnull(text):
        return np.nan
    text = re.sub(r'Downloaded by.*', '', text, flags=re.I)
    if remove_parentheses:
        text = re.sub(r'\([^)]*\)', '', text)
    if remove_brackets:
        text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# GROBID

def grobid_alive():
    try:
        r = requests.get(GROBID_ALIVE_URL, timeout=5)
        return r.text.strip() == "true"
    except requests.exceptions.RequestException as e:
        print(f"GROBID not reachable: {e}")
        return False


# send a PDF to grobid, return the TEI xml (string) or None
def parse_pdf_grobid(pdf_file, retries=3):
    name = os.path.basename(pdf_file)
    for _ in range(retries):
        try:
            with open(pdf_file, "rb") as pdf:
                response = requests.post(GROBID_URL, files={"input": pdf}, timeout=300)
        except requests.exceptions.RequestException as e:
            print(f"   GROBID request failed for {name}: {e}")
            return None

        if response.status_code == 200:
            return response.text
        if response.status_code == 503:  # GROBID is busy, wait and try again
            time.sleep(5)
            continue

        print(f"   GROBID returned status {response.status_code} for {name}")
        return None

    print(f"   GROBID stayed busy (503) for {name}")
    return None


# body text of the TEI (used as the 'full text')
def tei_to_fulltext(xml):
    soup = BeautifulSoup(xml, "xml")
    body = soup.find("body")
    if body is None:
        return ""
    return re.sub(r'\s+', ' ', body.get_text(" ", strip=True))


# lower-case a heading and remove numbering / letter spacing so it can be matched
def normalize_heading(title):
    title = title.lower().strip()

    # "a b s t r a c t" -> "abstract"
    tokens = title.split()
    if len(tokens) >= 4 and all(len(t) == 1 for t in tokens):
        title = "".join(tokens)

    # leading numbering: "3", "3.1", "2.1.", "iii.", "(4)"
    title = re.sub(r'^\(?(?:\d+(?:\.\d+)*|[ivxlc]+(?=[\.\)]))[\.\):]*\s*', '', title)
    return title


# select the sections we want from the GROBID TEI
def tei_to_sections(xml):
    soup = BeautifulSoup(xml, "xml")

    extracted = []
    headers = []

    abstract = soup.find("abstract")
    if abstract is not None:
        abstract_text = abstract.get_text(" ", strip=True)
        if abstract_text:
            extracted.append(abstract_text)
            headers.append("abstract")

    # only walk the body: acknowledgements, annexes and references live in <back>
    body = soup.find("body")
    if body is not None:
        keep = False
        for div in body.find_all("div", recursive=False):
            head = div.find("head", recursive=False)
            
            if head is not None:
                raw_title = head.get_text(" ", strip=True).lower()
                headers.append(raw_title)
                title = normalize_heading(raw_title)

                if KEEP_RE.search(title):
                    keep = True
                elif STOP_RE.search(title):
                    keep = False

            if keep:
                # paragraphs only: leaves out figure and table content
                for p in div.find_all("p"):
                    extracted.append(p.get_text(" ", strip=True))

    section_text = re.sub(r'\s+', ' ', " ".join(extracted))
    return section_text, list(dict.fromkeys(headers))


def tei_to_case_text(xml):
    soup = BeautifulSoup(xml, "xml")
    pieces = []
    seen = set()

    abstract = soup.find("abstract")
    if abstract is not None:
        abstract_text = re.sub(r'\s+', ' ', abstract.get_text(" ", strip=True)).strip()
        if abstract_text:
            pieces.append("SECTION ABSTRACT: " + abstract_text)
            seen.add(abstract_text.lower())

    body = soup.find("body")
    if body is None:
        return " ".join(pieces)

    for div in body.find_all("div"):
        head = div.find("head", recursive=False)
        heading = normalize_heading(head.get_text(" ", strip=True)) if head is not None else ""

        paragraphs = []
        for paragraph in div.find_all("p", recursive=False):
            paragraph_text = re.sub(r'\s+', ' ', paragraph.get_text(" ", strip=True)).strip()
            if paragraph_text:
                paragraphs.append(paragraph_text)

        # Fallback for leaf divisions whose paragraphs are wrapped in another tag.
        if not paragraphs and div.find("div") is None:
            for paragraph in div.find_all("p"):
                paragraph_text = re.sub(r'\s+', ' ', paragraph.get_text(" ", strip=True)).strip()
                if paragraph_text:
                    paragraphs.append(paragraph_text)

        if CASE_SECTION_RE.search(heading):
            for paragraph_text in paragraphs:
                key = paragraph_text.lower()
                if key not in seen:
                    pieces.append("SECTION " + heading.upper() + ": " + paragraph_text)
                    seen.add(key)
            continue

        is_generic_method = bool(re.search(r'\b(methods?|methodology|materials? and methods?|data and methods?|empirical analysis|research setting|study setting)\b', heading, re.I))

        if is_generic_method:
            for paragraph_text in paragraphs:
                if CASE_PARAGRAPH_RE.search(paragraph_text):
                    key = paragraph_text.lower()
                    if key not in seen:
                        label = heading.upper() if heading else "METHODS"
                        pieces.append("SECTION " + label + "; CASE EVIDENCE: " + paragraph_text)
                        seen.add(key)

    return re.sub(r'\s+', ' ', " ".join(pieces)).strip()

# grobid + section selection in one go. Returns None if GROBID "failed"
def try_grobid(pdf_file):
    xml = parse_pdf_grobid(pdf_file)

    if xml is None:
        return None

    origtext = tei_to_fulltext(xml)

    if len(origtext) <= MIN_BODY_CHARS:
        print("   GROBID returned too little body text")
        return None

    section_text, headers = tei_to_sections(xml)
    case_text = tei_to_case_text(xml)

    return origtext, section_text, case_text, headers

# PyMuPDF / Tesseract

# extract text (and span-level text/font properties) from machine readable pdfs
def extract_pdf_fallback(pdf_file):
    pagetext = []
    filetxt = []
    fileprop = []

    try:
        doc = fitz.open(pdf_file)
    except Exception as e:
        print(f"   PyMuPDF could not open {os.path.basename(pdf_file)}: {e}")
        return "", filetxt, fileprop

    try:
        for page in doc:
            pagetext.append(page.get_text().replace('-\n', ''))

            try:
                blocks = page.get_text("dict", flags=1)["blocks"]
                for b in blocks:
                    if "lines" not in b:
                        continue
                    for l in b["lines"]:
                        for s in l["spans"]:
                            filetxt.append(s["text"])
                            # a new dict for every span
                            fileprop.append({
                                "font": s["font"],
                                "flags": flags_decomposer(s["flags"]),
                                "size": s["size"],
                            })
            except Exception:
                pass
    except Exception as e:
        print(f"   PyMuPDF failed while reading {os.path.basename(pdf_file)}: {e}")
    finally:
        doc.close()

    return " ".join(pagetext), filetxt, fileprop


def text_is_unusable(text):
    return len(text) < MIN_TEXT_CHARS or text.count(chr(65533)) > MAX_BAD_CHARS


# re-save the pdf with PyMuPDF (repairs broken structure, removes encryption flags)
def save_clean_pdf(pdf_file, out_pdf):
    doc = fitz.open(pdf_file)
    try:
        doc.save(out_pdf, garbage=4, clean=True, deflate=True,
                 encryption=fitz.PDF_ENCRYPT_NONE)
    finally:
        doc.close()
    return out_pdf


# OCR a pdf with tesseract and write a searchable pdf (image + invisible text layer)
def ocr_to_pdf(pdf_file, out_pdf, dpi=OCR_DPI):
    n_pages = pdfinfo_from_path(pdf_file)["Pages"]
    out = fitz.open()
    try:
        for p in range(1, n_pages + 1):
            # one page at a time to keep memory low
            img = convert_from_path(pdf_file, dpi=dpi, first_page=p, last_page=p)[0]
            page_pdf = pytesseract.image_to_pdf_or_hocr(img, extension="pdf")
            page_doc = fitz.open(stream=page_pdf, filetype="pdf")
            out.insert_pdf(page_doc)
            page_doc.close()
        tmp_path = out_pdf + ".part"
        out.save(tmp_path, garbage=3, deflate=True)
    finally:
        out.close()
    os.replace(tmp_path, out_pdf)  # only appears under its final name when complete
    return out_pdf


# Workflow 

def process_pdf(pdf_file, intermediate_dir):
    os.makedirs(intermediate_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(pdf_file))[0]

    # GROBID on the original file
    res = try_grobid(pdf_file)
    if res is not None:
        origtext, section_text, case_text, headers = res
        if not case_text.strip():
            case_text = section_text if section_text.strip() else origtext
        print("   GROBID worked on the original PDF")
        return {
            'text': clean_text(origtext),
            'section_text': clean_text(section_text),
            'case_text': clean_text(case_text, remove_parentheses=False, remove_brackets=False),
            'headers': headers,
            'method': 'grobid',
        }

    # PyMuPDF
    print("   GROBID failed -> PyMuPDF")
    text, filetxt, fileprop = extract_pdf_fallback(pdf_file)
    method = 'pymupdf'
    candidate = None

    if text_is_unusable(text):
        # Tesseract
        print("   Text unusable -> Tesseract")
        ocr_pdf = os.path.join(intermediate_dir, stem + "__ocr.pdf")
        try:
            if not os.path.exists(ocr_pdf):
                ocr_to_pdf(pdf_file, ocr_pdf)
            text, filetxt, fileprop = extract_pdf_fallback(ocr_pdf)
            candidate = ocr_pdf
            method = 'tesseract'
        except Exception as e:
            print(f"   OCR failed: {e}")
            text = ' '
            method = 'failed'
    else:
        # Cleaned copy from PyMuPDF
        clean_pdf = os.path.join(intermediate_dir, stem + "__pymupdf.pdf")
        try:
            save_clean_pdf(pdf_file, clean_pdf)
            candidate = clean_pdf
        except Exception as e:
            print(f"   Could not write a cleaned PDF: {e}")

    # GROBID again on the repaired or OCR file
    if candidate is not None:
        print(f"   Trying GROBID again on {os.path.basename(candidate)}")
        res = try_grobid(candidate)
        if res is not None:
            origtext, section_text, case_text, headers = res
            if not case_text.strip():
                case_text = section_text if section_text.strip() else origtext
            return {
                'text': clean_text(origtext),
                'section_text': clean_text(section_text),
                'case_text': clean_text(case_text, remove_parentheses=False, remove_brackets=False),
                'headers': headers,
                'splittext': filetxt,
                'fonts': fileprop,
                'method': method + '+grobid',
            }

    # No GROBID structure available
    print("   No GROBID sections, keeping the full text")
    return {
        'text': clean_text(text),
        'section_text': clean_text(text),
        'case_text': clean_text(text, remove_parentheses=False, remove_brackets=False),
        'headers': [],
        'splittext': filetxt,
        'fonts': fileprop,
        'method': method,
        }


def process_folder(folder, intermediate_dir):
    os.makedirs(intermediate_dir, exist_ok=True)
    pdfs = sorted(glob.glob(os.path.join(folder, '*.pdf')))

    results = defaultdict(dict)
    for i, path in enumerate(pdfs):
        name = os.path.basename(path)
        print(f"[{i + 1}/{len(pdfs)}] {name}")
        try:
            results[name] = process_pdf(path, intermediate_dir)
        except Exception as e:  # one bad file should not stop the whole run
            print(f"   ERROR on {name}: {e}")
            results[name] = {
                'text': ' ',
                'section_text': ' ',
                'case_text': ' ',
                'headers': [],
                'method': 'error',
            }
    return results


# write the pickles / csv the same way as before
def save_outputs(results, dict_name, suffix, ocrfiles, cleandata):
    with open(os.path.join(ocrfiles, dict_name), 'wb') as f:
        pickle.dump(results, f)

    names = list(results.keys())

    dfcases = pd.DataFrame({
        'txt_title': names,
        'origtext': [results[name].get('text', '') for name in names],
        'text': [results[name].get('section_text', '') for name in names],
        'case_text': [results[name].get('case_text', '') for name in names],
        'headers': [results[name].get('headers', []) for name in names],
        'method': [results[name].get('method', 'unknown') for name in names],
    })

    for column in ['origtext', 'text', 'case_text']:
        dfcases[column] = dfcases[column].fillna('').astype(str)

    empty_case_text = dfcases['case_text'].str.strip().eq('')
    print('Empty case texts before fallback:', int(empty_case_text.sum()))
    dfcases.loc[empty_case_text, 'case_text'] = dfcases.loc[empty_case_text, 'text']

    still_empty_case_text = dfcases['case_text'].str.strip().eq('')
    dfcases.loc[still_empty_case_text, 'case_text'] = dfcases.loc[still_empty_case_text, 'origtext']

    tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-scico')

    dfcases['n_tokens'] = dfcases['origtext'].apply(lambda value: len(tokenizer.encode(value, add_special_tokens=False, truncation=False)))
    dfcases['n_tokens_selected'] = dfcases['text'].apply(lambda value: len(tokenizer.encode(value, add_special_tokens=False, truncation=False)))
    dfcases['n_tokens_case'] = dfcases['case_text'].apply(lambda value: len(tokenizer.encode(value, add_special_tokens=False, truncation=False)))
    
    dfcases = dfcases.drop_duplicates(subset=['origtext'], keep='first').reset_index(drop=True)
    txkeep = dict(zip(dfcases['txt_title'], dfcases['text']))
    casekeep = dict(zip(dfcases['txt_title'], dfcases['case_text']))
    headings = {name: results[name].get('headers', []) for name in names}
    dfheads = pd.DataFrame.from_dict(headings, orient='index')

    with open(os.path.join(cleandata, 'headers' + suffix + '.p'), 'wb') as f:
        pickle.dump(dfheads, f)

    with open(os.path.join(cleandata, 'text' + suffix + '.p'), 'wb') as f:
        pickle.dump(txkeep, f)

    with open(os.path.join(cleandata, 'case_text' + suffix + '.p'), 'wb') as f:
        pickle.dump(casekeep, f)

    nonempty_origtext = dfcases['origtext'].str.strip().ne('')
    nonempty_selected = dfcases['text'].str.strip().ne('')
    nonempty_case = dfcases['case_text'].str.strip().ne('')

    print('Documents before duplicate removal:', len(dfcases))
    print('Nonempty full texts:', int(nonempty_origtext.sum()))
    print('Nonempty selected texts:', int(nonempty_selected.sum()))
    print('Nonempty case texts:', int(nonempty_case.sum()))
    print('Empty selected texts:', int((~nonempty_selected).sum()))
    print('Empty case texts:', int((~nonempty_case).sum()))
    print('Extraction methods:')
    print(dfcases['method'].value_counts(dropna=False))

    dfcases = dfcases.drop_duplicates(subset=['origtext'], keep='first').reset_index(drop=True)

    with open(os.path.join(cleandata, 'extracted_df' + suffix + '.p'), 'wb') as f:
        pickle.dump(dfcases, f)

    dfcases[['txt_title', 'text', 'case_text']].to_csv(os.path.join(cleandata, 'extracts' + suffix + '.csv'), index=False)

    print('Documents after duplicate removal:', len(dfcases))
    print('Full-text token distribution:')
    print(dfcases['n_tokens'].describe())
    print('Selected-text token distribution:')
    print(dfcases['n_tokens_selected'].describe())
    print('Case-text token distribution:')
    print(dfcases['n_tokens_case'].describe())
    print('Case texts > 512:', int((dfcases['n_tokens_case'] > 512).sum()))
    print('Case texts > 1024:', int((dfcases['n_tokens_case'] > 1024).sum()))
    print('Case texts > 2048:', int((dfcases['n_tokens_case'] > 2048).sum()))
    print('Case texts > 4096:', int((dfcases['n_tokens_case'] > 4096).sum()))


def preflight():
    if not grobid_alive():
        raise SystemExit("GROBID is not running. Start it first (./gradlew run in the grobid folder) and check curl http://localhost:8070/api/isalive")

    try:
        BeautifulSoup("<a/>", "xml")
        print("XML parser OK")
    except Exception:
        raise SystemExit("BeautifulSoup needs lxml to read GROBID XML: pip install lxml")

    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        print(f"WARNING: Tesseract not found ({e}). Scanned PDFs will not be OCR'd.")

'''
Script
'''



dwdir = '/CommonsSynthProject/DataCoded/Articles/PDF'
trainfiles = '/CommonsSynthProject/DataCoded/Articles/TextForTraining/train_pdf'
ocrfiles = '/CommonsSynthProject//DataCoded/Articles/OCR'
cleandata = '/CommonsSynthProject/jbaggio/Documents/AAA_Study/AAA_Work/CommonsSynthProject/NLPSynth/Data'

checkfiles = '/CommonsSynthProject/DataCoded/Articles/subset'

intermediate_main = os.path.join(ocrfiles, 'intermediate_pdf', 'main')
intermediate_train = os.path.join(ocrfiles, 'intermediate_pdf', 'train')

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

print('GROBID STATUS:', requests.get(GROBID_ALIVE_URL).text)


preflight()


'''

Process the main corpus

'''

mudict = process_folder(dwdir, intermediate_main)

save_outputs(
    results=mudict,
    dict_name='mudict.p',
    suffix='',
    ocrfiles=ocrfiles,
    cleandata=cleandata
)

'''

process training corpus

'''

traindict = process_folder(trainfiles, intermediate_train)

save_outputs(
    results=traindict,
    dict_name='traindict.p',
    suffix='_train',
    ocrfiles=ocrfiles,
    cleandata=cleandata
)

print('\n')
print('====================================================================')
print('SUMMARY')
print('====================================================================')
print('\n')

print('MAIN SET')

print(
    pd.Series(
        [v.get('method', 'unknown') for v in mudict.values()]
    ).value_counts()
)

print('\n')

print('TRAINING SET')

print(
    pd.Series(
        [v.get('method', 'unknown') for v in traindict.values()]
    ).value_counts()
)

print('\n')
print('====================================================================')
print('TOKEN STATISTICS')
print('====================================================================')
print('\n')

os.chdir(cleandata)
dfcases = pickle.load(open('extracted_df.p', 'rb'))

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-scico")
dfcases['n_tokens_selected_check'] = dfcases['text'].apply(lambda x: len(tokenizer.encode(str(x), add_special_tokens=False, truncation=False)))
dfcases['n_tokens_case_check'] = dfcases['case_text'].apply(lambda x: len(tokenizer.encode(str(x), add_special_tokens=False, truncation=False)))

print('\nSELECTED TEXT')
print(dfcases['n_tokens_selected_check'].describe())
print('Docs > 4096:', int((dfcases['n_tokens_selected_check'] > 4096).sum()))
print('Percent > 4096:', 100 * (dfcases['n_tokens_selected_check'] > 4096).mean())

print('\nCASE TEXT')
print(dfcases['n_tokens_case_check'].describe())
print('Docs > 512:', int((dfcases['n_tokens_case_check'] > 512).sum()))
print('Docs > 1024:', int((dfcases['n_tokens_case_check'] > 1024).sum()))
print('Docs > 2048:', int((dfcases['n_tokens_case_check'] > 2048).sum()))
print('Docs > 4096:', int((dfcases['n_tokens_case_check'] > 4096).sum()))
print('Percent > 2048:', 100 * (dfcases['n_tokens_case_check'] > 2048).mean())
