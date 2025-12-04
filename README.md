# Synthesis of Ostrom Institutional Design Principles
Analysis code for the following paper: Baggio, Epstein, Lambert, Joel, Gordian: Blueprints for success: leveraging natural language processing to assess the effect of Ostrom institutional design principles on resources, conflict, and inequality in the commons.

This repository contains the python scripts used for the analysis of institutional design principles. The scripts contained here should be used in order.
1. CSP_GetArticles.py:

      This is the entry script for the following paper: A blueprint for Success: leveraging NLP to assess Ostrom Institutional Design Principles
2.  CSP_PdfToCleanText.py:

      This script takes full text in pdf and parses them in machine readable text either via tesseract or via pyMuPDF
     Further it cleans the pdf files by eliminating section starting with the following headings:['keywords','introduction', 'methods',  'methodology', 'references', 'bibliography', 'literature cited', 'acknowledgements', 'funding']
     Hence the resulting text for each article should be composed by abstract, results, discussion and conclusions
3. CSP_FewShotLearner.py:
 
     Build an embedding model to work with a few shot learner to classify papers

4. CSP_EmbeddedTopics.py:

      This script extracts main topics for all the full texts that are considered case studies predicted via the CSP_FewShot
     Learner script. It does so via embedded topic modelling
5. CSP_Locations_VF.py:

      Find locations in papers extraced and cleaned via CSP_ToCleanText and that result in being a case study based on CSP_FewShotLearner

6. CSP_AnalysisDiffVF2.py:
 
    This script is used after Few Shot Learner and the Embedded Topic Model script in order to analyze the resulting
   probability of presence/absence and missing in relation to outcomes and by socio-ecological context and provides individual DP importance figures.

7. CSP_PSL_ModelBuilder.py:
 
    This script contains functions used in the PSL_Analsysi file. The functions allow the analysis of association rules via probabilistic soft logic. Rules are pre-screened using LASSO regression.
   
8. CSP_PSL_Analysis.py:
 
    This script is used after Few Shot Learner and the Embedded Topic Model script and reuires the CPS_PSL_ModelBuilder.py and is used to analyze the results via probabilistic soft logic  in relation to outcomes and by socio-ecological context and provides the key configurations figures in the paper.
