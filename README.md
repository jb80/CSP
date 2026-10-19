# Synthesis of Ostrom Institutional Design Principles
Baggio, Epstein, Lambert, Joel, and Gordian. Sustainable resource management in the commons: a synthesis based on institutional design principles leveraging natural language processing.

This repository contains the python scripts used for the analysis of institutional design principles. The scripts contained here should be used in order.

## Workflow:

<img src="Analysis_Flow.png" alt="Diagram of the seven-step analysis workflow" width="500">

## File layout

Paths are relative to this repository, regardless of the directory from which a script is run. Put existing inputs in the folders below, keeping their original filenames:

```text
data/
  searching/                 matched_corpus.csv, paper_corpus.csv, papers_dois.csv, training2.xlsx
  articles/
    pdf/                     main-corpus PDFs
    training/                matching.xlsx
      train_pdf/             training PDFs
    subset/                  optional article subset
  ocr/                       intermediate OCR outputs
  processed/                 extracted_df.p, extracted_df_train.p, and related files
results/
  main/                      predicted codes, topic handoff, analysis data
  models/                    trained models and checkpoints
  figures/topics/            topic figures and tables
  shapley/                   SHAP analysis
  psl/                       LASSO and PSL analysis
```

Step 1 writes `doilist.csv` to `data/searching/`; PDFs are obtained manually. Steps 2 through 7 write to `data/processed/` and `results/`. Input data and generated results are ignored by Git. To keep them outside the checkout, set `CSP_WORKSPACE` to an absolute directory path with the same `data/` and `results/` layout before running the scripts.

Tesseract is found on `PATH` by default; set `TESSERACT_CMD` if needed. Step 7 requires `PSL_JAR_DIR` pointing to the PSL JAR folder, or `JAVA_HOME` with JARs in `lib/psl-jars`. This layout change does not install the Python dependencies or replace the GROBID service needed for a full run.

## Scripts:

1. CSP_1_GetArticles.py

Combines article titles and DOIs from the Scopus and Web of Science exports, removes duplicates, and writes `doilist.csv`. The PDF articles are downloaded manually afterward. This is the entry point for reproducing the study Blueprints for Success: Leveraging Natural Language Processing to Assess the Effect of Ostrom Institutional Design Principles on Resources, Conflict, and Inequality in the Commons.

2. CSP_2_PdfToCleanText_VF.py

Converts PDF files into machine-readable text using GROBID, PyMuPDF, and Tesseract OCR. The script extracts article structure, identifies relevant sections, removes non-substantive content (e.g., references and acknowledgements), and produces the text corpora used in downstream analyses.

3. CSP_3_Newton_FewShotLearner_AnnVF.py

Trains and evaluates few-shot learning models for:

* case-study identification,
* institutional design principle (DP) coding,
* outcome coding (resource conditions, conflict, and inequality).

The script uses SciCo/Longformer embeddings, SetFit, Optuna-based hyperparameter optimization, and nested cross-validation. Outputs include hard classifications, class probabilities, and uncertainty-aware probability scores.

4. CSP_4_EmbeddedTopics.py

Applies Embedded Topic Modeling (ETM) to texts classified as single-case studies. The script identifies dominant social-ecological contexts, estimates topic distributions, and produces the contextual variables used in subsequent analyses.

5. CSP_5_XGBoost_Shapley_VF.py

Analyzes relationships between institutional design principles and outcomes using XGBoost and SHAP. The script:

* evaluates alternative uncertainty thresholds,
* estimates aggregate and context-specific models,
* calculates SHAP values and SHAP interaction values,
* produces individual DP importance and effect figures.

6. CSP_6_PSL_ModelBuilder.py

Contains the functions required for the PSL analysis. The module:

* constructs Probabilistic Soft Logic (PSL) models,
* generates and evaluates candidate configurations,
* calculates configuration coverage, consistency, effect, and importance,
* builds co-occurrence networks,
* performs network and community analyses.

Candidate configurations are pre-screened using repeated LASSO interaction models.

7. CSP_7_PSL_Analysis.py

Evaluates institutional configurations using Probabilistic Soft Logic (PSL). The script:

* learns PSL rule weights,
* ranks and filters competing configurations,
* performs sensitivity analyses,
* constructs DP co-occurrence networks,
* identifies communities using the Leiden algorithm,
* generates the configurational and network figures reported in the paper.
