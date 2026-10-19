"""Shared, repository-relative locations for the CSP analysis pipeline."""

import os
from pathlib import Path


def workspace_root():
    """Use this checkout by default, or a separate workspace when configured."""
    configured = os.environ.get("CSP_WORKSPACE")
    if not configured:
        return Path(__file__).resolve().parent
    root = Path(configured).expanduser()
    if not root.is_absolute():
        raise ValueError("CSP_WORKSPACE must be an absolute path")
    return root.resolve()


ROOT = workspace_root()
DATA = ROOT / "data"
SEARCHING = DATA / "searching"
ARTICLES = DATA / "articles"
PDF = ARTICLES / "pdf"
TRAINING = ARTICLES / "training"
TRAIN_PDF = TRAINING / "train_pdf"
SUBSET = ARTICLES / "subset"
OCR = DATA / "ocr"
CLEAN = DATA / "processed"

RESULTS = ROOT / "results"
MAIN = RESULTS / "main"
MODELS = RESULTS / "models"
FIGURES = RESULTS / "figures"
TOPICS = FIGURES / "topics"
SHAPLEY = RESULTS / "shapley"
PSL = RESULTS / "psl"
