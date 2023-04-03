"""Functions for loading simple nlp models"""
import pickle as pkl
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def _load_bow_model() -> CountVectorizer:
    """Loads the bag-of-words model from a pickle file"""
    
    project_root = Path(__file__).resolve().parents[3]
    filename = (
        project_root
        / "coercion-feature-generation"
        / "application"
        / "modules"
        / "text_feature_models"
        / "bow_model"
    )

    with Path(filename).open("rb") as f:
        return pkl.load(f)


def _load_tfidf_model() -> TfidfVectorizer:
    """Loads the tfidf model from a pickle file"""
    
    project_root = Path(__file__).resolve().parents[3]
    filename = (
        project_root
        / "coercion-feature-generation"
        / "application"
        / "modules"
        / "text_feature_models"
        / "bow_model"
    )

    with Path(filename).open("rb") as f:
        return pkl.load(f)
