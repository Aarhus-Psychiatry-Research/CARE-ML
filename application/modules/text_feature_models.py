"""Script for creating a bag-of-words model and a tfidf model on text data from coercion cohort"""

import pickle as pkl
from pathlib import Path
from typing import Any, List, Sequence
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from psycop_feature_generation.loaders.raw.load_text import load_all_notes


def load_txt_data() -> List[str]:
    """
    Loads the text data and returns the text
    """
    # load all text data?
    all_notes = load_all_notes(n_rows=1000) # None 

    return all_notes["text"].dropna().tolist()


def train_bow_model(corpus: Sequence[str]) -> CountVectorizer:
    """
    Trains a bag-of-words model on text data

    Args:
        corpus (Sequence[str]): The corpus to train on
    """
    model = CountVectorizer(lowercase=True)  # , max_features=10)
    model.fit(corpus)

    return model


def train_tfidf_model(corpus: Sequence[str]) -> TfidfVectorizer:
    """
    Trains a tfidf model on  data

    Args:
        corpus (Sequence[str]): The corpus to train on
    """
    model = TfidfVectorizer(lowercase=True)
    model.fit(corpus)

    return model


def save_model_to_dir(
    model: Any,
    filename: str,
):  # pylint: disable=missing-type-doc
    """
    Saves the model to a pickle file

    Args:
        model: The model to save
        filename: The filename to save the model to
    """
    project_root = Path(__file__).resolve().parents[3]
    filename = (
        project_root
        / "coercion-feature-generation"
        / "application"
        / "modules"
        / "text_feature_models"
        / filename
    )

    with Path(filename).open("wb") as f:
        pkl.dump(model, f)


if __name__ == "__main__":
    corpus = load_txt_data()
    bow_model = train_bow_model(corpus)
    tfidf_model = train_tfidf_model(corpus)

    save_model_to_dir(bow_model, "bow_model.pkl")
    save_model_to_dir(tfidf_model, "tfidf_model.pkl")
