"""Fit and save text models"""
from psycop_feature_generation.text_models.text_model_pipeline import (
    text_model_pipeline,
)

if __name__ == "__main__":
    text_model_pipeline(
        model="bow",
        view="psycop_train_all_sfis_all_years_lowercase_stopwords_and_symbols_removed",
        max_features=100,
        max_df=1.0,
        min_df=1,
        ngram_range=(1, 2),
    )

    text_model_pipeline(
        model="bow",
        view="psycop_train_all_sfis_all_years_lowercase_stopwords_and_symbols_removed",
        sfi_type=["Aktuelt psykisk"],
        max_features=100,
        max_df=1.0,
        min_df=1,
        ngram_range=(1, 2),
    )

    text_model_pipeline(
        model="tfidf",
        view="psycop_train_all_sfis_all_years_lowercase_stopwords_and_symbols_removed",
        max_features=100,
        max_df=1.0,
        min_df=1,
        ngram_range=(1, 2),
    )

    text_model_pipeline(
        model="tfidf",
        view="psycop_train_all_sfis_all_years_lowercase_stopwords_and_symbols_removed",
        sfi_type=["Aktuelt psykisk"],
        max_features=100,
        max_df=1.0,
        min_df=1,
        ngram_range=(1, 2),
    )
