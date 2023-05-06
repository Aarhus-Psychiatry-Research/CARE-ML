from psycop_feature_generation.text_models.utils import load_text_model
from psycop_feature_generation.loaders.raw.sql_load import sql_load
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def plot_word_freq(
    model_file_name: str,
    corpus: pd.DataFrame,
):
    """Plot words/bigrams in order of frequency

    Args:
        model_file_name (str): file name of fitted text model
        view (str, optional): Name of view/SQL table with text data. Defaults to "psycop_train_all_sfis_all_years_lowercase_stopwords_and_symbols_removed".

    """

    # load model
    print("–––––––– Loading text model ––––––––")
    text_model = load_text_model(filename=model_file_name)

    # transform corpus
    print("–––––––– Transform corpus ––––––––")
    docs = text_model.transform(corpus["text"])  # type: ignore

    # prepare plot
    plt.style.use("ggplot")
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["figure.figsize"] = (20.0, 5.0)
    plt.rcParams["xtick.labelsize"] = 10
    plot = plt

    df = pd.DataFrame(
        docs.sum(axis=0).T, index=text_model.get_feature_names_out(), columns=["freq"]
    ).sort_values(by="freq", ascending=False)

    df.index = [str(i + 1) + " " + df.index[i] for i in range(len(df.index))]  # type: ignore

    # plot
    if isinstance(text_model, TfidfVectorizer):
        model = "TF-IDF"
    elif isinstance(text_model, CountVectorizer):
        model = "Bag-of-words"

    plot = df.plot(kind="bar", title=f"Most Freq Words: {model}")

    # save
    print("–––––––– Saving plot at path: ––––––––")
    print(
        f"–––––––– application/modules/text_models/plots/word_freq_{model_file_name[:-4]}.png ––––––––"
    )
    plot.figure.savefig(
        f"application/modules/text_models/plots/word_freq_{model_file_name[:-4]}.png",
        bbox_inches="tight",
    )


# create dict with vocab and count
# bow_model_transform = bow_model.transform(corpus["text"].tolist())
# bow_counts = bow_model_transform.toarray().sum(axis=0)

if __name__ == "__main__":
    # load corpus
    print("–––––––– Loading corpus ––––––––")
    corpus = sql_load(
        query="SELECT * FROM fct.psycop_train_all_sfis_all_years_lowercase_stopwords_and_symbols_removed"
    )

    # plot word frequencies
    plot_word_freq(
        model_file_name="bow_psycop_train_all_sfis_all_years_lowercase_stopwords_and_symbols_removed_sfi_type_All_sfis_ngram_range_12_max_df_10_min_df_1_max_features_100.pkl",
        corpus=corpus,  # type: ignore
    )
    plot_word_freq(
        model_file_name="tfidf_psycop_train_all_sfis_all_years_lowercase_stopwords_and_symbols_removed_sfi_type_All_sfis_ngram_range_12_max_df_10_min_df_1_max_features_100.pkl",
        corpus=corpus,  # type: ignore
    )
