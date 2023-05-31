import matplotlib.pyplot as plt
import pandas as pd
import plotnine as pn
from pandas.io import parquet
from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.common.feature_generation.text_models.text_model_paths import (
    PREPROCESSED_TEXT_DIR,
)
from psycop.common.feature_generation.text_models.utils import load_text_model
from psycop_coercion.model_evaluation.config import (
    COLOURS,
    GENERAL_ARTIFACT_PATH,
    PN_THEME,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def plot_word_freq(
    model_file_name: str,
    corpus: pd.DataFrame,
):
    """Plot words/bigrams in order of frequency

    Args:
        model_file_name (str): file name of fitted text model
        corpus (pd.DataFrame): dataframe with text data

    """

    # load model
    print("-------- Loading text model --------")
    text_model = load_text_model(filename=model_file_name)

    # transform corpus
    print("-------- Transform corpus --------")
    docs = text_model.transform(corpus["value"])  # type: ignore

    # prepare plot
    plt.style.use("ggplot")
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["figure.figsize"] = (20.0, 5.0)
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    plot = plt

    df = pd.DataFrame(
        docs.sum(axis=0).T,  # type: ignore
        index=text_model.get_feature_names_out(),
        columns=["freq"],
    ).sort_values(by="freq", ascending=False)

    df.index = [str(i + 1) + " " + df.index[i] for i in range(len(df.index))]  # type: ignore
    df = df.iloc[0:100]  # .reset_index()

    # plot
    if isinstance(text_model, TfidfVectorizer):
        model = "TF-IDF"
    elif isinstance(text_model, CountVectorizer):
        model = "Bag-of-words"

    plot = df.plot(kind="bar", title=f"Most Freq Words: {model}", color=COLOURS["blue"])

    # p = (
    #     pn.ggplot(df, pn.aes(x="index", y="freq"))
    #     + pn.geom_col()
    #     + pn.ylab("Frequency")
    #     + pn.ggtitle(f"Most Freq Words: {model}")
    #     + PN_THEME
    #     + pn.theme(axis_text_x=pn.element_text(rotation=90, hjust=1))
    # )

    # p.save(GENERAL_ARTIFACT_PATH.parent.parent / f"word_freq_{model_file_name[:-4]}.jpg", dpi=600)

    # save
    output_path = (
        GENERAL_ARTIFACT_PATH.parent.parent / f"word_freq_{model_file_name[:-4]}.png"
    )
    plot.figure.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
    )


def table_word_freq(
    model_file_name: str,
):
    """Plot words/bigrams in order of frequency

    Args:
        model_file_name (str): file name of fitted text model
        corpus (pd.DataFrame): dataframe with text data

    """

    # load model
    print("-------- Loading text model --------")
    text_model = load_text_model(filename=model_file_name)

    # create data frame from dictionary
    vocab = pd.DataFrame(text_model.vocabulary_, index=[0]).T.reset_index()
    vocab = vocab.sort_values(by=0, ascending=True)[["index"]]

    output_path = (
        GENERAL_ARTIFACT_PATH.parent.parent
        / f"word_freq_table_{model_file_name[:-4]}.csv"
    )

    vocab.to_csv(output_path, index=False)


if __name__ == "__main__":
    # load corpus
    print("-------- Loading corpus --------")
    filter_list = [("overskrift", "=", "Aktuelt psykisk")]

    corpus = pd.read_parquet(
        path=PREPROCESSED_TEXT_DIR / "psycop_train_all_sfis_preprocessed.parquet",
        filters=filter_list,
    )

    # plot word frequencies
    plot_word_freq(
        model_file_name="bow_psycop_train_all_sfis_preprocessed_sfi_type_Aktueltpsykisk_ngram_range_12_max_df_10_min_df_1_max_features_500.pkl",
        corpus=corpus,  # type: ignore
    )
    plot_word_freq(
        model_file_name="tfidf_psycop_train_all_sfis_preprocessed_sfi_type_Aktueltpsykisk_ngram_range_12_max_df_10_min_df_1_max_features_500.pkl",
        corpus=corpus,  # type: ignore
    )
    # save tables
    table_word_freq(
        model_file_name="bow_psycop_train_all_sfis_preprocessed_sfi_type_Aktueltpsykisk_ngram_range_12_max_df_10_min_df_1_max_features_500.pkl",
    )
