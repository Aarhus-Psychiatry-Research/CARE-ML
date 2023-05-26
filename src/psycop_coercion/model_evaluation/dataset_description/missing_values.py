import pandas as pd
from psycop.common.model_training.application_modules.process_manager_setup import setup
from psycop.common.model_training.data_loader.data_loader import DataLoader


def main():
    # load train and test splits using config
    cfg, _ = setup(
        config_file_name="default_config.yaml",
        application_config_dir_relative_path="../../../../../../psycop_coercion/model_training/application/config/",
    )

    train_df = DataLoader(data_cfg=cfg.data).load_dataset_from_dir(split_names="train")
    test_df = DataLoader(data_cfg=cfg.data).load_dataset_from_dir(split_names="val")

    train_df["dataset"] = "train"
    test_df["dataset"] = "test"

    df = pd.concat([train_df, test_df])

    text_features = df[
        [col for col in df.columns if col.startswith("pred_aktuelt_psykisk")]
    ]

    count_vectorizer_30_day = text_features[
        [
            col
            for col in text_features.columns
            if col.endswith("CountVectorizer_within_30_days_concatenate_fallback_nan")
        ]
    ]
    percent_missing = (
        (count_vectorizer_30_day == 0).sum() / count_vectorizer_30_day.shape[0] * 100.00
    )
    pd.DataFrame(
        {
            "column_name": count_vectorizer_30_day.columns,
            "percent_missing": percent_missing,
        },
    )
