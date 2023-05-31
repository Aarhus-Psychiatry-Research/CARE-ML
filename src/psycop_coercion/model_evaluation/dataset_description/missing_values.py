import pandas as pd
from psycop_coercion.model_evaluation.config import (
    GENERAL_ARTIFACT_PATH,
)
from psycop_coercion.model_evaluation.dataset_description.utils import load_feature_set


def main():
    df = load_feature_set()

    text_features = df[
        [
            col
            for col in df.columns
            if col.startswith("pred_aktuelt_psykisk")
            and "type_token_ratio" not in col
            and "mean_number_of_characters" not in col
        ]
    ]
    lb_7 = text_features[
        [col for col in text_features.columns if "within_7_days" in col]
    ]
    lb_30 = text_features[
        [col for col in text_features.columns if "within_30_days" in col]
    ]

    lb_7_na_zero_percentages = (((lb_7.isna() | (lb_7 == 0)).mean()) * 100).mean()
    lb_30_na_zero_percentages = (((lb_30.isna() | (lb_30 == 0)).mean()) * 100).mean()


if __name__ == "__main__":
    main()
