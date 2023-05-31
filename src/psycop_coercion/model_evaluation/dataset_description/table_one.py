import pandas as pd
from psycop.common.model_training.application_modules.process_manager_setup import setup
from psycop.common.model_training.data_loader.data_loader import DataLoader
from psycop_coercion.model_evaluation.config import TABLES_PATH
from psycop_coercion.model_evaluation.dataset_description.utils import (
    load_feature_set,
    table_one_coercion,
    table_one_demographics,
)


def main():
    # load train and test splits
    df = load_feature_set()

    # create table one - demographics
    table_one_d = table_one_demographics(df)
    table_one_d.to_csv(
        TABLES_PATH.parent.parent.parent / "table_one_demographics.csv",
        index=False,
    )

    # create table one - coericon
    table_one_c = table_one_coercion(df)
    table_one_c.to_csv(
        TABLES_PATH.parent.parent.parent / "table_one_coercion.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
