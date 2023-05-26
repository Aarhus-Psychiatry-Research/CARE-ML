import pandas as pd
from psycop.common.model_training.application_modules.process_manager_setup import setup
from psycop.common.model_training.data_loader.data_loader import DataLoader
from psycop_coercion.model_evaluation.config import TABLES_PATH
from psycop_coercion.model_evaluation.dataset_description.utils import (
    table_one_coercion,
    table_one_demographics,
)


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
