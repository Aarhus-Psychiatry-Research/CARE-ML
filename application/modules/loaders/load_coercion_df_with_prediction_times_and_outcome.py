import pandas as pd

# feature generation module for some reason not on path, so fix to add here
# import sys
# sys.path.insert(12,'E:\\sara\\coercion-feature-generation\\src\\psycop-feature-generation\\src')

from psycop_feature_generation.loaders.raw import sql_load
from wasabi import msg


class LoadCoercion:
    """Class for loading data frames with prediction times and outcomes for
    coercion data."""

    def coercion_df(
        timestamps_only: bool = True,
    ) -> pd.DataFrame:

        df = sql_load(
            "SELECT * FROM [fct].[psycop_coercion_within_2_days_feb2022]",
            database="USR_PS_FORSK",
            chunksize=None,
        )

        if timestamps_only:
            df = df[["dw_ek_borger", "timestamp"]]

        msg.good(
            "Finished loading data frame for coercion with prediction times and outcomes.",
        )

        return df.reset_index(drop=True)
