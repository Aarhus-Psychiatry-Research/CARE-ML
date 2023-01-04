import pandas as pd

# feature generation module for some reason not on path, so fix to add here
import sys
sys.path.insert(12,'E:\\sara\\coercion-feature-generation\\src\\psycop-feature-generation\\src')

from psycop_feature_generation.loaders.raw import sql_load
from wasabi import msg

# class LoadCoercion():
#     """Class for loading data frames with prediction times and outcome for
#     forced admissions."""

#     def forced_admissions_inpatient(timestamps_only=True):

#         df = LoadCoercion.process_forced_pred_dfs(
#             visit_type="Indlagt",
#             prediction_times_col_name="datotid_slut",
#             timestamps_only=timestamps_only,
#         )

#         msg.good(
#             "Finished loading data frame for forced admissions with prediction times and outcome for all inpatient admissions",
#         )

#         return df.reset_index(drop=True)

#     def forced_admnissions_outpatient(timestamps_only=True):

#         df = LoadCoercion.process_forced_pred_dfs(
#             visit_type="Ambulant",
#             prediction_times_col_name="datotid_predict",
#             timestamps_only=timestamps_only
#         )

#         msg.good(
#             "Finished loading data frame for forced admissions with prediction times and outcome for all outpatient visits",
#         )

#         return df.reset_index(drop=True)

#     def process_forced_pred_dfs(
#         visit_type: str,
#         prediction_times_col_name: str,
#         timestamps_only: bool = True,
#     ) -> pd.DataFrame:

#         df = sql_load(
#             f"SELECT * FROM [fct].[psycop_fa_outcome_all_disorders_tvangsindlaeg_{visit_type}_2y_0f_2015-2021]",
#             database="USR_PS_FORSK",
#             chunksize=None,
#         )

#         df = df[["dw_ek_borger", prediction_times_col_name, "six_month"]]

#         df.rename(
#             columns={
#                 prediction_times_col_name: "timestamp",
#                 "six_month": "outcome_forced_admission_within_6_months",
#             },
#             inplace=True,
#         )
#         df["timestamp"] = pd.to_datetime(df["timestamp"])

#         if timestamps_only:
#             df = df[["dw_ek_borger", "timestamp"]]

#         return df

class LoadCoercion:
    """Class for loading data frames with prediction times and outcomes for
    coercion data."""
    
    def coercion_df(
        timestamps_only: bool = True,
        ) -> pd.DataFrame:

        df = sql_load(
            "SELECT * FROM [fct].[psycop_coercion_within_2_days]",
            database="USR_PS_FORSK",
            chunksize=None,
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if timestamps_only:
            df = df[["dw_ek_borger", "timestamp"]]

        msg.good(
            "Finished loading data frame for coercion with prediction times and outcomes.",
        )

        return df.reset_index(drop=True)