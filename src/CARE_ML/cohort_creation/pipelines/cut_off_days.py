"""In this script, we cut off days in the cohort that are after the mean+std admission duration"""

from datetime import date

import pandas as pd
from psycop.common.feature_generation.loaders.raw.sql_load import sql_load

# ---------------------------------
# LOAD DATA
# ---------------------------------

# Load cohort
df_cohort = sql_load(query="SELECT * FROM fct.[psycop_coercion_within_2_days_feb2022]")

# Load train (to find cut-off)
df_train = pd.read_parquet(
    path="E:/shared_resources/coercion/feature_sets/psycop_coercion_adminsignbe_features_2023_03_31_13_39/psycop_coercion_adminsignbe_features_2023_03_31_13_39_train.parquet",
)

df_val = pd.read_parquet(
    path="E:/shared_resources/coercion/feature_sets/psycop_coercion_adminsignbe_features_2023_03_31_13_39/psycop_coercion_adminsignbe_features_2023_03_31_13_39_val.parquet",
)

df_train = pd.concat([df_train, df_val])

# ---------------------------------
# Train: Admission durations
# ---------------------------------

df_adm_grain = df_train[
    [
        "adm_id",
        "dw_ek_borger",
        "timestamp_admission",
        "timestamp_discharge",
        "outcome_timestamp",
    ]
].drop_duplicates(keep="first")

# calculate adm duration
df_adm_grain["adm_duration"] = (
    df_adm_grain["timestamp_discharge"] - df_adm_grain["timestamp_admission"]
)
df_train["adm_duration"] = (
    df_train["timestamp_discharge"] - df_train["timestamp_admission"]
)

# ---------------------------------
# Train: Cut-off definition
# ---------------------------------


# How many days and coericon instances will we lose?
def std_check(df: pd.DataFrame, df_adm_grain: pd.DataFrame, times: int = 1):
    cut_off = (
        df_adm_grain["adm_duration"].mean() + df_adm_grain["adm_duration"].std() * times
    )
    print(f"Cut_off: Mean + Std x {times} =", cut_off)

    n_excl_days = df[pd.to_timedelta(df["pred_adm_day_count"], "days") > cut_off].shape[  # type: ignore
        0
    ]
    n_days = df.shape[0]

    n_excl_days_with_outcome = df[
        (pd.to_timedelta(df["pred_adm_day_count"], "days") > cut_off)  # type: ignore
        & (df["outcome_coercion_bool_within_2_days"] == 1)
    ].shape[0]
    n_days_with_outcome = df[df["outcome_coercion_bool_within_2_days"] == 1].shape[0]

    print(
        "We will exclude",
        n_excl_days,
        "days out of",
        n_days,
        "days, corresponding to",
        round((n_excl_days / n_days) * 100, 2),
        "% of days/observations",
    )
    print(
        "We will lose",
        n_excl_days_with_outcome,
        "days with outcome out of",
        n_days_with_outcome,
        ", corresponding to",
        round((n_excl_days_with_outcome / n_days_with_outcome) * 100, 2),
        "% of days with outcome",
    )

    print(
        "We will include",
        n_days - n_excl_days,
        "days and",
        n_days_with_outcome - n_excl_days_with_outcome,
        "days with outcome",
    )


std_check(df_cohort, df_adm_grain, 1)
std_check(df_cohort, df_adm_grain, 2)
std_check(df_cohort, df_adm_grain, 3)

# ---------------------------------
# Cohort: Cut off days after cut-off
# ---------------------------------

cut_off = df_adm_grain["adm_duration"].mean() + df_adm_grain["adm_duration"].std()

df_cohort_exclude_days_after_cut_off = df_cohort[
    (pd.to_timedelta(df_cohort["pred_adm_day_count"], "days") <= cut_off)
]


# ---------------------------------
# Plot cohort: admission length before and after excluding days after cut-off
# ---------------------------------


# to be done


# ---------------------------------
# WRITE CSV / WRITE TO SQL DB
# ---------------------------------

# write csv named with today's date
today = date.today().strftime("%d%m%y")
lookahead_days = 2
df_cohort_exclude_days_after_cut_off.to_csv(
    f"psycop_coercion_within_{lookahead_days}_days_feb2022_exclude_days_after_cut_off_run_{today}.csv",
)
