"""
This script creates the labels for the psycop coercion project.

Labels: Kig to dage frem
- Hierarchy of coercion instances
"""

from datetime import date

import pandas as pd
from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycopmlutils.sql.writer import write_df_to_sql

df_cohort = sql_load(
    "psycop_coercion_cohort_with_all_days_without_labels_feb2022",
)  ## change to load from sql database when we have uploaded the cohort

lookahead_days = 2


# First coercion to pred_time
df_cohort["diff_first_coercion"] = pd.to_datetime(
    df_cohort["datotid_start_sei"],
) - pd.to_datetime(df_cohort["pred_time"])

# First mechanical restraint to pred_time
df_cohort["diff_first_mechanical_restraint"] = pd.to_datetime(
    df_cohort["first_mechanical_restraint"],
) - pd.to_datetime(df_cohort["pred_time"])

# First forced medication to pred_time
df_cohort["diff_first_forced_medication"] = pd.to_datetime(
    df_cohort["first_forced_medication"],
) - pd.to_datetime(df_cohort["pred_time"])

# First manual restraint to pred_time
df_cohort["diff_first_manual_restraint"] = pd.to_datetime(
    df_cohort["first_manual_restraint"],
) - pd.to_datetime(df_cohort["pred_time"])


# function for creating labels depending on lookahead window (in days)
def create_labels(df: pd.DataFrame, lookahead_days: int) -> pd.DataFrame:
    """Depending on the lookahead window, create label columns indicating
    - outcome_coercion_bool_within_{lookahead_days}_days (int): 0 or 1, depending on whether a coercion instance happened within the lookahead window.
    - outcome_coercion_type_within_{lookahead_days}_days (int): which coercion type happened within the lookahead window.
    - outcome_mechanical_restraint_bool_within_{lookagead_days}_days (int): 0 or 1, depending on whether mechanical restrained happened within the lookahead window.

    If multiple coercion types happened within the lookahead window, the most "severe" coercion type will be chosen.
    Coercion hierarchy/classes (least to most severe coercion type):
    0) No coercion
    1) Manual restraint
    2) Forced medication
    3) Mechanial restraint

    Args:
        df (pd.DataFrame): coercion cohort dataframe
        lookahead_days (int): How far to look for coercion instances in days

    Returns:
        pd.DateFrame: df with three added columns (outcome_coercion_bool_within_{lookahead_days}_days, outcome_coercion_type_within_{lookahead_days}_days, and outcome_mechanical_restraint_bool_within_{lookagead_days}_days)
    """

    # Outcome bool
    df[f"outcome_coercion_bool_within_{lookahead_days}_days"] = 0
    df.loc[
        (df["diff_first_coercion"] < pd.Timedelta(f"{lookahead_days} days"))
        & (df["include_pred_time"] == 1),
        f"outcome_coercion_bool_within_{lookahead_days}_days",
    ] = 1

    # Outcome type
    df[f"outcome_coercion_type_within_{lookahead_days}_days"] = 0

    # Mechanical restraint (3)
    df.loc[
        (df[f"outcome_coercion_bool_within_{lookahead_days}_days"] == 1)
        & (
            df["diff_first_mechanical_restraint"]
            < pd.Timedelta(f"{lookahead_days} days")
        ),
        f"outcome_coercion_type_within_{lookahead_days}_days",
    ] = 3

    # Forced medication (2)
    df.loc[
        (df[f"outcome_coercion_bool_within_{lookahead_days}_days"] == 1)
        & (df[f"outcome_coercion_type_within_{lookahead_days}_days"] != 3)
        & (df["diff_first_forced_medication"] < pd.Timedelta(f"{lookahead_days} days")),
        f"outcome_coercion_type_within_{lookahead_days}_days",
    ] = 2

    # Manual restraint (1)
    df.loc[
        (df[f"outcome_coercion_bool_within_{lookahead_days}_days"] == 1)
        & (df[f"outcome_coercion_type_within_{lookahead_days}_days"] != 2)
        & (df[f"outcome_coercion_type_within_{lookahead_days}_days"] != 3)
        & (df["diff_first_manual_restraint"] < pd.Timedelta(f"{lookahead_days} days")),
        f"outcome_coercion_type_within_{lookahead_days}_days",
    ] = 1

    # Outcome col with only mechanical restraint
    df[f"outcome_mechanical_restraint_bool_within_{lookahead_days}_days"] = 0
    df.loc[
        (df[f"outcome_coercion_type_within_{lookahead_days}_days"] == 3),
        f"outcome_mechanical_restraint_bool_within_{lookahead_days}_days",
    ] = 1

    return df


# apply create_labels function to data
df_cohort = create_labels(df_cohort, lookahead_days=lookahead_days)


# only include admission days at which coercion has not happened yet
df_cohort = df_cohort[df_cohort["include_pred_time"] == 1]


# Rename columns, drop irrelevant columns, and reset index
df_cohort = (
    df_cohort.rename(
        columns={
            "datotid_start": "timestamp_admission",
            "datotid_slut": "timestamp_discharge",
            "datotid_start_sei": "outcome_timestamp",
            "pred_time": "timestamp",
        },
    )
    .drop(
        columns=(
            [
                "behandlingsomraade",
                "first_mechanical_restraint",
                "first_forced_medication",
                "first_manual_restraint",
                "include_pred_time",
                "diff_first_coercion",
                "diff_first_mechanical_restraint",
                "diff_first_forced_medication",
                "diff_first_manual_restraint",
            ]
        ),
    )
    .reset_index(drop=True)
)

# add admission id
df_cohort.insert(
    loc=0,
    column="adm_id",
    value=df_cohort.dw_ek_borger.astype(str)
    + "-"
    + df_cohort.timestamp_admission.astype(str),
)


# write csv named with today's date
today = date.today().strftime("%d%m%y")
df_cohort.to_csv(
    f"psycop_coercion_within_{lookahead_days}_days_feb2022_run_{today}.csv",
)

# Write to sql database
write_df_to_sql(
    df=df_cohort,
    table_name=f"psycop_coercion_within_{lookahead_days}_days_feb2022",
    if_exists="replace",
    rows_per_chunk=5000,
)
