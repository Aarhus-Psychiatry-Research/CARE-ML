"""
This script creates the cohort for the psycop coercion project.
"""

from datetime import date

import numpy as np
import pandas as pd
from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop_ml_utils.sql.writer import write_df_to_sql

# load data
df_adm = sql_load(
    "SELECT * FROM fct.[FOR_kohorte_indhold_pt_journal_inkl_2021_feb2022]",
)  # only includes admissions in psychiatry (shak code starts with 6600)
df_coercion = sql_load(
    "SELECT * FROM fct.[FOR_tvang_alt_hele_kohorten_inkl_2021_feb2022]",
)  # includes coercion in both psychiatry and somatic


# ADMISSIONS DATA
# only keep admissions (not ambulatory visits)
df_adm = df_adm[df_adm["pt_type"] == "Indlagt"]

# only keep age >= 18 at the start of contact
df_adm = df_adm[df_adm["alder_start"] >= 18]

# only keep admissions after January 1st 2015 (so we can use use lookbehind windows of two years)
df_adm = df_adm[df_adm["datotid_start"] >= "2015-01-01"]

# only keep relevant columns
df_adm = df_adm[["dw_ek_borger", "datotid_start", "datotid_slut"]]

# COERCION DATA
# only include target coercion types: manual restraint, forced medication, and mechanical restraint, excluding voluntary mechanical restraint (i.e., "fastholdelse", "beroligende medicin", and "bæltefiksering", excluding "frivillig bæltefiksering")
df_coercion = df_coercion[
    (
        (df_coercion.typetekst_sei == "Bælte")
        & (df_coercion.begrundtekst_sei != "Frivillig bæltefiksering")
    )
    | (df_coercion.typetekst_sei == "Fastholden")
    | (df_coercion.typetekst_sei == "Beroligende medicin")
]

# only keep relevant columns
df_coercion = df_coercion[
    ["dw_ek_borger", "datotid_start_sei", "typetekst_sei", "behandlingsomraade"]
]


def concat_readmissions(
    df_patient: pd.DataFrame,
    readmission_interval_hours: int = 4,
) -> pd.DataFrame:
    """
    Concatenates individual readmissions into continuous admissions. An admission is defined as a readmission when the admission starts less than a specifiec number of
    hours after the previous admission. According to the Danish Health Data Authority, subsequent admissions should be regarded as one continious admission if the interval
    between admissions is less than four hours.
    In the case of multiple subsequent readmissions, all readmissions are concatenated into one admission.

    Args:
        df_patient (pd.DateFrame): A data frame containing an ID column, an admission time column and a discharge time column for the admissions of a unique ID.
        readmission_interval_hours (int): Number of hours between admissions determining whether an admission is considered a readmission. Defaults to 4, following
        advice from the Danish Health Data Authority.

    Returns:
        pd.DateFrame: A data frame containing an ID column, and admission time column and a discharge time column for all admissions of a
        unique ID with readmissions concatenated.
    """

    # 'end_readmission' indicates whether the end of the admission was followed be a readmission less than four hours later
    df_patient = df_patient.assign(
        end_readmission=lambda x: x["datotid_start"].shift(-1) - x["datotid_slut"]
        < pd.Timedelta(readmission_interval_hours, "hours"),
    )
    # 'start_readmission' indicates whether the admission started less than four hours later after the previous admission
    df_patient = df_patient.assign(
        start_readmission=lambda x: x["datotid_start"] - x["datotid_slut"].shift(1)
        < pd.Timedelta(readmission_interval_hours, "hours"),
    )

    # if the patients have any readmissions, the affected rows are subsetted
    if df_patient["end_readmission"].any() & df_patient["start_readmission"].any():
        readmissions = df_patient[
            (df_patient["end_readmission"] is True)
            | (df_patient["start_readmission"] is True)
        ]

        # if there are multiple subsequent readmissions (i.e., both 'end_readmission' and 'start_readmission' == True), all but the first and last are excluded
        readmissions_subset = readmissions[
            (readmissions.end_readmission is False)
            | (readmissions.start_readmission is False)
        ]

        # insert discharge time from the last readmission into the first
        readmissions_subset.loc[
            readmissions_subset.start_readmission is False,
            "datotid_slut",
        ] = readmissions_subset["datotid_slut"].shift(-1)

        # keep only the first admission
        readmissions_subset = readmissions_subset[
            readmissions_subset.end_readmission is True
        ]

        # remove readmissions from the original data
        df_patient_no_readmissions = df_patient.merge(
            readmissions[["dw_ek_borger", "datotid_start"]],
            how="outer",
            on=["dw_ek_borger", "datotid_start"],
            indicator=True,
        )
        df_patient_no_readmissions = df_patient_no_readmissions.loc[
            df_patient_no_readmissions["_merge"] != "both"
        ]

        # merge the new rows with the rest of the admissions
        df_patient_concatenated_readmissions = df_patient_no_readmissions.merge(
            readmissions_subset,
            how="outer",
            on=["dw_ek_borger", "datotid_start", "datotid_slut"],
        )

    else:
        return df_patient[
            ["dw_ek_borger", "datotid_start", "datotid_slut"]
        ].sort_values(["dw_ek_borger", "datotid_start"])

    return df_patient_concatenated_readmissions[
        ["dw_ek_borger", "datotid_start", "datotid_slut"]
    ].sort_values(["dw_ek_borger", "datotid_start"])


# sort based on patient and start of admission
df_adm = df_adm.sort_values(["dw_ek_borger", "datotid_start"])

# group by patient
df_patients = df_adm.groupby("dw_ek_borger")

# list of dfs; one for each patient
df_patients_list = [df_patients.get_group(key) for key in df_patients.groups]

# concatenate dataframes for individual patients
df_adm = pd.concat([concat_readmissions(patient) for patient in df_patients_list])

# for all patients, join all instances of coercion onto all admissions
df_cohort = df_adm.merge(df_coercion, how="left", on="dw_ek_borger")


# exclude admission if there has been an instance of coercion between 0 and 365 days before admission start (including 0 and 365)
df_excluded_admissions = df_cohort[
    (df_cohort.datotid_start - df_cohort.datotid_start_sei >= pd.Timedelta(0, "days"))
    & (
        df_cohort.datotid_start - df_cohort.datotid_start_sei
        <= pd.Timedelta(365, "days")
    )
][["dw_ek_borger", "datotid_start"]]

# remove duplicate rows, so we have one row per admission (instead of multiple rows for admissions with multiple coercion instances)
df_excluded_admissions = df_excluded_admissions.drop_duplicates(keep="first")

# outer join of admissions and excluded admissions with and indicator column ("_merge") denoting whether and observation occurs in both datasets
df_cohort = df_cohort.merge(
    df_excluded_admissions,
    how="outer",
    on=["dw_ek_borger", "datotid_start"],
    indicator=True,
)

# exclude rows that are in both datasets (i.e., exclude admissions in "df_excluded_admissions")
df_cohort = df_cohort.loc[df_cohort["_merge"] != "both"]


# only keep instances of coercion that occured during the particular admission
df_cohort_with_coercion = df_cohort[
    (df_cohort["datotid_start_sei"] > df_cohort["datotid_start"])
    & (df_cohort["datotid_start_sei"] < df_cohort["datotid_slut"])
]

# keep first time of coercion for each admission
# group by admission
df_admissions = df_cohort_with_coercion.groupby(["dw_ek_borger", "datotid_start"])
df_admissions_list = [df_admissions.get_group(key) for key in df_admissions.groups]


def first_coercion_within_admission(admission: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        admission (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    admission = admission.assign(
        first_mechanical_restraint=admission.datotid_start_sei[
            admission.typetekst_sei == "Bælte"
        ].min(),
    )
    admission = admission.assign(
        first_forced_medication=admission.datotid_start_sei[
            admission.typetekst_sei == "Beroligende medicin"
        ].min(),
    )
    admission = admission.assign(
        first_manual_restraint=admission.datotid_start_sei[
            admission.typetekst_sei == "Fastholden"
        ].min(),
    )

    return admission.drop(columns=["typetekst_sei", "_merge"])[
        (admission.datotid_start_sei == admission.datotid_start_sei.min())
    ].drop_duplicates()


df_cohort_with_coercion = pd.concat(
    [first_coercion_within_admission(admission) for admission in df_admissions_list],
)

# remove irrelevant columns from df_cohort, drop duplicates
df_cohort = df_cohort[
    ["dw_ek_borger", "datotid_start", "datotid_slut"]
].drop_duplicates()


# merge with df_cohort_coercion
df_cohort = df_cohort.merge(
    df_cohort_with_coercion,
    how="left",
    on=["dw_ek_borger", "datotid_start", "datotid_slut"],
)


# we exclude admissions with na discharge day and discharge day > 2021-11-22 due to legal restrictions
df_cohort = df_cohort[
    (df_cohort.datotid_slut.notna()) & (df_cohort.datotid_slut <= "2021-11-22")
]


# for each admission, we want to make a prediction every day
def unpack_adm_days(
    idx: int,
    row: pd.Series,
    pred_hour: int = 6,
) -> pd.DataFrame:
    """Unpack admissions to long format (one row per day in the admission)

    Args:
        idx (int): row index
        row (pd.DataFrame): one admission
        pred_hour (int): prediction hour. Defaults to 6.

    Returns:
        pd.DataFrame: _description_
    """

    row = pd.DataFrame(row).transpose()  # type: ignore

    # expand admission days between admission start and discharge
    adm_day = pd.DataFrame(
        pd.date_range(
            row.loc[idx, "datotid_start"].date(),
            row.loc[idx, "datotid_slut"].date(),
        ),
    )

    # add admission start to every day of admission
    adm_day["datotid_start"] = row.loc[idx, "datotid_start"]

    # join adm_day with row
    days_unpacked = pd.merge(row, adm_day, how="left", on="datotid_start")

    # add counter for days
    days_unpacked["pred_adm_day_count"] = (
        adm_day.groupby(by="datotid_start").cumcount() + 1
    )

    # add prediction time to prediction dates
    days_unpacked = days_unpacked.assign(
        pred_time=lambda x: x[0] + pd.Timedelta(hours=pred_hour),
    )

    # exclude admission start days where admission happens after prediction
    if days_unpacked.loc[0, "datotid_start"] >= days_unpacked.loc[0, "pred_time"]:  # type: ignore
        days_unpacked = days_unpacked.iloc[1:, :]

    # if admission is longer than 1 day and if time is <= pred_hour
    if (len(days_unpacked) > 1) and (
        days_unpacked.iloc[-1, 2].time() <= days_unpacked.iloc[-1, 10].time()  # type: ignore
    ):
        days_unpacked = days_unpacked.iloc[:-1, :]

    return days_unpacked.drop(columns=0)


# Apply the function unpack_adm_days to all patients
df_cohort = pd.concat([unpack_adm_days(idx, row) for idx, row in df_cohort.iterrows()])  # type: ignore


# Create include_pred_time_column (pred times were coercion hasn't happened yet or no coercion in the admission)
df_cohort["include_pred_time"] = np.where(
    (df_cohort.pred_time < df_cohort.datotid_start_sei)
    | (df_cohort.datotid_start_sei.isna()),
    1,
    0,
)


# load admission data again
df_adm = sql_load(
    "SELECT * FROM fct.[FOR_kohorte_indhold_pt_journal_inkl_2021_feb2022]",
)  # only includes admissions in psychiatry (shak code starts with 6600)

# only keep admission contacts
df_adm = df_adm[df_adm["pt_type"] == "Indlagt"]

# only keep age >= 18 at the start of contact
df_adm = df_adm[df_adm["alder_start"] >= 18]

# only keep admissions after January 1st 2015 (so we can use use lookbehind windows of two years)
df_adm = df_adm[df_adm["datotid_start"] >= "2015-01-01"]

# only keep relevant columns
df_adm = df_adm[["dw_ek_borger", "datotid_start", "shakkode_ansvarlig"]]

# left join df_adm on df_cohort
df_cohort = df_cohort.merge(
    df_adm,
    how="left",
    on=["dw_ek_borger", "datotid_start"],
)

# remove admissions in the department of forensic psychiatry (shak code 6600021 and 6600310)
df_cohort = df_cohort[
    (df_cohort["shakkode_ansvarlig"] != "6600310")
    & (df_cohort["shakkode_ansvarlig"] != "6600021")
]


# remove coercion in somatics
df_cohort = df_cohort[df_cohort["behandlingsomraade"] != "Somatikken"]


# write csv with today's date
today = date.today().strftime("%d%m%y")
df_cohort.to_csv(f"cohort_{today}.csv")

# Write to sql database
write_df_to_sql(
    df=df_cohort,
    table_name="psycop_coercion_cohort_with_all_days_without_labels_feb2022",
    if_exists="replace",
    rows_per_chunk=5000,
)
