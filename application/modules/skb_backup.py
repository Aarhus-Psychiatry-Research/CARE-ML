"""Feature specification module."""
import logging
from sys import prefix

import numpy as np

from psycop_feature_generation.application_modules.project_setup import ProjectInfo
from timeseriesflattener.feature_spec_objects import (
    BaseModel,
    OutcomeSpec,
    PredictorGroupSpec,
    PredictorSpec,
    StaticSpec,
    _AnySpec,
)


log = logging.getLogger(__name__)


class SpecSet(BaseModel):
    """A set of unresolved specs, ready for resolving."""

    temporal_predictors: list[PredictorSpec]
    static_predictors: list[StaticSpec]
    outcomes: list[OutcomeSpec]
    metadata: list[_AnySpec]


class FeatureSpecifier:
    """Specify features based on prediction time."""

    def __init__(self, project_info: ProjectInfo, min_set_for_debug: bool = False):
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info

    def _get_static_predictor_specs(self):
        """Get static predictor specs."""
        return [
            StaticSpec(
                values_loader="sex_female",
                input_col_name_override="sex_female",
                prefix=self.project_info.prefix.predictor,
            ),
        ]

    def _get_visits_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get visits specs."""
        log.info("–––––––– Generating visits specs ––––––––")

        visits = PredictorGroupSpec(
            values_loader=(
                "physical_visits",
                "physical_visits_to_psychiatry",
                "physical_visits_to_somatic",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return visits

    def _get_admissions_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get admissions specs."""
        log.info("–––––––– Generating admissions specs ––––––––")

        admissions = PredictorGroupSpec(
            values_loader=(
                "admissions",
                "admissions_to_psychiatry",
                "admissions_to_somatic",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return admissions

    def _get_medication_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get medication specs."""
        log.info("–––––––– Generating medication specs ––––––––")

        psychiatric_medications = PredictorGroupSpec(
            values_loader=(
                "antipsychotics",
                "anxiolytics",
                "hypnotics and sedatives",
                "antidepressives",
                "olanzapine",
                "clozapine",
                "lithium",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return psychiatric_medications

    def _get_coercion_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get coercion specs."""
        log.info("–––––––– Generating coercion specs ––––––––")

        coercion = PredictorGroupSpec(
            values_loader=(
                "skema_1",
                "tvangstilbageholdelse",
                "skema_2_without_nutrition",
                "medicinering",
                "ect",
                "af_legemlig_lidelse",
                "skema_3",
                "fastholden",
                "baelte",
                "remme",
                "farlighed",
            ),
            resolve_multiple_fn=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return coercion

    def _get_beroligende_medicin_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get beroligende medicin specs."""
        log.info("–––––––– Generating beroligende medicin specs ––––––––")

        beroligende_medicin = PredictorGroupSpec(
            values_loader=("beroligende_medicin",),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return beroligende_medicin

    def _get_structured_sfi_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get structured sfi specs."""
        log.info("–––––––– Generating structured sfi specs ––––––––")

        structured_sfi = PredictorGroupSpec(
            values_loader=(
                "broeset_violence_checklist",
                "selvmordsrisiko",
                "hamilton_d17",
                "mas_m",
            ),
            resolve_multiple_fn=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return structured_sfi

    def _get_bvc_physical_threats_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get Brøset Violence Checklist physical threats specs."""
        log.info("–––––––– Generating BVC physical threats specs ––––––––")

        bvc_physical_threats = PredictorGroupSpec(
            values_loader=("broeset_violence_checklist_physical_threats",),
            resolve_multiple_fn=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return bvc_physical_threats

    def _get_temporal_predictor_specs(self) -> list[PredictorSpec]:
        """Generate predictor spec list."""
        log.info("–––––––– Generating temporal predictor specs ––––––––")

        if self.min_set_for_debug:
            return [
                PredictorSpec(
                    values_loader="f3_disorders",
                    lookbehind_days=100,
                    resolve_multiple_fn="bool",
                    fallback=np.nan,
                    allowed_nan_value_prop=0,
                    prefix=self.project_info.prefix.predictor,
                )
            ]

        interval_days = [10, 30, 180]
        allowed_nan_value_prop = [0]

        visits = self._get_visits_specs(
            resolve_multiple=["count"],
            interval_days=[30, 180, 365],
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        admissions = self._get_admissions_specs(
            resolve_multiple=["count", "sum"],
            interval_days=[10, 30, 180],
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        diagnoses = self._get_diagnoses_specs(
            resolve_multiple=["count", "bool"],
            interval_days=[30, 180],  # but also "max"? is that two years = 730?
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        medications = self._get_medication_specs(
            resolve_multiple=["count", "bool"],
            interval_days=interval_days,
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        beroligende_medicin = self._get_beroligende_medicin_specs(
            resolve_multiple=["count", "bool"],
            interval_days=interval_days,
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        coercion = self._get_coercion_specs(
            resolve_multiple=["count", "sum", "bool"],
            interval_days=interval_days,
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        structured_sfi = self._get_structured_sfi_specs(
            resolve_multiple=["mean", "max", "min", "change_per_day", "variance"],
            interval_days=[1, 3, 10, 30, 180],
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        bvc_physical_threats = self._get_bvc_physical_threats_specs(
            resolve_multiple=["count", "bool"],
            interval_days=[1, 3, 10, 30, 180],
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        return (
            visits
            + admissions
            + medications
            # + diagnoses
            + beroligende_medicin
            + coercion
            + structured_sfi
            + bvc_physical_threats
        )

    def get_feature_specs(self) -> list[_AnySpec]:
        """Get a spec set."""

        if self.min_set_for_debug:
            return (
                self._get_temporal_predictor_specs()
                # + self._get_outcome_specs()
            )

        return self._get_temporal_predictor_specs() + self._get_static_predictor_specs()

class FeatureSpecifier_AdmissionDay:
    """Specify features based on prediction time."""

    def __init__(self, project_info: ProjectInfo, min_set_for_debug: bool = False):
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info

    def _get_diagnoses_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get diagnoses specs."""
        log.info("–––––––– Generating diagnoses specs ––––––––")

        psychiatric_diagnoses = PredictorGroupSpec(
            values_loader=(  
                "f0_disorders",
                "dementia",  # ["f00", "f01", "f02", "f03", "f04"]
                "delirium",  # "f05"
                "miscellaneous_organic_mental_disorders",  # ["f06", "f07", "f09"],
                # f08 is missing? because it doesn't exist?
                "f1_disorders",
                "alcohol_dependency",  # "f10"
                "opioid_dependency",  # f11
                "cannabinoid_dependency",  # f12
                "sedative_dependency",  # f13
                "stimulant_dependencies",  # ["f14", "f15"]
                "hallucinogen_dependency",  # f16
                "tobacco_dependency",  # f17
                "miscellaneous_drug_dependencies",  # ["f18", "f19"]
                "f2_disorders",
                "schizophrenia",  # f20
                "schizoaffective",  # f25
                "miscellaneous_psychotic_disorders",  # ["f21", "f22", "f23", "f24", "f28", "f29"]
                # f26 and f27 missing? because they don't exist?
                "f3_disorders",
                "manic_and_bipolar",  # ["f30", "f31"]
                "depressive_disorders",  # ["f32", "f33", "f34", "f38"]
                "miscellaneous_affective_disorders",  # ["f38", "f39"],
                # f35, f37 missing? because they don't exist? - and f38 (Other mood [affective] disorders) twice??
                "f4_disorders",
                "f6_disorders",
                "cluster_a",  # ["f600", "f601"]
                "cluster_b",  # ["f602", "f603", "f604"]
                "cluster_c",  # ["f605", "f606", "f607"]
                "miscellaneous_personality_disorders",  # ["f608", "f609", "f61", "f62", "f63", "f68", "f69"]
                "sexual_disorders",  # ["f65", "f66"]
                # f64 sexual identity disorders is excluded (by Martin/Lasse/Kenneth/Frida/Jakob), but why?
                # f68 (Other disorders of adult personality and behaviour) and f69 (Unspecified disorder of adult personality and behaviour) missing
                "f7_disorders",
                "f8_disorders",
                "f9_disorders",
            ),
            resolve_multiple_fn=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return psychiatric_diagnoses

    def _get_temporal_predictor_specs(self) -> list[PredictorSpec]:
        """Generate predictor spec list."""
        log.info("–––––––– Generating temporal predictor specs ––––––––")

        if self.min_set_for_debug:
            return [
                PredictorSpec(
                    values_loader="f3_disorders",
                    lookbehind_days=100,
                    resolve_multiple_fn="bool",
                    fallback=np.nan,
                    allowed_nan_value_prop=0,
                    prefix=self.project_info.prefix.predictor,
                )
            ]

        # interval_days = [10, 30, 180]
        allowed_nan_value_prop = [0]

        psychiatric_diagnoses = self._get_diagnoses_specs(
            resolve_multiple=["count", "bool"],
            interval_days=[30, 180],  # but also "max"? is that two years = 730?
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        return psychiatric_diagnoses

    def get_feature_specs(self) -> list[_AnySpec]:
        """Get a spec set."""

        if self.min_set_for_debug:
            return self._get_temporal_predictor_specs()

        return self._get_temporal_predictor_specs()


"""main"""

"""Main feature generation."""

import logging

# had to add application - something's up with the paths
from application.modules.specify_features import (
    FeatureSpecifier,
    FeatureSpecifier_AdmissionDay,
)
from application.modules.loaders.load_coercion_df_with_prediction_times_and_outcome import (
    LoadCoercion,
)
from psycop_feature_generation.application_modules.describe_flattened_dataset import (
    save_flattened_dataset_description_to_disk,
)
from psycop_feature_generation.application_modules.flatten_dataset import (
    create_flattened_dataset,
)
from psycop_feature_generation.application_modules.loggers import init_root_logger
from psycop_feature_generation.application_modules.project_setup import (
    get_project_info,
    init_wandb,
)
from psycop_feature_generation.application_modules.save_dataset_to_disk import (
    split_and_save_dataset_to_disk,
)
from psycop_feature_generation.application_modules.wandb_utils import (
    wandb_alert_on_exception,
)
from psycop_feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)

log = logging.getLogger()


@wandb_alert_on_exception
def main():
    """Main function for loading, generating and evaluating a flattened
    dataset."""
    feature_specs = FeatureSpecifier(
        project_info=project_info,
        min_set_for_debug=True,  # Remember to set to False when generating full dataset
    ).get_feature_specs()

    flattened_df = create_flattened_dataset(
        feature_specs=feature_specs,
        prediction_times_df=LoadCoercion.coercion_df(timestamps_only=False),
        drop_pred_times_with_insufficient_look_distance=False,
        project_info=project_info,
        quarantine_df=load_move_into_rm_for_exclusion(),
        quarantine_days=730,
    )

    # specify admission day features
    feature_specs_admission_day = FeatureSpecifier_AdmissionDay(
        project_info=project_info,
        min_set_for_debug=False,  # Remember to set to False when generating full dataset
    ).get_feature_specs()

    # prepare coercion_df_admissionday for admission day features
    coercion_df = LoadCoercion.coercion_df(timestamps_only=False)
    coercion_df_admission_day = coercion_df.iloc[:, 1:]
    coercion_df_admission_day = coercion_df_admission_day.rename(
        columns={"timestamp_admission": "timestamp"}
    )
    coercion_df_admission_day = coercion_df_admission_day.sort_values(
        by=["dw_ek_borger", "timestamp"]
    )
    coercion_df_admission_day = coercion_df_admission_day.drop_duplicates(
        subset=["timestamp", "dw_ek_borger"],
        keep="first",
    )

    # coercion_df_admission_day.admission_count_days.unique()
    # array([  2,   1,  35,   8,   9,  30,  12,   7,  11,   6,  14,   3,  98, 115], dtype=int64)

    # create flattened df for admission day features
    flattened_df_admission_day = create_flattened_dataset(
        feature_specs=feature_specs_admission_day,
        prediction_times_df=coercion_df_admission_day,
        drop_pred_times_with_insufficient_look_distance=False,
        project_info=project_info,
        quarantine_df=load_move_into_rm_for_exclusion(),
        quarantine_days=730,  # should we do something differently with quarantine days for admission days vs pred time?
        # timestamp_col_name=project_info.col_names.admission,
    )

    flattened_df_admission_day = flattened_df_admission_day.rename(
        columns={"timestamp": "timestamp_admission"}
    )

    # join flattened_df and flattened_df_admission_day
    """
    flattened_df_admission_day: 49773 rows x 22 columns, 
    flattened_df: [816221 rows x 8 columns]
    """

    merge = flattened_df.merge(
        flattened_df_admission_day, on=["timestamp_admission", "dw_ek_borger"]
    )
    """
    merge: [808373 rows x 28 columns]
    """
    split_and_save_dataset_to_disk(
        flattened_df=flattened_df,
        project_info=project_info,
    )

    save_flattened_dataset_description_to_disk(
        project_info=project_info,
    )


if __name__ == "__main__":
    # Run elements that are required before wandb init first,
    # then run the rest in main so you can wrap it all in
    # wandb_alert_on_exception, which will send a slack alert
    # if you have wandb alerts set up in wandb
    project_info = get_project_info(
        project_name="coercion",
    )

    init_root_logger(project_info=project_info)

    log.info(f"Stdout level is {logging.getLevelName(log.level)}")
    log.debug("Debugging is still captured in the log file")

    # Use wandb to keep track of your dataset generations
    # Makes it easier to find paths on wandb, as well as
    # allows monitoring and automatic slack alert on failure
    init_wandb(
        project_info=project_info,
    )

    main()




def load_from_codes(
    codes_to_match: Union[list[str], str],
    load_diagnoses: bool,
    code_col_name: str,
    source_timestamp_col_name: str,
    view: str,
    output_col_name: Optional[str] = None,
    match_with_wildcard: bool = True,
    n_rows: Optional[int] = None,
    exclude_codes: Optional[list[str]] = None,
    shak_location_col: Optional[str] = None,
    shak_code: Optional[int] = None,
    shak_sql_operator: Optional[str] = "=",
) -> pd.DataFrame:
    """Load the visits that have diagnoses that match icd_code or atc code from
    the beginning of their adiagnosekode or atc code string. Aggregates all
    that match.

    Args:
        codes_to_match (Union[list[str], str]): Substring(s) to match diagnoses or medications for.
            Diagnoses: Matches any diagnoses, whether a-diagnosis, b-diagnosis.
            Both: If a list is passed, will count as a match if any of the icd_codes or at codes in the list match.
        load_diagnoses (bool): Determines which mathing logic is employed. If True, will load diagnoses. If False, will load medications.
            Diagnoses must be able to split a string like this:
                A:DF431#+:ALFC3#B:DF329
            Which means that if match_with_wildcard is False, we must match on *icd_code# or *icd_code followed by nothing. If it's true, we can match on *icd_code*.
        code_col_name (str): Name of column containing either diagnosis (icd) or medication (atc) codes.
            Takes either 'diagnosegruppestreng' or 'atc' as input.
        source_timestamp_col_name (str): Name of the timestamp column in the SQL
            view.
        view (str): Name of the SQL view to load from.
        output_col_name (str, optional): Name of new column string. Defaults to
            None.
        match_with_wildcard (bool, optional): Whether to match on icd_code* / atc_code*.
            Defaults to true.
        n_rows: Number of rows to return. Defaults to None.
        exclude_codes (list[str], optional): Drop rows if their code is in this list. Defaults to None.
        shak_location_col (str, optional): Name of column containing shak code. Defaults to None. Combine with shak_code and shak_sql_operator.
            shak_code (int, optional): Shak code indicating where to keep/not keep visits from (e.g. 6600).
            shak_sql_operator, (str, optional): Operator indicating how to filter shak_code, e.g. "!= 6600". Defaults to "=".

    Returns:
        pd.DataFrame: A pandas dataframe with dw_ek_borger, timestamp and
            output_col_name = 1
    """
    fct = f"[{view}]"

    if isinstance(codes_to_match, list) and len(codes_to_match) > 1:
        match_col_sql_str = list_to_sql_logic(
            codes_to_match=codes_to_match,
            code_sql_col_name=code_col_name,
            load_diagnoses=load_diagnoses,
            match_with_wildcard=match_with_wildcard,
        )
    elif isinstance(codes_to_match, str):
        match_col_sql_str = str_to_sql_match_logic(
            code_to_match=codes_to_match,
            code_sql_col_name=code_col_name,
            load_diagnoses=load_diagnoses,
            match_with_wildcard=match_with_wildcard,
        )
    else:
        raise ValueError("codes_to_match must be either a list or a string.")

    sql = (
        f"SELECT dw_ek_borger, {source_timestamp_col_name}, {code_col_name} "
        + f"FROM [fct].{fct} WHERE {source_timestamp_col_name} IS NOT NULL AND ({match_col_sql_str})"
    )

    if shak_code is not None:
        sql += f" AND left({shak_location_col}, {len(str(shak_code))}) {shak_sql_operator} {str(shak_code)}"

    df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n_rows=n_rows)

    if exclude_codes:
        # Drop all rows whose code_col_name is in exclude_code
        df = df[~df[code_col_name].isin(exclude_codes)]

    if output_col_name is None:
        if isinstance(codes_to_match, list):
            output_col_name = "_".join(codes_to_match)
        else:
            output_col_name = codes_to_match

    df[output_col_name] = 1

    df.drop([f"{code_col_name}"], axis="columns", inplace=True)

    return df.rename(
        columns={
            source_timestamp_col_name: "timestamp",
        },
    )



def from_contacts(
    icd_code: Union[list[str], str],
    output_col_name: Optional[str] = "value",
    n_rows: Optional[int] = None,
    wildcard_icd_code: Optional[bool] = False,
    shak_location_col: Optional[str] = None,
    shak_code: Optional[int] = None,
    shak_sql_operator: Optional[str] = None
) -> pd.DataFrame:
    """Load diagnoses from all hospital contacts. If icd_code is a list, will
    aggregate as one column (e.g. ["E780", "E785"] into a ypercholesterolemia
    column).

    Args:
        icd_code (str): Substring to match diagnoses for. Matches any diagnoses, whether a-diagnosis, b-diagnosis etc. # noqa: DAR102
        output_col_name (str, optional): Name of new column string. Defaults to "value".
        n_rows: Number of rows to return. Defaults to None.
        wildcard_icd_code (bool, optional): Whether to match on icd_code*. Defaults to False.
        shak_location_col (str, optional): Name of column containing shak code. Defaults to None. Combine with shak_code and shak_sql_operator.  
            shak_code (int, optional): Shak code indicating where to keep/not keep visits from (e.g. 6600). 
            shak_sql_operator, (str, optional): Operator indicating how to filter shak_code, e.g. "!= 6600" or. Defaults to None.

    Returns:
        pd.DataFrame
    """

    df = load_from_codes(
        codes_to_match=icd_code,
        code_col_name="diagnosegruppestreng",
        source_timestamp_col_name="datotid_slut",
        view="FOR_kohorte_indhold_pt_journal_psyk_somatik_inkl_2021_feb2022",
        output_col_name=output_col_name,
        match_with_wildcard=wildcard_icd_code,
        n_rows=n_rows,
        load_diagnoses=True,
        shak_location_col="shakkode_ansvarlig", 
        shak_code=shak_code
        shak_sql_operator=shak_sql_operator
    )

    df = df.drop_duplicates(
        subset=["dw_ek_borger", "timestamp", output_col_name],
        keep="first",
    )

    return df.reset_index(drop=True)




## specify features anno 08/02
"""Feature specification module."""
import logging
from sys import prefix

import numpy as np

from psycop_feature_generation.application_modules.project_setup import ProjectInfo
from timeseriesflattener.feature_spec_objects import (
    BaseModel,
    OutcomeSpec,
    PredictorGroupSpec,
    PredictorSpec,
    StaticSpec,
    _AnySpec,
)


log = logging.getLogger(__name__)


class SpecSet(BaseModel):
    """A set of unresolved specs, ready for resolving."""

    temporal_predictors: list[PredictorSpec]
    static_predictors: list[StaticSpec]
    outcomes: list[OutcomeSpec]
    metadata: list[_AnySpec]


class FeatureSpecifier:
    """Specify features based on prediction time."""

    def __init__(self, project_info: ProjectInfo, min_set_for_debug: bool = False):
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info

    def _get_static_predictor_specs(self):
        """Get static predictor specs."""
        return [
            StaticSpec(
                values_loader="sex_female",
                input_col_name_override="sex_female",
                prefix=self.project_info.prefix.predictor,
            ),
        ]

    def _get_visits_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get visits specs."""
        log.info("–––––––– Generating visits specs ––––––––")

        visits = PredictorGroupSpec(
            values_loader=(
                "physical_visits",
                # "physical_visits_to_psychiatry_with_value",
                "physical_visits_to_somatic",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return visits

    def _get_admissions_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get admissions specs."""
        log.info("–––––––– Generating admissions specs ––––––––")

        admissions = PredictorGroupSpec(
            values_loader=(
                "admissions",
                "admissions_to_psychiatry",
                "admissions_to_somatic",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return admissions

    def _get_medication_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get medication specs."""
        log.info("–––––––– Generating medication specs ––––––––")

        psychiatric_medications = PredictorGroupSpec(
            values_loader=(
                "antipsychotics",
                "anxiolytics",
                "hypnotics and sedatives",
                "antidepressives",
                "olanzapine",
                "clozapine",
                "lithium",
                "alcohol_abstinence",
                # "opioid_dependency",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return psychiatric_medications

    def _get_diagnoses_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get diagnoses specs."""
        log.info("–––––––– Generating diagnoses specs ––––––––")

        psychiatric_diagnoses = PredictorGroupSpec(
            values_loader=(
                "f0_disorders",
                "f1_disorders",
                "f2_disorders",
                "f3_disorders",
                "f4_disorders",
                "f6_disorders",
                "f7_disorders",
                "f8_disorders",
                "f9_disorders",
            ),
            resolve_multiple_fn=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return psychiatric_diagnoses

    def _get_prior_coercion_specs(  ## ready for merge
        self,
        resolve_multiple,
        interval_days,
        allowed_nan_value_prop,
    ):
        """Get coercion specs."""
        log.info("–––––––– Generating prior coercion specs ––––––––")

        admission_time_coercion = PredictorGroupSpec(
            values_loader=(
                "skema_1",
                "tvangsindlaeggelse",
                "tvangstilbageholdelse",
                "paa_grund_af_farlighed",  # røde papirer
                "af_helbredsmaessige_grunde",  # gule papirer
                "skema_2_without_nutrition",
                "medicinering",
                "ect",
                "af_legemlig_lidelse",
            ),
            resolve_multiple_fn=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return admission_time_coercion

    # def _get_coercion_admission_time_specs(  ######### at admission time ##########
    #     self, resolve_multiple, interval_days, allowed_nan_value_prop
    # ):
    #     """Get coercion specs at admission time."""
    #     log.info("–––––––– Generating static/at admission time coercion specs ––––––––")

    #     admission_time_coercion = PredictorGroupSpec(
    #         values_loader=(
    #             "tvangsindlaeggelse" "paa_grund_af_farlighed",  # røde papirer
    #             "af_helbredsmaessige_grunde",  # gule papirer
    #         ),
    #         resolve_multiple_fn=resolve_multiple,
    #         lookbehind_days=interval_days,
    #         fallback=[0],
    #         allowed_nan_value_prop=allowed_nan_value_prop,
    #     ).create_combinations()

    # return admission_time_coercion

    def _get_coercion_schema_3_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get schema 3 coercion specs."""
        log.info("–––––––– Generating schema 3 coercion specs ––––––––")

        schema_3_coercion = PredictorGroupSpec(
            values_loader=(
                "skema_3",
                "fastholden",
                "baelte",
                "remme",
                "farlighed",
            ),
            resolve_multiple_fn=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return schema_3_coercion

    def _get_beroligende_medicin_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get beroligende medicin specs."""
        log.info("–––––––– Generating beroligende medicin specs ––––––––")

        beroligende_medicin = PredictorGroupSpec(
            values_loader=("beroligende_medicin",),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return beroligende_medicin

    def _get_structured_sfi_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get structured sfi specs."""
        log.info("–––––––– Generating structured sfi specs ––––––––")

        structured_sfi = PredictorGroupSpec(
            values_loader=(
                "broeset_violence_checklist",
                "selvmordsrisiko",
                "hamilton_d17",
                "mas_m",
            ),
            resolve_multiple_fn=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return structured_sfi

    # def _get_bvc_physical_threats_specs(
    #     self, resolve_multiple, interval_days, allowed_nan_value_prop
    # ):
    #     """Get Brøset Violence Checklist physical threats specs."""
    #     log.info("–––––––– Generating BVC physical threats specs ––––––––")

    #     bvc_physical_threats = PredictorGroupSpec(
    #         values_loader=("broeset_violence_checklist_physical_threats",),
    #         resolve_multiple_fn=resolve_multiple,
    #         lookbehind_days=interval_days,
    #         fallback=[0],
    #         allowed_nan_value_prop=allowed_nan_value_prop,
    #     ).create_combinations()

    #     return bvc_physical_threats

    def _get_temporal_predictor_specs(self) -> list[PredictorSpec]:
        """Generate predictor spec list."""
        log.info("–––––––– Generating temporal predictor specs ––––––––")

        if self.min_set_for_debug:
            return [
                PredictorSpec(
                    values_loader="f0_disorders",
                    lookbehind_days=100,
                    resolve_multiple_fn="bool",
                    fallback=np.nan,
                    allowed_nan_value_prop=0,
                    prefix=self.project_info.prefix.predictor,
                )
            ]

        allowed_nan_value_prop = [0]

        visits = self._get_visits_specs(
            resolve_multiple=["count"],
            interval_days=[30, 180, 365],
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        admissions = self._get_admissions_specs(
            resolve_multiple=["count", "sum"],
            interval_days=[10, 30, 180],
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        diagnoses = self._get_diagnoses_specs(
            resolve_multiple=["bool"],
            interval_days=[30, 180, 730],
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        medications = self._get_medication_specs(
            resolve_multiple=["count", "bool"],
            interval_days=[1, 3, 10, 30, 365, 730],
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        beroligende_medicin = self._get_beroligende_medicin_specs(
            resolve_multiple=["count", "bool"],
            interval_days=[1, 3, 7, 10, 30, 180],
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        prior_coercion = self._get_coercion_current_status_specs(
            resolve_multiple=["sum", "count", "bool"],
            interval_days=[7, 30, 180],
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        schema_3_coercion = self._get_coercion_current_status_specs(
            resolve_multiple=["sum", "count", "bool"],
            interval_days=[730],
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        structured_sfi = self._get_structured_sfi_specs(
            resolve_multiple=["mean", "max", "min", "change_per_day", "variance"],
            interval_days=[1, 3, 10, 30, 180],
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        # bvc_physical_threats = self._get_bvc_physical_threats_specs(
        #     resolve_multiple=["count", "bool"],
        #     interval_days=[1, 3, 10, 30, 180],
        #     allowed_nan_value_prop=allowed_nan_value_prop,
        # )

        return (
            visits
            + admissions
            + medications
            + diagnoses
            + beroligende_medicin
            + prior_coercion
            + schema_3_coercion
            + structured_sfi
            # + bvc_physical_threats
        )

    def get_feature_specs(self) -> list[_AnySpec]:
        """Get a spec set."""

        if self.min_set_for_debug:
            return (
                self._get_temporal_predictor_specs()
                # + self._get_outcome_specs()
            )

        return self._get_temporal_predictor_specs() + self._get_static_predictor_specs()




def expand_adm_period(df: pd.DataFrame, pred_time: int, CPR: int):
    """_summary_ SKB: fill this out

    Args:
        df_final (_type_): _description_
        kon_id (int): _description_
        CPR (int): _description_

    Returns:
        _type_: _description_
    """
    # OBS: Here it is possible to change the timestamp of the day to where one would like to make the prediction
    # We will do two versions:
    # one predicting at 17.00
    # one predicting at 06.00
    pred_time = 6

    # expand admission period to days + discharge day
    days = pd.date_range(dfr.iloc[0, 1].date(), dfr.iloc[0, 2].date())
    days = pd.DataFrame(days).rename(columns={0: "days"})

    # add day-of-admission column
    days["admission_count_days"] = 0
    for day in time.index:
        time["admission_count_days"][day] = day + 1

    for day in range(0, len(time)):
        time["period"][day] = time["period"][day].replace(hour=pred_time, minute=0)

    time["datotid_start_indlaeg"] = kon_id

    # join time period with df
    temp_period = pd.merge(time, temp, how="left", on="datotid_start_indlaeg")

    # exclude admission start days where admission happens after prediction
    if temp_period.iloc[0, 2].time() > pd.Timestamp(2020, 1, 1, pred_time).time():
        temp_period = temp_period.iloc[1:, :]

    # if admission is longer than 1 day
    if len(temp_period) > 1:  # SKB: rewrite this to one if statement
        # exclude admission end days where the admission ends before prediction
        if temp_period.iat[-1, 4].time() < pd.Timestamp(2020, 1, 1, pred_time).time():
            temp_period = temp_period.iloc[:-1, :]

    return temp_period





def unpack_time_intervals_to_days(
    df: pd.DataFrame,
    starttime_column: str = "datotid_start_sei",
    endtime_column: str = "datotid_slut_sei",
) -> pd.DataFrame:

    #### NB: WHAT TO DO WITH TIME OF DAY ##### -- missing info if we 'cut' the last date?

    """Transform df with starttime_column and endtime_column to day grain (one row per day in the interval starttime_column-endtime_column)

    Args:
        df (pd.DataFrame): dataframe with time interval in separate columns.
        starttime_column (str, optional): Name of column with start time. Defaults to "datotid_start_sei".
        endtime_column (str, optional): Name of column with end time. Defaults to "datotid_slut_slut".

    Returns:
        pd.DataFrame: Dataframe with time interval unpacked to day grain.

    """

    # for testing
    starttime_column = "value"  # "datotid_start_sei"
    endtime_column = "timestamp"  # "datotid_slut_sei"

    # remove rows that are either missing start or end time
    df = df[
        (df[f"{starttime_column}"].notnull()) & (df[f"{endtime_column}"].notnull())
    ]  # or does something

    # create a date range column between start_date and end_date for each visit_id
    df["date_range"] = df.apply(
        lambda x: pd.date_range(
            start=x[f"{starttime_column}"].date() + pd.DateOffset(1),
            end=x[f"{endtime_column}"].date(),
        ),
        axis=1,
    )

    # explode the date range column to create a new row for each date in the range
    df = df.explode("date_range")

    # add rows with start time and end time for each patient and start time
    df_group = df.groupby(["dw_ek_borger", f"{starttime_column}"])
    df_group_list = [df_group.get_group(key) for key in df_group.groups.keys()]

    def add_starttime_and_endtime_rows(df_test):
        # extract two rows from df
        df_rows = df_test.iloc[0:2, :].copy().reset_index().drop(columns=["index"])

        # if start and end is the same day, we will only have one row, but need two (one row for start time, one for end time)
        if df_test.shape[0] == 1:
            df_rows = (
                pd.concat([df_rows, df_rows]).reset_index().drop(columns=["index"])
            )

        # insert start time and end time in date_range column
        df_rows.at[0, "date_range"] = df_rows.at[0, f"{starttime_column}"]
        df_rows.at[1, "date_range"] = df_rows.at[1, f"{endtime_column}"]

        return pd.concat([df_test, df_rows]).sort_values("date_range")

    df_testing = pd.concat(
        [add_starttime_and_endtime_rows(df_group) for df_group in df_group_list]
    )

    # drop the date_range column and rename the exploded column to timestamp
    new_df = new_df.drop("date_range", axis=1).rename(
        columns={"date_range": "timestamp"}
    )

    # reset the index of the new dataframe
    new_df = new_df.reset_index(drop=True)

    # print the new dataframe
    return new_df
