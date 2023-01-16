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

    def _get_coercion_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get coercion specs."""
        log.info("–––––––– Generating coercion specs ––––––––")

        coercion = PredictorGroupSpec(
            values_loader=(
                "skema_1",
                "tvangstilbageholdelse",
                "skema_2",
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

        interval_days = [10, 30, 180]
        allowed_nan_value_prop = [0]

        visits = self._get_visits_specs(
            resolve_multiple=["count"],
            interval_days=interval_days,
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        admissions = self._get_admissions_specs(
            resolve_multiple=["count", "sum"],
            interval_days=interval_days,
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        diagnoses = self._get_diagnoses_specs(
            resolve_multiple=["count", "bool"],
            interval_days=interval_days,
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
            interval_days=interval_days,
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        return (
            visits
            + admissions
            + medications
            + diagnoses
            + beroligende_medicin
            + coercion
            + structured_sfi
        )

    def get_feature_specs(self) -> list[_AnySpec]:
        """Get a spec set."""

        if self.min_set_for_debug:
            return (
                self._get_temporal_predictor_specs()
                # + self._get_outcome_specs()
            )

        return self._get_temporal_predictor_specs() + self._get_static_predictor_specs()