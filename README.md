<a href="https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation"><img src="https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils/blob/main/docs/_static/icon_with_title.png?raw=true" width="220" align="right"/></a>

![python versions](https://img.shields.io/badge/Python-%3E=3.9-blue)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
[![github actions pytest]()

# Predicting use of restraint on psychiatric inpatients using electronic health data and machine learning

This repository was developed as a part of the product Master’s Thesis in Cognitive Science by: 

Signe Kirk Brødbæk (201707519) and Sara Kolding (201708816)

## Table of Contents 
[1. Motivation](#motivation)

[2. Terminology](#terminology)

[3. Installation](#installation)

[4. Project Organization](#project_organisation)

[5. Module 1: Cohort Generation](#mod1)

[6. Module 2: Feature Generation](#mod2)

[7. Module 3: Model Training](#mod3)

[8. Module 4: Model Evaluation ](#mod4)


 <a id="motivation"></a>
## 1. Motivation
The use of three types of restraint, _physical_, _chemical_, and _mechanical_ restraint, has been increasing in Danish psychiatric units, despite the objective from the Ministry of Health and the Danish Regions to decrease the use of mechanical restraint, which can be seen in Figure 1 below. In recent years, In recent years, the literature on machine learning (ML) and prognostic prediction models in clinical contexts has expanded (Islam et al., 2020; Placido et al., 2023; Shamout et al., 2021), including studies identifying individual patients at high risk of being coerced (Danielsen et al., 2019; Günther et al., 2020; Hotzy et al., 2018). By offering early detection of at-risk patients, such models could enable staff to reallocate resources to a subgroup of patients, to avoid coercive interventions.
   
For our thesis, we built this pipeline for training and evaluating prognostic supervised ML models for predicting the use of restraint on inpatients in the Central Denmark Region, building upon the study by Danielsen et al. (2019) and utilising the frameworks of the [timeseriesflattener](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener) package and the code base for the PSCYOP projects [psycop-common](https://github.com/Aarhus-Psychiatry-Research/psycop-common).

<img src="docs/figures/restraint_stats.jpg" alt= “” width="70%" height="70%" class="center">

Our focus has been to build a tool that is sound and transparent, including evaluations to examine the relationship between the most important features and the outcome, as well as potential biases. Due to the sensitivity of the data infrastructures utilised in the current study, the packages are designed for very specific use cases within the department of psychiatry in CDR. As a consequence, the pipeline is intended for a small target audience, and not generalisable across other regions in Denmark or in other countries. 

The specific pipeline can be utilised and adapted for future research by researchers in the CDR. However, the framework and considerations implemented in this pipeline, such as the temporal considerations, evaluating on a held-out test set and thorough evaluation, is generalisable and can be utilised in other ML contexts. 

In the following sections, we will present 0) the terminology at the core of this pipeline, 1) how to install this package, 2) the project organisation, and 3) go through the functionality within each of module.

<a id="terminology"></a>
## 2. Terminology
We adopt the terminology used in the [timeseriesflattener](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener) package, which includes _lookbehind and lookahead windows_, and _aggregation functions_. 

### 2.1 Lookahead and lookbehind windows 
In A) below, the prediction time describes the time of prediction and acts as the reference point for the lookbehind and lookahead windows. The lookbehind window denotes how far back in time to look for feature values, while the lookahead window denotes how far into the future to look for outcome values. B) shows that when the outcome is found within the lookahead window, this constitutes a true positive within this framework. If the outcome occurs later than the lookahead window, this is a true negative

<img src="docs/figures/tsf_terminology.jpg" alt= "" class="center" width="60%" height="60%">

_Note. Visualisation of the timeseriesflattener terminology adapted from Bernstoff et al. (2023), reprinted with permission from the original authors_

In the current project, we utilise lookbehind windows of varying lengths (between 1 day and 730 days) to create features. The labels were created with a lookahead of 2 days/48 hours. 

### 2.2 Aggregation functions
When multiple feature values occur within a lookbehind window, there are several ways we can aggregate them. 
The figuer figure denotes how features can be "flattened" when multiple data entries exist within a lookbehind window. In the blue lookbehind, the three hospital contactsoccur and these entries are aggregated into a tabular format by counting the number of contacts and summing the duration (in hours). Similarly, two values appear in the green lookbehind window, which is also aggregated as the count of hospital contacts and the sum of hours. 

<img src="docs/figures/feature_flattening.jpg" alt= "" class="center" width="60%" height="60%">

_Note. Figure developed in collaboration with the PSYCOP group._

The aggregation functions utilised in this project include: 
- Latest value
- Count
- Sum of hours
- Boolean
- Mean
- Maximum
- Minimum
- Change per day (slope of linear regression)
- Variance
- Concatenate (for text features)

<a id="installation"></a>
## 3. Installation

To install this repository, 

1) Clone this repo 
```
git clone https://github.com/Aarhus-Psychiatry-Research/psycop-restraint.git
```

2) Go to the project root and use the pyproject.toml to install dependencies: 
```
python -m pip install -e .
```

3) Optional: Install timeseriesflattener and psycop-common in your 'src' folder as their own repos
```
pip install --src ./src -r src-requirements.txt
```
<a id="project_organisation"></a>
## 4. Project Organization

The project consist of the following overall structure, including four modules for 1) cohort generation, 2) feature generation, 3) model training, and 4) model evaluation. In the following sections, we will delineate the functionality of each module. 

    ├── README.md
    │
    ├── docs
    │
    ├── src                      <- Source code for use in this project.
    │   ├── __init__.py          
    │   │
    │   ├── cohort_generation    <- Module for creating the cohort wih labels
    │   │
    │   ├── feature_generation   <- Module for feature generation
    │   │
    │   ├── model_evaluation     <- Module for model evaluation
    │   │
    │   ├── model_training       <- Module for model training
    │
    ├── pyproject.toml           <- Poetry file handling all dependendencies
    │ 
    └── Other configuration files

<a id="mod1"></a>
## 5. Module 1: Cohort Generation 

First, the cohort was defined with the following inclusion/exclusion criteria: 

1. The patient had a minimum of one psychiatric admission which started between 1 January 2015 and 22 November 2021.
2. The patient was >= 18 years at the time of admission.
3. The patient experienced no instances of physical, chemical, or mechanical restraint in the 365 days before the admission start date. 

Then, target days were defined as: 

1. Either physical, chemical, or mechanical restraint occuring within 48 hours of the time of prediction
3. Days after the first outcome instance was excluded, only 11.68% of admission with coercion in the PSYCOP cohort have only one instance of restraint), and predicting days after the first outcome offers less information to healthcare professionals. 
4. Prediction days after mean admission length + 1 standard deviation (mean=16 days, sd=44 days, cut-off = 60 days) was excluded to remedy the imbalance in classes.

As discussed in the thesis, these criteria could influence model prediction and clinical applicability. Within this module, you can change these criteria.

In this module, admissions start out as being one row and is unpacked to include 1 row per day in the admission with the prediction time, excluding the first admisison day if it is after the prediction time of the current day and the last admission day if the patient is discharged before the time of prediction.

See example below with the unpacking of an admission, where the time of prediction 6:00 a.m. and lookahead is 48 hours.  
- The first day of the admission is removed, since the patient was admitted after the time of prediction. 
- The fifth/last day of admisision is removed, since it is after the outcome. 
- Note that since the lookahead is 48 hours, two days are denoted target days (outcome = 1). 

Before: 

| adm_id | patient_id | admission_timestamp | discharge_timestamp |  outcome_timestamp   |
| :----- | :--------- | :------------------ | :------------------ |  :-----------------  |
| 1      |     1      | 2021-01-01 10:00:00 | 2021-01-05 16:00:00 |  2021-01-04 16:33:00 |


After: 

| adm_id | patient_id | admission_timestamp | discharge_timestamp |  outcome_timestamp   | prediction_timestamp | admission_day_counter | outcome | 
| :----- | :--------- | :------------------ | :------------------ |  :-----------------  | :------------------- | :-------------------- | :-----  |
| 1      |     1      | 2021-01-01 10:00:00 | 2021-01-05 16:00:00 |  2021-01-04 16:33:00 | 2021-01-02 06:00:00  |  2                    |  0      |
| 1      |     1      | 2021-01-01 10:00:00 | 2021-01-05 16:00:00 |  2021-01-04 16:33:00 | 2021-01-03 06:00:00  |  3                    |  1      |
| 1      |     1      | 2021-01-01 10:00:00 | 2021-01-05 16:00:00 |  2021-01-04 16:33:00 | 2021-01-04 06:00:00  |  4                    |  1      |

<a id="mod2"></a>
## 6. Module 2: Feature Generation

In this module, the cohort is linked to other variables to create features based on the defined lookbehind windows and aggregation functions, using the _timeseriesflattener_ package and data loaders from the _psycop-commn_ package. 

_timeseriesflattener_ was created to handle data from electronic health records, which might have many missing values and are sampled irregularly. By defining windows to look for values and how such values should be aggregated, _timeseriesflattener_ *flattens* the data, as described in Section [2. Terminology](#terminology). In addition to aggregation functions, a _fallback_, i.e. a value to insert when no observations is found within a window, is chosen. We used fallbacks of 0 (e.g., for hospital contacts) and NA (for texts and structured SFI's where a 0 score is different from a missing value). 




For example, 


<a id="mod3"></a>
## 7. Module 3: Model Training 



<a id="mod4"></a>
## 8. Module 4: Model Evaluation 

```py
import numpy as np
import pandas as pd

if __name__ == "__main__":

    # Load a dataframe with times you wish to make a prediction
    prediction_times_df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "date": ["2020-01-01", "2020-02-01", "2020-02-01"],
        },
    )
    # Load a dataframe with raw values you wish to aggregate as predictors
    predictor_df = pd.DataFrame(
        {
            "id": [1, 1, 1, 2],
            "date": [
                "2020-01-15",
                "2019-12-10",
                "2019-12-15",
                "2020-01-02",
            ],
            "value": [1, 2, 3, 4],
        },
    )
    # Load a dataframe specifying when the outcome occurs
    outcome_df = pd.DataFrame({"id": [1], "date": ["2020-03-01"], "value": [1]})

    # Specify how to aggregate the predictors and define the outcome
    from timeseriesflattener.feature_spec_objects import OutcomeSpec, PredictorSpec
    from timeseriesflattener.resolve_multiple_functions import maximum, mean

    predictor_spec = PredictorSpec(
        values_df=predictor_df,
        lookbehind_days=30,
        fallback=np.nan,
        entity_id_col_name="id",
        resolve_multiple_fn=mean,
        feature_name="test_feature",
    )
    outcome_spec = OutcomeSpec(
        values_df=outcome_df,
        lookahead_days=31,
        fallback=0,
        entity_id_col_name="id",
        resolve_multiple_fn=maximum,
        feature_name="test_outcome",
        incident=False,
    )

    # Instantiate TimeseriesFlattener and add the specifications
    from timeseriesflattener import TimeseriesFlattener

    ts_flattener = TimeseriesFlattener(
        prediction_times_df=prediction_times_df,
        entity_id_col_name="id",
        timestamp_col_name="date",
        n_workers=1,
        drop_pred_times_with_insufficient_look_distance=False,
    )
    ts_flattener.add_spec([predictor_spec, outcome_spec])
    df = ts_flattener.get_df()
    df
```
Output:

|      |   id | date                | prediction_time_uuid  | pred_test_feature_within_30_days_mean_fallback_nan | outc_test_outcome_within_31_days_maximum_fallback_0_dichotomous |
| ---: | ---: | :------------------ | :-------------------- | -------------------------------------------------: | --------------------------------------------------------------: |
|    0 |    1 | 2020-01-01 00:00:00 | 1-2020-01-01-00-00-00 |                                                2.5 |                                                               0 |
|    1 |    1 | 2020-02-01 00:00:00 | 1-2020-02-01-00-00-00 |                                                  1 |                                                               1 |
|    2 |    2 | 2020-02-01 00:00:00 | 2-2020-02-01-00-00-00 |                                                  4 |                                                               0 |


## 📖 Documentation

| Documentation          |                                                                                        |
| ---------------------- | -------------------------------------------------------------------------------------- |
| 🎓 **[Tutorial]**       | Simple and advanced tutorials to get you started using `timeseriesflattener`           |
| 🎛 **[General docs]** | The detailed reference for timeseriesflattener's API. |
| 🙋 **[FAQ]**            | Frequently asked question                                                              |
| 🗺️ **[Roadmap]**        | Kanban board for the roadmap for the project                                           |

[Tutorial]: https://aarhus-psychiatry-research.github.io/timeseriesflattener/tutorials.html
[General docs]: https://Aarhus-Psychiatry-Research.github.io/timeseriesflattener/
[FAQ]: https://Aarhus-Psychiatry-Research.github.io/timeseriesflattener/faq.html
[Roadmap]: https://github.com/orgs/Aarhus-Psychiatry-Research/projects/11/views/1


## 🎓 Sources 

Danielsen, A. A., Fenger, M.H.J., Østerggard, S.D., Nielbo, K.L., & Mors, O. (2019). Predicting mechanical restraint of psychiatric inpatients by applying machine learning on electronic health data. Acta Psychiatrica Scandinavica, 147–157. https://doi.org/10.1111/acps.13061
