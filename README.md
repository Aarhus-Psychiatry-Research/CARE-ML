<a href="https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation"><img src="https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils/blob/main/docs/_static/icon_with_title.png?raw=true" width="220" align="right"/></a>

![python versions](https://img.shields.io/badge/Python-%3E=3.9-blue)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
[![github actions pytest]()

## Predicting use of restraint on psychiatric inpatients using electronic health data and machine learning

This repository was developed as a part of the product Master’s Thesis in Cognitive Science by: 

Signe Kirk Brødbæk (201707519) and Sara Kolding (201708816)

### Motivation
The use of three types of restraint, _physical_, _chemical_, and _mechanical_ restraint, has been increasing in Danish psychiatric units, despite the objective from the Ministry of Health and the Danish Regions to decrease the use of mechanical restraint, which can be seen in Figure 1 below. In recent years, In recent years, the literature on machine learning (ML) and prognostic prediction models in clinical contexts has expanded (Islam et al., 2020; Placido et al., 2023; Shamout et al., 2021), including studies identifying individual patients at high risk of being coerced (Danielsen et al., 2019; Günther et al., 2020; Hotzy et al., 2018). By offering early detection of at-risk patients, such models could enable staff to reallocate resources to a subgroup of patients, to avoid coercive interventions.
   
For our thesis, we built this pipeline for training and evaluating prognostic supervised ML models for predicting the use of restraint on inpatients in the Central Denmark Region, building upon the study by Danielsen et al. (2019) and utilising the frameworks of the [timeseriesflattener](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener) package and the code base for the PSCYOP projects [psycop-common](https://github.com/Aarhus-Psychiatry-Research/psycop-common).


![My Image](docs/figures/restraint_stats.jpg)

Our focus has been to build a tool that is sound and transparent, including evaluations to examine the relationship between the most important features and the outcome, as well as potential biases. Due to the sensitivity of the data infrastructures utilised in the current study, the packages are designed for very specific use cases within the department of psychiatry in CDR. As a consequence, the pipeline is intended for a small target audience, and not generalisable across other regions in Denmark or in other countries. 

The specific pipeline can be utilised and adapted for future research by researchers in the CDR. However, the framework and considerations implemented in this pipeline, such as the temporal considerations, evaluating on a held-out test set and thorough evaluation, is generalisable and can be utilised in other ML contexts. 

In the following sections, we will present 0) the terminology at the core of this pipeline, 1) how to install this package, 2) the project organisation, and 3) go through the functionality within each of module.

## Terminology
We adopt the terminology used in the [timeseriesflattener](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener) package, which includes _lookbehind windows_, _lookahead windows_, and _aggregation functions_. 

### Lookahead and lookbehind windows 
![My_image](docs/figures/tsf_terminology.jpg)


## Installation

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

## Project Organization

The project consist of the following overall structure, including four modules for 1) cohort generation, 2) feature generation, 3) model training, and 4) model evaluation. In the following sections, we will delineate the functionality of each module. 

    ├── LICENSE
    ├── README.md                <- The top-level README introducing this repository
    │
    ├── docs
    │
    ├── src                      <- Source code for use in this project.
    │   ├── __init__.py          
    │   │
    │   ├──  cohort_generation   <- Module for creating the cohort wih labels
    │   │
    │   ├── feature_generation   <- Module for feature generation
    │   │
    │   ├──  model_evaluation    <- Module for model evaluation
    │   │
    │   ├──  model_training      <- Module for model training
    │
    ├──  pyproject.toml          <- Poetry file handling all dependendencies
    │ 
    └──  Other configuration files


## Cohort Generation 

First, the cohort was defined with the following inclusion/exclusion criteria: 

1. The patient had a minimum of one psychiatric admission which started between 1 January 2015 and 22 November 2021.
2. The patient was >= 18 years at the time of admission.
3. The patient experienced no instances of physical, chemical, or mechanical restraint in the 365 days before the admission start date. 

We defined the lookahead window as

## Feature Generation


Time series from e.g. electronic health records often have a large number of variables, are sampled at irregular intervals and tend to have a large number of missing values. Before this type of data can be used for prediction modelling with machine learning methods such as logistic regression or XGBoost, the data needs to be reshaped. 

In essence, the time series need to be *flattened* so that each prediction time is represented by a set of predictor values and an outcome value. These predictor values can be constructed by aggregating the preceding values in the time series within a certain time window. 

`timeseriesflattener` aims to simplify this process by providing an easy-to-use and fully-specified pipeline for flattening complex time series. 

## Model Training 


## Model Evaluation 

## 🔧 Installation
To get started using timeseriesflattener simply install it using pip by running the following line in your terminal:

```
pip install timeseriesflattener
```

## ⚡ Quick start

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
