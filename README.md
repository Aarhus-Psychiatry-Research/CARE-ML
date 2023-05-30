<a href="https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation"><img src="https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils/blob/main/docs/_static/icon_with_title.png?raw=true" width="220" align="right"/></a>

![python versions](https://img.shields.io/badge/Python-%3E=3.9-blue)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
[![github actions pytest]()

## Predicting use of restraint on psychiatric inpatients using electronic health data and machine learning

(Hvad er det, hvad er formålet, hvad kan den, hvem er den til)

This repository was developed as a part of the product Master’s Thesis in Cognitive Science by: Signe Kirk Brødbæk (201707519) and Sara Kolding (201708816)

For our thesis, we have developed this pipeline for training and evaluating prognostic supervised ML models for predicting the use of restraint on inpatients in the Central Denmark Region.

The motivation for the current study was twofold: we wanted to 1) build upon a sound framework focusing on pre-processing, data analysis, as well as transparent training and evaluation practices to increase generalisability to new data to 2) obtain interpretable ML models predicting restraint with the aim of helping staff identify at-risk inpatients to allocate scarce resources. 

This repository builds upon the framework in [timeseriesflattener](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener) and the code base for the PSCYOP projects [psycop-common](https://github.com/Aarhus-Psychiatry-Research/psycop-common), as well as previous work by Danielsen et al. (2019).  

Due data infrastructure of Central Denmark Region, we rely upon, the functionality of this project is tied to this specific use case. However, the framework is generalisable. Since the nature of psychiatric data is sensitive, we have created tutorials to showcase the pipeline. 

In the following sections, we will present how to 1) install this package, 2) he project organisation, and 3= go through the functionality within each of the modules. 

Hope you enjoy, 

Signe and Sara 

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
------------

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

First, the

1. Use the template

![image](https://user-images.githubusercontent.com/8526086/208095705-81baa10b-b396-4fd7-a549-3b920ec18322.png)

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
