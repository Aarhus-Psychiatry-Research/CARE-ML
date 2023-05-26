Current priorities are on the 
[Board 🎬](https://github.com/orgs/Aarhus-Psychiatry-Research/projects/6).

psycop-coercion
==============================
![python versions](https://img.shields.io/badge/Python-%3E=3.7-blue)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)

Prediction of coercion among patients admittied to the hospital psychiatric department. Encompasses predicting mechanical restraint, sedative medication and manual restraint 48 hours before coercion occurs. 

## Installing pre-commit hooks
`pre-commit install`

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs        org for details
    ├── src                <- Source code for use in this project.       <- A default Sphinx project; see sphinx-doc.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── forced_admissions_cohort.py    <- Script that generates data frames for projects
    │   │
    │   ├── loaders           <- Scripts to load in data from SQL database
    │   │   └── load_data_function.py
    │   │
    │   └── writers.py  <- Scripts to create exploratory and results oriented visualizations
    │
    ├──  pyproject.toml            <- Poetry file handling all dependendencies
    └──  Other configuration files


--------


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
