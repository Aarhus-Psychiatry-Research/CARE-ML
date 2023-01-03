<a href="https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation"><img src="https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils/blob/main/docs/_static/icon_with_title.png?raw=true" width="220" align="right"/></a>

# Feature generation for the PSYCOP [TEMPLATE] project

![python versions](https://img.shields.io/badge/Python-%3E=3.9-blue)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)

This is application scripts for feature generation for the [TEMPLATE] project. 

Main functionality lies in [psycop-feature-generation](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation) and [timeseriesflattener](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener).

## Installation
`pip install --src ./src -r requirements.txt`

This will install the requirements in your `src` folder as their own repos. 

For example, this means that it install the `timeseriesflattener` repository in `src/timeseriesflattener`. You can make edits there, checkout to a new branch, and submit PRs to the `timeseriesflattener` repo - all within the VS Code editor.

![image](https://user-images.githubusercontent.com/8526086/208070436-a52fef2c-16c8-4e7e-830b-8cff6dba44c2.png)

## Usage
1. Use the template

![image](https://user-images.githubusercontent.com/8526086/208095705-81baa10b-b396-4fd7-a549-3b920ec18322.png)

2. Open up `application/main.py`.
3. Change the project name in the call to `get_project_info`
4. Update the arguments to `create_flattened_dataset` to fit your situation
5. Update feature specs in `modules/specify_features.py`
Whichever featuers you specify will need a corresponding loader which returns the raw values for flattening. 

Note that there are quite a few loaders in `/src/psycop-feature-generation/src/psycop_feature_generation/loaders`. Definitely use them as much as possibl to build and fix together. 

Also, if you need to add loaders that are likely to generalise, feel free to add them here. If they are specific to your project, add them to `modules/loaders/your_loader_file.py`.

6. Generate with a tiny set of features (keep `FeatureSpecifier`'s `min_set_for_debug` as `True`.
7. When everything works, set `min_set_for_debug` to `False` and generate a full data set!

## Before publication
- [ ] Lock the dependencies in `requirements.txt` to a specific version
