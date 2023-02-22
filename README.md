<a href="https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation"><img src="https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils/blob/main/docs/_static/icon_with_title.png?raw=true" width="220" align="right"/></a>

# Model training for the PSYCOP [TEMPLATE] project

![python versions](https://img.shields.io/badge/Python-%3E=3.9-blue)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)

This is application scripts for feature generation for the [TEMPLATE] project. 

All the shared functionality across projects lies in [psycop-model-training](https://github.com/Aarhus-Psychiatry-Research/psycop-model-training)'s `application_modules` folder.

## Installation
`pip install --src ./src -r requirements.txt`

This will install the requirements in your `src` folder as their own repos. 

For example, this means that it install the `psycop-model-training` repository in `src/psycop-model-training`. You can make edits there, checkout to a new branch, and submit PRs to the `psycop-model-training` repo - all within the VS Code editor.

![image](https://user-images.githubusercontent.com/8526086/208070436-a52fef2c-16c8-4e7e-830b-8cff6dba44c2.png)

## Usage
1. Use the template

![image](https://user-images.githubusercontent.com/8526086/208095705-81baa10b-b396-4fd7-a549-3b920ec18322.png)

2. Modify the configs in `application/config`
3. To test that everything works, train a single model with `train_model_from_application_module`
4. When you're satisfied, run a full hyperparameter search using `train_models_in_parallel`


## Adding custom plots
The basic idea of custom plots is to create functions that can take an `EvalDataset` as input and which output a `png` file. We then wrap those in an `ArtifactContainer` object, and pass it to the evaluation module. 

To see what this looks like in practice, see `application/artifacts/custom_artifacts`.

## Before publication
- [ ] Lock the dependencies in `requirements.txt` to a specific version
