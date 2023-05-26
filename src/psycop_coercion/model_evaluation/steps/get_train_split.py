import pandas as pd
from psycop_coercion.model_evaluation.best_runs import BestRun
from psycop_coercion.model_evaluation.data.load_true_data import load_fullconfig
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.data_loader.utils import load_and_filter_split_from_cfg
from zenml.steps import BaseParameters, step


class TrainSplitConf(BaseParameters):
    best_runs: BestRun


from joblib import Memory

# Create a memory object that will cache the results of the function
memory = Memory(location=".", verbose=0)


@step
def get_train_split_step(params: TrainSplitConf) -> pd.DataFrame:
    cfg: FullConfigSchema = load_fullconfig(
        wandb_group=params.best_runs.wandb_group,
        wandb_run=params.best_runs.model,
    )
    cfg.Config.allow_mutation = True
    cfg.debug = None

    pass

    df = load_and_filter_split_from_cfg(cfg=cfg, split="train")
    return df


@memory.cache
def get_train_split(best_run: BestRun) -> pd.DataFrame:
    cfg: FullConfigSchema = load_fullconfig(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run.model,
    )

    df = load_and_filter_split_from_cfg(cfg=cfg, split="train")

    return df
