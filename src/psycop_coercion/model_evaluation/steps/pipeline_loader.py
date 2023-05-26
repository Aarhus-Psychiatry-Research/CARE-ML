from psycop_coercion.model_evaluation.data.load_true_data import load_pipe
from psycop_coercion.model_evaluation.steps.get_train_split import TrainSplitConf
from sklearn.pipeline import Pipeline
from zenml.steps import step


@step
def pipeline_loader(params: TrainSplitConf) -> Pipeline:
    return load_pipe(
        wandb_group=params.best_runs.wandb_group,
        wandb_run=params.best_runs.model,
    )
