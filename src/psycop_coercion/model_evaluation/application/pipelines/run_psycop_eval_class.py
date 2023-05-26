from psycop_coercion.model_evaluation.best_runs import best_run
from psycop_coercion.model_evaluation.steps.get_train_split import (
    TrainSplitConf,
    get_train_split_step,
)
from psycop_coercion.model_evaluation.steps.model_evaluator import (
    evaluate_model,
)
from psycop_coercion.model_evaluation.steps.pipeline_loader import pipeline_loader
from zenml.pipelines import pipeline
from zenml.steps.base_step import BaseStep


@pipeline(enable_cache=True)
def base_evaluation_pipeline(
    training_data_loader: BaseStep,
    pipeline_loader: BaseStep,
    model_evaluator: BaseStep,
):
    train_split = training_data_loader()
    pipe = pipeline_loader()
    model_evaluator(pipe=pipe, train_split=train_split)  # type: ignore


if __name__ == "__main__":
    BASE_EVAL_PIPELINE_INSTANCE = base_evaluation_pipeline(
        training_data_loader=get_train_split_step(TrainSplitConf(best_runs=best_run)),
        pipeline_loader=pipeline_loader(TrainSplitConf(best_runs=best_run)),
        model_evaluator=evaluate_model(),
    )

    BASE_EVAL_PIPELINE_INSTANCE.run(unlisted=True)
