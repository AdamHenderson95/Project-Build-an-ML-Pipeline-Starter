import json

import mlflow
import tempfile
import os
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]
root_path = os.path.abspath(os.path.dirname(__file__))
# This automatically reads in the configuration
@hydra.main(config_name='config')

def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    mlruns_path = os.path.join(root_path, "mlruns")
    mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")
    mlflow.set_experiment(config["main"]["experiment_name"])

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            from pathlib import Path

            project_root = Path(__file__).resolve().parent
            mlflow.run(
                str(project_root / "components" / "get_data"),
                entry_point="main",
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Data with outliers and null values removed",
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                },
            )

        if "data_check" in active_steps:
            mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                parameters={
                    "csv": "adamjhenderson95-western-governors-university/project-ml-pipeline-src_basic_cleaning/clean_sample.csv:latest",
                    "ref": "adamjhenderson95-western-governors-university/project-ml-pipeline-src_basic_cleaning/clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["data_check"]["min_price"],
                    "max_price": config["data_check"]["max_price"],
                },
            )

        if "data_split" in active_steps:
            mlflow.run(
                os.path.join(root_path, "components/train_val_test_split"),
                "main",
                parameters={
                    "input": "adamjhenderson95-western-governors-university/project-ml-pipeline-src_basic_cleaning/clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                },
            )

        if "train_random_forest" in active_steps:
            rf_config_path = os.path.join(tmp_dir, "rf_config.json")
            with open(rf_config_path, "w") as f:
                json.dump(omegaconf.OmegaConf.to_container(config["modeling"]["random_forest"], resolve=True), f)

            mlflow.run(
                os.path.join(root_path, "src/train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": "adamjhenderson95-western-governors-university/nyc_airbnb/trainval_data.csv:v0",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config_path,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": config["modeling"]["output_artifact"],
                },
            )

        if "test_regression_model" in active_steps:
            mlflow.run(
                os.path.join(root_path, "components/test_regression_model"),
                "main",
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_dataset": "test_data.csv:latest"
                }
            )


if __name__ == "__main__":
    go()
