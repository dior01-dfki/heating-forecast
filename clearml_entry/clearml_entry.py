from clearml import Task
from forcateri.data.dataprovider import DataProvider, SeriesRole
import os
from dotenv import load_dotenv
from .test_pipeline import main
from pathlib import Path
import argparse
import yaml

load_dotenv()
def exec_remotely():

    my_task = Task.init(
        project_name="ForeSightNEXT/BaltBest",
        task_name="Forcateri Pipeline Test",
    )
    print("test local")
    my_task.execute_remotely(
        queue_name="default",
        clone=False,
        exit_process=True,
    )
    print("Test on remote")
    results = main()
    Task.current_task().upload_artifact(name='results', artifact_object=results)
def exec_taskenq(*args):
    token = os.environ["GIT_OAUTH_TOKEN"]

    task = Task.create(
        project_name="ForeSightNEXT/BaltBest",
        task_name="test",
        add_task_init_call=True,
        branch="main",
        repo="git@github.com:connected-intelligent-systems/forcateri.git",
        script="clearml_entry/test_pipeline.py",
        #docker = "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
        docker="dior00002/forcateri:v1",
        docker_args=(
            f"-e CLEARML_AGENT_GIT_USER=oauth2 -e CLEARML_AGENT_GIT_PASS={token}"
        ),
        argparse_args=args,
    )
    Task.enqueue(task=task, queue_name="default")

def extract_config(config: dict) -> list[tuple]:
    args = []
    for section, section_content in config.items():
        if section == "Models":
            for model_name, params in section_content.items():
                for param_name, param_value in params.items():
                    arg_key = f"model_{model_name}_{param_name}"
                    args.append((arg_key, param_value))
        elif section == "Dataset":
            for dataset_name, dataset_content in section_content.items():
                if isinstance(dataset_content, dict):
                    for subkey, subcontent in dataset_content.items():
                        if subkey == "roles":
                            for role, features in subcontent.items():
                                arg_key = f"Dataset_{dataset_name}_{role}"
                                if isinstance(features, list):
                                    args.append((arg_key, ",".join(features)))
                                else:
                                    args.append((arg_key, features))
                        else:
                            arg_key = f"{dataset_name}_{subkey}"
                            args.append((arg_key, subcontent))
        elif section == "Metrics":
            for metric_name, params in section_content.items():
                for param_name, param_value in params.items():
                    arg_key = f"Metric_{metric_name}_{param_name}"
                    if isinstance(param_value, list):
                        args.append((arg_key, ",".join(map(str, param_value))))
                    else:
                        args.append((arg_key, param_value))
    return args



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline config parser")
    parser.add_argument(
        '--config',
        type = str,
        #required = True,
        default = 'tft_pipeline',
        help = 'Specify the config for the training process'
    )
    args = parser.parse_args()
    project_root=Path(__file__).parent.parent
    config_path = project_root.joinpath("configs")

    with open(config_path.joinpath(args.config + '.yaml'),"r") as infile:
            parsed_config = yaml.safe_load(infile)
    argss = extract_config(parsed_config)
    argss.append(("config", args.config))
    print("args\n\n")
    print(argss)
    exec_taskenq(*argss)
    
    