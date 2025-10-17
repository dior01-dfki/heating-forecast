from forcateri.data.dataprovider import SeriesRole
from forcateri.data.timeseries import TimeSeries
#from forcateri import project_root
import argparse
import yaml
from pathlib import Path

def extract_config(config: dict) -> list[tuple]:
    args = []
    for section, section_content in config.items():
        if section == "ClearML":
            continue
        elif section == "Models":
            for model_name, params in section_content.items():
                for param_name, param_value in params.items():
                    arg_key = f"model.{model_name}.{param_name}"
                    args.append((arg_key, param_value))
        elif section == "Dataset":
            for dataset_name, dataset_content in section_content.items():
                if isinstance(dataset_content, dict):
                    for subkey, subcontent in dataset_content.items():
                        if subkey == "roles":
                            for role, features in subcontent.items():
                                arg_key = f"Dataset.{dataset_name}.{role}"
                                if isinstance(features, list):
                                    args.append((arg_key, ",".join(features)))
                                else:
                                    args.append((arg_key, features))
                        else:
                            arg_key = f"{dataset_name}.{subkey}"
                            args.append((arg_key, subcontent))
        elif section == "Metrics":
            for metric_name, params in section_content.items():
                for param_name, param_value in params.items():
                    arg_key = f"Metric.{metric_name}.{param_name}"
                    if isinstance(param_value, list):
                        args.append((arg_key, ",".join(map(str, param_value))))
                    else:
                        args.append((arg_key, param_value))
    return args

def from_args_to_kwargs(*args) -> dict:
    kwargs = {"Models": {}, "Dataset": {}, "Metrics": {}}
    for key, value in args:
        if key.startswith("model"):
            keysplit = key.split(".", 2)
            model_name, param = keysplit[1], keysplit[2]
            kwargs["Models"].setdefault(model_name, {})[param] = value
        elif key.startswith("Dataset"):
            _, dataset_name, role_key = key.split(".", 2)
            kwargs["Dataset"].setdefault(dataset_name, {"roles": {}})
            if "," in value:
                features = value.split(",")
            else:
                features = [value]
            role_enum = getattr(SeriesRole, role_key)
            for f in features:
                kwargs["Dataset"][dataset_name]["roles"][f] = role_enum
        elif key.startswith("Metric"):
            keysplit = key.split(".", 2)
            metric_name, param = keysplit[1], keysplit[2]
            # If value is a comma-separated string, split to list
            if isinstance(value, str) and "," in value:
                value = value.split(",")
            kwargs["Metrics"].setdefault(metric_name, {})[param] = value
    return kwargs

def arg_parser(project_root):
    parser = argparse.ArgumentParser()
    #project_root=Path(__file__).parent.parent
    config_path = project_root.joinpath("configs")
    parser.add_argument(
        '--config',
        type=str,
        #required=True,
        help="Configuration file name without .yaml extension"
    )
    args,remaining_args = parser.parse_known_args()
    print(args.config)
    with open(config_path.joinpath('pipeline' + '.yaml'),"r") as infile:
            parsed_config = yaml.safe_load(infile)
    args = extract_config(parsed_config)
    for k, v in args:
        if isinstance(v, list):
            v = ",".join(map(str, v))
        elif v is None:
            v = "None"
        parser.add_argument(f"--{k}", default=v)
    return parser

def load_config(config_name: str,project_root) -> dict:
    """
    Parse command line args and load YAML config file from configs/ directory.
    Returns: (config_name: str, parsed_config: dict)
    """
    parser = argparse.ArgumentParser(description="Pipeline config parser")
    parser.add_argument(
        '--config',
        type=str,
        default='pipeline',
        help='Specify the config name (without .yaml) from configs/ directory'
    )
    args = parser.parse_args()

    #project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / f"{config_name}.yaml"

    with open(config_path, "r") as infile:
        parsed_config = yaml.safe_load(infile)

    return parsed_config