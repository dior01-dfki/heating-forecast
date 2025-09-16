from forcateri.model.dartsmodels.dartstcnmodel import DartsTCNModel
from forcateri.model.dartsmodels.dartstftmodel import DartsTFTModel
from forcateri.baltbestapi.baltbestaggregatedapidata import BaltBestAggregatedAPIData
import pandas as pd
import yaml
from forcateri.data.dataprovider import DataProvider, SeriesRole
# from darts.models import TCNModel
# from darts.utils.likelihood_models import QuantileRegression
from forcateri.data.timeseries import TimeSeries
from forcateri.reporting.dimwiseaggregatedmetric import DimwiseAggregatedMetric
from forcateri.reporting.dimwiseaggregatedquantileloss import DimwiseAggregatedQuantileLoss
from forcateri.reporting.resultreporter import ResultReporter
from forcateri.controls.pipeline import Pipeline
from pathlib import Path
import argparse
import sys
#from .clearml_entry import extract_config


OFFSET, TIME_STEP = TimeSeries.ROW_INDEX_NAMES
FEATURE, REPRESENTATION = TimeSeries.COL_INDEX_NAMES
DATASET_CLASSES = {
    "BaltBestAggregatedAPIData": BaltBestAggregatedAPIData,
    #Other datasets to be added here
}
METRIC_CLASSES = {
    "DimwiseAggregatedQuantileLoss": DimwiseAggregatedQuantileLoss,
    "DimwiseAggregatedMetric": DimwiseAggregatedMetric,
    # Other metrics to be added here
}

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

def from_args_to_kwargs(*args) -> dict:
    kwargs = {"Models": {}, "Dataset": {}, "Metrics": {}}
    for key, value in args:
        if key.startswith("model"):
            keysplit = key.split("_", 2)
            model_name, param = keysplit[1], keysplit[2]
            kwargs["Models"].setdefault(model_name, {})[param] = value
        elif key.startswith("Dataset"):
            _, dataset_name, role_key = key.split("_", 2)
            kwargs["Dataset"].setdefault(dataset_name, {"roles": {}})
            if "," in value:
                features = value.split(",")
            else:
                features = [value]
            role_enum = getattr(SeriesRole, role_key.split(".")[-1])
            for f in features:
                kwargs["Dataset"][dataset_name]["roles"][f] = role_enum
        elif key.startswith("Metric"):
            keysplit = key.split("_", 2)
            metric_name, param = keysplit[1], keysplit[2]
            # If value is a comma-separated string, split to list
            if isinstance(value, str) and "," in value:
                value = value.split(",")
            kwargs["Metrics"].setdefault(metric_name, {})[param] = value
    return kwargs

def arg_parser():
    parser = argparse.ArgumentParser()
    project_root=Path(__file__).parent.parent
    config_path = project_root.joinpath("configs")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Configuration file name without .yaml extension"
    )
    args,remaining_args = parser.parse_known_args()
    with open(config_path.joinpath(args.config + '.yaml'),"r") as infile:
            parsed_config = yaml.safe_load(infile)
    args = extract_config(parsed_config)
    for k, v in args:
        if isinstance(v, list):
            v = ",".join(map(str, v))
        elif v is None:
            v = "None"
        parser.add_argument(f"--{k}", default=v)
    return parser

def main(*args):



    kwargs = from_args_to_kwargs(*args)
    dataset_names = list(kwargs['Dataset'].keys())
    data_sources = []
    roles = []
    for dataset_name in dataset_names:
        dataset_class = DATASET_CLASSES.get(dataset_name)
        if dataset_class is None:
            raise ValueError(f"Dataset class '{dataset_name}' not found in DATASET_CLASSES.")
        ds = dataset_class()
        data_sources.append(ds)
        roles = kwargs['Dataset'][dataset_name]['roles']
    
    
    dp = DataProvider(data_sources=data_sources, roles=[roles])

    model_adapters = []
    for model_name, params in kwargs['Models'].items():
        model_class = globals().get(model_name)
        if model_class is None:
            raise ValueError(f"Model class '{model_name}' not found in global namespace.")
        model_adapters.append(model_class(kwargs=params))
    metrics = []
    for metric_name, params in kwargs['Metrics'].items():
        metric_class = METRIC_CLASSES.get(metric_name)
        if metric_class is None:
            raise ValueError(f"Metric class '{metric_name}' not found in METRIC_CLASSES.")
        # Convert axes strings to actual values
        axes = [OFFSET if ax == "OFFSET" else TIME_STEP for ax in params.get("axes", [])]
        metrics.append(metric_class(axes=axes))

    test_set = dp.get_test_set()
    rep = ResultReporter(test_set,models=model_adapters,metrics=metrics)
    #rep.report_all()
    pipe = Pipeline(dp,model_adapter=model_adapters,reporter=rep)
    #results = pipe.run()
    pipe.run()
    #return results

if __name__ == "__main__":
    
    parser = arg_parser()
    args = parser.parse_args()
    print("pipeline args\n\n")
    print(args)

    print("\n\n")
    print(*list(vars(args).items()))

    # Parse the command-line arguments


    main(*list(vars(args).items()))
