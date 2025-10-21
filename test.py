from src.dartsmodels.dartstcnmodel import DartsTCNModel
from src.dartsmodels.dartstftmodel import DartsTFTModel
from src.baltbestapi.baltbestaggregatedapidata import BaltBestAggregatedAPIData
import pandas as pd
import yaml
from forcateri.data.dataprovider import DataProvider, SeriesRole

# from darts.models import TCNModel
# from darts.utils.likelihood_models import QuantileRegression
from forcateri.data.timeseries import TimeSeries
from forcateri.reporting.dimwiseaggregatedmetric import DimwiseAggregatedMetric
from forcateri.reporting.dimwiseaggregatedquantileloss import (
    DimwiseAggregatedQuantileLoss,
)
from forcateri.reporting.clearmlreporter import ClearMLReporter
from forcateri.controls.clearmlsingletaskpipeline import ClearMLSingleTaskPipeline

from forcateri.utils.config_utils import extract_config, load_config, arg_parser, from_args_to_kwargs
from pathlib import Path
#from src import project_root
# import argparse
# import sys


OFFSET, TIME_STEP = TimeSeries.ROW_INDEX_NAMES
FEATURE, REPRESENTATION = TimeSeries.COL_INDEX_NAMES
DATASET_CLASSES = {
    "BaltBestAggregatedAPIData": BaltBestAggregatedAPIData,
    # Other datasets to be added here
}
METRIC_CLASSES = {
    "DimwiseAggregatedQuantileLoss": DimwiseAggregatedQuantileLoss,
    "DimwiseAggregatedMetric": DimwiseAggregatedMetric,
    # Other metrics to be added here
}


def main(*args):

    kwargs = from_args_to_kwargs(*args)
    print(list(args))
    dataset_names = list(kwargs["DataSources"].keys())
    data_sources = []
    roles = []
    for dataset_name in dataset_names:
        dataset_class = DATASET_CLASSES.get(dataset_name)
        if dataset_class is None:
            raise ValueError(
                f"Dataset class '{dataset_name}' not found in DATASET_CLASSES."
            )
        ds = dataset_class()
        data_sources.append(ds)
        roles = kwargs["DataSources"][dataset_name]["roles"]
    splits = kwargs.get("DataProvider", {}).get("splits", [0.33, 0.66])
    print(kwargs['DataProvider'])
    dp = DataProvider(data_sources=data_sources, roles=[roles], splits=splits)

    model_adapters = []
    for model_name, params in kwargs["Models"].items():
        model_class = globals().get(model_name)
        if model_class is None:
            raise ValueError(
                f"Model class '{model_name}' not found in global namespace."
            )
        model_adapters.append(model_class(kwargs=params))
    metrics = []
    for metric_name, params in kwargs["Metrics"].items():
        metric_class = METRIC_CLASSES.get(metric_name)
        if metric_class is None:
            raise ValueError(
                f"Metric class '{metric_name}' not found in METRIC_CLASSES."
            )
        # Convert axes strings to actual values
        axes = [
            OFFSET if ax == "OFFSET" else TIME_STEP for ax in params.get("axes", [])
        ]
        metrics.append(metric_class(axes=axes))

    test_set = dp.get_test_set()
    rep = ClearMLReporter(test_set, models=model_adapters, metrics=metrics)
    cml_pipe = ClearMLSingleTaskPipeline(
        dp=dp,
        model_adapter=model_adapters,
        reporter=rep,
        config_path="configs/pipeline.yaml",
        param_args=list(args),
        requirements="./requirements.txt",
        docker = "dior00002/heating-forecast2:v1"
    )
    cml_pipe.run()

def test(*args):
    kwargs = from_args_to_kwargs(*args)
    print(kwargs)
    print("\n\n pipeline args\n")
    print(*args)

if __name__ == "__main__":
    parser = arg_parser(config_path="configs/pipeline.yaml")
    args = parser.parse_args()
    
    # print("\n\n")
    # print(*list(vars(args).items()))
    main(*list(vars(args).items()))

    print(type(args))
    #test(*list(vars(args).items()))
