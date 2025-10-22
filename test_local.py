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
from forcateri.reporting.resultreporter import ResultReporter
from forcateri.controls.pipeline import Pipeline
from forcateri.utils.config_utils import (
    extract_config,
    from_args_to_kwargs,
    load_config,
    arg_parser
)
from pathlib import Path
from src import project_root
import argparse
import sys
from darts.utils.likelihood_models import *

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

    print(list(args))
    kwargs = from_args_to_kwargs(*args)
    
    #print(kwargs)
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
    splits = kwargs["DataProvider"].get("splits", [0.33, 0.66])
    print(type(splits))
    dp = DataProvider(data_sources=data_sources, roles=[roles], splits=splits)

    model_adapters = []
    for model_name, params in kwargs["Models"].items():
        model_class = globals().get(model_name)
        if model_class is None:
            raise ValueError(
                f"Model class '{model_name}' not found in global namespace."
            )
        likelihood_name = params.get("likelihood")
        likelihood = globals().get(likelihood_name) if likelihood_name else None
        if likelihood:
            params["likelihood"] = likelihood(params.get("quantiles", [0.1, 0.5, 0.9]))
        model_adapters.append(model_class(**params))
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
    rep = ResultReporter(test_set, models=model_adapters, metrics=metrics)
    cml_pipe = Pipeline(
        dp=dp,
        model_adapter=model_adapters,
        reporter=rep,
    )
    cml_pipe.run()


if __name__ == "__main__":
    parser = arg_parser('configs/pipeline.yaml')
    args = parser.parse_args()
    main(*list(vars(args).items()))

