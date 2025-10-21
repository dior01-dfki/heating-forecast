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
from forcateri.reporting.resultreporter import ResultReporter
from forcateri.controls.pipeline import Pipeline
from forcateri.controls.clearmlsingletaskpipeline import ClearMLSingleTaskPipeline

from forcateri.utils.config_utils import extract_config, load_config, arg_parser, from_args_to_kwargs
from pathlib import Path
#from src import project_root
# import argparse
# import sys


OFFSET, TIME_STEP = TimeSeries.ROW_INDEX_NAMES
FEATURE, REPRESENTATION = TimeSeries.COL_INDEX_NAMES



def main():

    model_adapters = []
    model_adapters.append(DartsTFTModel())
    model_adapters.append(DartsTCNModel())
    data_sources = []
    data_sources.append(BaltBestAggregatedAPIData())
    roles = {
        'q_hca': SeriesRole.TARGET,
        'temperature_outdoor_avg': SeriesRole.KNOWN,
        'temperature_1_max': SeriesRole.OBSERVED,
        'temperature_2_max': SeriesRole.OBSERVED,
        'temperature_room_avg': SeriesRole.OBSERVED,
    }
    metrics = [] 
    metrics.append(
        DimwiseAggregatedQuantileLoss(axes=[OFFSET])
    )
    metrics.append(
        DimwiseAggregatedMetric(axes=[TIME_STEP])
    )
    dp = DataProvider(data_sources=data_sources, roles=[roles])
    test_set = dp.get_test_set()

    rep = ResultReporter(
        test_set,
        models=model_adapters,
        metrics=metrics,
    )
    pipe = Pipeline(
        dp=dp,
        model_adapter=model_adapters,
        reporter=rep,
    )

    # clearml_rep = ClearMLReporter(test_set, models=model_adapters, metrics=metrics)
    # cml_pipe = ClearMLSingleTaskPipeline(
    #     dp=dp,
    #     model_adapter=model_adapters,
    #     reporter=clearml_rep,
    #     config_path="configs/pipeline.yaml",
    #     param_args='',
    #     requirements="./requirements.txt",
    #     docker = "dior00002/heating-forecast2:v1"
    # )
    # cml_pipe.run()
    pipe.run()


if __name__ == "__main__":
    main()
    #test(*list(vars(args).items()))
