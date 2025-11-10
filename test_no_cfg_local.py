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
from forcateri.reporting.localresultreporter import LocalResultReporter
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

    n_epochs = 10
    model_adapters = []
    init_args = []
    init_args.append(("n_epochs", n_epochs))
    #model_adapters.append(DartsTFTModel(n_epochs=n_epochs))
    
    
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
    # metrics.append(
    #     DimwiseAggregatedQuantileLoss(axes=[OFFSET, FEATURE])
    # )
    metrics.append(
        DimwiseAggregatedMetric(axes=[TIME_STEP])
    )
    dp = DataProvider(data_sources=data_sources, roles=[roles])
    test_set = dp.get_test_set()
    model_adapters.append(DartsTCNModel(n_epochs=1, scaler_data=dp.get_train_set(), predict_likelihood_parameters=True))
    # rep = ResultReporter(
    #     test_set,
    #     models=model_adapters,
    #     metrics=metrics,
    # )
    # pipe = Pipeline(
    #     dp=dp,
    #     model_adapter=model_adapters,
    #     reporter=rep,
    # )
    #pipe.run()

    rep = LocalResultReporter(test_set, models=model_adapters, metrics=metrics)
    pipe = Pipeline(
        dp=dp,
        model_adapter=model_adapters,
        reporter=rep,
    )
    pipe.run()
    


if __name__ == "__main__":
    main()
    #test(*list(vars(args).items()))
