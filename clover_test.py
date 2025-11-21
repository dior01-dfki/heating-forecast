from argparse import ArgumentParser
from ast import literal_eval
from cmath import log
from inspect import Parameter, signature
import logging
from typing import List

from forcateri.reporting.dimwiseaggregatedmetric import DimwiseAggregatedMetric
from forcateri.reporting.dimwiseaggregatedquantileloss import (
    DimwiseAggregatedQuantileLoss,
)
from forcateri.data.timeseries import TimeSeries

from src.baltbestapi.baltbestaggregatedapidata import BaltBestAggregatedAPIData
from src.dartsmodels.dartstcnmodel import DartsTCNModel
from forcateri.data.dataprovider import DataProvider
from forcateri.reporting.clearmlreporter import ClearMLReporter
from forcateri.controls.clearmlsingletaskpipeline import ClearMLSingleTaskPipeline


OFFSET, TIME_STEP = TimeSeries.ROW_INDEX_NAMES

def main():
    model_adapters = []
    model_adapters.append(DartsTCNModel())

    #kwargs = {**defaults,  **kwargs}


    
    data_sources = []
    data_sources.append(BaltBestAggregatedAPIData())
    roles = {
      'TARGET': ['q_hca'],
      'KNOWN': ['temperature_outdoor_avg'],
      'OBSERVED': ['temperature_1_max', 'temperature_2_max','temperature_room_avg']
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

    #args = kwargs
    clearml_rep = ClearMLReporter(test_set, models=model_adapters, metrics=metrics)

    
    
    cml_pipe = ClearMLSingleTaskPipeline(
        dp=dp,
        model_adapter=model_adapters,
        reporter=clearml_rep,
        project_name='ForeSightNEXT/BaltBest',
        task_name='DartsTCNModel_NoConfig',
        #config_path="configs/pipeline.yaml",
        init_args=[],
        requirements="./requirements.txt",
        docker = "dior00002/heating-forecast2:v1",
        repo="git@github.com:dior01-dfki/heating-forecast.git",
        branch="iss47"
    )
    cml_pipe.run()

if __name__ == "__main__":
    main()

