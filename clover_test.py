from argparse import ArgumentParser
from ast import literal_eval
from cmath import log
from inspect import Parameter, signature
import logging
from typing import List
import yaml

from forcateri.reporting.dimwiseaggregatedmetric import DimwiseAggregatedMetric
from forcateri.reporting.dimwiseaggregatedquantileloss import (
    DimwiseAggregatedQuantileLoss,
)
from forcateri.data.timeseries import TimeSeries

from src.baltbestapi.baltbestaggregatedapidata import BaltBestAggregatedAPIData
from src.dartsmodels.dartstcnmodel import DartsTCNModel
from forcateri.data.dataprovider import DataProvider
from forcateri.reporting.clearmlreporter import ClearMLReporter
from forcateri.reporting.localresultreporter import LocalResultReporter
from forcateri.controls.clearmlsingletaskpipeline import ClearMLSingleTaskPipeline, Pipeline

from forcateri.utils.decorators import clover, connect_config, global_cfg_dct, clover_parser
from pathlib import Path
from forcateri.utils.config_utils import extract_config


from darts.utils.likelihood_models import *

OFFSET, TIME_STEP = TimeSeries.ROW_INDEX_NAMES


def main_v2():

    darts_config_path = './configs/dartstcn.yaml'
    connect_config(Path(__file__).parent.joinpath(darts_config_path))
    config = yaml.safe_load(open(darts_config_path, 'r'))

    #args = extract_config(config)


    likelihood_name = config.get("likelihood")
    likelihood = globals().get(likelihood_name) if likelihood_name else None
    if likelihood:
        config["likelihood"] = likelihood(config.get("quantiles", [0.1, 0.5, 0.9]))

    model_adapters = []
    model_adapters.append(DartsTCNModel(**config))

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

@clover
def main(config_path:str = './configs/test_cfg.yaml'):

    config = yaml.safe_load(open(config_path, 'r'))
    print(config['Models'])

    connect_config(Path(__file__).parent.joinpath(config_path))
    dartstcn_config = config['Models'].get('DartsTCNModel').copy()

    likelihood_name = dartstcn_config.get("likelihood")
    likelihood = globals().get(likelihood_name) if likelihood_name else None
    if likelihood:
        dartstcn_config["likelihood"] = likelihood(dartstcn_config.get("quantiles", [0.1, 0.5, 0.9]))

    model_adapters = []
    model_adapters.append(DartsTCNModel(**dartstcn_config))

    #kwargs = {**defaults,  **kwargs}
    # args = clover_parser.parse_args()

    # print(vars(args))
    
    data_sources = []
    data_sources.append(BaltBestAggregatedAPIData())
    roles = config['DataSources']['BaltBestAggregatedAPIData']['roles']
    # roles = {
    #   'TARGET': ['q_hca'],
    #   'KNOWN': ['temperature_outdoor_avg'],
    #   'OBSERVED': ['temperature_1_max', 'temperature_2_max','temperature_room_avg']
    # }

    metrics = [] 
    axes1 = config['Metrics']['DimwiseAggregatedQuantileLoss']['axes']
    axes2 = config['Metrics']['DimwiseAggregatedMetric']['axes']
    
    metrics.append(
        DimwiseAggregatedQuantileLoss(axes=axes1)
    )
    metrics.append(
        DimwiseAggregatedMetric(axes=axes2)
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


def main_local(config_path:str = './configs/test_cfg.yaml'):

    config = yaml.safe_load(open(config_path, 'r'))
    print(config['Models'])

    dartstcn_config = config['Models'].get('DartsTCNModel')
    likelihood_name = dartstcn_config.get("likelihood")
    likelihood = globals().get(likelihood_name) if likelihood_name else None
    if likelihood:
        dartstcn_config["likelihood"] = likelihood(dartstcn_config.get("quantiles", [0.1, 0.5, 0.9]))
    print(type(dartstcn_config["likelihood"]))
    model_adapters = []
    model_adapters.append(DartsTCNModel(**dartstcn_config))

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
    rep = LocalResultReporter(test_set, models=model_adapters, metrics=metrics)
    pipe = Pipeline(
        dp=dp,
        model_adapter=model_adapters,
        reporter=rep,
    )
    pipe.run()

if __name__ == "__main__":
    main()

