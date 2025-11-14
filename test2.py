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
from forcateri.reporting.localresultreporter import LocalResultReporter
from forcateri.controls.pipeline import Pipeline

from forcateri.utils.config_utils import extract_config, load_config, arg_parser, from_args_to_kwargs
from pathlib import Path
from darts.utils.likelihood_models import *
from hydra import main
from omegaconf import OmegaConf
import fire
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

#MOVE TO SEPARATE PARAM NAMES INSTEAD OF RELYING ON CFG
#
#CONNECT DEFAULT PARAMS TO UI WITHOUT EXTRA STEPS AND CODES
#
#check the FIRE

# def main_cml(cfg_path,cfg_name, dataset_names, models, **kwargs):
#     kwargs_from_cfg = read_config(cfg_path, cfg_name)
#     kwargs.update(kwargs_from_cfg)
#     dataset_names = ['Baltbest','BaltTheBest']
#     models = ['DartsTCNModel','DartsTFTModel']


@main(config_path='configs',config_name='pipeline', version_base=None)
def main_cml(cfg, dataset_names, models, **kwargs):

    cfg = OmegaConf.to_container(cfg,resolve=True)
    dataset_names = list(cfg["DataSources"].keys())
    data_sources = []
    roles = []
    print(dataset_names)
    for dataset_name in dataset_names:
        dataset_class = DATASET_CLASSES.get(dataset_name)
        if dataset_class is None:
            raise ValueError(
                f"Dataset class '{dataset_name}' not found in DATASET_CLASSES."
            )
        ds = dataset_class()
        data_sources.append(ds)
        roles = cfg["DataSources"][dataset_name]["roles"]
    splits = cfg["DataProvider"].get("splits", [0.33, 0.66])
    print(roles)
    dp = DataProvider(data_sources=data_sources, roles=[roles], splits=splits)

    model_adapters = []
    for model_name, params in cfg["Models"].items():
        model_class = globals().get(model_name)
        if model_class is None:
            raise ValueError(
                f"Model class '{model_name}' not found in global namespace."
            )
        likelihood_name = params.get("likelihood")
        likelihood = globals().get(likelihood_name) if likelihood_name else None
        if likelihood:
            params["likelihood"] = likelihood(params.get("quantiles", [0.1, 0.5, 0.9]))
            print(params)
        model_adapters.append(model_class(**params))
    metrics = []
    for metric_name, params in cfg["Metrics"].items():
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
    #print(args)
    rep = ClearMLReporter(test_set, models=model_adapters, metrics=metrics)
    cml_pipe = ClearMLSingleTaskPipeline(
        dp=dp,
        model_adapter=model_adapters,
        reporter=rep,
        #config_path="configs/pipeline.yaml",
        project_name=cfg['ClearML']['task']['project_name'],
        task_name=cfg['ClearML']['task']['task_name'],
        init_args=cfg,
        requirements="./requirements.txt",
        docker = "dior00002/heating-forecast2:v1",
        repo="git@github.com:dior01-dfki/heating-forecast.git",
        branch="iss47"
    )
    cml_pipe.run()



@main(config_path='configs',config_name='pipeline', version_base=None)
def main_local(cfg):
    cfg = OmegaConf.to_container(cfg,resolve=True)
    #print(type(cfg))
    dataset_names = list(cfg["DataSources"].keys())
    data_sources = []
    roles = []
    print(dataset_names)
    for dataset_name in dataset_names:
        dataset_class = DATASET_CLASSES.get(dataset_name)
        if dataset_class is None:
            raise ValueError(
                f"Dataset class '{dataset_name}' not found in DATASET_CLASSES."
            )
        ds = dataset_class()
        data_sources.append(ds)
        roles = cfg["DataSources"][dataset_name]["roles"]
    splits = cfg["DataProvider"].get("splits", [0.33, 0.66])
    print(roles)
    dp = DataProvider(data_sources=data_sources, roles=[roles], splits=splits)

    model_adapters = []
    for model_name, params in cfg["Models"].items():
        model_class = globals().get(model_name)
        if model_class is None:
            raise ValueError(
                f"Model class '{model_name}' not found in global namespace."
            )
        likelihood_name = params.get("likelihood")
        likelihood = globals().get(likelihood_name) if likelihood_name else None
        if likelihood:
            params["likelihood"] = likelihood(params.get("quantiles", [0.1, 0.5, 0.9]))
            print(params)
        model_adapters.append(model_class(**params))
    metrics = []
    for metric_name, params in cfg["Metrics"].items():
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
    rep = LocalResultReporter(test_set, models=model_adapters, metrics=metrics)
    pipe = Pipeline(
        dp=dp,
        model_adapter=model_adapters,
        reporter=rep,
    )
    pipe.run()


def main_fire(**kwargs):
    n_epochs = kwargs.get("n_epochs",10)
    model_adapters = []
    # init_args = []
    # init_args.append(("n_epochs", n_epochs))
    #model_adapters.append(DartsTFTModel(n_epochs=n_epochs))
    model_adapters.append(DartsTCNModel(n_epochs=n_epochs))
    
    data_sources = []
    data_sources.append(BaltBestAggregatedAPIData())
    roles = {
      'TARGET': ['q_hca'],
      'KNOWN': ['temperature_outdoor_avg'],
      'OBSERVED': ['temperature_1_max', 'temperature_2_max','temperature_room_avg']
    }

    metrics = [] 
    metrics.append(
        DimwiseAggregatedQuantileLoss(axes=[OFFSET, FEATURE])
    )
    metrics.append(
        DimwiseAggregatedMetric(axes=[TIME_STEP])
    )
    dp = DataProvider(data_sources=data_sources, roles=[roles])
    test_set = dp.get_test_set()

    args = kwargs
    clearml_rep = ClearMLReporter(test_set, models=model_adapters, metrics=metrics)

    

    cml_pipe = ClearMLSingleTaskPipeline(
        dp=dp,
        model_adapter=model_adapters,
        reporter=clearml_rep,
        project_name='ForeSightNEXT/BaltBest',
        task_name='DartsTCNModel_NoConfig',
        #config_path="configs/pipeline.yaml",
        init_args=args,
        requirements="./requirements.txt",
        docker = "dior00002/heating-forecast2:v1",
        repo="git@github.com:dior01-dfki/heating-forecast.git",
        branch="iss47"
    )
    cml_pipe.run()

if __name__ == "__main__":
    # parser = arg_parser(config_path="configs/pipeline.yaml")
    # args = parser.parse_args()
    #main_cml()
    #main_local()
    fire.Fire(main_fire)

