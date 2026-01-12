
from forcateri.reporting.dimwiseaggregatedmetric import DimwiseAggregatedMetric
from forcateri.reporting.dimwiseaggregatedquantileloss import (
    DimwiseAggregatedQuantileLoss,
)
from forcateri.data.timeseries import TimeSeries
import pandas as pd

from src.baltbestapi.baltbestaggregatedapidata import BaltBestAggregatedAPIData
from src.dartsmodels.dartstcnmodel import DartsTCNModel
from forcateri.data.dataprovider import DataProvider
from forcateri.reporting.clearmlreporter import ClearMLReporter
from forcateri.reporting.localresultreporter import LocalResultReporter
from forcateri.controls.clearmlsingletaskpipeline import ClearMLSingleTaskPipeline, Pipeline

#from forcateri.utils.decorators import clover, connect_config, global_cfg_dct, clover_parser
from pathlib import Path
#from forcateri.utils.config_utils import extract_config
from forcateri.data.seriesrole import SeriesRole
from clearml import Dataset

from darts.utils.likelihood_models import *
import yaml
from clover import clover
from clover.decorator import connect_config

import logging 


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

OFFSET, TIME_STEP = TimeSeries.ROW_INDEX_NAMES
FEATURE, REPRESENTATION = TimeSeries.COL_INDEX_NAMES

@clover
def main_for_reference(config_path:str = './configs/test_cfg.yaml'):

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

# def fetch_data():
#     dataset = Dataset.get(dataset_name="ResampledData", dataset_project="ForeSightNEXT/BaltBest/resampled")
#     dataset_path = dataset.get_local_copy()
#     resampled_data = pd.read_csv(f"{dataset_path}/resampled_data.csv")

#     dataset_qa = Dataset.get(dataset_name='BaltBestMetadata', dataset_project='ForeSightNEXT/BaltBest')
#     dataset_qa_path = dataset_qa.get_local_copy()
#     data_qa_report = pd.read_csv(f"{dataset_qa_path}/data_qa_report.csv", index_col=[0,1])

#     acceptable_rooms = (
#         data_qa_report
#         .groupby(level='room_id')
#         .apply(lambda g: ((g['non_nan_ratio'] >= 0.8) & (g['non_zero_ratio'] > 0.3)).all())
#     )
#     acceptable_rooms_ids = acceptable_rooms.index[acceptable_rooms].tolist()
#     filtered_data = resampled_data[resampled_data['room_id'].isin(acceptable_rooms_ids)].copy()
#     return filtered_data

def main():
    #data = fetch_data()
    data_source = []
    ds = BaltBestAggregatedAPIData(
        group_col='room_id',
        known='outside_temp',
        time_col='ts',
        observed=[
            "inside_temp",
            "heater_side_hca_temp",
            "room_side_hca_temp"
        ],
        target = "hca_units"
        )
    data_source.append(ds)
    roles = {
        SeriesRole.TARGET: ['hca_units'],
        SeriesRole.KNOWN: ['outside_temp'],
        SeriesRole.OBSERVED: [
            "inside_temp",
            "heater_side_hca_temp",
            "room_side_hca_temp"
        ]
    }
    dp = DataProvider(data_sources=data_source, roles=[roles])
    model_adapters = []
    test_set = dp.get_test_set()
    #print(f"Test set: {test_set}")
    metrics = []
    metrics.append(
        DimwiseAggregatedQuantileLoss(axes=[OFFSET, FEATURE])
    )
    model_adapters.append(DartsTCNModel(n_epochs=1, scaler_data=dp.get_train_set(), predict_likelihood_parameters=True))
    rep = LocalResultReporter(test_set, models=model_adapters, metrics=metrics)
    pipe = Pipeline(
        dp=dp,
        model_adapter=model_adapters,
        reporter=rep,
    )
    pipe.run()
if __name__ == "__main__":
    print("Starting main")
    main()