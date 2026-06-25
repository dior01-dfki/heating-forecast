
from forcateri.model.dartsmodeladapter import Scaler
from forcateri.reporting.dimwiseaggregatedmetric import DimwiseAggregatedMetric
from forcateri.reporting.dimwiseaggregatedquantileloss import (
    DimwiseAggregatedQuantileLoss,
)
from forcateri.data.timeseries import TimeSeries
import pandas as pd
from src.dartsmodels.dartstftmodel import DartsTFTModel
from src.baltbestapi.baltbestaggregatedapidata import BaltBestAggregatedAPIData
from src.dartsmodels.dartstcnmodel import DartsTCNModel
from src.dartsmodels.xgbmodel import dartsXGB
from src.dartsmodels.lrmodel import DartsLRModel
from forcateri.data.dataprovider import DataProvider
from forcateri.reporting.clearmlreporter import ClearMLReporter
from forcateri.reporting.localresultreporter import LocalResultReporter
from forcateri.controls.clearmlsingletaskpipeline import ClearMLSingleTaskPipeline, Pipeline
from forcateri.reporting.metric_aggregations import column_wise_mae, column_wise_mape, column_wise_wmape
#from forcateri.utils.decorators import clover, connect_config, global_cfg_dct, clover_parser
from pathlib import Path
#from forcateri.utils.config_utils import extract_config
from forcateri.data.seriesrole import SeriesRole
from clearml import Dataset, Task

from darts.utils.likelihood_models import *
import yaml
from clover import clover
from clover.decorator import connect_config
import os
import logging 

from src.baltbestapi.baltdataprovider import BaltDataProvider
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

OFFSET, TIME_STEP = TimeSeries.ROW_INDEX_NAMES
FEATURE, REPRESENTATION = TimeSeries.COL_INDEX_NAMES


def main():
    #data = fetch_data()
    dataset = Dataset.get(dataset_name='Temp_data', dataset_project='ForeSightNEXT/BaltBest/resampled')
    local_path = dataset.get_local_copy()
    
    ts_static_data = pd.read_csv(local_path + '/ses_report.csv')
    ts_static_data = ts_static_data[ts_static_data['overlap_season'] > 0]
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
        target = "hca_units",
        #local_copy='data/',
        static_data=ts_static_data,
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
    ds_test = BaltBestAggregatedAPIData(
        group_col='room_id',
        known='outside_temp',
        time_col='ts',
        observed=[
            "inside_temp",
            "heater_side_hca_temp",
            "room_side_hca_temp"
        ],
        target = "hca_units",
        #local_copy='data/',
        static_data=ts_static_data,
        dataset_name="TestData",
        file_name="test_data.csv"
    )
    data_source.append(ds_test)


    dp = BaltDataProvider(data_sources=data_source, roles = [roles,roles], data_purposes=[1,2],splits=0.6)
    model_adapters = []

    metrics = []
    metrics.append(
        DimwiseAggregatedMetric(axes=[TIME_STEP], reduction=column_wise_mae)
    )
    metrics.append(
        DimwiseAggregatedMetric(axes=[TIME_STEP], reduction=column_wise_mape)
    )
    metrics.append(
        DimwiseAggregatedMetric(axes=[TIME_STEP], reduction=column_wise_wmape)
    )
    metrics.append(
        DimwiseAggregatedMetric(axes=[OFFSET], reduction=column_wise_mae)
    )
    metrics.append(
        DimwiseAggregatedMetric(axes=[OFFSET], reduction=column_wise_mape)
    )
    metrics.append(
        DimwiseAggregatedMetric(axes=[OFFSET], reduction=column_wise_wmape)
    )

    metrics.append(
        DimwiseAggregatedQuantileLoss(axes=[OFFSET])
    )
    

    def encode_year(idx):
        return (idx.year - 1950) / 50

    add_encoders={
        'cyclic': {'future': ['month']},
        'datetime_attribute': {'future': ['hour', 'dayofweek']},
        'position': {'past': ['relative'], 'future': ['relative']},
        'custom': {'past': [encode_year]},
        'transformer': Scaler(),
        'tz': 'CET'
    }
    dartstcn = DartsTCNModel(
        name="TCN_model",
        input_chunk_length=48,
        quantiles=[0.1, 0.5, 0.9],
        output_chunk_length=24,
        kernel_size=3,
        num_filters=32,
        is_likelihood=True,
        n_epochs=5
    )
    model_adapters.append(dartstcn)


    dartstft = DartsTFTModel(
        name="TFT_Model",
        input_chunk_length=48,
        quantiles=[0.1, 0.5, 0.9],
        output_chunk_length=24,
        is_likelihood=True,
        n_epochs=5,
        add_encoders=add_encoders
    )
    model_adapters.append(dartstft)

    clearml_rep = ClearMLReporter( models=model_adapters, metrics=metrics)

    
    
    cml_pipe = ClearMLSingleTaskPipeline(
        data_provider=dp,
        model_adapter=model_adapters,
        result_reporter=clearml_rep,
        project_name='ForeSightNEXT/BaltBest',
        task_name=f'{model_adapters[0].model_name} training',
        init_args=[],
        requirements="./requirements.txt",
        docker = "python:3.12-slim",
        repo="git@github.com:dior01-dfki/heating-forecast.git",
        branch="edz_train"
    )
    cml_pipe.run()

if __name__ == "__main__":
    print("Starting main")
    main()
    