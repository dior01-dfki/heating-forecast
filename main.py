
from forcateri.reporting.dimwiseaggregatedmetric import DimwiseAggregatedMetric
from forcateri.reporting.dimwiseaggregatedquantileloss import (
    DimwiseAggregatedQuantileLoss,
)
from forcateri.data.timeseries import TimeSeries
import pandas as pd

from src.baltbestapi.baltbestaggregatedapidata import BaltBestAggregatedAPIData
from src.dartsmodels.dartstcnmodel import DartsTCNModel
from src.dartsmodels.xgbmodel import dartsXGB
from src.dartsmodels.lrmodel import DartsLRModel
from forcateri.data.dataprovider import DataProvider
from forcateri.reporting.clearmlreporter import ClearMLReporter
from forcateri.reporting.localresultreporter import LocalResultReporter
from forcateri.controls.clearmlsingletaskpipeline import ClearMLSingleTaskPipeline, Pipeline
from forcateri.reporting.metric_aggregations import column_wise_mae, column_wise_mape
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

    #dp = DataProvider(data_sources=data_source, roles=[roles])
    dp = BaltDataProvider(data_sources=data_source, roles = [roles,roles], data_purposes=[1,2],splits=0.6)
    model_adapters = []
    #train_set = dp.get_train_set()
    #test_set = dp.get_test_set()
    #val_set = dp.get_val_set()
    #print(f"Test set: {test_set}")
    metrics = []
    metrics.append(
        DimwiseAggregatedMetric(axes=[OFFSET], metric_func=column_wise_mae)
    )
    metrics.append(
        DimwiseAggregatedMetric(axes=[OFFSET], metric_func=column_wise_mape)
    )
    #print(dp.get_test_set())
    # metrics.append(
    #     DimwiseAggregatedQuantileLoss(axes=[OFFSET])
    # )
    #model_adapters.append(DartsTCNModel(n_epochs=1, scaler_data=dp.get_train_set(), predict_likelihood_parameters=True))
    #model_adapters.append(dartsXGB())
    # kwargs = {'predict_likelihood_parameters': True}
    # lrmodel = DartsLRModel(
    #     lags_past_covariates=24,
    #     lags_future_covariates=list(range(24)),
    #     output_chunk_length=24,
    #     likelihood='quantile',
    #     quantiles=[0.1, 0.5, 0.9],
    #     model_name="LR_baseline_model",
    #     kwargs=kwargs,
    # )
    # model_adapters.append(lrmodel)
    # dartstcn = DartsTCNModel(
    #     model_name="TCN_model",
    #     input_chunk_length=48,
    #     quantiles=[0.1, 0.5, 0.9],
    #     output_chunk_length=24,
    #     kernel_size=3,
    #     num_filters=32,
    #     predict_likelihood_parameters=True,
    #     n_epochs=5
    # )
    # #print(f"DartsTCN.is_likelihood: {dartstcn.is_likelihood}")
    # model_adapters.append(dartstcn)
    dartsXGB_model = dartsXGB(
        model_name="XGB_model",
        input_chunk_length=48,
        output_chunk_length=24,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        lags=24,
        lags_past_covariates=24,
        random_state=42
    )
    model_adapters.append(dartsXGB_model)
    #rep = LocalResultReporter(  models=model_adapters, metrics=metrics)
    # pipe = Pipeline(
    #     dp=dp,
    #     model_adapter=model_adapters,
    #     reporter=rep,
    # )
    # pipe.run()
    # train_set = dp.get_train_set()
    # val_set = dp.get_val_set()
    # test_set = dp.get_test_set()
    #print(f"training a LR model")
    # lrmodel.fit(train_set, val_set)
    # print(f"predicting with LR model")
    # #kwargs = {'predict_likelihood_parameters': True}
    # kwargs = {}
    # predictions = lrmodel.predict(data=test_set,n=12, **kwargs)
    # print(len(predictions))
    clearml_rep = ClearMLReporter( models=model_adapters, metrics=metrics)

    
    
    cml_pipe = ClearMLSingleTaskPipeline(
        data_provider=dp,
        model_adapter=model_adapters,
        result_reporter=clearml_rep,
        project_name='ForeSightNEXT/BaltBest',
        task_name=f'{model_adapters[0].model_name} training',
        #config_path="configs/pipeline.yaml",
        init_args=[],
        requirements="./requirements.txt",
        #docker = "dior00002/heating-forecast2:v1",
        docker = "python:3.12-slim",
        #docker = "unit8/darts",
        repo="git@github.com:dior01-dfki/heating-forecast.git",
        branch="edz_train"
    )
    cml_pipe.run()
    # out_path = "model_artifacts/DartsTCNModel"
    # os.makedirs(out_path, exist_ok=True)
    # dartstcn.save(out_path)
    # Task.current_task().upload_artifact(
    # name="DartsTCNModel",
    # artifact_object=str(out_path),
    #)
if __name__ == "__main__":
    print("Starting main")
    main()
    