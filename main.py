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

#from forcateri.utils.decorators import clover, connect_config, global_cfg_dct, clover_parser
from pathlib import Path
from forcateri.utils.config_utils import extract_config


from darts.utils.likelihood_models import *

from clover import clover, connect_config

