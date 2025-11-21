import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from darts import TimeSeries as DartsTimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TCNModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from datetime import datetime
from forcateri.data.adapterinput import AdapterInput
from forcateri.model.modelexceptions import InvalidModelTypeError, ModelAdapterError
from forcateri.model.dartsmodeladapter import DartsModelAdapter
from forcateri.data.timeseries import TimeSeries

from forcateri.utils.decorators import clover

# from forcateri import project_root

from src import project_root

# @dataclass
# class DartsTCNModel_config:
#     input_chunk_length: int = 7
#     output_chunk_length: int = 5
#     kernel_size: int = 3
#     num_filters: int = 32
#     dilation_base: int = 2
#     num_layers: int = 3
#     dropout: float = 0.1
#     weight_norm: bool = True
#     n_epochs: int = 100
#     batch_size: int = 32
#     optimizer_kwargs: dict = {"lr": 1e-3}
#     random_state: Optional[int] = None
#     likelihood: Optional[QuantileRegression] = QuantileRegression([0.1, 0.5, 0.9])

class DartsTCNModel(DartsModelAdapter):

    @clover
    def __init__(        self,
        model: Optional[TCNModel] = None,
        quantiles=[0.1, 0.5, 0.9],
        input_chunk_length=7,
        output_chunk_length=5,
        kernel_size=3,
        num_filters=32,
        dilation_base=2,
        num_layers=3,
        dropout=0.1,
        weight_norm=True,
        n_epochs=1,
        batch_size=8,
        optimizer_kwargs={"lr": 0.001},
        random_state=None,
        forecast_horizon=1,
        predict_likelihood_parameters=False,
        *args,
        **kwargs):
        """
        Initializes the Darts TCNModel with specified parameters and scalers.
        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments to configure the TCNModel. Supported keys include:
            - input_chunk_length (int): Length of the input sequence. Default is 7.
            - output_chunk_length (int): Length of the output sequence. Default is 5.
            - kernel_size (int): Size of the convolutional kernel. Default is 3.
            - num_filters (int): Number of filters in the convolutional layers. Default is 32.
            - dilation_base (int): Base of the dilation factor. Default is 2.
            - num_layers (int): Number of convolutional layers. Default is 3.
            - dropout (float): Dropout rate for regularization. Default is 0.1.
            - weight_norm (bool): Whether to apply weight normalization. Default is True.
            - n_epochs (int): Number of training epochs. Default is 100.
            - batch_size (int): Batch size for training. Default is 32.
            - optimizer_kwargs (dict): Additional arguments for the optimizer. Default is {'lr': 1e-3}.
            - random_state (int, optional): Random seed for reproducibility. Default is None.
            - likelihood (Likelihood, optional): Likelihood model for probabilistic forecasting.
              Default is QuantileRegression([0.1, 0.5, 0.9]).
        Attributes
        ----------
        model : TCNModel
            The initialized TCNModel with the specified parameters.
        scaler_target : Scaler
            Scaler for the target variable.
        scaler_cov : Scaler
            Scaler for the covariates.
        """

        super().__init__(*args, **kwargs)
        self.quantiles = quantiles
        if model is not None:
            self.model = model
        else:
            self.input_chunk_length = input_chunk_length
            self.output_chunk_length = output_chunk_length
            self.kernel_size = kernel_size
            self.num_filters = num_filters
            self.dilation_base = dilation_base
            self.num_layers = num_layers
            self.dropout = dropout
            self.weight_norm = weight_norm
            self.n_epochs = n_epochs
            self.batch_size = batch_size
            self.optimizer_kwargs = optimizer_kwargs
            self.random_state = random_state
            self.forecast_horizon = forecast_horizon
            log_dir = project_root.joinpath(
                f"logs/dartstcn/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            )
            logger = TensorBoardLogger(save_dir=log_dir)
            trainer_kwargs = dict(logger=[logger])
            self.model = TCNModel(
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                kernel_size=self.kernel_size,
                num_filters=self.num_filters,
                dilation_base=self.dilation_base,
                num_layers=self.num_layers,
                dropout=self.dropout,
                weight_norm=self.weight_norm,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                optimizer_kwargs=self.optimizer_kwargs,
                random_state=self.random_state,
                likelihood=kwargs.get("likelihood", QuantileRegression(self.quantiles)),
                pl_trainer_kwargs=trainer_kwargs,
            )
        self.forecast_horizon = kwargs.get("forecast_horizon", 1)


    def fit(
        self,
        train_data: List[AdapterInput],
        val_data: Optional[List[AdapterInput]],
    ):
        """
        Fits the model using the provided training and validation data.

        Parameters:
            train_data (List[AdapterInput]): The training data to be used for fitting the model.
            val_data (Optional[List[AdapterInput]]): The validation data to be used for evaluating the model during training.
                This parameter is optional and can be None.
            **kwargs: Additional keyword arguments to be passed to the parent class's fit method.

        Raises:
            ModelAdapterError: If the model fitting process fails due to invalid parameters or other issues.

        Logs:
            An error message is logged if the model fitting process fails.
        """

        super().fit(train_data=train_data, val_data=val_data)



    @classmethod
    def load(cls, path: Union[Path, str]) -> "DartsTCNModel":
        try:
            model = TCNModel.load(path)
            # if not isinstance(model, ForecastingModel):
            #     raise InvalidModelTypeError(
            #         "The loaded model is not a valid Darts model."
            #     )
            # else:

            logging.info(f"Model loaded from {path}")
            return cls(model=model)
        except Exception as e:
            logging.error(f"Failed to load the model from {path}, check the model path")
            raise ModelAdapterError("Failed to load the model.") from e

    def __repr__(self):
        return "DartsTCNModel"
