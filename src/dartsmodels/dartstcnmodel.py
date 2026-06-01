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

# from forcateri import project_root
from clover import clover
from src import project_root


class DartsTCNModel(DartsModelAdapter):

    @clover
    def __init__(
        self,
        model: Optional[TCNModel] = None,
        model_name: Optional[str] = None,
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
        is_likelihood=True,
        add_encoders=None
    ):
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

        super().__init__(name=model_name, quantiles=quantiles,is_likelihood=is_likelihood)
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
            self.is_likelihood = is_likelihood
            self.add_encoders = add_encoders
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
                likelihood=QuantileRegression(self.quantiles) if self.is_likelihood else None,
                pl_trainer_kwargs=trainer_kwargs,
                add_encoders=add_encoders
            )
        self.forecast_horizon = 24
        # self.scaler_target = Scaler()
        # self.scaler_cov = Scaler()

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

        try:

            super().fit(train_data=train_data, val_data=val_data)

        except ModelAdapterError as e:
            logging.error("Failed to fit a model, check the model params")
            raise ModelAdapterError(f"Failed to fit model: {e}")

    def predict(
        self,
        data: List[AdapterInput],
        n: Optional[int] = 24,
        use_rolling_window: bool = True,
        #**kwargs,
    ):

        return super().predict(
            data=data,
            n=n,
            use_rolling_window=use_rolling_window,
            #**kwargs,
        )

    # def convert_input(self, input: List[AdapterInput]) -> Tuple[
    #     List[DartsTimeSeries],
    #     List[DartsTimeSeries],
    #     List[DartsTimeSeries],
    #     Optional[pd.DataFrame],
    # ]:
    #     """
    #     Converts the input data into the required format for the model, applying scaling transformations
    #     to the target and observed time series.

    #     Parameters:
    #         data (List[AdapterInput]): A list of AdapterInput objects containing the input data.

    #     Returns:
    #         Tuple[List[DartsTimeSeries], List[DartsTimeSeries], List[DartsTimeSeries], Optional[pd.DataFrame]]:
    #             - target: A list of scaled DartsTimeSeries objects representing the target time series.
    #             - known: A list of DartsTimeSeries objects representing the known covariates.
    #             - observed: A list of scaled DartsTimeSeries objects representing the observed covariates.
    #             - static: An optional pandas DataFrame containing static covariates, if available.
    #     """

    #     target, known, observed, static = super().convert_input(input)
    #     target = self.scaler_target.fit_transform(target)
    #     observed = self.scaler_cov.fit_transform(observed)
    #     return target, known, observed, static

    # def predict(
    #     self,
    #     data: Union[AdapterInput, List[AdapterInput]],
    #     n: Optional[int] = 1,
    #     historical_forecast=True,
    #     predict_likelihood_parameters=True,
    #     forecast_horizon=5,
    # ) -> List[TimeSeries]:
    #     """
    #     Predict using the model and provided data.
    #     """

    #     target, known, observed, static = self.convert_input(data)
    #     self._prepare_predict_args(target, known, observed, static)
    #     self._predict_args.update(
    #         {"predict_likelihood_parameters": predict_likelihood_parameters}
    #     )
    #     if historical_forecast:
    #         # If historical forecast is True, use the model's historical_forecast method
    #         last_points_only = False
    #         prediction = self.model.historical_forecasts(
    #             **self._predict_args,
    #             forecast_horizon=forecast_horizon,
    #             last_points_only=last_points_only,
    #             retrain=False,
    #         )
    #         prediction = self.scaler_target.inverse_transform(prediction)
    #     else:
    #         if n is not None:
    #             self._predict_args["n"] = n
    #         prediction = self.model.predict(**self._predict_args)
    #         prediction = self.scaler_target.inverse_transform(prediction)
    #     # self.isquantile = predict_likelihood_parameters
    #     if isinstance(data, list):
    #         # print(type(prediction[0][0]))

    #         prediction_ts_format = [
    #             DartsModelAdapter.to_time_series(ts=pred, quantiles=self.quantiles)
    #             for pred in prediction
    #         ]
    #     else:
    #         prediction_ts_format = DartsModelAdapter.to_time_series(
    #             ts=prediction, quantiles=self.quantiles
    #         )
    #     return prediction_ts_format

    def tune(
        self,
        train_data: List[AdapterInput],
        val_data: Optional[List[AdapterInput]],
        **kwargs,
    ):
        raise NotImplementedError("Tune method is not implemented yet.")

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
