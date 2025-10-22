import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from darts import TimeSeries as DartsTimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime
from src import project_root

from forcateri.data.adapterinput import AdapterInput
from forcateri.model.modelexceptions import InvalidModelTypeError, ModelAdapterError
from forcateri.model.dartsmodeladapter import DartsModelAdapter
from forcateri.data.timeseries import TimeSeries


class DartsTFTModel(DartsModelAdapter):

    def __init__(self, model: Optional[TFTModel] = None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.quantiles = kwargs.get("quantiles", [0.1, 0.5, 0.9])
        if model is not None:
            self.model = model
        else:
            self.input_chunk_length = kwargs.get("input_chunk_length", 7)
            log_dir = project_root.joinpath(
                f'logs/darts_tft/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
            )
            logger = TensorBoardLogger(save_dir=log_dir)
            my_stopper = EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min",
            )
            trainer_kwargs = dict(logger=[logger])
            trainer_kwargs["callbacks"] = [my_stopper]
            self.model = TFTModel(
                input_chunk_length=kwargs.get("input_chunk_length", 7),
                output_chunk_length=kwargs.get("output_chunk_length", 5),
                hidden_size=kwargs.get("hidden_size", 32),
                dropout=kwargs.get("dropout", 0.1),
                lstm_layers=kwargs.get("lstm_layers", 2),
                batch_size=kwargs.get("batch_size", 32),
                n_epochs=kwargs.get("n_epochs", 5),
                random_state=kwargs.get("random_state", 42),
                likelihood=QuantileRegression(self.quantiles),
                pl_trainer_kwargs=trainer_kwargs,
            )
        self.forecast_horizon = kwargs.get("forecast_horizon", 5)
        self.scaler_target = Scaler()
        self.scaler_cov = Scaler()

    def convert_input(self, input: List[AdapterInput]) -> Tuple[
        List[DartsTimeSeries],
        List[DartsTimeSeries],
        List[DartsTimeSeries],
        Optional[pd.DataFrame],
    ]:
        """
        Converts the input data into the required format for the model, applying scaling transformations
        to the target and observed time series.

        Parameters:
            data (List[AdapterInput]): A list of AdapterInput objects containing the input data.

        Returns:
            Tuple[List[DartsTimeSeries], List[DartsTimeSeries], List[DartsTimeSeries], Optional[pd.DataFrame]]:
                - target: A list of scaled DartsTimeSeries objects representing the target time series.
                - known: A list of DartsTimeSeries objects representing the known covariates.
                - observed: A list of scaled DartsTimeSeries objects representing the observed covariates.
                - static: An optional pandas DataFrame containing static covariates, if available.
        """

        target, known, observed, static = super().convert_input(input)
        self.target_col_names = [t.components[0] for t in target]
        target = self.scaler_target.fit_transform(target)
        observed = self.scaler_cov.fit_transform(observed)
        
        return target, known, observed, static

    def fit(
        self,
        train_data: List[AdapterInput],
        val_data: Optional[List[AdapterInput]],
    ):

        try:

            super().fit(train_data=train_data, val_data=val_data)

        except ModelAdapterError as e:
            logging.error("Failed to fit a model, check the model params")
            raise ModelAdapterError(f"Failed to fit model: {e}")

    def predict(
        self,
        data: Union[AdapterInput, List[AdapterInput]],
        n: Optional[int] = 1,
        historical_forecast=True,
        predict_likelihood_parameters=True,
        forecast_horizon=5,
    ) -> List[TimeSeries]:
        """
        Predict using the model and provided data.
        """

        super().prepare_predict_args(data=data)
        self._predict_args.update(
            {"predict_likelihood_parameters": predict_likelihood_parameters}
        )
        if historical_forecast:
            # If historical forecast is True, use the model's historical_forecast method
            last_points_only = False
            prediction = self.model.historical_forecasts(
                **self._predict_args,
                forecast_horizon=forecast_horizon,
                last_points_only=last_points_only,
                retrain=False,
            )
            prediction = self.scaler_target.inverse_transform(prediction)
        else:
            if n is not None:
                self._predict_args["n"] = n
            prediction = self.model.predict(**self._predict_args)
            prediction = self.scaler_target.inverse_transform(prediction)
        # self.isquantile = predict_likelihood_parameters
        if isinstance(data, list):
            

            prediction_ts_format = [
                DartsModelAdapter.to_time_series(ts=pred, quantiles=self.quantiles)
                for pred in prediction
            ]
            for ts, new_name in zip(prediction_ts_format, self.target_col_names):
                ts.data.columns = pd.MultiIndex.from_tuples(
                    [(new_name, q) for q in ts.data.columns.get_level_values(1)],
                    names=TimeSeries.COL_INDEX_NAMES
            )
        else:
            prediction_ts_format = DartsModelAdapter.to_time_series(
                ts=prediction, quantiles=self.quantiles
            )
        return prediction_ts_format

    def tune(
        self,
        train_data: List[AdapterInput],
        val_data: Optional[List[AdapterInput]],
        **kwargs,
    ):
        raise NotImplementedError("Tune method is not implemented yet.")

    @classmethod
    def load(cls, path: Union[Path, str]) -> "DartsTFTModel":
        try:
            model = TFTModel.load(path)
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
        return "DartsTFTModel"
