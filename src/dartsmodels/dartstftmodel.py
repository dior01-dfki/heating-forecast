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

    def __init__(
        self,
        model: Optional[TFTModel] = None,
        name: Optional[str] = None,
        quantiles=[0.1, 0.5, 0.9],
        input_chunk_length=7,
        output_chunk_length=5,
        output_chunk_shift=0,
        hidden_size=16,
        lstm_layers=1,
        num_attention_heads=4,
        full_attention=False,
        feed_forward="GatedResidualNetwork",
        dropout=0.1,
        hidden_continuous_size=8,
        categorical_embedding_sizes=None,
        add_relative_index=False,
        #skip_interpolation=False,
        loss_fn=None,
        norm_type="LayerNorm",
        use_static_covariates=True,
        is_likelihood=True,
        add_encoders=None,
        n_epochs=1,
        batch_size = 8
    ):

        super().__init__(
            name=name, quantiles=quantiles, is_likelihood=is_likelihood
        )
        
        if model is not None:
            self.model = model 
        else:
            self.input_chunk_length = input_chunk_length
            self.output_chunk_length = output_chunk_length
            self.output_chunk_shift = output_chunk_shift
            self.hidden_size = hidden_size 
            self.lstm_layers = lstm_layers
            self.num_attention_heads = num_attention_heads
            self.full_attention = full_attention
            self.feed_forward = feed_forward
            self.dropout = dropout
            self.hidden_continuous_size = hidden_continuous_size
            self.categorical_embedding_sizes = categorical_embedding_sizes
            self.add_relative_index = add_relative_index
            self.loss_fn = loss_fn
            self.norm_type = norm_type
            self.use_static_covariates = use_static_covariates
            self.n_epochs = n_epochs
            self.add_encoders = add_encoders
            self.batch_size = batch_size
            log_dir = project_root.joinpath(
                f"logs/dartstcn/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            )
            logger = TensorBoardLogger(save_dir=log_dir)
            trainer_kwargs = dict(logger=[logger])
            self.model = TFTModel(
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                output_chunk_shift=self.output_chunk_shift,
                hidden_size=self.hidden_size,
                lstm_layers=self.lstm_layers,
                num_attention_heads=self.num_attention_heads,
                full_attention=self.full_attention,
                feed_forward=self.feed_forward,
                dropout=self.dropout,
                hidden_continuous_size=self.hidden_continuous_size,
                categorical_embedding_sizes=self.categorical_embedding_sizes,
                add_relative_index=self.add_relative_index,
                loss_fn=self.loss_fn,
                norm_type=self.norm_type,
                batch_size=self.batch_size,
                add_encoders=self.add_encoders,
                use_static_covariates=self.use_static_covariates,
                likelihood=QuantileRegression(self.quantiles) if self.is_likelihood else None,
                pl_trainer_kwargs=trainer_kwargs,
                n_epochs = self.n_epochs
            )
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
        # **kwargs,
    ):

        return super().predict(
            data=data,
            n=n,
            use_rolling_window=use_rolling_window,
            # **kwargs,
        )



    def __repr__(self):
        return "DartsTFTModel"
