from forcateri.model.dartsmodeladapter import DartsModelAdapter
from darts.models import XGBModel
from darts.utils.missing_values import extract_subseries
from typing import Optional
from darts.dataprocessing.transformers import Scaler
from forcateri.data.adapterinput import AdapterInput
from typing import List
import pandas as pd
import pickle
from darts import TimeSeries as DartsTimeSeries
from forcateri.data.timeseries import TimeSeries
from forcateri.model.modelexceptions import ModelAdapterError
from clover import clover


class dartsXGB(DartsModelAdapter):

    @clover
    def __init__(self, 
                 model: Optional[XGBModel] = None, 
                 model_name: Optional[str] = None,
                 input_chunk_length: int = 7,
                 output_chunk_length: int = 5,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 6,
                 random_state: Optional[int] = None,
                 lags: int = 7,
                 lags_past_covariates: int = 7,
                 add_encoders: Optional[dict] = None,
                 *args, 
                 **kwargs):
        super().__init__(name=model_name,*args,**kwargs)
        if model is not None:
            self.model = model
        else:
            self.model = XGBModel(
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                lags=lags,
                lags_past_covariates=lags_past_covariates,
                add_encoders=add_encoders
            )
            self.model_name = model_name if model_name else "dartsXGB"

    def fit(self, train_data, val_data):
        super().fit(train_data, val_data)

    def convert_input(self, input):
        """
        Converts input data to Darts format and applies scaling transformations.

        This method extends the parent class's convert_input method by adding optional
        scaling transformations to the target series and covariates. Scalers are applied
        if they were configured during initialization.

        Parameters
        ----------
        input : List[AdapterInput]
            A list of AdapterInput objects containing the input data with target series,
            known/future covariates, observed/past covariates, and static covariates.

        Returns
        -------
        tuple
            A tuple containing four elements:
            - target : List[DartsTimeSeries] or DartsTimeSeries
                The target series, scaled if scaler_target is configured.
            - known : List[DartsTimeSeries] or DartsTimeSeries or None
                The known/future covariates, scaled if scaler_known is configured.
            - observed : List[DartsTimeSeries] or DartsTimeSeries or None
                The observed/past covariates, scaled if scaler_observed is configured.
            - static : pd.DataFrame or None
                The static covariates (not scaled).

        Notes
        -----
        - Scaling is only applied if the corresponding scaler was fitted during
          initialization (when scaler_data was provided).
        - The parent class's convert_input method handles the conversion from
          TimeSeries format to Darts format.
        - Scalers transform data to have zero mean and unit variance by default.
        """
        target, known, observed, static = super().convert_input(input)

        target_chunks, known_chunks, observed_chunks = [], [], []

        for t_idx, t in enumerate(target):
            t_subs = extract_subseries(t, min_gap_size=4, mode="any")

            for sub in t_subs:
                target_chunks.append(sub)

                # Slice covariates at the same time index range as this target subseries
                start, end = sub.start_time(), sub.end_time()

                if known:
                    k_sub = known[t_idx].slice(start, end)
                    known_chunks.append(k_sub)
                if observed:
                    o_sub = observed[t_idx].slice(start, end)
                    observed_chunks.append(o_sub)

        print(f"Number of target chunks after extraction: {len(target_chunks)}")
        print(
            f"Number of known chunks after extraction: {len(known_chunks) if known_chunks is not None else 'N/A'}"
        )
        print(
            f"Number of observed chunks after extraction: {len(observed_chunks) if observed_chunks is not None else 'N/A'}"
        )

        target = self.scaler_target.transform(target)
        if self.scaler_known:
            known = self.scaler_known.transform(known)
        if self.scaler_observed:
            observed = self.scaler_observed.transform(observed)

        return target_chunks, known_chunks, observed_chunks, static

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
            #logging.error("Failed to fit a model, check the model params")
            raise ValueError(f"Failed to fit model: {e}")
    
    def predict(
        self,
        data: List[AdapterInput],
        n: Optional[int] = 24,
        use_rolling_window: bool = True,
    ):

        return super().predict(
            data=data,
            n=n,
            use_rolling_window=use_rolling_window,
        
        )
