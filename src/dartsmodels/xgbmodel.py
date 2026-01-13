from forcateri.model.dartsmodeladapter import DartsModelAdapter
from darts.models import XGBModel
from darts.utils.missing_values import extract_subseries
from typing import Optional
from darts.dataprocessing.transformers import Scaler

class dartsXGB(DartsModelAdapter):
    def __init__(self, *args, model: Optional[XGBModel] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if model is not None:
            self.model = model
        else:
            self.model = XGBModel(
                input_chunk_length=kwargs.get("input_chunk_length", 7),
                output_chunk_length=kwargs.get("output_chunk_length", 5),
                n_estimators=kwargs.get("n_estimators", 100),
                learning_rate=kwargs.get("learning_rate", 0.1),
                max_depth=kwargs.get("max_depth", 6),
                random_state=kwargs.get("random_state", None),
                    lags=7,
    lags_past_covariates=7,
            )
    
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

        target_chunks = extract_subseries(target, min_gap_size=4, mode="any")
        known_chunks = extract_subseries(known, min_gap_size=4, mode="any") if known is not None else None
        observed_chunks = extract_subseries(observed, min_gap_size=4, mode="any") if observed is not None else None
        print(f"Number of target chunks after extraction: {len(target_chunks)}")
        print(f"Number of known chunks after extraction: {len(known_chunks) if known_chunks is not None else 'N/A'}")
        print(f"Number of observed chunks after extraction: {len(observed_chunks) if observed_chunks is not None else 'N/A'}")
        self.scaler_target = Scaler().fit(target)
        self.scaler_known = Scaler().fit(known) if known is not None else None
        self.scaler_observed = Scaler().fit(observed) if observed is not None else None
        
        target = self.scaler_target.transform(target)
        if self.scaler_known:
            known = self.scaler_known.transform(known)
        if self.scaler_observed:
            observed = self.scaler_observed.transform(observed)

        return target_chunks, known_chunks, observed_chunks, static