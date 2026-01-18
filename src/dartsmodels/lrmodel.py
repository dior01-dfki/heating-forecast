from forcateri.model.dartsmodeladapter import DartsModelAdapter
from darts.models import LinearRegressionModel
from typing import Optional
from clover import clover
from darts.utils.missing_values import extract_subseries, fill_missing_values


class DartsLRModel(DartsModelAdapter):

    @clover
    def __init__(self, 
                 model: Optional[LinearRegressionModel] = None, 
                 model_name: Optional[str] = None,
                 likelihood = None,
                 lags_past_covariates: int = 0,
                 lags_future_covariates: int = 0,
                 output_chunk_length: int = 1,
                 output_chunk_shift: int = 0,
                  quantiles: Optional[list] = None,
                 *args, **kwargs,
                ):
        super().__init__(*args, **kwargs)
        if model is not None:
            self.model = model
        else:
            self.model = LinearRegressionModel(
                likelihood=likelihood,
                lags_past_covariates=lags_past_covariates,
                lags_future_covariates=lags_future_covariates,
                output_chunk_length=output_chunk_length,
                output_chunk_shift=output_chunk_shift,
                quantiles=quantiles,
            )
            self.model_name = model_name if model_name else "DartsLRModel"

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
                sub = fill_missing_values(sub)
                # Slice covariates at the same time index range as this target subseries
                start, end = sub.start_time(), sub.end_time()

                if known:
                    k_sub = known[t_idx].slice(start, end)
                    k_sub = fill_missing_values(k_sub)
                    known_chunks.append(k_sub)
                if observed:
                    o_sub = observed[t_idx].slice(start, end)
                    o_sub = fill_missing_values(o_sub)
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
    def fit(self, train_data, val_data):
        super().fit(train_data, val_data)