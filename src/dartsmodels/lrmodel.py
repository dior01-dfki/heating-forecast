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
        self.quantiles = quantiles
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


    def fit(self, train_data, val_data):
        super().fit(train_data, val_data)

    def predict(self,data, n: int=12, rolling_window=True, *args, **kwargs):
        #return self.model.predict(n, predict_likelihood_parameters=predict_likelihood_parameters, *args, **kwargs)
        print(f"predict_likelihood_parametrs:{self.is_likelihood}")
        return super().predict(n=n, data=data, rolling_window=rolling_window, *args, **kwargs)