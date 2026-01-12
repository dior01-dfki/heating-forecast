from forcateri.model.dartsmodeladapter import DartsModelAdapter
from darts.models import XGBModel
from typing import Optional

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