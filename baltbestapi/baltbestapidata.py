from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..data.cachedapidata import CachedAPIData
from ..data.clearmldatamixin import ClearmlDataMixin
from ..data.timeseries import TimeSeries


class BaltBestAPIData(ClearmlDataMixin, CachedAPIData):

    def __init__(
        self,
        url: str = "https://edc.baltbest.de/public",
        local_copy: Optional[Path] = None,
    ):

        super().__init__()
        self.url = url
        self.local_copy = local_copy

    def get_data(self):

        if self.local_copy:
            if self.is_up2date():
                self._fetch_from_cache()
            else:
                self.update_local_copy()
        else:
            self.local_copy = self.get_from_clearml()
            
            self._fetch_from_cache()

    def _fetch_data_from_api(self):
        # TODO For now, the logic does not get the data from API
        raise NotImplementedError("Subclasses must implement this method.")

    def _fetch_from_cache(self):
        raise NotImplementedError("Subclasses must implement this method.")
