import logging
from datetime import datetime
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.baltbestapi.baltbestapidata import BaltBestAPIData
from pathlib import Path

from forcateri.data.timeseries import TimeSeries

from clearml import Dataset, Task

logger = logging.getLogger(__name__)


class BaltBestAggregatedAPIData(BaltBestAPIData):
    # dataset_project: str = "ForeSightNEXT/BaltBest/resampled"
    # dataset_name: str = "AcceptableRooms"
    # file_name: str = "acceptable_rooms.csv"
    dataset_project: str = "ForeSightNEXT/BaltBest/resampled"
    dataset_name: str = "ResampledData"
    file_name = 'resampled_data.csv'
    
    def __init__(
        self,
        group_col: str = "room_id",
        time_col: str = "datetime",
        freq: str = "h",
        known="temperature_outdoor_avg",
        observed: List[str] = [
            "temperature_1_max",
            "temperature_2_max",
            "temperature_room_avg",
        ],
        target: str = "q_hca",
        static: Optional[Union[str, List[str]]] = None,
        url: str = "https://edc.baltbest.de/public",
        local_copy: Optional[str] = None,
    ):
        super().__init__(
            url=url,
            local_copy=local_copy,
        )
        self.ts = []
        self.link_dataset(
            dataset_project=BaltBestAggregatedAPIData.dataset_project,
            dataset_name=BaltBestAggregatedAPIData.dataset_name,
            file_name=BaltBestAggregatedAPIData.file_name,
        )
        self.target: str = target
        self.group_col: str = group_col
        self.time_col: str = time_col
        self.freq: str = freq
        self.known: Union[str, List[str]] = known
        self.observed: Union[str, List[str]] = observed
        self.static: Union[str, List[str]] = static
        # self.value_cols: List[str] = self._get_value_cols(
        #     self.target, self.known, self.observed, self.static
        # )
        self.value_cols: List[str] = self._get_value_cols(
            'target', self.known, self.observed, self.static
        )
        self.ts_dict = {}

    def get_data(self):
        super().get_data()
        return self.ts

    @staticmethod
    def room_subset():

        metadata = Dataset.get(
            dataset_project='ForeSightNEXT/BaltBest',
            dataset_name='BaltBestMetadata',
            dataset_version='0.0.2',)
        data_qa_path = metadata.get_local_copy()
        data_qa_report = pd.read_csv(f"{data_qa_path}/data_qa_report.csv", index_col=[0,1])
        acceptable_rooms = (
            data_qa_report
            .groupby(level='room_id')
            .apply(lambda g: ((g['non_nan_ratio'] >= 0.8) & (g['non_zero_ratio'] > 0.3)).all())
        )

        # Get the list of room_ids that are True, empty if none
        acceptable_room_ids = acceptable_rooms.index[acceptable_rooms].tolist()
        return acceptable_room_ids

    def _fetch_from_cache(self):
        """
        Fetch data from a local CSV file, process it by resampling and grouping, and store it as a TimeSeries instance.

        The method performs the following operations:
        - Reads the CSV file into a DataFrame.
        - Converts the time column to datetime format.
        - Groups the data by a specified column and resamples it into 60-minute intervals.
        - Drops the grouping column and resets the index.
        - Converts the resulting DataFrame into a TimeSeries instance.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This method modifies the instance's `ts` attribute to store the resulting TimeSeries object.

        Raises
        ------
        FileNotFoundError
            If the specified local CSV file does not exist.
        ValueError
            If the time column is not present in the DataFrame.
        """
        logger.debug("Fetching data from local cache.")
        # print(f"Local copy set to: {self.local_copy}")
        df = pd.read_csv(Path(self.local_copy) / self.file_name)

        room_ids = self.room_subset()
        df = df[df['room_id'].isin(room_ids)]
        out_path = Path(self.local_copy) / 'acceptable_rooms.csv'
        df.to_csv(out_path, index=False)

        Task.current_task().upload_artifact(name='acceptable_rooms', artifact_object=out_path)
        #HERE ALIGNING THE COLUMN NAMES with prediction
        df.rename(columns={self.target:'target'},inplace=True)
        logger.debug(f"DataFrame columns after renaming: {df.columns.tolist()}")
        df[self.time_col] = pd.to_datetime(df[self.time_col]).dt.tz_localize(None)
        df = (
            df.set_index(self.time_col)
            .groupby(self.group_col)
            .resample("60min")
            .asfreq()
            .drop(columns=[self.group_col])
            .reset_index()
        )
        self.ts, self.ts_dict = self._from_group_df(
            df=df,
            group_col=self.group_col,
            time_col=self.time_col,
            value_cols=self.value_cols,
            freq=self.freq,
        )

    def _get_value_cols(self, *args: Union[str, List[str], None]) -> List[str]:
        """
        Safely merges input variables into a single flat list.
        """
        result = []
        for arg in args:
            if arg is None:
                continue
            if isinstance(arg, list):
                result.extend(arg)
            else:
                result.append(arg)
        return result

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        time_col: Optional[str] = "time_stamp",
        value_cols: Optional[Union[List[str], str]] = None,
        freq: Optional[Union[str, int]] = "h",
        ts_type: Optional[str] = "determ",
    ) -> TimeSeries:
        logger.info("Creating TimeSeries from DataFrame via class method.")
        logger.debug(f"Input DataFrame columns: {df.columns.tolist()}")
        formatted = BaltBestAggregatedAPIData._build_internal_format(
            df, time_col, value_cols, freq=freq, ts_type=ts_type
        )

        return TimeSeries(formatted)

    def _from_group_df(
        self,
        df: pd.DataFrame,
        group_col: str,
        time_col: Optional[str] = None,
        value_cols: Optional[Union[List[str], str]] = None,
        freq: Optional[Union[str, int]] = "h",
        ts_type: Optional[str] = "determ",
    ) -> List[pd.DataFrame]:
        """
        Build `TimeSeries` instances for each group in the DataFrame.

        This method groups the DataFrame by the specified `group_col` and applies the logic
        from `from_dataframe` to each group. Each group is expected to contain a time column (or index)
        and one or more value columns representing the time series data.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame from which to initialize the instances.
        group_col : str
            The column name used for grouping the data. Each unique value in this column will
            result in a separate `TimeSeries` instance.
        time_col : Optional[str], default None
            The name of the column in the DataFrame that contains time information.
            If provided, this column must exist in the DataFrame.
        value_cols : Optional[Union[List[str], str]], default None
            The name(s) of the column(s) in the DataFrame that contain the values.
            Can be a single column name or a list of column names.
        freq : Optional[Union[str, int]], default 'h'
            The frequency of the time series data.
        ts_type : Optional[str], default 'determ'
            The type of the time series to create. Use 'quantile' for quantile forecasts,
            'determ' for deterministic series, and 'sampled' for sampled series.

        Returns
        -------
        List[TimeSeries]
            A list of `TimeSeries` instances, one for each unique group in the DataFrame.

        Raises
        ------
        ValueError
            If `group_col` is not found in the DataFrame.
        """
        if group_col not in df.columns:
            # logger.error("Initialization failed: group_col not found in the DataFrame.")
            raise ValueError(f"Column {group_col} not found in the DataFrame.")
        # if value_cols is None:
        #         value_cols = df.columns[df.columns != time_col]
        unique_group = df[group_col].unique()
        ts_dict = {}
        ts_list = []
        for i, group_id in enumerate(unique_group):
            df_group = df[df[group_col] == group_id]
            ts_instance = BaltBestAggregatedAPIData.from_dataframe(
                df_group, time_col, value_cols, freq, ts_type
            )
            # ts_dict[group_id] = ts_instance
            ts_list.append(ts_instance)
            ts_dict[i] = group_id
        return ts_list, ts_dict

    @staticmethod
    def _build_internal_format(
        df: pd.DataFrame,
        time_col: Optional[str],
        value_cols: Optional[Union[List[str], str]],
        freq: Optional[Union[str, int]] = "h",
        ts_type: Optional[str] = "determ",
    ) -> pd.DataFrame:
        df = df.copy()
        if isinstance(df.index, pd.MultiIndex):
            if time_col in df.index.names:
                time_stamp_values = df.index.get_level_values(time_col)
                df = df.reset_index(
                    level=[name for name in df.index.names if name != time_col],
                    drop=True,
                )
                df[time_col] = time_stamp_values
            else:
                df = df.reset_index(drop=True)
        else:
            df = df.reset_index()

        df[time_col] = pd.to_datetime(df[time_col])
        if not isinstance(time_col, str):
            raise TypeError("time_col must be a string.")
        if time_col not in df.columns:
            raise ValueError(f"Column {time_col} not found in DataFrame.")

        t0_index = pd.date_range(
            start=df[time_col].min(), end=df[time_col].max(), freq=freq
        )
        if value_cols is None:
            value_cols = df.columns[df.columns != time_col]
        elif isinstance(value_cols, str):
            value_cols = [value_cols]
        
        logger.debug(f"Building internal format with time_col: {time_col}, value_cols: {value_cols}, freq: {freq}, ts_type: {ts_type}")
        features = value_cols
        row_dim_names = ["offset", "time_stamp"]
        col_dim_names = ["feature", "representation"]

        if ts_type == "determ":
            point_0_index = [pd.Timedelta(0)]
            point_0_row_index = pd.MultiIndex.from_product(
                [point_0_index, t0_index], names=row_dim_names
            )
            determ_col_index = pd.MultiIndex.from_product(
                [features, ["value"]], names=col_dim_names
            )
            df = df[features]
            return pd.DataFrame(
                df.values, index=point_0_row_index, columns=determ_col_index
            )

        elif ts_type == "sampled":
            sampled_cols = [f"s_{i}" for i in range(16)]
            point_1_row_index = pd.MultiIndex.from_product(
                [pd.to_timedelta([1], unit="h"), t0_index], names=row_dim_names
            )
            sampled_col_index = pd.MultiIndex.from_product(
                [features, sampled_cols], names=col_dim_names
            )
            return pd.DataFrame(
                df[features].values, index=point_1_row_index, columns=sampled_col_index
            )

        elif ts_type == "quantile":
            quant_cols = ["q_0.1", "q_0.5", "q_0.9"]
            range_index = pd.to_timedelta(np.arange(1, 25), unit="h")
            range_row_index = pd.MultiIndex.from_product(
                [range_index, t0_index], names=row_dim_names
            )
            quant_col_index = pd.MultiIndex.from_product(
                [features, quant_cols], names=col_dim_names
            )
            return pd.DataFrame(
                df[features].values, index=range_row_index, columns=quant_col_index
            )

        else:
            raise ValueError(
                "Invalid ts_type provided. Use 'determ', 'sampled', or 'quantile'."
            )

    def is_up2date(self):
        # TODO update the logic later
        self.last_updated = datetime.now()
        return True

    def update_local_copy(self):
        # TODO update the logic later
        pass

    def _fetch_data_from_api(self):
        raise NotImplementedError("Subclasses must implement this method.")
