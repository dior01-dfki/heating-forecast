from forcateri import DataProvider, SeriesRole, DataSource, AdapterInput
from forcateri.data.dataprovider import Cutoff
from typing import List, Dict, Union



    
class BaltDataProvider(DataProvider):
    def __init__(
        self,
        data_sources: List[DataSource],
        roles: List[Dict[SeriesRole, List[str]]],
        data_purposes: List[int],
        splits: float = 0.8,  # fraction for train/val
    ):
        super().__init__(data_sources=data_sources, roles=roles, splits=splits)
        self.data_purposes = data_purposes
        self.test_target = []
        self.test_known = []
        self.test_observed = []
        self.is_test_separated = False


    def _separate_ts(self):
        self.target = []
        self.known = []
        self.observed = []
        self.test_target = []
        self.test_known = []
        self.test_observed = []

        for ds, role, purpose in zip(self.data_sources, self.roles, self.data_purposes):
            columns_target = role.get(SeriesRole.TARGET) or []
            columns_known = role.get(SeriesRole.KNOWN) or []
            columns_observed = role.get(SeriesRole.OBSERVED) or []

            data_list = ds.get_data()
            for ts_obj in data_list:
                target_slice = ts_obj.get_feature_slice(index=columns_target)
                known_slice = ts_obj.get_feature_slice(index=columns_known) if columns_known else None
                observed_slice = ts_obj.get_feature_slice(index=columns_observed) if columns_observed else None

                if purpose == 1:
                    self.target.append(target_slice)
                    self.known.append(known_slice)
                    self.observed.append(observed_slice)
                else:  #
                    self.test_target.append(target_slice)
                    self.test_known.append(known_slice)
                    self.test_observed.append(observed_slice)

        self.is_separated = True
        self.is_test_separated = True

    
    def _get_split_set(self, split_type: str) -> List[AdapterInput]:
        """
        Retrieves a dataset split (train, validation, or test) based on the split type.

        Args:
            split_type (str): The type of dataset split to retrieve.
                              It can be "train", "val", or "test".

        Returns:
            List[AdapterInput]: A list of AdapterInput objects representing the requested dataset split.
        """
        #logger.debug(f"Retrieving {split_type} dataset split.")
        
        
        list_of_tuples = []

        for target_ts, known_ts, observed_ts in zip(
            self.target, self.known, self.observed
        ):
            if split_type == "train":
                # logger.debug(
                #     "Processing training split. List[AdapterInput] length: %d",
                #     len(list_of_tuples),
                # )
                list_of_tuples.append(
                    AdapterInput(
                        target=target_ts[:self.splits] if target_ts is not None else None,
                        known=known_ts[:self.splits] if known_ts is not None else None,
                        observed=(
                            observed_ts[:self.splits] if observed_ts is not None else None
                        ),
                        static=self.static,
                    )
                )
            elif split_type == "val":
                # logger.debug(
                #     "Processing validation split. List[AdapterInput] length: %d",
                #     len(list_of_tuples),
                # )
                list_of_tuples.append(
                    AdapterInput(
                        target=(
                            target_ts[self.splits:] if target_ts is not None else None
                        ),
                        known=known_ts[self.splits:] if known_ts is not None else None,
                        observed=(
                            observed_ts[self.splits:]
                            if observed_ts is not None
                            else None
                        ),
                        static=self.static,
                    )
                )

        return list_of_tuples
    
    def get_test_set(self):
        if not self.is_test_separated:
            self._separate_ts()
        test_list = []
        for t, k, o in zip(self.test_target, self.test_known, self.test_observed):
            test_list.append(AdapterInput(
                target=t,
                known=k,
                observed=o,
                static=self.static
            ))
        return test_list
