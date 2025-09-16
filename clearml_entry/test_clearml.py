# from clearml import Task 
# from .baltbestaggregatedapidata import BaltBestAggregatedAPIData 
from clearml import Dataset, Task

dataset_project: str = "ForeSightNEXT/BaltBest/Forcateri"
dataset_name: str = "ForcateriPipelineTest"


dataset = Dataset.create(
    dataset_project=dataset_project,
    dataset_name=dataset_name,
)
dataset.add_files('/home/dior00002/dfki/forcateri/_data/pipeline_test.csv')
dataset.upload()      # Uploads the data to the ClearML server/storage
dataset.finalize()    # Locks and versions the dataset

# data_root = Dataset.get(
#     dataset_project=dataset_project,
#     dataset_name=dataset_name,
#     #dataset_version=self.dataset_version,
# ).get_local_copy()
# print(data_root)