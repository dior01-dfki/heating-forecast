from clearml import Task, Dataset


from .baltbestaggregatedapidata import BaltBestAggregatedAPIData


project_name = "ForeSightNEXT/BaltBest/Forcateri"
dataset_name = "BaltBestAggregatedAPIData"

# baltbest = BaltBestAggregatedAPIData()


# data = baltbest.get_data()
# print(baltbest.local_copy)
# print(baltbest.ts)
ds = Dataset.create(
    dataset_project=project_name,   
    dataset_name=dataset_name,
)
ds.add_files(path = '/home/user/DFKI/forcateri/_data/showcase_data.csv')
ds.upload(show_progress=True)
ds.finalize()
#print(data_root)
