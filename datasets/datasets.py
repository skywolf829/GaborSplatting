from datasets.ImageDataset import ImageDataset
from datasets.SceneDataset import SceneDataset

def create_dataset(opt):
    if(opt['training_data_type'] == "image"):
        return ImageDataset(opt)
    if(opt['training_data_type'] == "scene"):
        return SceneDataset(opt)