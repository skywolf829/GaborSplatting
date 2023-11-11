from models.PeriodicPrimitives2D import PeriodicPrimitives2D
from models.PeriodicPrimitives3D import PeriodicPrimitives3D
import os
project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "data")
output_folder = os.path.join(project_folder_path, "output")
save_folder = os.path.join(project_folder_path, "savedModels")

def create_model(opt):
    if(opt['num_dims'] == 2):
        return PeriodicPrimitives2D(opt)
    if(opt['num_dims'] == 3):
        return PeriodicPrimitives3D(opt)
    
def load_model(opt):
    if(opt['num_dims'] == 2):
        model = PeriodicPrimitives2D(opt)
    if(opt['num_dims'] == 3):
        model = PeriodicPrimitives3D(opt)
    model.load(os.path.join(save_folder, opt['save_name']))
    return model