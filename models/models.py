from models.PeriodicPrimitives2D import PeriodicPrimitives2D
import os
project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "data")
output_folder = os.path.join(project_folder_path, "output")
save_folder = os.path.join(project_folder_path, "savedModels")

def create_model(opt):
    if(opt['num_dims'] == 2):
        return PeriodicPrimitives2D(opt)
    
def load_model(opt):
    if(opt['num_dims'] == 2):
        model = PeriodicPrimitives2D(opt)
    model.load(os.path.join(save_folder, opt['save_name']))
    return model