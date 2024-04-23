from models.PeriodicPrimitives2D import PeriodicPrimitives2D
import os
from models.iNGP import iNGP

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "data")
output_folder = os.path.join(project_folder_path, "output")
save_folder = os.path.join(project_folder_path, "savedModels")

def create_model(opt):
    if(opt['model'] == "splats"):
        return PeriodicPrimitives2D(opt)
    elif(opt['model'] == "iNGP"):
        return iNGP(opt)
    
def load_model(opt, location):
    if(opt['model'] == "splats"):
        model = PeriodicPrimitives2D(opt)
    elif(opt['model'] == "iNGP"):
        model = iNGP()
    model.load(location)
    return model