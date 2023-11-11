import torch
from utils.data_generators import load_img
import os
project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "data")
output_folder = os.path.join(project_folder_path, "output")
save_folder = os.path.join(project_folder_path, "savedModels")

class ImageDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, opt):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.batch_size = opt['batch_size']
        self.device = opt['data_device']
        self.opt = opt
        training_img = load_img(os.path.join(data_folder, opt['training_data']))
        img_shape = list(training_img.shape)[0:2]
        self.og_img_shape = list(training_img.shape)
        
        g_x = torch.arange(0, img_shape[0], dtype=torch.float32, device=self.device) / (img_shape[0]-1)
        g_y = torch.arange(0, img_shape[1], dtype=torch.float32, device=self.device) / (img_shape[1]-1)
        training_img_positions = torch.stack(torch.meshgrid([g_x, g_y], indexing='ij'), 
                                            dim=-1).reshape(-1, 2).type(torch.float32)
        training_img_colors = torch.tensor(training_img, dtype=torch.float32, device=self.device).reshape(-1,training_img.shape[-1])

        self.x = training_img_positions
        self.y = training_img_colors 
        
        max_img_reconstruction_dim_size = 1024    
        xmin = self.x.min(dim=0).values
        xmax = self.x.max(dim=0).values

        img_scale = min(max_img_reconstruction_dim_size, max(img_shape))/max(img_shape)
        self.training_preview_img_shape = [int(img_shape[i]*img_scale) for i in range(xmin.shape[0])]
        g = [torch.linspace(xmin[i], xmax[i], self.training_preview_img_shape[i], device=self.device) for i in range(xmin.shape[0])]
        self.training_preview_positions = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)

        # Shuffle the training data once
        idx = torch.randperm(len(self), device=self.device, dtype=torch.long)
        self.training_samples = self.x.to(self.device)[idx]
        self.training_colors = self.y.to(self.device)[idx]

    def get_output_shape(self):
        return self.og_img_shape

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if(self.batch_size < len(self)):
            start = (self.batch_size*idx) % len(self)
            end = min(len(self), start+self.batch_size)
            return self.training_samples[start:end].to(self.opt['device']), \
                self.training_colors[start:end].to(self.opt['device'])
        else:
            return self.x.to(self.opt['device']), self.y.to(self.opt['device'])
    