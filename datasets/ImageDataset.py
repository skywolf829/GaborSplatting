import torch
from utils.data_generators import load_img
import os
project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "data")
output_folder = os.path.join(project_folder_path, "output")
save_folder = os.path.join(project_folder_path, "savedModels")

def make_coord_grid(shape, device, flatten=True, align_corners=False, use_half=False):
    """ 
    Make coordinates at grid centers.
    return (shape.prod, 3) matrix with (z,y,x) coordinate
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        left = -1.0
        right = 1.0
        if(align_corners):
            r = (right - left) / (n-1)
            seq = left + r * \
            torch.arange(0, n, 
            device=device, 
            dtype=torch.float32).float()

        else:
            r = (right - left) / (n+1)
            seq :torch.Tensor = left + r + r * \
            torch.arange(0, n, 
            device=device, 
            dtype=torch.float32).float()
            
        if(use_half):
                seq = seq.half()
        coord_seqs.append(seq)

    ret = torch.meshgrid(*coord_seqs, indexing="ij")
    ret = torch.stack(ret, dim=-1)
    if(flatten):
        ret = ret.view(-1, ret.shape[-1])
    return ret.flip(-1)

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
        self.img = torch.tensor(load_img(os.path.join(data_folder, opt['training_data'])), device=self.opt['data_device'])      
        
    def __len__(self):
        return self.opt['train_iterations']
    
    def shape(self):
        return self.img.shape

    def forward(self, x):
        samples = torch.nn.functional.grid_sample(self.img.permute(2, 0, 1)[None,...], 
                                                  x[None,None,...].to(self.opt['data_device'])*2-1,
                                                  mode="bilinear",
                                                  align_corners=True)[0, :, 0, :].T
        return samples.to(x.device)
    
    def __getitem__(self, idx):
        points = torch.rand([self.opt['batch_size'], 2], 
                dtype=torch.float32, device=self.opt['data_device'])*2 - 1
        samples = torch.nn.functional.grid_sample(self.img.permute(2, 0, 1)[None,...], 
                                                  points[None,None,...],
                                                  mode="bilinear",
                                                  align_corners=True)[0, :, 0, :].T
        points += 1.
        points /= 2.
        return points.to(self.opt['device']), samples.to(self.opt['device'])
        