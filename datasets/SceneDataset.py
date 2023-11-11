import torch
from utils.data_generators import load_img

class SceneDataset(torch.utils.data.Dataset):
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
        training_img = load_img("./data/"+opt['training_data'])
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

        self.training_samples = torch.rand(len(self), device=self.device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if(self.batch_size < len(self)):
            sample = self.training_samples.random_().topk(self.batch_size, dim=0).indices
            return self.x[sample], self.y[sample]
        else:
            return self.x, self.y
    