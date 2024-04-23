import torch
import tinycudann as tcnn
import numpy as np
import os

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "data")
output_folder = os.path.join(project_folder_path, "output")
save_folder = os.path.join(project_folder_path, "savedModels")

# Config that has roughly the same # parameters as the 3mil Gabor model (~30mil params)
config = {
	"loss": {
		"otype": "RelativeL2"
	},
	"optimizer": {
		"otype": "Adam",
		"learning_rate": 1e-2,
		"beta1": 0.9,
		"beta2": 0.99,
		"epsilon": 1e-15,
		"l2_reg": 1e-6
	},
	"encoding": {
		"otype": "HashGrid",
		"n_levels": 19,
		"n_features_per_level": 2,
		"log2_hashmap_size": 20,
		"base_resolution": 16
	},
	"network": {
		"otype": "FullyFusedMLP",
		"activation": "ReLU",
		"output_activation": "None",
		"n_neurons": 64,
		"n_hidden_layers": 2
	}
}

class iNGP(torch.nn.Module):
    def __init__(self, opt):
        super(iNGP, self).__init__()
        self.opt = opt
        self.model = tcnn.NetworkWithInputEncoding(
            2, 3,
            config["encoding"], config["network"]
        ).to(self.opt['device'])

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
            lr=config['optimizer']['learning_rate'], 
            eps=config['optimizer']['epsilon'], 
            betas=[config['optimizer']['beta1'], config['optimizer']['beta2']])
    
    def update_learning_rate(self, iteration):
        pass

    def training_routine_updates(self, iteration, writer=None):
        pass
    
    def update_cumulative_gradients(self):
        pass

    def loss(self, x, y):
        # x is our output, y is the ground truth
        model_out = self(x)
        mse = torch.nn.functional.mse_loss(model_out,y)
        final_loss = mse
        losses = {
            "final_loss": final_loss,
            "mse": mse,
        }
        return losses, model_out
    
    def param_count(self):
        total = 0
        for group in self.optimizer.param_groups:    
            total += group['params'][0].numel()
        return total
    
    def effective_param_count(self):
        return self.param_count()
    
    def get_num_primitives(self):
        return 1
    
    def vis_heatmap(self, points):
        return None
    
    def save(self, path):
        torch.save({'state_dict': self.model.state_dict()}, 
            os.path.join(path, "model.ckpt.tar"),
            pickle_protocol=4
        )

    def load(self, path):
        ckpt = torch.load(os.path.join(path, 'model.ckpt.tar'), 
            map_location = self.opt['device'])   
        self.model.load_state_dict(ckpt['state_dict'])
    
    def forward(self, x):
        return self.model(x).float()