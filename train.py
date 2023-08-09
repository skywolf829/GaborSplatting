from models.GaussianSplatting2D import GaussianSplatting2D
from models.PeriodicPrimitives2D import PeriodicPrimitives2D
from utils.data_generators import load_img
import torch
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as imageio
from tqdm import tqdm

def to_img(x):
    x_img = x.clone().detach().cpu().numpy() * 255
    x_img = np.clip(x_img, 0, 255)
    x_img = x_img.astype(np.uint8)
    return x_img

if __name__ == '__main__':

    device = "cpu"
    torch.random.manual_seed(42)
    np.random.seed(42)

    model = PeriodicPrimitives2D(10, device=device)
    training_img = load_img("./data/tablecloth_zoom2.jpg")
    training_img_copy = (training_img.copy() * 255).astype(np.uint8)
    og_img_shape = training_img.shape

    g_x = torch.arange(0, og_img_shape[0], dtype=torch.float32, device=device) / og_img_shape[0]
    g_y = torch.arange(0, og_img_shape[1], dtype=torch.float32, device=device) / og_img_shape[1]
    training_img_positions = torch.stack(torch.meshgrid([g_x, g_y], indexing='ij'), 
                                        dim=-1).reshape(-1, 2).type(torch.float32)
    training_img_colors = torch.tensor(training_img, dtype=torch.float32, device=device).flatten()[:,None]

    optim = model.create_optimizer()

    training_imgs = []
    
    n_iters = 1000
    t = tqdm(range(n_iters))
    for i in t:
        optim.zero_grad()
        loss, model_out = model.loss(training_img_positions, training_img_colors)
        loss.backward()
        optim.step()

        if i % 10 == 0 or i == n_iters-1:
            #print(f"[{i}/{n_iters}] loss: {l.item():0.04f}")
            to_append = np.concatenate([
                to_img(model_out).reshape(og_img_shape),
                training_img_copy
                ], axis=1)
            training_imgs.append(to_append)

        t.set_description(f"[{i+1}/{n_iters}] loss: {loss.item():0.04f}")
    
    imageio.imwrite("training_imgs.gif", training_imgs)