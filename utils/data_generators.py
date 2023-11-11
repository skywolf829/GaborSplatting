import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as imageio
from scipy.interpolate import RegularGridInterpolator

def generate_1D_random_peroidic_data(resolution=512, extents=[-1, 1]):
    a = np.random.rand()
    b = np.random.rand()
    c = np.random.rand()*4
    d = np.random.rand()
    e = np.random.randn(resolution)*0.8*b*0
    x = np.sort(extents[0]+np.random.rand(resolution)*(extents[1]-extents[0]))
    y = a + b*np.sin(2 * np.pi * c*x + d) + e
    #y += np.sin(2*np.pi*4 * x)
    print(f"Generated sin wave: {a:0.02f}+{b:0.02f}sin(2pi*{c:0.02f}x + {d:0.02f})")
    plt.plot(x, y)
    plt.show()
    return x[:,None], y[:,None]

def generate_1D_random_peroidic_data_square(resolution=512, extents=[-1, 1]):
    a = np.random.rand()
    b = np.random.rand()*0.5
    c = np.random.rand()*4
    d = np.random.rand()
    e = np.random.randn(resolution)*0.8*b*0
    x = np.sort(extents[0]+np.random.rand(resolution)*(extents[1]-extents[0]))
    y = a + (b*np.sin(2 * np.pi * c*x + d))**0.01 + e
    y = np.nan_to_num(y)
    #y += np.sin(2*np.pi*4 * x)
    print(f"Generated sin wave: {a:0.02f}+{b:0.02f}sin(2pi*{c:0.02f}x + {d:0.02f})")
    plt.plot(x, y)
    plt.show()
    return x[:,None], y[:,None]

def generate_2D_random_peroidic_data_simple(resolution=512, extents=[-1, 1]):
    a = np.random.rand()
    b = np.random.rand()
    c = np.random.rand()
    d = np.random.rand()
    e = np.random.randn(resolution)*0.8*b
    x = extents[0]+np.random.rand(resolution, 2)*(extents[1]-extents[0])
    y = a + b*np.sin(2 * np.pi * c*x[:,0] + d) + e
    print(f"Generated sin wave: {a:0.02f}+{b:0.02f}sin(2pi*{c:0.02f}x + {d:0.02f})")
    return x, y[:,None]

def generate_2D_random_peroidic_data_simple_square(resolution=512, extents=[-5, 5], plot=True):
    a = np.random.rand()
    b = np.random.rand()
    c = np.random.rand()
    d = np.random.rand()
    e = np.random.randn(resolution)
    o = 0.8*b
    x = extents[0]+np.random.rand(resolution, 2)*(extents[1]-extents[0])
    y = a + b*(np.sin(2 * np.pi * c*x[:,0] + d)**0.2) + e*o
    y = np.nan_to_num(y)
    print(f"Generated sin wave: {a:0.02f}+{b:0.02f}sin(2pi*{c:0.02f}x + {d:0.02f})")
    if(plot):
        x_plot = np.linspace(extents[0], extents[1], 512)
        x_plot = np.stack([x_plot[:,None].repeat(512,axis=1), 
                           x_plot[None,:].repeat(512,axis=0)], axis=-1).reshape([-1, 2])
        y_plot = a + b*(np.sin(2 * np.pi * c * x_plot[:,0] + d)**0.2) + np.random.randn(512*512)*o
        y_plot = y_plot.reshape([512, 512])
        y_plot = np.nan_to_num(y_plot)
        plt.imshow(y_plot, extent=[extents[0], extents[1], extents[0], extents[1]])
        plt.show()
    return x, y[:,None]

def generate_2D_random_peroidic_data_moderate(resolution=512, extents=[-1, 1], plot=False):
    a = np.random.rand()
    b = np.random.rand()
    c = np.random.rand()
    d = np.random.rand()
    o = b*4
    e = np.random.randn(resolution)*o
    x = extents[0]+np.random.rand(resolution, 2)*(extents[1]-extents[0])
    r = np.random.rand()*np.pi*0.7
    R = np.array([[np.cos(r), -np.sin(r)],
                  [np.sin(r), np.cos(r)]])
    x_rotated = x[...,None].swapaxes(-1, -2) @ R[None,...] 
    x_rotated = x_rotated.squeeze()
    y = a + b*np.sin(2 * np.pi * c * x_rotated[:,0] + d) + e
    print(f"Generated sin wave: {a:0.02f}+{b:0.02f}sin(2pi*{c:0.02f}x + {d:0.02f}) + N(0, {o:0.02f})")
    print(f"at angle {r:0.02f} = {r/np.pi:0.02f} rad.")
    if(plot):
        x_plot = np.linspace(extents[0], extents[1], resolution)
        x_plot = np.stack([x_plot[:,None].repeat(resolution,axis=1), 
                           x_plot[None,:].repeat(resolution,axis=0)], axis=-1).reshape([-1, 2])
        x_plot_rotated = x_plot[...,None].swapaxes(-1, -2) @ R[None,...] 
        x_plot_rotated = x_plot_rotated.squeeze()
        y_plot = a + b*np.sin(2 * np.pi * c * x_plot_rotated[:,0] + d) + np.random.randn(resolution*resolution)*o
        y_plot = y_plot.reshape([resolution, resolution])
        plt.imshow(y_plot, extent=[extents[0], extents[1], extents[0], extents[1]])
        plt.show()
    return x, y

def generate_2D_random_peroidic_data_moderate_square(resolution=512, extents=[-5, 5], plot=True):
    a = np.random.rand()
    b = np.random.rand()
    c = np.random.rand()
    d = np.random.rand()
    e = np.random.randn(resolution)
    o = 0.8*b
    r = np.random.rand()*np.pi
    R = np.array([[np.cos(r), -np.sin(r)],
                  [np.sin(r), np.cos(r)]])
    x = extents[0]+np.random.rand(resolution, 2)*(extents[1]-extents[0])
    x_rotated = x[...,None].swapaxes(-1, -2) @ R[None,...] 
    x_rotated = x_rotated.squeeze()
    y = a + b*(np.sin(2 * np.pi * c*x_rotated[:,0] + d)**0.2) + e*o
    y = np.nan_to_num(y)
    print(f"Generated sin wave: {a:0.02f}+{b:0.02f}sin(2pi*{c:0.02f}x + {d:0.02f})")
    print(f"at angle {r:0.02f} = {r/np.pi:0.02f} rad.")
    if(plot):
        x_plot = np.linspace(extents[0], extents[1], resolution)
        x_plot = np.stack([x_plot[:,None].repeat(resolution,axis=1), 
                           x_plot[None,:].repeat(resolution,axis=0)], axis=-1).reshape([-1, 2])
        x_plot_rotated = x_plot[...,None].swapaxes(-1, -2) @ R[None,...] 
        x_plot_rotated = x_plot_rotated.squeeze()
        y_plot = a + b*(np.sin(2 * np.pi * c * x_plot_rotated[:,0] + d)**0.2) + np.random.randn(resolution*resolution)*o
        y_plot = y_plot.reshape([resolution, resolution])
        y_plot = np.nan_to_num(y_plot)
        plt.imshow(y_plot, extent=[extents[0], extents[1], extents[0], extents[1]])
        plt.show()
    return x, y[:,None]

def generate_2D_random_peroidic_data_hard(resolution=512, extents=[-1, 1], plot=False):
    
    num_waves = 5

    a = np.random.rand()
    b = np.random.rand(num_waves)
    c = np.random.rand(num_waves)*6
    d = np.random.rand(num_waves)
    o = b.max()*4
    e = np.random.randn(resolution)*o
    x = extents[0]+np.random.rand(resolution, 2)*(extents[1]-extents[0])
    r = np.random.rand(num_waves)*np.pi
    
    y = a + e
    eq_str = f"N({a:0.02f}, {o:0.02f})"
    for i in range(num_waves):
        R = np.array([[np.cos(r[i]), -np.sin(r[i])],
                  [np.sin(r[i]), np.cos(r[i])]])
        x_rotated = x[...,None].swapaxes(-1, -2) @ R[None,...] 
        x_rotated = x_rotated.squeeze()
        y += b[i]*np.sin(2 * np.pi * c[i] * x_rotated[:,0] + d[i])
        eq_str = f"{eq_str} + {b[i]:0.02f}*sin(2pi*{c[i]:0.02f}x + {d[i]:0.02f})"
    print(f"Generated sin wave: {eq_str})")
    print(f"at angles {r} = {r/np.pi} rad.")

    if(plot):
        x_plot = np.linspace(extents[0], extents[1], 512)
        x_plot = np.stack([x_plot[:,None].repeat(512,axis=1), 
                           x_plot[None,:].repeat(512,axis=0)], axis=-1).reshape([-1, 2])
        y_plot = a + np.random.randn(512*512)*o
        for i in range(num_waves):
            R = np.array([[np.cos(r[i]), -np.sin(r[i])],
                  [np.sin(r[i]), np.cos(r[i])]])
            x_plot_rotated = x_plot[...,None].swapaxes(-1, -2) @ R[None,...] 
            x_plot_rotated = x_plot_rotated.squeeze()
            y_plot += b[i]*np.sin(2 * np.pi * c[i] * x_plot_rotated[:,0] + d[i])
        y_plot = y_plot.reshape([512, 512])
        plt.imshow(y_plot, extent=[extents[0], extents[1], extents[0], extents[1]])
        plt.show()
    return x, y

def generate_2D_random_peroidic_data_hard_square(resolution=512, extents=[-5, 5], plot=False):
    
    num_waves = 2

    a = np.random.rand()
    b = np.random.rand(num_waves)
    c = np.random.rand(num_waves)
    d = np.random.rand(num_waves)
    o = b.max()*4
    e = np.random.randn(resolution)*o
    x = extents[0]+np.random.rand(resolution, 2)*(extents[1]-extents[0])
    r = np.random.rand(num_waves)*np.pi
    
    y = 0
    eq_str = f"N({a:0.02f}, {o:0.02f})"
    for i in range(num_waves):
        R = np.array([[np.cos(r[i]), -np.sin(r[i])],
                  [np.sin(r[i]), np.cos(r[i])]])
        x_rotated = x[...,None].swapaxes(-1, -2) @ R[None,...] 
        x_rotated = x_rotated.squeeze()
        y += b[i]*(np.sin(2 * np.pi * c[i] * x_rotated[:,0] + d[i])**0.2)
        eq_str = f"{eq_str} + {b[i]:0.02f}*sin(2pi*{c[i]:0.02f}x + {d[i]:0.02f})"
    
    y = a + e + np.nan_to_num(y)
    print(f"Generated sin wave: {eq_str})")
    print(f"at angles {r} = {r/np.pi} rad.")

    if(plot):
        x_plot = np.linspace(extents[0], extents[1], 512)
        x_plot = np.stack([x_plot[:,None].repeat(512,axis=1), 
                           x_plot[None,:].repeat(512,axis=0)], axis=-1).reshape([-1, 2])
        y_plot = 0
        for i in range(num_waves):
            R = np.array([[np.cos(r[i]), -np.sin(r[i])],
                  [np.sin(r[i]), np.cos(r[i])]])
            x_plot_rotated = x_plot[...,None].swapaxes(-1, -2) @ R[None,...] 
            x_plot_rotated = x_plot_rotated.squeeze()
            y_plot += b[i]*(np.sin(2 * np.pi * c[i] * x_plot_rotated[:,0] + d[i])**0.2)
        y_plot = a + np.random.randn(512*512)*o + np.nan_to_num(y_plot)
        y_plot = y_plot.reshape([512, 512])
        plt.imshow(y_plot, extent=[extents[0], extents[1], extents[0], extents[1]])
        plt.show()
    return x, y

def generate_2D_random_peroidic_data_harder_square(resolution=512, extents=[-5, 5], plot=False):
    
    num_waves = 2

    a = np.random.rand()
    b = np.random.rand(num_waves)
    c = np.random.rand(num_waves)*5
    d = np.random.rand(num_waves)
    o = b.max()*4
    e = np.random.randn(resolution)*o
    x = extents[0]+np.random.rand(resolution, 2)*(extents[1]-extents[0])
    r = np.random.rand(num_waves)*np.pi
    
    y = 0
    eq_str = f"N({a:0.02f}, {o:0.02f})"
    for i in range(num_waves):
        R = np.array([[np.cos(r[i]), -np.sin(r[i])],
                  [np.sin(r[i]), np.cos(r[i])]])
        x_rotated = x[...,None].swapaxes(-1, -2) @ R[None,...] 
        x_rotated = x_rotated.squeeze()
        y += b[i]*(np.sin(2 * np.pi * c[i] * x_rotated[:,0] + d[i])**0.2)
        eq_str = f"{eq_str} + {b[i]:0.02f}*sin(2pi*{c[i]:0.02f}x + {d[i]:0.02f})"
    
    y = a + e + np.nan_to_num(y)
    print(f"Generated sin wave: {eq_str})")
    print(f"at angles {r} = {r/np.pi} rad.")

    if(plot):
        x_plot = np.linspace(extents[0], extents[1], 512)
        x_plot = np.stack([x_plot[:,None].repeat(512,axis=1), 
                           x_plot[None,:].repeat(512,axis=0)], axis=-1).reshape([-1, 2])
        y_plot = 0
        for i in range(num_waves):
            R = np.array([[np.cos(r[i]), -np.sin(r[i])],
                  [np.sin(r[i]), np.cos(r[i])]])
            x_plot_rotated = x_plot[...,None].swapaxes(-1, -2) @ R[None,...] 
            x_plot_rotated = x_plot_rotated.squeeze()
            y_plot += b[i]*(np.sin(2 * np.pi * c[i] * x_plot_rotated[:,0] + d[i])**0.2)
        y_plot = a + np.random.randn(512*512)*o + np.nan_to_num(y_plot)
        y_plot = y_plot.reshape([512, 512])
        plt.imshow(y_plot, extent=[extents[0], extents[1], extents[0], extents[1]])
        plt.show()
    return x, y

def generate_2D_random_peroidic_data_hardest_square(resolution=512, extents=[-5, 5], plot=False):
    
    num_waves = 2

    a = np.random.rand()
    b = np.random.rand(num_waves)
    c = np.random.rand(num_waves)*5
    d = np.random.rand(num_waves)
    o = b.max()*4
    e = np.random.randn(resolution)*o
    x = extents[0]+np.random.rand(resolution, 2)*(extents[1]-extents[0])
    r = np.random.rand(num_waves)*np.pi
    
    y = 0
    eq_str = f"N({a:0.02f}, {o:0.02f})"
    for i in range(num_waves):
        R = np.array([[np.cos(r[i]), -np.sin(r[i])],
                  [np.sin(r[i]), np.cos(r[i])]])
        x_rotated = x[...,None].swapaxes(-1, -2) @ R[None,...] 
        x_rotated = x_rotated.squeeze()
        y += b[i]*(np.sin(2 * np.pi * c[i] * x_rotated[:,0] + d[i])**0.2)
        eq_str = f"{eq_str} + {b[i]:0.02f}*sin(2pi*{c[i]:0.02f}x + {d[i]:0.02f})"
    
    y = a + e + np.nan_to_num(y)
    mask1 = (x[:,0]<3) * (x[:,0]>2) *(x[:,1] <0.5) * (x[:,1]>-0.5)
    mask2 = (x[:,0]<5) * (x[:,0]>1) *(x[:,1] <-1.5) * (x[:,1]>-4.5)
    mask3 = (x[:,0]>-4) * (x[:,0]<-2) *(x[:,1] >-3) * (x[:,1]<3)
    y[mask1] = 0
    y[mask2] = 0
    y[mask3] = 0
    print(f"Generated sin wave: {eq_str})")
    print(f"at angles {r} = {r/np.pi} rad.")

    if(plot):
        x_plot = np.linspace(extents[0], extents[1], 512)
        x_plot = np.stack([x_plot[:,None].repeat(512,axis=1), 
                           x_plot[None,:].repeat(512,axis=0)], axis=-1).reshape([-1, 2])
        y_plot = 0
        for i in range(num_waves):
            R = np.array([[np.cos(r[i]), -np.sin(r[i])],
                  [np.sin(r[i]), np.cos(r[i])]])
            x_plot_rotated = x_plot[...,None].swapaxes(-1, -2) @ R[None,...] 
            x_plot_rotated = x_plot_rotated.squeeze()
            y_plot += b[i]*(np.sin(2 * np.pi * c[i] * x_plot_rotated[:,0] + d[i])**0.2)
        y_plot = a + np.random.randn(512*512)*o + np.nan_to_num(y_plot)
        mask1 = (x_plot[:,0]<3) * (x_plot[:,0]>2) * (x_plot[:,1]<0.5) * (x_plot[:,1]>-0.5)
        mask2 = (x_plot[:,0]<5) * (x_plot[:,0]>1) *(x_plot[:,1] <-1.5) * (x_plot[:,1]>-4.5)
        mask3 = (x_plot[:,0]>-4) * (x_plot[:,0]<-2) *(x_plot[:,1] >-3) * (x_plot[:,1]<3)
        y_plot[mask1] = 0
        y_plot[mask2] = 0
        y_plot[mask3] = 0
        y_plot = y_plot.reshape([512, 512])
        plt.imshow(y_plot, extent=[extents[0], extents[1], extents[0], extents[1]])
        plt.show()
    return x, y

def load_img(path):
    import PIL.Image
    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    img = imageio.imread(path).astype(np.float32)/ 255.0
    #if(len(img.shape) == 3):
    #    img = img[:,:,0] 
    if(img.shape[2] > 3):
        img = img[:,:,0:3]
    print(f"Img shape: {img.shape}")
    #img = smooth_signal2D(img, window_size=100)
    return img


def sample_img_points(img, n_points=512, plot=False):
    g_x = np.arange(0, img.shape[0], dtype=np.float32) / (img.shape[0]-1)
    g_y = np.arange(0, img.shape[1], dtype=np.float32) / (img.shape[1]-1)
    f = RegularGridInterpolator((g_x, g_y), img)
    samples = np.random.rand(n_points, 2)
    interp_data = f(samples)
    if(plot):
        plt.imshow(img, cmap="gray")
        #plt.scatter(samples[:,1], -samples[:,0], c=interp_data, cmap='gray')
        plt.show()
    if(len(interp_data.shape) == 1):
        interp_data = interp_data[:,None]
    return samples, interp_data