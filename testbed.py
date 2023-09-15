
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as imageio
from utils.data_generators import load_img
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

def thing1():
     x = np.linspace(-10., 10., 400)[None,:].repeat(400, axis=0)
     y = np.linspace(-10., 10., 400)[:,None].repeat(400, axis=1)

     g = np.stack([x,y], axis=-1)

     g_f = g.reshape(-1, 2)
     theta = np.pi/4
     R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]).reshape(2, 2)
     S = np.array([1, 0, 0, 5]).reshape(2,2)
     RSSR = S @ S.T 
     cov = np.linalg.inv(RSSR)

     a = 0.7
     b = 0.7
     w1 = 2*np.pi*0.15
     wavelength = 2*np.pi/w1

     x1 = g_f[:,0]
     x2 = g_f[:,1]
     g1_coeff = (a**2+b**2)**0.5
     phase_shift = 1-np.arctan(a/(b+1e-8))/np.pi
     print(phase_shift)
     print(wavelength)
     wave1 = a*np.sin(w1*(x1*np.cos(theta)+x2*np.sin(theta))) + b*np.cos(w1*(x1*np.cos(theta)+x2*np.sin(theta)))
     gaussian = g1_coeff*np.exp(-(g_f[:,None,:] @ g_f[:,:,None]/2))

     print(g_f.copy()[:,None,:].shape)
     print(R[None,...].shape)
     g_x = -wavelength/2 + (((g_f.copy()[:,None,:] @ R[None,...])[:,0,:]+(phase_shift*wavelength/2.)) % wavelength)

     gaussian_wave = g1_coeff*np.exp(-(g_x[:,None,:] @cov[None,None,...] @ g_x[:,:,None]/2))

     wave1 = wave1.reshape(400,400)
     gaussian = gaussian.reshape(400,400)
     gaussian_wave = gaussian_wave.reshape(400,400)

     summed = wave1 + gaussian
     multed = wave1 * gaussian

     im_top_row = np.concatenate([wave1, gaussian], axis=1)
     im_bottom_row = np.concatenate([gaussian_wave, wave1*gaussian_wave], axis=1)

     im = np.concatenate([im_top_row, im_bottom_row], axis=0)
     plt.imshow(im)
     plt.show()

def thing2():
     x = np.linspace(-10., 10., 400)[None,:].repeat(400, axis=0)
     y = np.linspace(-10., 10., 400)[:,None].repeat(400, axis=1)

     g = np.stack([x,y], axis=-1)
     theta1 = 0
     theta2 = np.pi*150/180
     R1 = np.array([[np.cos(theta1), -np.sin(theta1)],[np.sin(theta1), np.cos(theta1)]]).reshape(2, 2)
     R2 = np.array([[np.cos(theta2), -np.sin(theta2)],[np.sin(theta2), np.cos(theta2)]]).reshape(2, 2)
     w1 = np.sin(2*g @ R1)[...,0]
     w2 = np.sin(2*g @ R2)[...,1]
     print(w1.shape)

     comp = w1+w2
     plt.imshow(comp)
     plt.show()
     import imageio.v3 as imageio
     plt.imsave("synthetic2.jpg", comp)

def thing3():
     x = np.linspace(-10., 10., 400)[None,:].repeat(400, axis=0)
     y = np.linspace(-10., 10., 400)[:,None].repeat(400, axis=1)

     g = np.stack([x,y], axis=-1)

     a = 0.7
     b = 0.7
     w1 = 2*np.pi*0.2
     w2 = 2*np.pi*0.4
     wavelength = np.array([2*np.pi/w1, 2*np.pi/w2])

     theta1 = 0
     theta2 = np.pi/4
     imgs = []
     for theta2 in np.linspace(np.pi/4, np.pi*3/4, 100):
          g_f = g.reshape(-1, 2)
          S = np.array([.4, 0, 0, .4]).reshape(2,2)
          angle_diff = (theta2-theta1) % np.pi
          shear = np.tan(angle_diff - np.pi/2) / np.pi

          R = np.array([[np.cos(theta1), -np.sin(theta1)],
                        [np.sin(theta1), np.cos(theta1)]]).reshape(2, 2)
          SH = np.array([[1, shear*wavelength[1]],
                        [0, 1]]).reshape(2, 2)
          SHn = np.array([[1, -shear*wavelength[1]],
                        [0, 1]]).reshape(2, 2)
          RSSR =  S @ S.T
          cov = np.linalg.inv(RSSR)


          g1_coeff = (a**2+b**2)**0.5
          phase_shift = np.array([1-np.arctan(a/(b+1e-8))/np.pi, 1-np.arctan(a/(b+1e-8))/np.pi])

          g_x = -wavelength/2 + (((g_f.copy()[:,None,:] @ R[None,...]@ SH[None,...])[:,0,:]+(phase_shift*wavelength[None,:]/2.)) % wavelength)
          g_x = g_x[:,None,:] @ SHn[None,None,...]
          gaussian_wave = g1_coeff*np.exp(-(g_x @ cov[None,None,...] @ g_x.swapaxes(-1, -2)/2))

          gaussian_wave = gaussian_wave.reshape(400,400)
          gaussian_wave -= gaussian_wave.min()
          gaussian_wave /= gaussian_wave.max()
          imgs.append(gaussian_wave.copy()*255)
          #plt.imshow(gaussian_wave)
          #plt.show()
          
     imageio.imwrite("./output/shears.gif", imgs)

def thing4():
     x = np.linspace(-10., 10., 400)[None,:].repeat(400, axis=0)
     y = np.linspace(-10., 10., 400)[:,None].repeat(400, axis=1)

     g = np.stack([x,y], axis=-1)

     g_f = g.reshape(-1, 2)
     theta = np.pi/4

     wave1 = 1.5*np.sin(6.0*(g_f[:,0]*np.cos(theta)+g_f[:,1]*np.sin(theta))) + \
          3.2*np.cos(6.0*(g_f[:,0]*np.cos(theta)+g_f[:,1]*np.sin(theta)))
     g = np.exp(-0.5*(g_f**10).sum(axis=-1)**1)
     wave1 = wave1.reshape(400,400)
     g = g.reshape(400, 400)
     wave1 = wave1*g

     im = wave1
     plt.imshow(im)
     plt.show()
     im -= im.min()
     im /= im.max()
     im *= 255
     im = im.astype(np.uint8)
     imageio.imwrite("synthetic3.jpg", im)

def thing5():
     im = load_img("./data/tablecloth.jpg")


     x = np.linspace(0, 1.0, im.shape[1])
     y = np.linspace(0, 1.0, im.shape[0])
     g = np.stack(np.meshgrid(y,x, indexing='ij'), axis=-1)
     print(g.shape)
     aspect=im.shape[1]/im.shape[0]
     m1 = (g[:,:,0] > aspect*0.35*g[:,:,1] - 50/im.shape[0])
     m2 = (g[:,:,0] < aspect*0.4*g[:,:,1] + 760/im.shape[0])
     m3 = (g[:,:,0] < aspect*-1.55*g[:,:,1] + 2270/im.shape[0])
     m4 = (g[:,:,1] < 1040/im.shape[1])

     # for top of table
     m5 = (g[:,:,0] < aspect*0.375*g[:,:,1] + 420/im.shape[0])
     m6 = (g[:,:,0] < aspect*-1.25*g[:,:,1] + 1620/im.shape[0])
     
     im *= m1 * m2 * m3 * m4 * m5 * m6
     plt.imshow(im)
     plt.show()

def thing6():
     x = np.linspace(-np.pi, np.pi, 1000)
     
     imgs = []
     fig = Figure(figsize=(5, 4), dpi=100)
     canvas = FigureCanvasAgg(fig)
     ax = fig.add_subplot()

     for p in np.linspace(-5, 5, 200):
          y = np.sin(np.pi*x + 0)
          y = np.sign(y)*(np.abs(y)**np.exp(p))
          ax.plot(x, y)
          canvas.draw()  # Draw the canvas, cache the renderer
          
          s, (width, height) = canvas.print_to_buffer()
          # Option 2a: Convert to a NumPy array.
          image = np.fromstring(s, np.uint8).reshape((height, width, 4))
          imgs.append(image)
          ax.clear()
     imageio.imwrite("./output/Wave_power.mp4", imgs)

def thing7():
     x = np.linspace(-10., 10., 400)[None,:].repeat(400, axis=0)
     y = np.linspace(-10., 10., 400)[:,None].repeat(400, axis=1)

     g = np.stack([x,y], axis=-1)
     w1 = np.sin(2*g[...,0])
     img = np.zeros([400, 400, 3])
     img[w1<0,:] = np.array([1., 0, 0])
     img[w1>=0,:] = np.array([0, 1., 0])
     plt.imshow(img)
     plt.show()
     import imageio.v3 as imageio
     plt.imsave("synthetic4.jpg", img)

def thing8():

     fig = Figure(figsize=(5, 4), dpi=100)
     canvas = FigureCanvasAgg(fig)
     ax = fig.add_subplot()

     x = np.linspace(-5., 5., 4096)

     gaussian_period = 2.
     gaussian_width = 0.3
     gaussian_flat_top = 1.

     def compute_gaussian(x, p, w, f):
          #print(f"{p} {w} {f}")
          eval_spot = ((x + p/2) % p) - p/2
          return np.exp(-((eval_spot/w)**2)**f)
     
     imgs = []
     for gaussian_flat_top in np.linspace(-1, 4, 100):
          gaussian_flat_top = np.exp(gaussian_flat_top)
          
          y = compute_gaussian(x, gaussian_period, gaussian_width, gaussian_flat_top)
          ax.plot(x, y)
          
          canvas.draw()  # Draw the canvas, cache the renderer
          s, (width, height) = canvas.print_to_buffer()
          # Option 2a: Convert to a NumPy array.
          image = np.frombuffer(s, np.uint8).reshape((height, width, 4)).copy()
          imgs.append(image)
          ax.clear()
          
     imageio.imwrite("./output/periodic_gaussians.gif", imgs)

def thing9():

     fig = Figure(figsize=(5, 4), dpi=100)
     canvas = FigureCanvasAgg(fig)
     ax = fig.add_subplot()

     x = np.linspace(-5., 5., 256)
     y = np.linspace(-5., 5., 256)
     g = np.stack(np.meshgrid(y,x, indexing='ij'), axis=-1)
     
     imgs = []
     p_x = 2
     for p_y in np.linspace(.1, 2, 100):
          s_x = np.sin(g[:,:,0]*p_x) + np.cos(g[:,:,0]*p_x)
          s_y = np.sin(g[:,:,1]*p_y) + np.cos(g[:,:,1]*p_y) 
          result = s_x*s_y    
          ax.imshow(result)
          
          canvas.draw()  # Draw the canvas, cache the renderer
          s, (width, height) = canvas.print_to_buffer()
          # Option 2a: Convert to a NumPy array.
          image = np.frombuffer(s, np.uint8).reshape((height, width, 4)).copy()
          imgs.append(image)
          ax.clear()
          
     imageio.imwrite("./output/waves.gif", imgs)

def thing10():
     x = np.linspace(-10., 10., 400)[None,:].repeat(400, axis=0)
     y = np.linspace(-10., 10., 400)[:,None].repeat(400, axis=1)

     g = np.stack([x,y], axis=-1)
     theta1 = 0
     theta2 = np.pi*150/180
     R1 = np.array([[np.cos(theta1), -np.sin(theta1)],[np.sin(theta1), np.cos(theta1)]]).reshape(2, 2)
     R2 = np.array([[np.cos(theta2), -np.sin(theta2)],[np.sin(theta2), np.cos(theta2)]]).reshape(2, 2)
     w1 = np.sin(0.5*g @ R1)[...,0]
     w2 = np.sin(2*g @ R2)[...,1]
     print(w1.shape)

     comp = w1*w2
     plt.imshow(comp)
     plt.show()
     import imageio.v3 as imageio
     plt.imsave("synthetic5.jpg", comp)

def thing11():
     fig = Figure(figsize=(5, 4), dpi=100)
     canvas = FigureCanvasAgg(fig)
     ax = fig.add_subplot()

     x = np.linspace(-5., 5., 256)
     y = np.linspace(-5., 5., 256)
     g = np.stack(np.meshgrid(y,x, indexing='ij'), axis=-1)
     
     imgs = []
     p_x = 2
     p_y = 4
     for rot in np.linspace(0, 2*np.pi, 100):
          s_x = np.sin(g[:,:,0]*p_x)*0.5+0.5
          s_y = np.sin(g[:,:,1]*p_y)*0.5+0.5
          s_x, s_y = s_x*np.cos(rot) + s_y*np.sin(rot), -s_x*np.sin(rot)+s_y*np.cos(rot)
          s_result = s_x*s_y    

          g_x = (1/(1-1/np.e))*((np.exp(-(1*(1-s_x))))-(1/np.e))
          g_y = (1/(1-1/np.e))*((np.exp(-(1*(1-s_y))))-(1/np.e))
          g_result = g_x*g_y
          ax.imshow(np.concatenate([s_result, g_result], axis=0))
          
          canvas.draw()  # Draw the canvas, cache the renderer
          s, (width, height) = canvas.print_to_buffer()
          # Option 2a: Convert to a NumPy array.
          image = np.frombuffer(s, np.uint8).reshape((height, width, 4)).copy()
          imgs.append(image)
          ax.clear()
          
     imageio.imwrite("./output/sines.gif", imgs)

thing11()