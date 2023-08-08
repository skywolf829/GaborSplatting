import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import warnings
from skimage.metrics import structural_similarity
import torch


def smooth_signal(sig, window_size=7):
    gauss_kernel = np.exp(-(np.linspace(-2, 2, window_size)**2)/2.)/np.sqrt(np.pi*2)
    pad_size = window_size//2
    return np.convolve(sig, gauss_kernel)[pad_size:sig.shape[0]+pad_size]

def smooth_signal2D(sig, window_size=7):
    x = np.linspace(-2, 2, window_size)[None, :].repeat(window_size, axis=0)
    y = np.linspace(-2, 2, window_size)[:, None].repeat(window_size, axis=1)
    gauss_kernel = np.exp(-(x*x+y*y)/2.)/np.sqrt(np.pi*2)
    return convolve2d(sig, gauss_kernel)

def fftPlot(sig, dt=None, plot=True):
    # Here it's assumes analytic signal (real signal...) - so only half of the axis is required

    if dt is None:
        dt = 1
        t = np.arange(0, sig.shape[-1])
        xLabel = 'samples'
    else:
        t = np.arange(0, sig.shape[-1]) * dt
        xLabel = 'freq [Hz]'

    if sig.shape[0] % 2 != 0:
        warnings.warn("signal preferred to be even in size, autoFixing it...")
        t = t[0:-1]
        sig = sig[0:-1]

    sigFFT = np.fft.fft(sig) / t.shape[0]  # Divided by size t for coherent magnitude

    freq = np.fft.fftfreq(t.shape[0], d=dt)

    # Plot analytic signal - right half of frequence axis needed only...
    firstNegInd = np.argmax(freq < 0)
    freqAxisPos = freq[0:firstNegInd]
    sigFFTPos = 2 * sigFFT[0:firstNegInd]  # *2 because of magnitude of analytic signal

    if plot:
        plt.figure()
        plt.plot(freqAxisPos, np.abs(sigFFTPos))
        plt.xlabel(xLabel)
        plt.ylabel('mag')
        plt.title('Analytic FFT plot')
        plt.show()

    return sigFFTPos, freqAxisPos

def ssim2D(x, y):
    return structural_similarity(x, y)

def psnr(x, y, range=1.0):
    return 20*torch.log10(range) - 10*torch.log10(((y-x)**2).sum())