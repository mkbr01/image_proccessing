import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# Load data
g = np.load('blur1.npy')
G = np.fft.fftshift(np.fft.fft2(g))

# PSF/OTF (Motion blur) Eq5.7-1
[u, v] = np.meshgrid(np.arange(0, 40), np.arange(0, 40))
k = 1E-6
M = 40
N = 0
H = np.fft.fftshift(np.pad(np.exp(-k * ((u - M) ** 2 + (v - N) ** 2) ** (5 / 6)), ((34, 34), (34, 34))))

# Inverse filtering
F_inv = G / H
f_inv = np.abs(np.fft.ifft2(np.fft.ifftshift(F_inv)))

# Pseudo Inverse filtering
eps = 1E-6
F_pseudo = G / (H + eps)
f_pseudo = np.abs(np.fft.ifft2(np.fft.ifftshift(F_pseudo)))

# Show images
plt.figure(1)
plt.clf()
plt.imshow(f_inv)
plt.title('Blurred image with inverse filter')

plt.figure(2)
plt.clf()
plt.imshow(g)
plt.title('Image with motion blur')

plt.figure(3)
plt.clf()
plt.imshow(f_pseudo)
plt.title('Blurred image with pseudo inverse filter')

plt.show()
