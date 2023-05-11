import numpy as np
import matplotlib.pyplot as plt

# Load data
LPsyst = np.load('LPSYST.npy')
f = np.load('f.npy')
g = np.load('g.npy')
n = np.load('n.npy')

# Perform Fourier Transform and shift
F = np.fft.fftshift(np.fft.fft2(f))
G = np.fft.fftshift(np.fft.fft2(g))
N = np.fft.fftshift(np.fft.fft2(n))
LPsyst = np.fft.fftshift(LPsyst)
eps = 1E-6

# Pseudo filter
F_pseudo = F / (LPsyst + eps)
f_pseudo = np.abs(np.fft.ifft2(np.fft.ifftshift(F_pseudo)))

# Noise removal
F_noise = G / (LPsyst + eps) - N / (LPsyst + eps)
f_noise = np.abs(np.fft.ifft2(np.fft.ifftshift(F_noise)))

# Plotting
plt.figure(1)
plt.clf()
plt.imshow(f_pseudo)
plt.title('Image one using Pseudo filter')

plt.figure(2)
plt.clf()
plt.imshow(f_noise)
plt.title('Image two using Pseudo filter with noise removed')

plt.figure(3)
plt.clf()
plt.imshow(f)
plt.title('Original image one')

plt.figure(4)
plt.clf()
plt.imshow(g)
plt.title('Original image two (with noise)')

plt.show()
