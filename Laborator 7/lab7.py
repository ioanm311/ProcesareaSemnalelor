from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

# Exercitiul 1
def x1(n1, n2):
    return np.sin(2 * np.pi * n1 + 3 * np.pi * n2)

def x2(n1, n2):
    return np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)

def Y1(N1, N2):
    Y = np.zeros((N1, N2))
    Y[0, 5] = Y[N1-1, N2-5] = 1
    return Y

def Y2(N1, N2):
    Y = np.zeros((N1, N2))
    Y[5, 0] = Y[N1-5, 0] = 1
    return Y

def Y3(N1, N2):
    Y = np.zeros((N1, N2))
    Y[5, 5] = Y[N1-5, N2-5] = 1
    return Y

# parametrii pt functii
N1, N2 = 256, 256
n1 = np.arange(N1)
n2 = np.arange(N2)

# semnalele și funcțiile
x1_semnal = x1(n1[:, None], n2[None, :])
x2_semnal = x2(n1[:, None], n2[None, :])
Y1_func = Y1(N1, N2)
Y2_func = Y2(N1, N2)
Y3_func = Y3(N1, N2)

# transformata
X1_spectru = np.fft.fftshift(np.fft.fft2(x1_semnal))
X2_spectru = np.fft.fftshift(np.fft.fft2(x2_semnal))
Y1_spectru = np.fft.fftshift(np.fft.fft2(Y1_func))
Y2_spectru = np.fft.fftshift(np.fft.fft2(Y2_func))
Y3_spectru = np.fft.fftshift(np.fft.fft2(Y3_func))

plt.figure(figsize=(15, 12))

plt.subplot(321)
plt.title('x1 (n1, n2)')
plt.imshow(x1_semnal, cmap='gray')

plt.subplot(322)
plt.title('Spectru x1')
plt.imshow(np.log(1 + np.abs(X1_spectru)), cmap='gray')

plt.subplot(323)
plt.title('x2 (n1, n2)')
plt.imshow(x2_semnal, cmap='gray')

plt.subplot(324)
plt.title('Spectru x2')
plt.imshow(np.log(1 + np.abs(X2_spectru)), cmap='gray')

plt.subplot(325)
plt.title('Y1 (n1, n2)')
plt.imshow(Y1_func, cmap='gray')

plt.subplot(326)
plt.title('Spectru Y1')
plt.imshow(np.log(1 + np.abs(Y1_spectru)), cmap='gray')

plt.tight_layout()
plt.show()

# Y2 și Y3
plt.figure(figsize=(15, 12))

plt.subplot(321)
plt.title('Y2 (n1, n2)')
plt.imshow(Y2_func, cmap='gray')

plt.subplot(322)
plt.title('Spectru Y2')
plt.imshow(np.log(1 + np.abs(Y2_spectru)), cmap='gray')

plt.subplot(323)
plt.title('Y3 (n1, n2)')
plt.imshow(Y3_func, cmap='gray')

plt.subplot(324)
plt.title('Spectru Y3')
plt.imshow(np.log(1 + np.abs(Y3_spectru)), cmap='gray')

plt.tight_layout()
plt.show()

# Exercitiul 2

# raton
X = misc.face(gray=True)

# imagine originala
plt.figure(figsize=(6, 6))
plt.title('Imaginea originală')
plt.imshow(X, cmap=plt.cm.gray)
plt.show()

# transformata Fourier
X_spectru = np.fft.fftshift(np.fft.fft2(X))

# calcul spectru în scala logaritmică
spectru_db = 20 * np.log10(np.abs(X_spectru))

plt.figure(figsize=(6, 6))
plt.title('Spectrul imaginii în scala logaritmică')
plt.imshow(spectru_db, cmap=plt.cm.gray)
plt.colorbar(label='Amplitudine (dB)')
plt.show()

# setarea pragului SNR autoimpus
prag_SNR = 54

X_comprimat = X_spectru.copy()
X_comprimat[spectru_db < (np.max(spectru_db) - prag_SNR)] = 0

X_comprimat = np.fft.ifft2(np.fft.ifftshift(X_comprimat)).real

plt.figure(figsize=(6, 6))
plt.title('Imaginea comprimată')
plt.imshow(X_comprimat, cmap=plt.cm.gray)
plt.show()

# Exercitiul 3

# conversie la tip de date float
X = X.astype(float)
X_comprimat = X_comprimat.astype(float)

# raport SNR înainte și după compresie
interval = X.max() - X.min()
SNR_inainte = peak_signal_noise_ratio(X, X_comprimat, data_range=interval)
SNR_dupa = peak_signal_noise_ratio(X, X_comprimat + np.random.normal(0, 20, size=X.shape), data_range=interval)

print(f'Raport SNR înainte de compresie: {SNR_inainte:.2f} dB')
print(f'Raport SNR după compresie: {SNR_dupa:.2f} dB')