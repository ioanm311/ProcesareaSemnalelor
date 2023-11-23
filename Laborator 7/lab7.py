from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

X = misc.face(gray=True)
plt.imshow(X, cmap=plt.cm.gray)
plt.show()

Y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(Y))

plt.imshow(freq_db)
plt.colorbar()
plt.show()

rotate_angle = 45
X45 = ndimage.rotate(X, rotate_angle)
plt.imshow(X45, cmap=plt.cm.gray)
plt.show()

Y45 = np.fft.fft2(X45)
plt.imshow(20*np.log10(abs(Y45)))
plt.colorbar()
plt.show()

freq_x = np.fft.fftfreq(X.shape[1])
freq_y = np.fft.fftfreq(X.shape[0])

plt.stem(freq_x, freq_db[:][0])
plt.show()

freq_cutoff = 120

Y_cutoff = Y.copy()
Y_cutoff[freq_db > freq_cutoff] = 0
X_cutoff = np.fft.ifft2(Y_cutoff)
X_cutoff = np.real(X_cutoff)    # avoid rounding erros in the complex domain,
                                # in practice use irfft2
plt.imshow(X_cutoff, cmap=plt.cm.gray)
plt.show()

pixel_noise = 200

noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Original')
plt.show()
plt.imshow(X_noisy, cmap=plt.cm.gray)
plt.title('Noisy')
plt.show()

# Exercitiul 1

def function1(n1, n2):
    return np.sin(2 * np.pi * n1 + 3 * np.pi * n2)

def function2(n1, n2):
    return np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)

# Dimensiunea imaginii
N1, N2 = 64, 64

# Crearea unei grile pentru n1 și n2
n1, n2 = np.meshgrid(np.arange(N1), np.arange(N2))

x1 = function1(n1, n2)
x2 = function2(n1, n2)

# Transformatei Fourier
X1 = np.fft.fft2(x1)
X2 = np.fft.fft2(x2)

plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(x1, cmap='gray')
plt.title('x1(n1, n2)')

plt.subplot(232)
plt.imshow(np.fft.fftshift(np.log(np.abs(X1) + 1)), cmap='viridis')
plt.title('Spectru x1')

plt.subplot(233)
plt.imshow(np.angle(X1), cmap='hsv')
plt.title('Faza x1')

plt.subplot(234)
plt.imshow(x2, cmap='gray')
plt.title('x2(n1, n2)')

plt.subplot(235)
plt.imshow(np.fft.fftshift(np.log(np.abs(X2) + 1)), cmap='viridis')
plt.title('Spectru x2')

plt.subplot(236)
plt.imshow(np.angle(X2), cmap='hsv')
plt.title('Faza x2')

plt.show()



