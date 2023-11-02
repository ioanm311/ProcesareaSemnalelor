import numpy as np
import time
import matplotlib.pyplot as plt


def transformata_fourier(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for omega in range(N):
        X[omega] = 0
        for n in range(N):
            X[omega] += x[n] * np.exp(-2j * np.pi * omega * n / N)

    return X

# dimensiuni vectori
dimensiuni = [128, 256, 512, 1024, 2048, 4096, 8192]

# timp executie implementare 1
timp_executie = []

# timpul de execuție pentru numpy.fft
timp_executie_numpy = []


for N in dimensiuni:
    timp = np.linspace(0, 1, N)
    # un semnal pentru a testa (cos) de frecventa 7 Hz
    x = np.cos(2 * np.pi * 7 * timp)

    # masurare timp de executie
    timp_start = time.time()
    transformata_fourier(x)
    timp_final = time.time()
    timp_executie.append(timp_final - timp_start)

    # masurare timp numpy
    timp_start = time.time()
    np.fft.fft(x)
    timp_final = time.time()
    timp_executie_numpy.append(timp_final - timp_start)

plt.figure(figsize=(8, 6))
plt.plot(dimensiuni, timp_executie, label="Implementare proprie")
plt.plot(dimensiuni, timp_executie_numpy, label="numpy.fft")
plt.xlabel('Dimensiunea vectorului N')
plt.ylabel('Timp de execuție')
plt.title('Compararea timpului de execuție DFT')
plt.legend()
plt.show()

























