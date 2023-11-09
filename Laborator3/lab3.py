import numpy as np
import matplotlib.pyplot as plt

N = 8  # numarul de pct în transformata
f = 0.25 # sin

n = np.arange(N)
x = np.sin(2 * np.pi * f * n)

# Transformatei Fourier
omega = np.linspace(0, 2 * np.pi, N, endpoint=False) # [0, 2/pi]
X = np.zeros(N, dtype=complex) # vector de complexe
for k in range(N):
    X[k] = np.sum(x * np.exp(-2j * np.pi * k * n / N))

# Generarea matricei
F = np.zeros((N, N), dtype=complex)
for k in range(N):
    for n in range(N):
        F[k, n] = np.exp(-2j * np.pi * k * n / N) / np.sqrt(N) # fiecare element din matrice

# Verificarea unitarității
F_conjugata = np.conj(F.T)
matricea_i = np.identity(N)
unitaritate = np.allclose(F_conjugata.dot(F), matricea_i)

print("Transformata Fourier:\n", X)
print("Matricea Fourier F:\n", F)
print("Unitară?", unitaritate)

# Desenarea părții reale și imaginare
plt.figure(figsize=(8, 5))

plt.subplot(2, 1, 1)
plt.stem(omega, X.real)
plt.title('Partea Reala a Transformatei Fourier')
plt.xlabel('Omega')
plt.ylabel('Partea Reala')

plt.subplot(2, 1, 2)
plt.stem(omega, X.imag)
plt.title('Partea Imaginara a Transformatei Fourier')
plt.xlabel('Omega')
plt.ylabel('Partea Imaginara')

plt.tight_layout()
plt.show()