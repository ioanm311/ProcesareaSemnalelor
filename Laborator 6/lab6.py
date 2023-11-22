import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Exercitiul 1

# vector aleator
N = 100
x = np.random.rand(N)

plt.figure(figsize=(12, 4))
plt.subplot(141)
plt.plot(x)
plt.title('Vector initial')

# x ← x * x de trei ori
for i in range(3):
    x = x * x

    # afisare grafic pentru fiecare iteratie
    plt.subplot(142 + i)
    plt.plot(x)
    plt.title(f'Iteratia {i + 1}')

plt.tight_layout()
plt.show()

# Exercitiul 2

# convultia directa p(x) si q(x)
def convultie_directa(p, q):
    m, n = len(p), len(q)
    r = np.zeros(m + n - 1, dtype=int)

    for i in range(m):
        for j in range(n):
            r[i + j] += p[i] * q[j]

    return r

# convultie folosind fft
def convolutie_fft(p, q):
    N = 2**int(np.ceil(np.log2(len(p) + len(q) - 1)))
    p_pad = np.pad(p, (0, N - len(p))) # adauga 0 la sfarsitul polinomului -> dimensiune N
    q_pad = np.pad(q, (0, N - len(q)))

    # transformata Fourier
    P = np.fft.fft(p_pad)
    Q = np.fft.fft(q_pad)

    # inmultirea frecventelor
    R = P * Q

    # transformata inversa
    r = np.fft.ifft(R).real

    return np.rint(r).astype(int)

# polinoame aleatoare
N = 5  # grad maxim polinoame
p = np.random.randint(-5, 6, N + 1) # luam intervalul [-5, 5]
q = np.random.randint(-5, 6, N + 1)

rezultat_direct = convultie_directa(p, q)
rezultat_fft = convolutie_fft(p, q)

print("Coeficientii polinomului p(x):", p)
print("Coeficientii polinomului q(x):", q)
print("Rezultatul convolutiei directe:", rezultat_direct)
print("Rezultatul convolutiei folosind FFT:", rezultat_fft)

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.stem(p, basefmt="b-", markerfmt="bo", label="p(x)")
plt.stem(q, basefmt="r-", markerfmt="ro", label="q(x)")
plt.title("Polinoamele p(x) si q(x)")
plt.legend()

plt.subplot(132)
plt.stem(rezultat_direct, basefmt="g-", markerfmt="go", label="Direct")
plt.title("Convolutie Directa")
plt.legend()

plt.subplot(133)
plt.stem(rezultat_fft, basefmt="m-", markerfmt="mo", label="FFT")
plt.title("Convolutie cu FFT")
plt.legend()

plt.tight_layout()
plt.show()

# Exercitiul 3

df = pd.read_csv('Train.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')

# perioada de 3 zile
start_date = '2012-09-25'
end_date = '2012-09-27'

date = df[(df['Datetime'] >= start_date) & (df['Datetime'] < end_date)]
print(date)















