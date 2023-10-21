import numpy as np
import matplotlib.pyplot as plt

# Exercitiul 1

def x(t):
    return np.cos(520 * np.pi * t + np.pi / 3)

def y(t):
    return np.cos(280 * np.pi * t - np.pi / 3)

def z(t):
    return np.cos(120 * np.pi * t + np.pi / 3)

t = np.arange(0, 0.03, 0.0005)

semnal_1 = x(t)
semnal_2 = y(t)
semnal_3 = z(t)

plt.figure(figsize=(8, 4))

# Semnalul x(t)
plt.subplot(3, 1, 1)
plt.plot(t, semnal_1)
plt.xlabel('Timp')
plt.ylabel('Valoare semnal')
plt.title('Semnalul x(t)')

# Semnalul y(t)
plt.subplot(3, 1, 2)
plt.plot(t, semnal_2)
plt.xlabel('Timp')
plt.ylabel('Valoare semnal')
plt.title('Semnalul y(t)')

# Semnalul z(t)
plt.subplot(3, 1, 3)
plt.plot(t, semnal_3)
plt.xlabel('Timp')
plt.ylabel('Valoare semnal')
plt.title('Semnalul z(t)')

plt.show()


hertz = 200 # I don't hardcode it bcs I want to have an overview
timp = np.arange(0, 0.03, 1 / hertz)

# discrete x[n], y[n], z[n]
x_n = x(timp)
y_n = y(timp)
z_n = z(timp)

plt.figure(figsize=(8, 4))

# Subplot pentru x[n]
plt.subplot(3, 1, 1)
plt.stem(timp, x_n)
plt.title('x[n]')
plt.xlabel('Timp')

# Subplot pentru y[n]
plt.subplot(3, 1, 2)
plt.stem(timp, y_n)
plt.title('y[n]')
plt.xlabel('Timp')

# Subplot pentru z[n]
plt.subplot(3, 1, 3)
plt.stem(timp, z_n)
plt.title('z[n]')
plt.xlabel('Timp')

plt.tight_layout()
plt.show()

# Exercitiul 2

# a)
perioada = 1 / 400 # pentru 400 hertz
timp_total = 1600 * perioada # esantioane * perioada de timp
# interval 0 ----> timp_total
axa_timp = np.linspace(0, timp_total, 1600)

def semnal_sin():
    return np.sin(2 * np.pi * 400 * axa_timp)

plt.figure(figsize=(8, 4))
plt.plot(axa_timp, semnal_sin())
plt.xlabel('Timp')
plt.ylabel('Amplitudine')
plt.title('Semnal Sinusoidal de 400 Hz cu 1600 de Eșantioane')
plt.grid(True)
plt.show()

# b)
esantioane_total = int(3 * 800) # durata * frecventa
axa_timp2 = np.linspace(0, 3, esantioane_total) # de la 0 ---> 3 secunde --> step nr de esantioane

def semnal_sin2():
    return np.sin(2 * np.pi * 800 * axa_timp2)

plt.figure(figsize=(8, 4))
plt.plot(axa_timp2, semnal_sin2())
plt.xlabel('Timp')
plt.ylabel('Amplitudine')
plt.title('Semnal Sinusoidal de 800 Hz cu Durata de 3 Secunde')
plt.grid(True)
plt.show()

# c)
# luam frecventa 240 pentru o durata de 7 secunde
esantioane_total2 = int(240 * 7)
axa_timp3 = np.linspace(0, 7, esantioane_total2)

def semnal_sawtooth():
    return 2 * (240 * axa_timp3 - np.floor(240 * axa_timp3 + 0.5))

plt.figure(figsize=(8, 4))
plt.plot(axa_timp3, semnal_sawtooth())
plt.xlabel('Timp')
plt.ylabel('Amplitudine')
plt.title('Semnalul Sawtooth de 240 Hz cu durata de 7 secunde')
plt.grid(True)
plt.show()

# d)
# luam frecventa 300 pentru o durata de 6 secunde
esantioane_total3 = int(300 * 6)
axa_timp4 = np.linspace(0, 6, esantioane_total3)

def semnal_square():
    return np.sign(np.sin(2 * np.pi * 300 * axa_timp4))

plt.figure(figsize=(8, 4))
plt.plot(axa_timp4, semnal_square())
plt.xlabel('Timp')
plt.ylabel('Amplitudine')
plt.title('Semnal Square de 300 Hz cu durata de 6 secunde')
plt.grid(True)
plt.show()

# e)
# Dimensiunea matricei
a = 128
b = 128
# np.random.rand genereaza un array de 128x128 si-l initializeaza aleator
semnalAleator = np.random.rand(a, b)

plt.imshow(semnalAleator)
plt.title('Semnal 2D Aleator')
plt.colorbar()  # bara de culoare
plt.show()

# Exercitiul 3
# Intervalul de timp intre 2 esantioane este 1 / frecventa de esntionare masurata in Hertz
# deci rezultatul va fi 1 / 2000 = 0.0005 hz























