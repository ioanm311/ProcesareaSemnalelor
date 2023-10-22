import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# Exercitiul 1:

# Semnalul sinusoidal
ampl = 1.0
frecv = 2.0 # hertz
faza = 0.0  # în radiani
durata = 2.0  # secunde
pas_timp = 0.001  # pasul de timp între eșantioane

# sir de esantionae de la 0 la durata cu pasul specificat
timp = np.arange(0, durata, pas_timp)

# Generăm semnalul sinusoidal
sin = ampl * np.sin(2 * np.pi * frecv * timp + faza)

# Generăm semnalul cosinusoidal
cos = ampl * np.cos(2 * np.pi * frecv * timp + faza - np.pi / 2)

plt.figure(figsize=(8, 5))

# Subplot sin
plt.subplot(2, 1, 1)
plt.plot(timp, sin)
plt.title('Semnal Sinusoidal')
plt.xlabel('Timp')
plt.ylabel('Amplitudine')

# Subplot cos
plt.subplot(2, 1, 2)
plt.plot(timp, cos)
plt.title('Semnal Cosinusoidal')
plt.xlabel('Timp')
plt.ylabel('Amplitudine')

plt.tight_layout()
plt.show()


# Exercitiul 2

amplitudine2 = 1.0
frecventa2 = 5.0
faze = [0, np.pi/2, np.pi, 4*np.pi/5]

# Dimensiune semnal
semnal = 100

# timp + semnal sin pentru fiecare faza
timp2 = np.arange(0, 1, 1 / semnal)
semnale = [amplitudine2 * np.sin(2 * np.pi * frecventa2 * timp2 + faza) for faza in faze]

# zgomot Gaussian
z = np.random.randn(semnal)

# SNR
snrVal = [0.1, 1, 10, 100]

plt.figure(figsize=(8, 5))

for i, snr in enumerate(snrVal):
    # normele L2 ale semnalului original și ale zgomotului
    norma_x = np.linalg.norm(semnale[i])
    norma_z = np.linalg.norm(z)

    # amplitudinea zgomotului
    gama = np.sqrt((norma_x ** 2) / (snr * (norma_z ** 2)))

    # semnalul original si zgomotul + amplitudinea
    semnal_zgomot = semnale[i] + gama * z

    plt.subplot(4, 1, i+1)
    plt.plot(timp2, semnal_zgomot, label=f'SNR = {snr}')
    plt.title(f'Semnal cu SNR = {snr}')
    plt.xlabel('Timp')
    plt.ylabel('Amplitudine')
    plt.legend()

plt.tight_layout()
plt.show()

# Exercitiul 4

frecventa_sin = 10

# perioada sawtooth
sawthoot_perioada = 5 # secunde

amplitudine_sin = 1.5
amplitudine_sawtooth = 1.5

durata2 = 8 # durata semnalului in secunde

esantioane_frecv = 1000
esantioane = durata2 * esantioane_frecv
timp3 = np.linspace(0, durata2, int(esantioane))

sin2 = amplitudine_sin * np.sin(2 * np.pi * frecventa_sin * timp3)
sawtooth = amplitudine_sawtooth * (2 * (timp3 / sawthoot_perioada - np.floor(0.5 + timp3 / sawthoot_perioada)) - 1)
suma = sin2 + sawtooth

plt.figure(figsize=(8, 5))

plt.subplot(3, 1, 1)
plt.plot(timp3, sin2)
plt.title('Semnal Sinusoidal')
plt.xlabel('Timp')
plt.ylabel('Amplitudine')

plt.subplot(3, 1, 2)
plt.plot(timp3, sawtooth)
plt.title('Semnal Sawtooth')
plt.xlabel('Timp')
plt.ylabel('Amplitudine')

plt.subplot(3, 1, 3)
plt.plot(timp3, suma)
plt.title('Suma Semnalelor')
plt.xlabel('Timp')
plt.ylabel('Amplitudine')

plt.tight_layout()
plt.show()


# Exercitiul 5

frecv_semnal_audio1 = 600
frecv_semnal_audio2 = 700
durata3 = 4

frecv_esantionare = 3000
esantioane2 = durata3 * frecv_esantionare
timp4 = np.linspace(0, durata3, int(esantioane2))

semnal_audio1 = np.sin(2 * np.pi * frecv_semnal_audio1 * timp4)
semnal_audio2 = np.sin(2 * np.pi * frecv_semnal_audio2 * timp4)

# concatenam semnalele, am gasit functia concatenate
semnale_concatenate = np.concatenate((semnal_audio1, semnal_audio2))

sd.play(semnale_concatenate, frecv_esantionare)
sd.wait()























