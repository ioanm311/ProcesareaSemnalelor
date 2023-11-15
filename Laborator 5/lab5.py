import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Exercitiul 1 / # Exercitiul 2
## In fisierul Train.csv datele sunt organizate in intervale de o ora ---> frecventa de esantionare este de o ora
## Intervalul de timp pe care il acopera esantioanele din fisier este de ---> 2 ani si 1 luna

df = pd.read_csv(r'C:\Users\Ioan\PycharmProjects\ProcesareaSemnalelor\Laborator 5\Train.csv')
# Converteste coloana Datetime la formatul corect
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')

plt.figure(figsize=(12, 6))
plt.stem(df['Datetime'], df['Count'], linefmt='-', markerfmt='o', basefmt='k')
plt.title('Graficul semnalului în funcție de timp')
plt.xlabel('Datetime')
plt.ylabel('Count')
plt.show()

# Exercitiul 3

# sortam după coloana 'Datetime'
df.sort_values('Datetime', inplace=True)

# calculam diferența de timp între eșantioane consecutive
df['Datetime'].diff().mean()

# frecvența maximă
timp_total = (df['Datetime'].iloc[-1] - df['Datetime'].iloc[0]).total_seconds() / 3600
frecventa_maxima = 1 / (timp_total / (len(df) - 1))

print(f"Frecvența maximă prezenta în semnal: {frecventa_maxima:.5f} Hz")

# Exercitiul 4

frecventa_esantionare = len(df) / timp_total

# transformata Fourier
transformata_fourier = np.fft.fft(df['Count'])

# frecvente corespunzătoare la fiecare punct din transformata
frecvente = np.fft.fftfreq(len(df), d=1/frecventa_esantionare)

plt.plot(frecvente, np.abs(transformata_fourier))
plt.title('Modulul Transformatei Fourier')
plt.xlabel('Frecventa')
plt.ylabel('Amplitudine')
plt.show()

# Exercitiul 5
# aici luam indicele corespunzător frecvenței 0 Hz
frecventa0 = np.where(frecvente == 0)[0][0]

# eliminam componenta continua
transformata_fourier[frecventa0] = 0

# semnal fara componenta continua
semnal_nou = np.fft.ifft(transformata_fourier)

plt.plot(df['Datetime'], semnal_nou.real)
plt.title('Semnal fără Componentă Continuă')
plt.xlabel('Datetime')
plt.ylabel('Count')
plt.show()

plt.plot(frecvente, np.abs(transformata_fourier))
plt.title('Modulul Transformatei Fourier (fără componenta continua)')
plt.xlabel('Frecvența')
plt.ylabel('Amplitudine')
plt.show()