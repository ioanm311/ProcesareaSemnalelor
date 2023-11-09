import pandas as pd
import matplotlib.pyplot as plt


# Exercitiul 1
## In fisierul Train.csv datele sunt organizate in intervale de o ora ---> frecventa de esantionare este de o ora

df = pd.read_csv(r'C:\Users\Ioan\PycharmProjects\ProcesareaSemnalelor\Laborator 5\Train.csv')
# Converteste coloana Datetime la formatul corect
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')

plt.figure(figsize=(12, 6))
plt.stem(df['Datetime'], df['Count'], linefmt='-', markerfmt='o', basefmt='k')
plt.title('Graficul semnalului în funcție de timp')
plt.xlabel('Datetime')
plt.ylabel('Count')
plt.show()

# Exercitiul 2


