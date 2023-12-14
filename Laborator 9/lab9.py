import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# seed-ul pentru reproducibilitate
np.random.seed(42)

# dimensiune
N = 1000

t = np.arange(N)

# componenta de trend de gradul 2
trend = 0.02 * t**2

# componenta de sezon cu două frecvențe
seas1 = 5 * np.sin(2 * np.pi * t / 50)
seas2 = 3 * np.cos(2 * np.pi * t / 30)
seasonality = seas1 + seas2

# componenta de variabilitate mica
noise = np.random.normal(0, 1, N)

X = np.random.normal(0, 0.5, N)

# serie de timp ca suma a celor patru componente
time_series = trend + seasonality + noise + X

plt.figure(figsize=(12, 18))

# Componenta de trend
plt.subplot(4, 1, 1)
plt.plot(t, trend, label='Trend', linestyle='--')
plt.title('Componenta de Trend')
plt.xlabel('Timp')
plt.ylabel('Valoare')
plt.legend()

# Componenta de sezon
plt.subplot(4, 1, 2)
plt.plot(t, seasonality, label='Sezon')
plt.title('Componenta de Sezon')
plt.xlabel('Timp')
plt.ylabel('Valoare')
plt.legend()

# Componenta de zgomot alb
plt.subplot(4, 1, 3)
plt.plot(t, noise, label='Zgomot Alb', linestyle='--')
plt.title('Componenta de Zgomot Alb')
plt.xlabel('Timp')
plt.ylabel('Valoare')
plt.legend()

# Componenta adițională (variabilă X)
plt.subplot(4, 1, 4)
plt.plot(t, X, label='Variabilă X', linestyle='--')
plt.title('Variabilă X')
plt.xlabel('Timp')
plt.ylabel('Valoare')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(t, time_series, label='Seria de Timp')
plt.title('Seria de Timp și Componentele Sale')
plt.xlabel('Timp')
plt.ylabel('Valoare')
plt.legend()
plt.show()

# model AR la seria de timp
order = 2  # ordinul modelului AR
ar_model = sm.tsa.AutoReg(time_series, lags=order)
result = ar_model.fit()

print(result.summary())

predictions = result.predict(start=0, end=N-1)

plt.figure(figsize=(12, 6))
plt.plot(t, time_series, label='Seria de Timp')
plt.plot(t, predictions, label='Predicții AR', linestyle='--')
plt.title('Seria de Timp și Predicțiile Modelului AR')
plt.xlabel('Timp')
plt.ylabel('Valoare')
plt.legend()
plt.show()
