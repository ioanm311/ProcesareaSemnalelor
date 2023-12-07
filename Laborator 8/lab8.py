import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Setăm seed-ul pentru reproducibilitate
np.random.seed(42)

# Dimensiunea seriei de timp
N = 1000

# Generăm timpul
t = np.arange(N)

# Componenta de trend de gradul 2
trend = 0.02 * t**2

# Componenta de sezon cu două frecvențe
seas1 = 5 * np.sin(2 * np.pi * t / 50)
seas2 = 3 * np.cos(2 * np.pi * t / 30)
seasonality = seas1 + seas2

# Componenta de variabilitate mică (zgomot alb gaussian)
noise = np.random.normal(0, 1, N)

# Serie de timp ca suma a celor trei componente
time_series = trend + seasonality + noise

# Desenăm seria de timp și componentele sale
plt.figure(figsize=(12, 6))
plt.plot(t, time_series, label='Seria de Timp')
plt.plot(t, trend, label='Trend', linestyle='--')
plt.plot(t, seasonality, label='Sezon')
plt.plot(t, noise, label='Zgomot Alb', linestyle='--')

plt.title('Seria de Timp și Componentele Sale')
plt.xlabel('Timp')
plt.ylabel('Valoare')
plt.legend()
plt.show()

# Fităm un model ARIMA(p, d, q) la seria de timp
order = (2, 2, 0)  # Parametrii modelului ARIMA
arima_model = sm.tsa.ARIMA(time_series, order=order)
result = arima_model.fit()

# Sumarul modelului
print(result.summary())

# Diagnosticul reziduurilor
residuals = result.resid
plt.figure(figsize=(12, 6))
plt.plot(t, residuals)
plt.title('Reziduurile Modelului ARIMA')
plt.xlabel('Timp')
plt.ylabel('Reziduuri')
plt.show()
