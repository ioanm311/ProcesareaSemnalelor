import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Exercitiul 1

# seed-ul pentru reproducibilitate
np.random.seed(42)

# Dimensiune
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

# serie de timp ca suma a celor trei componente
time_series = trend + seasonality + noise

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

# Exercitiul 2

ar_params = np.array([0.8])  # Coeficientul AR
ma_params = None             # Coeficientul MA
ar_order = len(ar_params)
ma_order = 0 if ma_params is None else len(ma_params)
n_obs = 100

ar_model = sm.tsa.ArmaProcess(ar_params, ma_params)
serie_timp_ar = ar_model.generate_sample(nsample=n_obs)

# calculul vectorului de autocorelație
autocorrelation = sm.tsa.acf(serie_timp_ar, nlags=n_obs-1)

# desenarea vectorului de autocorelație
plt.stem(autocorrelation)
plt.title('Vector de Autocorelație pentru Modelul AR')
plt.xlabel('Lag')
plt.ylabel('Autocorelație')
plt.show()

# Exercitiul 3

ar_params = np.array([0.6, -0.4])  # Coeficienții AR
ma_params = None                   # Coeficientul MA
ar_order = len(ar_params)
ma_order = 0 if ma_params is None else len(ma_params)
n_obs = 100

ar_model = sm.tsa.ArmaProcess(ar_params, ma_params)
serie_timp_ar = ar_model.generate_sample(nsample=n_obs)

# modelului AR
model_ar = sm.tsa.AutoReg(serie_timp_ar, lags=ar_order)
result_ar = model_ar.fit()

# predicții
predictions = result_ar.predict(start=ar_order, end=n_obs-1)

plt.plot(serie_timp_ar, label='Serie de timp originală')
plt.plot(predictions, label='Predicții', linestyle='dashed')
plt.title('Serie de timp originală și predicții AR')
plt.legend()
plt.show()