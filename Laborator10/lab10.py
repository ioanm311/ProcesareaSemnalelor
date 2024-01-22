import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

# Exercitiul 1

medie_1d = 0  # medie distributie unidimensionala
varianta_1d = 1  # varianta distributie unidimensionala
medie_2d = [0, 0]  # medie distributie bidimensionala
matrice_covarianta = [[1, 0.5], [0.5, 1]]

numar_esantioane_1d = 1000
esantioane_1d = np.random.normal(medie_1d, np.sqrt(varianta_1d), numar_esantioane_1d)

numar_esantioane_2d = 1000
esantioane_2d = np.random.multivariate_normal(medie_2d, matrice_covarianta, numar_esantioane_2d)

plt.figure(figsize=(15, 6))

# distributia unidimensionala
plt.subplot(1, 2, 1)
plt.hist(esantioane_1d, bins=30, density=True)
plt.title('Histograma 1D')
plt.xlabel('Valoare esantion')
plt.ylabel('Densitate probabilitate')

# distributia bidimensionala
plt.subplot(1, 2, 2)
plt.scatter(esantioane_2d[:, 0], esantioane_2d[:, 1], alpha=0.5)
plt.title('2D Gaussian cu matricea de covarianta')
plt.xlabel('Valoare X')
plt.ylabel('Valoare Y')
plt.grid(True)
plt.axis('equal')
# scalare egala pe ambele axe

plt.tight_layout()
plt.show()

# Exercitiul 2
# a)

# ignoram comm-urile
date_co2 = pd.read_csv('co2_daily_mlo.csv', comment='#', header=None, sep=',')
date_co2.columns = ['An', 'Luna', 'Zi', 'Data_Decimala', 'CO2']

# convertim coloanele 'An', 'Luna' si 'Zi' intr-o singura coloana de tip data
date_co2['Data'] = pd.to_datetime(date_co2['An'].astype(str) + '-' +
                                   date_co2['Luna'].astype(str).str.zfill(2) + '-' +
                                   date_co2['Zi'].astype(str).str.zfill(2))

# grupam pe luni
date_co2['LunaAn'] = date_co2['Data'].dt.to_period('M')

media_lunara_co2 = date_co2.groupby(date_co2['Data'].dt.to_period('M'))['CO2'].mean().reset_index()
# convertim 'Data' inapoi in format datetime pentru a putea plota
media_lunara_co2['Data'] = media_lunara_co2['Data'].dt.to_timestamp()
media_lunara_co2.columns = ['Data', 'Media_CO2']

plt.figure(figsize=(15, 5))
plt.plot(media_lunara_co2['Data'], media_lunara_co2['Media_CO2'], marker='o', linestyle='-')
plt.title('Media Lunara a Nivelului de CO2')
plt.xlabel('Data')
plt.ylabel('CO2')
plt.grid(True)
plt.show()

# b)

# o var in care stocam fiecare luna
media_lunara_co2['Timp'] = np.arange(len(media_lunara_co2))
X = sm.add_constant(media_lunara_co2['Timp'])
y = media_lunara_co2['Media_CO2']
model = sm.OLS(y, X).fit()

# predictii
media_lunara_co2['Trend'] = model.predict(X)

# calculam seria scazand trendul calculat din valorile observate
media_lunara_co2['Detrended_CO2'] = media_lunara_co2['Media_CO2'] - media_lunara_co2['Trend']

plt.figure(figsize=(15, 7))
plt.plot(media_lunara_co2['Data'], media_lunara_co2['Media_CO2'], label='Seria originala', color='blue')
plt.plot(media_lunara_co2['Data'], media_lunara_co2['Trend'], label='Trend', color='red', linestyle='--')
plt.plot(media_lunara_co2['Data'], media_lunara_co2['Detrended_CO2'], label='Seria fara trend', color='green')
plt.title('Trend și Seria fara trend a Nivelului de CO2')
plt.xlabel('Data')
plt.ylabel('CO2')
plt.legend()
plt.grid(True)
plt.show()























