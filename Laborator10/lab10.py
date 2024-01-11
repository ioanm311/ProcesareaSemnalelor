import numpy as np
import matplotlib.pyplot as plt

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



































