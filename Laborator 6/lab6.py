import numpy as np
import matplotlib.pyplot as plt

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