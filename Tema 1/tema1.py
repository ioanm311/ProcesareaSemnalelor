import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from scipy.fft import dctn, idctn

# Exercitiul 1

# imaginea originala
X = misc.ascent()

# matricea JPEG de cuantizare
Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]]

# aplicam DCT pe blocuri de 8x8 px
def transformata_dct(imagine):
    inaltine, latime = imagine.shape
    blocuri = np.zeros_like(imagine)
    for i in range(0, inaltine, 8):
        for j in range(0, latime, 8):
            bloc = imagine[i:i+8, j:j+8]
            blocuri[i:i+8, j:j+8] = dctn(bloc, norm='ortho')
    return blocuri

# cuantizam un singur bloc
def cuantizare_bloc(bloc, matrice):
    return np.round(bloc / matrice) * matrice

# iteram si cuantizam fiecare bloc
def cuantizare_blocuri(blocuri, matrice):
    inaltime, latime = blocuri.shape
    blocuri_cuantizate = np.zeros_like(blocuri)
    for i in range(0, inaltime, 8):
        for j in range(0, latime, 8):
            blocuri_cuantizate[i:i+8, j:j+8] = cuantizare_bloc(blocuri[i:i+8, j:j+8], matrice)
    return blocuri_cuantizate

transformata_dct_blocuri = transformata_dct(X)
transformata_dct_cuantizare = cuantizare_blocuri(transformata_dct_blocuri, Q_jpeg)

# aplicam IDCT pe blocuri de 8x8 px
def transformata_idct_blocuri(blocuri):
    inaltime, latime = blocuri.shape
    blocuri_ = np.zeros_like(blocuri)
    for i in range(0, inaltime, 8):
        for j in range(0, latime, 8):
            bloc = blocuri[i:i+8, j:j+8]
            blocuri_[i:i+8, j:j+8] = idctn(bloc, norm='ortho')
    return blocuri_

# comprimarea imaginii
imagine_comprimata = transformata_idct_blocuri(transformata_dct_cuantizare)

# am scos axele
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Imaginea originala')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(imagine_comprimata, cmap=plt.cm.gray)
plt.title('Imaginea comprimata')
plt.axis('off')

plt.show()

# Exercitiul 2

# imaginea color
imagine_color = misc.face()

# conversie din RGB in Y'CbCr
# aici am facut un pic de researching pe Wikipedia (https://en.wikipedia.org/wiki/YCbCr) unde am gasit
# exact matricea de conversie cu un standard utilizat pentru HDTV
def conversie_ycbcr(imagine_rgb):
    matrice_conversie = np.array([[0.2126, 0.7152, 0.0722],
                                  [-0.1146, -0.3854, 0.5],
                                  [0.5, -0.4542, -0.0458]])
    # inmultire imagine cu RGB cu transpusa matricei de conversie
    ycbcr = np.dot(imagine_rgb, matrice_conversie.T)
    # modificam cromatica
    ycbcr[:, :, [1, 2]] += 128
    return ycbcr

# imaginea convertita in Y'CbCr
imagine_ycbcr = conversie_ycbcr(imagine_color)

# procesam Y, Cb și Cr separat
lista_componente = []
for componenta in range(3):
    blocuri_dct = transformata_dct(imagine_ycbcr[:, :, componenta])
    blocuri_cuantizate = cuantizare_blocuri(blocuri_dct, Q_jpeg)
    componenta_comprimata = transformata_idct_blocuri(blocuri_cuantizate)
    lista_componente.append(componenta_comprimata)

# combinam componentele inapoi intr-o singura imagine
imagine_comprimata_ycbcr = np.stack(lista_componente, axis=-1)

# conversie din Y'CbCr in RGB
# matricea de conversie este luata din researching ca mai sus
def conversie_rgb(imagine_ycbcr):
    matrice_conversie = np.array([[1, 0, 1.5784],
                                  [1, -0.1873, -0.4681],
                                  [1, 1.8556, 0]])
    rgb = imagine_ycbcr.copy()
    # restabilim cromatica
    rgb[:, :, [1, 2]] -= 128
    rgb = np.dot(rgb, matrice_conversie.T)
    # inteval pentru RGB
    np.clip(rgb, 0, 255, out=rgb)
    rgb = rgb.astype(np.uint8)
    return rgb

# imagine convertita inapoi in RGB
imagine_comprimata_rgb = conversie_rgb(imagine_comprimata_ycbcr)

# am scos axele
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(imagine_color)
plt.title('Imaginea originala')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(imagine_comprimata_rgb)
plt.title('Imaginea comprimata')
plt.axis('off')

plt.show()

# Exercitiul 3

# calculam MSE intre 2 imagini
def calculare_mse(im1, im2):
    mse = np.mean((im1 - im2) ** 2)
    return mse

prag_mse = 0.19

# am facut conversie pentru ca aveam niste erori
Q_jpeg_array = np.array(Q_jpeg, dtype='float64')

# cautare binara
minim = 0
maxim = 10
start = 5

while maxim - minim > 0.1:
    start = (minim + maxim) / 2
    # facem o matrice auxiliara prin inmultirea matricei cu start adica cuantizarea
    Q_jpeg_aux = Q_jpeg_array * start

    # facem DCT si cuatizarea
    blocuri_cuantizate = cuantizare_blocuri(transformata_dct(X), Q_jpeg_aux)
    # facem inversul lui DCT
    imagine_comprimata = transformata_idct_blocuri(blocuri_cuantizate)

    mse_actual = calculare_mse(X, imagine_comprimata)

    # resentam param pentru cautarea binara
    if mse_actual > prag_mse:
        minim = start
    elif (mse_actual < prag_mse):
        maxim = start

# am scos axele
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(X, cmap='gray')
plt.title('Imaginea originala')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(imagine_comprimata, cmap='gray')
plt.title('Imaginea comprimata')
plt.axis('off')

plt.show()



