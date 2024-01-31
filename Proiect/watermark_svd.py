import numpy as np
from PIL import Image

def calculeaza_entropia(bloc):
    histograma, _ = np.histogram(bloc.flatten(), bins=256, range=(0, 255))
    histogramaa_normalizata = histograma / histograma.sum()
    entropia = -np.sum(histogramaa_normalizata[histogramaa_normalizata > 0] *
                       np.log2(histogramaa_normalizata[histogramaa_normalizata > 0]))
    return entropia

def gaseste_bloc_cu_entropie_maxima(matrice, bloc_marime):
    y_marime, x_marime, _ = matrice.shape
    if y_marime % bloc_marime != 0 or x_marime % bloc_marime != 0:
        raise ValueError("Marimea blocului trebuie sa fie un divizor al dimensiunilor imaginii")

    entropie_maxima = 0
    coord_entropie_maxima = (0, 0)

    for i in range(0, y_marime, bloc_marime):
        for j in range(0, x_marime, bloc_marime):
            bloc = matrice[i:i + bloc_marime, j:j + bloc_marime]

            entropie = calculeaza_entropia(bloc)

            if entropie > entropie_maxima:
                entropie_maxima = entropie
                coord_entropie_maxima = (i // bloc_marime, j // bloc_marime)

    return coord_entropie_maxima

def incarca_imagine(cale_imagine):
    imagine = Image.open(cale_imagine)
    return np.array(imagine)

def descompunere_matrice_SVD(matrice):
    U, sigma, V = np.linalg.svd(matrice, full_matrices=False)
    return U, sigma, V

def inserare_watermark(sigma, watermark, alpha=280):
    for i in range(len(sigma)):
        sigma[i] += watermark * alpha
    return sigma

def imagine_cu_watermark(U, sigma_modificat, V):
    # np.diag - transforma vectorul sigma_modificat intr-o matrice diagonala
    matrice_diagonala = np.diag(sigma_modificat)

    # np.dot - inmulteste doua matrici
    matrice_imagine_rezultata = np.dot(U, np.dot(matrice_diagonala,V))

    # np.clip - ne asiguram ca valorile pixelilor sunt intre 0 si 255
    return np.clip(matrice_imagine_rezultata, 0, 255)

def ajusteaza_dimensiunea_imaginii_la_bloc(matrice, bloc_marime):
    inaltime, latime = matrice.shape[:2]
    # calculeaza inaltimea si latimea cea mai apropiata care sa se potriveasca cu marimea blocului
    inaltime_noua = (inaltime // bloc_marime) * bloc_marime
    latime_noua = (latime // bloc_marime) * bloc_marime
    # marim putin imaginea daca este cazul ca sa poata fi impartita in blocurile cu marimea selectata de noi
    return matrice[:inaltime_noua, :latime_noua, :]

# in aceasta funcite introducem watermark-ul intr-o imagine selectata de noi
def introducere_watermark_folosind_SVD_entropie(path, watermark, bloc_marime=30):
    matrice = incarca_imagine(path)
    matrice = ajusteaza_dimensiunea_imaginii_la_bloc(matrice, bloc_marime)
    coord = gaseste_bloc_cu_entropie_maxima(matrice, bloc_marime)
    # coordonatele pentru blocul cu entropia maxima, unde o sa punem watermark-ul
    y, x = coord
    bloc = matrice[y * bloc_marime:(y + 1) * bloc_marime, x * bloc_marime:(x + 1) * bloc_marime]

    # initializam blocul pe care o sa-l modificam pentru a introduce watermark-ul
    bloc_modificat = np.zeros_like(bloc)

    # aplicam svd-ul pe fiecare canal al blocului
    for i in range(3):  # procesam blocurile corespunzatoare culorilor RGB
        canal = bloc[:,:,i]
        U, sigma, V = descompunere_matrice_SVD(canal)
        sigma_modificat = inserare_watermark(sigma, watermark)
        canal_modificat = imagine_cu_watermark(U, sigma_modificat, V)
        # inseram canalul cu watermark in bloc
        bloc_modificat[:,:,i] = canal_modificat.astype('uint8') # convertim la un intreg fara semn de 8 biti

    # inlocuim blocul original cu entropie maxima, cu blocul nostru modificat
    matrice_modificata = matrice.copy()
    matrice_modificata[y * bloc_marime:(y + 1) * bloc_marime, x * bloc_marime:(x + 1) * bloc_marime] = bloc_modificat

    # convertim matricea modificata intr-o noua imagine si o salvam
    img_salvata = Image.fromarray(matrice_modificata, 'RGB')
    img_salvata.show()
    img_salvata.save("imagine_cu_watermark.png")

if __name__ == '__main__':
    introducere_watermark_folosind_SVD_entropie(r"C:\Users\Ioan\PycharmProjects\ProcesareaSemnalelor\Proiect\watermark.jpg", watermark=1)

