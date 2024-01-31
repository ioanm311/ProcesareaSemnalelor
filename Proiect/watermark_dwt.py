import numpy as np
import pywt
from PIL import Image
from skimage.measure import shannon_entropy

def calculeaza_entropia_blocuri(LL, dim_bloc):
    entropii = []
    for y in range(0, LL.shape[0], dim_bloc):
        for x in range(0, LL.shape[1], dim_bloc):
            bloc = LL[y:y+dim_bloc, x:x+dim_bloc]
            if bloc.shape[0] == dim_bloc and bloc.shape[1] == dim_bloc:
                entropia = shannon_entropy(bloc)
                entropii.append((entropia, (y, x)))
    return entropii


def gaseste_bloc_cu_entropie_maxima(entropii):
    entropii.sort(reverse=True)
    return entropii[0][1]

def aplica_dwt_pe_canal(canal, watermark, dim_bloc, mod='haar', nivel=1):
    canal_coeficienti = pywt.wavedec2(canal, mod, level=nivel)
    LL, detalii = canal_coeficienti

    entropii_blocuri = calculeaza_entropia_blocuri(LL, dim_bloc)
    y, x = gaseste_bloc_cu_entropie_maxima(entropii_blocuri)

    watermark_ajustat = np.resize(watermark, (dim_bloc, dim_bloc))

    LL_marcat = np.copy(LL)
    LL_marcat[y:y+dim_bloc, x:x+dim_bloc] += watermark_ajustat

    coeficienti_canal_marcat = (LL_marcat, detalii)
    canal_marcat = pywt.waverec2(coeficienti_canal_marcat, mod)
    return canal_marcat

def aplica_dwt_watermark(imagine_path, watermark_path, dim_bloc, mod='haar', nivel=1):
    imagine = Image.open(imagine_path)
    watermark = Image.open(watermark_path).convert('L')
    imagine_matrice = np.array(imagine)
    watermark_matrice = np.array(watermark) / 255.0  # Normalizare

    canale_marcate = []
    for i in range(3):
        canal_marcat = aplica_dwt_pe_canal(imagine_matrice[:,:,i], watermark_matrice, dim_bloc, mod, nivel)
        canale_marcate.append(canal_marcat)

    matrice_marcata_imagine = np.stack(canale_marcate, axis=-1)
    matrice_marcata_imagine = np.clip(matrice_marcata_imagine, 0, 255)

    Image.fromarray(matrice_marcata_imagine.astype(np.uint8)).save('imagine_cu_watermark_dwt.png')

if __name__ == '__main__':
    aplica_dwt_watermark(r"C:\Users\Ioan\PycharmProjects\ProcesareaSemnalelor\Proiect\watermark.jpg", r'C:\Users\Ioan\PycharmProjects\ProcesareaSemnalelor\Proiect\floare.jpg', dim_bloc=75)
