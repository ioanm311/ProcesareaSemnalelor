import time
from watermark_svd import introducere_watermark_folosind_SVD_entropie
from watermark_dwt import aplica_dwt_watermark
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# masurare timp executie - SVD
start_time = time.time()
introducere_watermark_folosind_SVD_entropie(r"C:\Users\Ioan\PycharmProjects\ProcesareaSemnalelor\Proiect\watermark.jpg", watermark=1)
svd_time = time.time() - start_time

# masurare timp executie - DWT
start_time = time.time()
aplica_dwt_watermark(r"C:\Users\Ioan\PycharmProjects\ProcesareaSemnalelor\Proiect\watermark.jpg", r'C:\Users\Ioan\PycharmProjects\ProcesareaSemnalelor\Proiect\floare.jpg', dim_bloc=20)
dwt_time = time.time() - start_time

metode = ['SVD', 'DWT']
timpi_executie = [svd_time, dwt_time]
culori = ['#1f77b4', '#ff7f0e']

fig, ax = plt.subplots()
bars = plt.bar(metode, timpi_executie, color=culori)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 4), ha='center', va='bottom')

plt.ylabel('Timp de executie (secunde)')
plt.title('Compararea timpului de executie intre SVD si DWT')
plt.tight_layout()
plt.show()
