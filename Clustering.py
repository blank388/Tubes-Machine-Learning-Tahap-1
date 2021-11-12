import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import random as rd
import math
#import dataset
trainDf = pd.read_csv('kendaraan_train_cleanforclustering.csv')

heatmap = sb.heatmap(trainDf.corr(), linewidths= 0.5, cmap='coolwarm',annot=True)
plt.title("Korelasi_Kendaraan_Train")
plt.show()

#ambil data untuk nilai x dan y yang memiliki korelasi tertinggi dan merupakan data numerik
#disini diambil data Kanal_Penjualan dan Umur

Dfcluster = trainDf[['Umur','Kanal_Penjualan']]
#Eksperimen 1
# Dfcluster = trainDf[['Sudah_Asuransi','Kendaraan_Rusak']]
#Eksperimen 2
# Dfcluster = trainDf[['Umur','Umur_Kendaraan']]
DfclusterCopy = Dfcluster.copy() #dipakai untuk evaluasi 

#fungsi euclidean distance
def jarak(x,y):
    return math.sqrt(sum((x - y)**2))

#fungsi Kmeans
def Kmeans (k, data):
    #membuat centroid awal dengan cara mengambil acak dari dataset
    centroid = {}
    for i in range(k):
        randCent = data[rd.randint(0, len(data))]
        centroid[i] = randCent
    print('Centroid awal(random Centroid) :')
    for j in range(k):
        print(f'Cluster {j}: ', centroid.get(j))
    

    centroidCopy = centroid.copy()#digunakan untuk menghentikan iterasi jika nilai centroid sekarang sama dengan centroid lama

    sama = False
    idx = 1
    while sama == False :
        #menginisialisasi cluster kosong sebanyak k
        cluster = {}
        for i in range(k):
            cluster[i] = []
        #mengisi cluster dengan data yang memiliki euclidean distance terkecil terhadap centroid yang ada
        for j in data :
            distance = []
            for m in centroid:
                distance.append(jarak(j, centroid[m]))
            cluster[distance.index(min(distance))].append(j)
        for l in cluster:
            centroid[l] = np.mean(cluster[l],axis=0)

        print('Centroid ke ',idx)
        for j in range(k):
            print(f'Cluster {j}: ', centroid.get(j))
        idx += 1
        #cek apakah cluster sekarang sama dengan cluster sebelumnya, jika sama maka iterasi dihentikan
        if ((centroidCopy.get(0) == centroid.get(0))[0] and
            (centroidCopy.get(1) == centroid.get(1))[0] and
            (centroidCopy.get(2) == centroid.get(2))[0] ):
            sama = True
        else:
            centroidCopy = centroid.copy()
    return centroid, cluster

#Main program untuk Kmeans
#
data = Dfcluster.to_numpy()
k = 3
centroid, cluster = Kmeans(k,data)

#Visualisasi Hasil Kmeans
Visualisasi = []

for key in cluster.keys():
    for i in cluster.get(key):
        Visualisasi.append((i[0], i[1], key))

Visualisasi = pd.DataFrame(Visualisasi)
Visualisasi.columns = ['Umur', 'Kanal_Penjualan', 'Cluster']
# print(Dfcluster.sample(10))
plt.title('Hasil Clustering')
warna = ['#BA55D3', '#00FA9A', '#DA70D6']
for i in range(k) :
    plt.scatter(Visualisasi['Umur'][Visualisasi['Cluster']==i],Visualisasi['Kanal_Penjualan'][Visualisasi['Cluster']==i],color= warna[i], label=f'Cluster {i+1}')
    plt.scatter(centroid[i][0],centroid[i][1],color= '#000000', marker='p')

# plt.scatter(centroid[0][0], centroid[0][1], color='#000000', marker='p',label='Centroid')
plt.xlabel('Umur')
plt.ylabel('Kanal_Penjualan')
plt.legend(loc='best')
plt.show()

#Evaluasi Model
from sklearn.cluster import KMeans as km
#Elbow Method
elbow = []

k = range(2, 10)

for i in k :
    modelEval = km(n_clusters=i)
    modelEval.fit(DfclusterCopy)
    elbow.append(modelEval.inertia_)

plt.plot(k, elbow,'bx-')
plt.xlabel('value K')
plt.ylabel('inertia')
plt.title('Elbow Method')
plt.show()

#Silhouette Score
from sklearn.metrics import silhouette_score as ss

for i in [3, 4]:
    clusterEval = km(n_clusters= i)
    clusterEval.fit_predict(DfclusterCopy)
    nilai_siluet = ss(DfclusterCopy, clusterEval.labels_, metric='euclidean')
    print(i,' Cluster, Silhouette Score = ', nilai_siluet,'\n')