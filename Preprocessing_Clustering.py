#import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
import seaborn as sb

#import csv file 
trainDf = pd.read_csv('kendaraan_train.csv')

# #Eksplorasi Data 
# #menampilkan sample 5 baris data dari keseluruhan data
# print(trainDf.sample(5))
# print("")
# #melihat type data tiap kolom
# print(trainDf.dtypes)
# print("")
# #melihat jumlah data keseluruhan
# print("Total Data Keseluruhan = ",len(trainDf))
# print("")
# #melihat jumlah data duplikasi
# trainDf.drop(['id'], axis=1, inplace=True)
# trainDf.drop(['Tertarik'], axis=1, inplace=True)
# cleanDuplicates= trainDf.drop_duplicates()
# print("Total Data Duplikasi = ",len(trainDf)- len(cleanDuplicates))
# print("")
# #melihat jumlah data kosong
# print("Total Data Kosong = ",trainDf.isnull().sum().sum())
# print("")



print("dataset sebelum drop")
print(trainDf.sample(5))
print("")
#terlihat dari 5 sample bahwa kolom id tidak diperlukan, sehingga bisa di drop
trainDf.drop(['id'], axis=1, inplace=True)
#drop kolom tertarik karena tidak dipakai di clustering
trainDf.drop(['Tertarik'], axis=1, inplace=True)
#cek dataset apakah id dan Tertarik sudah terdrop atau belum
print("dataset setelah drop")
print(trainDf.sample(5))

#cek apakah ada data yang sama atau duplikasi
cleanDuplicates= trainDf.drop_duplicates()
print("Total data duplikasi : ", len(trainDf)- len(cleanDuplicates))
#drop jika ada
trainDf.drop_duplicates(inplace=True)
#cek kembali apakah data sudah aman dari duplikasi
cleanDuplicates= trainDf.drop_duplicates()
print("Total data duplikasi : ", len(trainDf)- len(cleanDuplicates))

#ubah data categorical dengan angka (categorical encoding)
#pertama cek data type tiap kolom, jika object maka itu adalah categorical
print("")
print(trainDf.dtypes)
#buat procedure label encoding untuk mengubah data categorical menjadi angka (alternatif ada algoritma One-Hot Encoding)
def labelEncode(dataFrame):
    dataFrame['Jenis_Kelamin'] = dataFrame['Jenis_Kelamin'].replace(['Wanita','Pria'], [0,1])
    dataFrame['Umur_Kendaraan'] = dataFrame['Umur_Kendaraan'].replace(['< 1 Tahun', '1-2 Tahun', '> 2 Tahun'],[0,1,2])
    dataFrame['Kendaraan_Rusak'] = dataFrame['Kendaraan_Rusak'].replace(['Tidak', 'Pernah'], [0,1])
#implementasikan pada dataset train
print("")
labelEncode(trainDf)
#cek dataset apakah data categorical sudah berganti atau belum
print(trainDf.sample(5))

#Mengatasi data kosong / missing value
#pertama cek apakah ada data kosong atau tidak
print("Jumlah Data kosong =",trainDf.isnull().sum().sum())
#jika ada data kosong maka kita isi dengan mean/median (ditentukan oleh nilai skewness)
#untuk menentukan kita mengisinya dengan median atau mean perlu dilihat terlebih dahulu bentuk distplot nya
#jika penyebarannya merata maka bisa menggunakan mean, sendangkan jika tidak merata maka bisa memakai median
plt.subplot(2,5,1)
sb.distplot(trainDf['SIM'])
plt.title('SIM')

plt.subplot(2,5,2)
sb.distplot(trainDf[['Premi']])
plt.title('Premi')

plt.subplot(2,5,3)
sb.distplot(trainDf[['Lama_Berlangganan']])
plt.title('Lama_Berlangganan')

plt.subplot(2,5,4)
sb.distplot(trainDf[['Umur']])
plt.title('Umur')

plt.subplot(2,5,5)
sb.distplot(trainDf[['Kode_Daerah']])
plt.title('Kode_Daerah')

plt.subplot(2,5,6)
sb.distplot(trainDf[['Kanal_Penjualan']])
plt.title('Kanal_Penjualan')

plt.subplot(2,5,7)
sb.distplot(trainDf[['Jenis_Kelamin']])
plt.title('Jenis_Kelamin')

plt.subplot(2,5,8)
sb.distplot(trainDf[['Kendaraan_Rusak']])
plt.title('Kendaraan_Rusak')

plt.subplot(2,5,9)
sb.distplot(trainDf[['Sudah_Asuransi']])
plt.title('Sudah_Asuransi')

plt.subplot(2,5,10)
sb.distplot(trainDf[['Umur_Kendaraan']])
plt.title('Umur_Kendaraan')
# plt.show()
#data kosong pada sim diisi dengan median karena penyebaran datanya condong ke kanan atau left-skewed
trainDf['SIM'] = trainDf['SIM'].fillna(trainDf['SIM'].median())
trainDf['Premi'] = trainDf['Premi'].fillna(trainDf['Premi'].median())
#sisa data nya diisi mean karena berada di range -2 s/d 2 (untuk skewness kolom tertarik diabaikan karena tidak terdapat data kosong)
trainDf = trainDf.fillna(trainDf.mean())
#cek kembali apakah data kosong masih ada atau tidak
print("Jumlah Data kosong =",trainDf.isnull().sum().sum())

#mereduksi outlier
#cek apakah ada outlier pada dataset, dengan memvisualisasikan mengunakan boxplot
plt.figure()
plt.subplot(2,3,1)
plt.boxplot(trainDf['Premi'])
plt.title('Premi')

plt.subplot(2,3,2)
plt.boxplot(trainDf['Kanal_Penjualan'])
plt.title('Kanal_Penjualan')

plt.subplot(2,3,3)
plt.boxplot(trainDf['Lama_Berlangganan'])
plt.title('Lama_Berlangganan')

plt.subplot(2,3,4)
plt.boxplot(trainDf['Umur'])
plt.title('Umur')

plt.subplot(2,3,5)
plt.boxplot(trainDf['Kode_Daerah'])
plt.title('Kode_Daerah')

plt.show()
#jika terdapat outlier maka harus direduksi
#di sini akan direduksi menggunakan metode interquartile
#yang direduksi adalah dataset premi karena memiliki outlier yang berlebihan
Quartile1p = trainDf['Premi'].quantile(0.25)
Quartile3p = trainDf['Premi'].quantile(0.75)
interquartile = Quartile3p - Quartile1p

BatasB = Quartile1p - (1.5 * interquartile)
BatasA = Quartile3p + (1.5 * interquartile)

trainDf = trainDf[~((trainDf['Premi'] < BatasB) | (trainDf['Premi'] > BatasA))]

plt.figure()
plt.figure()
plt.subplot(2,3,1)
plt.boxplot(trainDf['Premi'])
plt.title('Premi')

plt.subplot(2,3,2)
plt.boxplot(trainDf['Kanal_Penjualan'])
plt.title('Kanal_Penjualan')

plt.subplot(2,3,3)
plt.boxplot(trainDf['Lama_Berlangganan'])
plt.title('Lama_Berlangganan')

plt.subplot(2,3,4)
plt.boxplot(trainDf['Umur'])
plt.title('Umur')

plt.subplot(2,3,5)
plt.boxplot(trainDf['Kode_Daerah'])
plt.title('Kode_Daerah')

plt.show()

#Normalisasi Data menggunakan Standard Scaler
#Agar pemrosesan data lebih cepat 
print(trainDf.head())
scale = StandardScaler()
trainDf[['Premi']] = scale.fit_transform(trainDf[['Premi']].values)
trainDf[['Lama_Berlangganan']] = scale.fit_transform(trainDf[['Lama_Berlangganan']].values)
trainDf[['Umur']] = scale.fit_transform(trainDf[['Umur']].values)
trainDf[['Kode_Daerah']] = scale.fit_transform(trainDf[['Kode_Daerah']].values)
trainDf[['Kanal_Penjualan']] = scale.fit_transform(trainDf[['Kanal_Penjualan']].values)
trainDf[['Jenis_Kelamin']] = scale.fit_transform(trainDf[['Jenis_Kelamin']].values)
trainDf[['SIM']] = scale.fit_transform(trainDf[['SIM']].values)
trainDf[['Kendaraan_Rusak']] = scale.fit_transform(trainDf[['Kendaraan_Rusak']].values)
trainDf[['Sudah_Asuransi']] = scale.fit_transform(trainDf[['Sudah_Asuransi']].values)
trainDf[['Umur_Kendaraan']] = scale.fit_transform(trainDf[['Umur_Kendaraan']].values)
print(trainDf.head())

trainDf.to_csv('kendaraan_train_cleanforclustering.csv', index=False)
