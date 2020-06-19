# -*- coding: utf-8 -*-

#Bu çalışmada el ile yazılmış sayı resimlerinin piksel verilerini kullanarak boyut indirgeme yapacağız
#Bir resmi ifade ederken piksel değerlerini kullanırız. Mesela 24x24 bir resmin 576 pikseli vardır.
#Bu piksellerin her biri eğer resim 3 kanallı ise (R,G,B) değerlerinden oluşur. Red, Green ve Blue olmak üzere...
#Eğer resim grayscale ise yani siyah beyaz ise piksel değerlerini 0,255 arasında bir değer olarak belirleyebiliriz.


#kütüphanelerin yüklenmesi
import numpy as np
import pandas as pd

#görselleştirmek için  
import seaborn as sns
import matplotlib.pyplot as plt

#kümelemede sayısal veriler ile işlem yapacağımızdan scaling gerekiyor
from sklearn.preprocessing import scale

import os
import warnings
warnings.filterwarnings('ignore')

#Veriyi Anlama ve Temizleme
#%%

#verinin okunup data frame e dönüştürülemesi
d0= pd.read_csv('train.csv')

#ilk değerlere bakalım
print(d0.head())

#verinin sınıflarını l değişkenine atayalım
l= d0['label']

#sınıfları yani etiketleri veri setinden atıp d değişkenine atayalım
d=d0.drop('label', axis=1)

#verinin boyutlarına bakarsak 784 kolon ve 42000 satırdan oluşan resimlerin 
#piksel değerlerini içeren yüksek boyutlu bir veri setidir.
print(d.shape)
print(l.shape)

#ilk satırdaki verinin rsim formatında gösterilmesi
plt.figure(figsize=(7, 7))
idx= 1

#1d'li veriyi 2d li resim olarak göstermesi
grid_data= d.iloc[idx].to_numpy().reshape(28, 28)
plt.imshow(grid_data, interpolation= 'none',cmap= 'gray')
plt.show()

print(l[idx])


#PCA kullanarak 2D görselleştirme
#%%

#sadece ilk 15k satır değerini performans amaçlı kullacağız
labels= l.head(15000)
data= d.head(15000)

print('the shape of sample data =' ,data.shape)

#Data-preprocessing: Numeric verilerle çalışacağımız için verinin standardize edilmesi gerekir
from sklearn.preprocessing import StandardScaler

standardized_data= StandardScaler().fit_transform(data)
print(standardized_data.shape)

#covariance matrisinin bulunması: A^T * A
sample_data= standardized_data

covar_matrix= np.matmul(sample_data.T, sample_data)
print('The shape of variance matrix =', covar_matrix.shape)


#En iyi iki eigen-value ve karşılık gelen eigen-vector lerini bulmak
#Buradaki amaç çok boyutlu verinin iki boyutlu olarak görselleştirilmesidir
from scipy.linalg import eigh

#'eigvals' düşük değerden yüksek seviyeye doğru tanımlıdır
#eigh fonksiyonu eigen valueları artan sırayla gösterir
#aşağıdaki kod bloğu verilen boyutların 2 eigen value ve vector değerlerini döndürür
values, vectors= eigh(covar_matrix, eigvals= (782,783))
print('Shape of eigen vectors =', vectors.shape)

#eigen value ve vectorlerin transpose unun alınması
vectors= vectors.T
print('Update shape of eigen vectors =', vectors.shape)

#vectors[1], 1. principle componentin eigen ve vectorünü simgeler
#vectors[0], 2. principle componentin eigen ve vectorünü simgeler


#orjinal verinin 2 boyutlu projeksiyonunu oluşturmak
#eigen vector ile veri setini de matris olarak düşünüp çarparsak 2x15000 boyutlarına
#sahip yeni indirgenmiş veri setine ulaşırız
import  matplotlib.pyplot as plt

new_coordinates= np.matmul(vectors, sample_data.T)
print('resultanat new data points shape', vectors.shape, 'X', sample_data.T.shape, '=', new_coordinates.shape)

import pandas as pd

#2 boyutlu veriye sınıflarının atanması
new_coordinates= np.vstack((new_coordinates, labels)).T

#indirgenmiş verinin sınıflarını, veri setine eklemek
dataframe= pd.DataFrame(data=new_coordinates, columns=('1st_principal', '2nd_principal', 'label'))
print(dataframe.head())

#2 boyutlu verinin seaborn ile görselleştirilmesş
import seaborn as sns

sns.FacetGrid(dataframe, hue= 'label', size= 6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()


#Scikit-Learn ile PCA
#%%

#PCA' in çağrılması
from sklearn import decomposition

pca= decomposition.PCA()

#parametrelerin ayarlanması componenti 2 olarak seçmiştik
pca.n_components = 2
pca_data= pca.fit_transform(sample_data)

#pca_reduced will contain the 2d projects of simple data
print('shape of pca_reduced.shape =', pca_data.shape)

#2 boyutlu veriye sınıflarının atanması
pca_data= np.vstack((pca_data.T, labels)).T

#2 boyutlu verinin görselleştirilmesi
pca_df= pd.DataFrame(data=pca_data, columns=('1st_principal', '2nd_principal', 'label'))
sns.FacetGrid(pca_df, hue= 'label', size= 6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()


#Boyut İndirgeme için PCA
#%%

#Bu bölümde kaç adet component kullanırsal orijinal veriden en az bilgi kaybederek boyut indirgeme
#yapabiliriz sorusuna cevap arayacağız
pca.n_components= 784
pca_data= pca.fit_transform(sample_data)

percentage_var_explained= pca.explained_variance_ / np.sum(pca.explained_variance_)

cum_var_explained= np.cumsum(percentage_var_explained)

#plot the PCA spectrum
plt.figure(1, figsize= (6, 4))

plt.clf()
plt.plot(cum_var_explained, linewidth= 2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()

#Farklı component değerlerine göre varyans değerini kontrol edersek. yaklaşık 250 veya 300 component ile
#veriyi en varyanslı ve en az bilgi kaybı içerecek şekilde ifade ederek sınıflama yapabiliriz


























