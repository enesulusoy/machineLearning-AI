# -*- coding: utf-8 -*-

#Müşteri Segmentasyonu

#kütüphanelerin yüklenmesi
import numpy as np
import pandas as pd

#görselleştirmek için  
import seaborn as sns
import matplotlib.pyplot as plt

#kümelemede sayısal veriler ile işlem yapacağımızdan scaling gerekiyor
from sklearn.preprocessing import scale

#k-means in çağrılması
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

import os
import warnings
warnings.filterwarnings('ignore')

#Veriyi Anlama ve Temizleme
#%%

#verinin okunup data frame e dönüştürülemesi
retail= pd.read_csv('Online+Retail.csv', sep= ',', encoding= 'ISO-8859-1', header= 0)

#ilk 5 veriye bakalım
print(retail.head())

#data formatının oluşturulması
retail['InvoiceDate']= pd.to_datetime(retail['InvoiceDate'], format= '%d-%m-%Y %H:%M')

#veriyi kontrol edelim
print(retail.head())

#verini boyutları
print(retail.shape)

#verinin istatiksel özeti
print(retail.describe())

#veri tiplerine bakalım
print(retail.info())

#NaN değer kontrolü
print(retail.isnull().values.any())

#toplam NaN değer sayısı
print(retail.isnull().values.sum())

#boş değere sahip satırların tüm verideki oranı
print(retail.isnull().sum()*100/retail.shape[0])

#NaN değer içeren satırların atılması
order_wise= retail.dropna()

#Verinin NaN değerlerini tekrar kontrol edelim
print(order_wise.shape)
print(order_wise.isnull().sum())


#RFM implementation
#%%

# Öncelikle order_wise veri setinden quantity ve unit price'ı kullanarak toplam satışı simgeleyen amount bağımsız değişkenini
# oluşturalım.
amount  = pd.DataFrame(order_wise.Quantity * order_wise.UnitPrice, columns = ["Amount"])
print(amount.head())

#Monetary Value
#amount'u zaten order_wise veri setindeki iki kolondan türeterek oluşturmuştuk.
# aynı şekilde bunları birleştirirsek herhangi bir problem yaşamayız.
order_wise = pd.concat(objs = [order_wise, amount], axis = 1, ignore_index = False)

#Monetary Fonksiyonu
# Her bir müşteri için toplam satın alma miktarını group_by yardımıyla gösterelim.
monetary = order_wise.groupby("CustomerID").Amount.sum()
monetary = monetary.reset_index()
print(monetary.head())

#Frequency
#Customer id ve invoice no'lardan oluşan bir data frame oluşturmak
frequency = order_wise[['CustomerID', 'InvoiceNo']]

# Her bir müşterinin sipariş sayısını invoice no'ları saydırarak bulabiliriz.
k = frequency.groupby("CustomerID").InvoiceNo.count()
k = pd.DataFrame(k)
k = k.reset_index()
k.columns = ["CustomerID", "Frequency"]
print(k.head())

#Amount ve Frequency kolonlarını ana veri seti ile join'liyoruz
master = monetary.merge(k, on = "CustomerID", how = "inner")
print(master.head())

#Recency Value
recency  = order_wise[['CustomerID','InvoiceDate']]
maximum = max(recency.InvoiceDate)

#Recency fonksiyonunu oluşturma
# recency için alt bir data frame oluşturmak
recency  = order_wise[['CustomerID','InvoiceDate']]

# Verinin en büyük tarihini bulmak
maximum = max(recency.InvoiceDate)

# Maksimum tarihe 1 gün ekliyoruz ki son siparişten sonra geçen süre hiçbir zaman 0 olmasın.
maximum = maximum + pd.DateOffset(days=1)
recency['diff'] = maximum - recency.InvoiceDate
print(recency.head())
print(recency)

# tekrar eden satırları silmek için group by kullanıyoruz.
a= recency.groupby('CustomerID')
print(a['diff'].min())

# müşterinin ne kadar süredir bizi ziyaret etmediğini recency değeri ile öğreniyoruz.
df = pd.DataFrame(recency.groupby('CustomerID')['diff'].min())
df = df.reset_index()
df.columns = ["CustomerID", "Recency"]
print(df.head())


#RFM ile birleştirme
#%%

# recency, frequency ve monetary metriklerinin bir araya getirilmesi.
RFM = k.merge(monetary, on = "CustomerID")
RFM = RFM.merge(df, on = "CustomerID")
print(RFM.head())

#Outlier Treatment (Aykırı Verileri Temizlemek)

# K means outlier'lara duyarlı olduğu için RFM değerlerini belirli quantile değerleri içerisinde tutuyoruz.
# Aşağıdaki kod bloğu Amount için outlier'ları siler.
plt.boxplot(RFM.Amount)
Q1 = RFM.Amount.quantile(0.25)
Q3 = RFM.Amount.quantile(0.75)
IQR = Q3 - Q1
RFM = RFM[(RFM.Amount >= Q1 - 1.5*IQR) & (RFM.Amount <= Q3 + 1.5*IQR)]

# frequency için outlier'ları silmek.
plt.boxplot(RFM.Frequency)
Q1 = RFM.Frequency.quantile(0.25)
Q3 = RFM.Frequency.quantile(0.75)
IQR = Q3 - Q1
RFM = RFM[(RFM.Frequency >= Q1 - 1.5*IQR) & (RFM.Frequency <= Q3 + 1.5*IQR)]

# Recency için outlier'ların silinmesi.
plt.boxplot(RFM.Recency)
Q1 = RFM.Recency.quantile(0.25)
Q3 = RFM.Recency.quantile(0.75)
IQR = Q3 - Q1
RFM = RFM[(RFM.Recency >= Q1 - 1.5*IQR) & (RFM.Recency <= Q3 + 1.5*IQR)]

#verinin ilk 20 satırına bakarsak.
print(RFM.head(20))


#RFM verinin scaling edilmesi
#%%

# RFM değerlerinin standardize edilmesi.
RFM_norm1 = RFM.drop("CustomerID", axis=1)
RFM_norm1.Recency = RFM_norm1.Recency.dt.days

from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
RFM_norm1 = standard_scaler.fit_transform(RFM_norm1)

RFM_norm1 = pd.DataFrame(RFM_norm1)
RFM_norm1.columns = ['Frequency','Amount','Recency']
print(RFM_norm1.head())

#Hopkins İstatikleri
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
 
def hopkins(X): # X bir data frame'dir.
    d = X.shape[1]
  
    n = len(X) # rows
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H

#veri setimizin hopkins istatistiği sonucu kümeleme eğilimi hayli yüksektir.
print(hopkins(RFM_norm1))


#K_MEANS
#%%

# Kmeans with K=5
model_clus5 = KMeans(n_clusters = 5, max_iter=50)
model_clus5.fit(RFM_norm1)

#Silhouette Analizi
from sklearn.metrics import silhouette_score

sse_ = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k).fit(RFM_norm1)
    sse_.append([k, silhouette_score(RFM_norm1, kmeans.labels_)])

#plot
plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1])
plt.show()

#görselde Silhouette değeri en yüksek olan K değeri 2'dir.


#Mesafelerin Karelerini Toplamları (Sum of squared Distances)
#%%

# k'yi seçmek için bir diğer metrik Sum of squared distance değeridir.
# Farklı K değerleri için sum of squared distances değeri oluşturulur.
# SSD değerinin marjinal olarak kırıldığı nokta optimal k değerini verir.
# o da Silhouette'de olduğu gibi 2 civarını göstermektedir.
ssd = []
for num_clusters in list(range(1,21)):
    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)
    model_clus.fit(RFM_norm1)
    ssd.append(model_clus.inertia_)

plt.plot(ssd)
plt.show()

# k'yi 5 seçelim ve kümelerin RFM değerlerine bakalım.
RFM.index = pd.RangeIndex(len(RFM.index))
RFM_km = pd.concat([RFM, pd.Series(model_clus5.labels_)], axis=1)
RFM_km.columns = ['CustomerID', 'Frequency', 'Amount', 'Recency', 'ClusterID']

RFM_km.Recency = RFM_km.Recency.dt.days
km_clusters_amount = 	pd.DataFrame(RFM_km.groupby(["ClusterID"]).Amount.mean())
km_clusters_frequency = 	pd.DataFrame(RFM_km.groupby(["ClusterID"]).Frequency.mean())
km_clusters_recency = 	pd.DataFrame(RFM_km.groupby(["ClusterID"]).Recency.mean())

df = pd.concat([pd.Series([0,1,2,3,4]), km_clusters_amount, km_clusters_frequency, km_clusters_recency], axis=1)
df.columns = ["ClusterID", "Amount_mean", "Frequency_mean", "Recency_mean"]
print(df.head())

sns.barplot(x=df.ClusterID, y=df.Amount_mean)
plt.show()

sns.barplot(x=df.ClusterID, y=df.Frequency_mean)
plt.show()

sns.barplot(x=df.ClusterID, y=df.Recency_mean)
plt.show()


#Heirarchical Clustering
#%%

# heirarchical clustering
mergings = linkage(RFM_norm1, method = "single", metric='euclidean')
dendrogram(mergings)
plt.show()

mergings = linkage(RFM_norm1, method = "complete", metric='euclidean')
dendrogram(mergings)
plt.show()

clusterCut = pd.Series(cut_tree(mergings, n_clusters = 5).reshape(-1,))
RFM_hc = pd.concat([RFM, clusterCut], axis=1)
RFM_hc.columns = ['CustomerID', 'Frequency', 'Amount', 'Recency', 'ClusterID']

#summarise
RFM_hc.Recency = RFM_hc.Recency.dt.days
km_clusters_amount = 	pd.DataFrame(RFM_hc.groupby(["ClusterID"]).Amount.mean())
km_clusters_frequency = 	pd.DataFrame(RFM_hc.groupby(["ClusterID"]).Frequency.mean())
km_clusters_recency = 	pd.DataFrame(RFM_hc.groupby(["ClusterID"]).Recency.mean())

df = pd.concat([pd.Series([0,1,2,3,4]), km_clusters_amount, km_clusters_frequency, km_clusters_recency], axis=1)
df.columns = ["ClusterID", "Amount_mean", "Frequency_mean", "Recency_mean"]
print(df.head())

#plotting barplot
sns.barplot(x=df.ClusterID, y=df.Amount_mean)
plt.show()

sns.barplot(x=df.ClusterID, y=df.Frequency_mean)
plt.show()

sns.barplot(x=df.ClusterID, y=df.Recency_mean)
plt.show()