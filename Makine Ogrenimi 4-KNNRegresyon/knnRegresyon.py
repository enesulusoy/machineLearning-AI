# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import neighbors

from matplotlib.colors import ListedColormap

def knn_comparison(data, n_neighbors= 15):
    '''
    Bu fonksiyon KNN ve verileri gösterir.
    '''
    X= data[:, :2]
    y= data[:, 2]
    
    #Grid hücre boyutu
    h= .02
    cmap_light= ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold= ListedColormap(['#FF0000', '#0000FF'])

    #KNN modelinin oluşturulması
    clf= neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(X, y)

    x_min, x_max= X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max= X[:, 1].min() - 1, X[:, 1].max() + 1
    
    
    #Mesh grid oluşturulması
    xx, yy= np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z= clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z= Z.reshape(xx.shape)
    
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap= cmap_light)
    
    #Verilen noktların plotlanması
    plt.scatter(X[:, 0], X[:, 1], c= y, cmap= cmap_bold)
    
    #x ve y eksenlerindeki limitlerin belirlenmesi
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    #title ekleme
    plt.title('K value = ' + str(n_neighbors))
    plt.show()
    

data= np.genfromtxt('demo_data/6.overlap.csv', delimiter=',')
knn_comparison(data, 1)    
knn_comparison(data, 5)    
knn_comparison(data, 15)    
knn_comparison(data, 30)        
knn_comparison(data, 50) 
#snıfların iç içe geçtiği bir veri setindeki farklı K değerleri ile sınıflandırma örnekleri.
#k = 1 için overfit denilebilir. Tüm sınıflar başarı ile sınıflanmış gözüküyor.
#büyük k değerleri için underfit durumu söze konusudur.
    
    
data= np.genfromtxt('demo_data/1.ushape.csv', delimiter=',')
knn_comparison(data, 1)    
knn_comparison(data, 5)    
knn_comparison(data, 15)    
knn_comparison(data, 30)         
#u şekline sahip bir veri setinde farklı k değerleri için KNN örnekleri.
#k = 1 için yine overfit söz konusudur.    
    
    
data= np.genfromtxt('demo_data/2.concerticcir1.csv', delimiter=',')
knn_comparison(data, 1)    
knn_comparison(data, 5)    
knn_comparison(data, 15)    
knn_comparison(data, 30)      
#iç içe geçmiş oval kümeleme problemlerinde de k=1 için overfit olmuş gözüküyor.
#büyük k değerleri için underfit bu veri setinde de devam etmektedir

    
data= np.genfromtxt('demo_data/3.concertriccir2.csv', delimiter=',')
knn_comparison(data, 1)    
knn_comparison(data, 5)    
knn_comparison(data, 15)         
#k = 1 overfit durumu

    
data= np.genfromtxt('demo_data/4.linearsep.csv', delimiter=',')
knn_comparison(data, 1)    
knn_comparison(data, 5)    
knn_comparison(data)     
    
    
data= np.genfromtxt('demo_data/7.xor.csv', delimiter=',')
knn_comparison(data, 1)    
knn_comparison(data, 5)    
knn_comparison(data)     
    
    
data= np.genfromtxt('demo_data/8.twospirals.csv', delimiter=',')
knn_comparison(data, 1)    
knn_comparison(data, 5)    
knn_comparison(data)  


data= np.genfromtxt('demo_data/9.random.csv', delimiter=',')
knn_comparison(data, 1)    
knn_comparison(data, 5)    
knn_comparison(data)  





