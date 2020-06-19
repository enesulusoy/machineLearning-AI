# -*- coding: utf-8 -*-

#Meme Kanseri Sınıflama

import sklearn
import numpy as np
import pandas as pd  
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import  cross_val_score
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits

import os
import warnings
warnings.filterwarnings('ignore')

#Veriyi Anlama ve Temizleme
#%%

#verinin okunup data frame e dönüştürülemesi
canser= load_breast_cancer()
digits= load_digits()

data= canser

df= pd.DataFrame(data= np.c_[data['data'], data['target']],
                 columns= list(data['feature_names']) + ['target'])

#bağımlı değişkenin veri tipinin belirlenmesi
df['target']= df['target'].astype('uint16')

#veri setine bakalım
print(df)
print(df.head())

#bağımsız ve bağımlı değişkenlerin birbirinden ayrılması
X= df.drop('target', axis= 1)
y= df[['target']]

#verinin train ve test setlere bölünmesi
X_train, X_test, y_train, y_test= train_test_split(X,
                                                   y,
                                                   test_size= 0.20,
                                                   random_state= 101)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#train ve test setlerdeki bağımlı değişkenin benzer şekilde dağılıp dağılmadığını kontrol edelim
print(y_train.mean())
print(y_test.mean())

#adaboost da derinliği olmayan ufak karar ağaçları kullanıyorduk. Maksimum kırılımı 2 olan karar ağacı oluşturalım
shallow_tree= DecisionTreeClassifier(max_depth=2, random_state= 100)

#modeli öğrenelim
shallow_tree.fit(X_train, y_train)

#test hatalarının ekrana basılması
y_pred= shallow_tree.predict(X_test)

#doğruluk oranı
score= metrics.accuracy_score(y_test, y_pred)
print(score)


#Şimdi de adaBoost ile 1-50 arası 3 er artan sayılarla weak learner ile sınıflama yapalım
#%%

#olası ağaç sayılarını tutacak olan listenin oluşturulması
estimators= list(range(1, 50, 3))

abc_scores= []

for n_est in estimators:
    ABC= AdaBoostClassifier(
                            base_estimator= shallow_tree,
                            n_estimators= n_est)
    
    ABC.fit(X_train, y_train)
    y_pred = ABC.predict(X_test)
    score= metrics.accuracy_score(y_test, y_pred)
    abc_scores.append(score)

#Farklı adaboost modellerinin tutarlılık sonuçları    
print(abc_scores)
    
#test accuracy lerinin plotuna bakarsak weak learner sayısı arttıkça modelin test verisindeki
#başarısının arttığını söyleyebiliriz.
plt.plot(estimators, abc_scores)
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.ylim([0.85, 1])
plt.show()




































