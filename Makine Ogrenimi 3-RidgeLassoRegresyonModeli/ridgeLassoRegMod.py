# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

import os
import warnings
warnings.filterwarnings('ignore')

#Veri setinin okunması
cars= pd.read_csv('CarPrice_Assignment.csv')

#Veri tipleri
print(cars.info())
print(cars.head())

#Tüm sayısal değişkenleri kullanarak bağımlı değişkeni tahminlemeye çalışalım.
cars_numeric= cars.select_dtypes(include= ['float64', 'int64'])
print(cars_numeric.head())

#'symboling' 6 adet kategoriye sahip bir kolon olduğu için veriden çıkarıyoruz.
cars_numeric= cars_numeric.drop(['symboling'], axis= 1)
print(cars_numeric.head())

#özelliklerin correlation grafiklerine bakılır
plt.figure(figsize=(20,10))
sns.pairplot(cars_numeric)
plt.show()

#Oluşturulan grafikten veri okunması zor olduğu için korelasyon matrisi kullanalım
cor= cars_numeric.corr()
print(cor)

#Korelasyon matrisinde de anlamak zor olduğu için heatmap üzerinde gösterelim.
plt.figure(figsize=(16,8))
sns.heatmap(cor, cmap= 'YlGnBu', annot= True)
plt.show()


#Veri Temizleme (Data Cleaning)
#%%

print(cars.info())

#'symboling' in kategorik değişkene çevrilmesi. 1 den 6 ya kadar rakamlar ile ifade ediliyordu
cars['symboling']= cars['symboling'].astype('object')
print(cars.info())

#Şirket adını CarName sütünundan atalım
print(cars['CarName'][:30])

#Sadece car nameleri almak
carnames= cars['CarName'].apply(lambda x: x.split(" ")[0])
print(carnames[:30])

#regular expression ile özel karakterleri silmek
import re

#regex: boşluktan önce herhangi bir alfasayısal sıra, kısa çizgi içerebilir
p= re.compile(r'\w+-?\w+')
carnames= cars['CarName'].apply(lambda x: re.findall(p, x)[0])
print(carnames)

#Company adını saklamak için yeni bir sütun oluşturalım
cars['car_company']= cars['CarName'].apply(lambda x: re.findall(p, x)[0])
print(cars['car_company'].astype('category').value_counts())

#yanlış yazılan car_company adlarının değiştirilmesi
cars.loc[(cars['car_company'] == 'vw') | 
         (cars['car_company'] == 'vokswagen'),
         'car_company'] = 'volkswagen'

cars.loc[cars['car_company'] == 'porcshce', 'car_company'] = 'porsche'
cars.loc[cars['car_company'] == 'toyouta', 'car_company'] = 'toyota'
cars.loc[cars['car_company'] == 'Nissan', 'car_company'] = 'nissan'
cars.loc[cars['car_company'] == 'maxda', 'car_company'] = 'mazda'

print(cars['car_company'].astype('category').value_counts())

#'CarName' değişkenini veri setinden çıkaralım
cars= cars.drop('CarName', axis= 1)
print(cars.info())

#outlier kontrolü: %75 ile max değer arasında çok  fark yoksa eğer outlier yok veya düşüktür.
print(cars.describe())
print(cars.columns)

#Veri Hazırlama (Data Preparation)
#%%

#Veri setinin X ve y değişkenlere ayrılması
X= cars.loc[:, ['symboling', 'fueltype', 'aspiration', 'doornumber',
       'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength',
       'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber',
       'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'compressionratio',
       'horsepower', 'peakrpm', 'citympg', 'highwaympg',
       'car_company']]

y= cars['price']

#Kategorik veriler için dummy variable ların oluşturulması
#Tüm kategorik verilerin alt kümesini ayrı bir veri setine atalım
cars_categorical= X.select_dtypes(include= ['object']) 
print(cars_categorical.head())

#kategorik verilerin dummy lere çevrilmesi
cars_dummies= pd.get_dummies(cars_categorical, drop_first= True)
print(cars_dummies.head())

#orjinal veri setinden kategorik değişkenlerin atılması ve yerine dummy variable ların getirilmesi
X= X.drop(list(cars_categorical.columns), axis= 1)

#birleştirme işlemleri
X= pd.concat([X, cars_dummies], axis= 1)

#sayısal veriler için scaling uygulayalım
from sklearn.preprocessing import scale
cols= X.columns
X= pd.DataFrame(scale(X))
X.columns = cols
print(X.columns)

#verinin training ve test set olarak ayrılması
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    train_size= 0.7,
                                                    test_size= 0.3,
                                                    random_state= 100)


#Model Oluşturma ve Değerlendirme (Model Building and Evaluation)
#%%

#Alfa hiperparametresi nin alternatif değerlerini yazalım
params= {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1,
                   0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                   0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                   7.0, 8.0, 9.0, 10.0, 20, 50, 100,
                   500, 1000]}

ridge= Ridge()

#Grid Search ile optimal alpha değerini bulalım
folds= 5
model_cv= GridSearchCV(estimator= ridge,
                       param_grid= params,
                       scoring= 'neg_mean_absolute_error',
                       cv= folds,
                       return_train_score= True,
                       verbose= 1)

model_cv.fit(X_train, y_train)

cv_results= pd.DataFrame(model_cv.cv_results_)
cv_results= cv_results[cv_results['param_alpha']<=200]
print(cv_results.head())

#bu parametleri grafik yardımı ile gözlemleyelim
cv_results['param_alpha']= cv_results['param_alpha'].astype('int32')

#plot
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')

plt.title('Negative Mean Absolute Error and alpha')
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()
#oluşan grafiğe bakaraj alfa yı 15 olarak seçebiliriz.
#test score nun azalmaya başladığı nokta

alpha= 15
ridge= Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
print(ridge.coef_)


#LASSO
#%%

lasso= Lasso()

#lasso regresyon içinde aynı alfa değerlerini deneyeceğiz 
model_cv= GridSearchCV(estimator= lasso,
                       param_grid= params,
                       scoring= 'neg_mean_absolute_error',
                       cv= folds,
                       return_train_score= True,
                       verbose= 1)

model_cv.fit(X_train, y_train)

cv_results = pd.DataFrame(model_cv.cv_results_)
print(cv_results.head())

#Alfa ile ortalama test ve train grafikte gösterimi
cv_results['param_alpha']= cv_results['param_alpha'].astype('float32')

#plot
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')

plt.title('Negative Mean Absolute Error and alpha')
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()
#oluşan grafiğe bakaraj alfa yı 100 olarak seçebiliriz.
#test score nun azalmaya başladığı nokta

alpha= 100
lasso= Lasso(alpha=alpha)

lasso.fit(X_train, y_train)
print(lasso.coef_)