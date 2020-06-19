# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

#Veriyi okumak için kullanılan method
advertising= pd.read_csv("tvmarketing.csv")

#Veri setinin ilk 5 satırına bakma
print(advertising.head())

#Veri setinin son 5 satırına bakma
print(advertising.tail())

#Verinin genel özellikleri için info() methodu kullanılır
print(advertising.info())

#Veri setinin satır ve kolon sayısıno öğrenme
print(advertising.shape)

#Verinin açıklayıcı istatistiklerini görmek için
print(advertising.describe())


#Seaborn kullanarak veri görselleştirme
import seaborn as sns
#%matplotlib inline

#Değişkenler arasındaki ilişkiyi görmek için Scatter Plot
sns.pairplot(advertising,
             x_vars=['TV'],
             y_vars='Sales',
             height=7,
             aspect=0.7,
             kind='scatter') 

#TV bağımsız değişkeni X feature değişkeni içerisine atıyoruz
X= advertising['TV']
print(X.head())

#Bağımlı değişkenimizi y değişkenine atıyoruz
y= advertising['Sales']
print(y.head())


#Veriyi traning ve test set olarak ayırıyoruz
#random_state rassal sayı üretimi için kullanılır
#Herhangi bir integer sayı olabilir
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,
                                                   y,
                                                   train_size=0.7,
                                                   random_state=100)

#Parçaladığımız veri setinin tiplerine bakalım
print(type(X_train))
print(type(X_test))
print(type(y_train))
print(type(y_test))

#Parçalanan veri setlerinin boyutlarına bakalım
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


#scikit-learn de gözlemler satır, featurelar kolonlardır
#aşağıdaki işlem sadece tek feature (kolon) varken yapılır
X_train= X_train[:, np.newaxis]
X_test= X_test[:, np.newaxis] 

#Düzenlenen veri setlerinin boyutlarına bakalım
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


#Linear Regresyon çağırmak
#%%
from sklearn.linear_model import LinearRegression

#linearRegresyon objesini lr değişkeninde tutuyoruz
lr= LinearRegression()

#lr fit ile train verisinden modeli öğreniyoruz
lr.fit(X_train, y_train)

#Katsayı Hesaplaması
#%%
print(lr.intercept_)
print(lr.coef_)

#Predictions
#%%
#Modelin gücünü ölçmek için x_test kısmını modele input olarak verelim
y_pred= lr.predict(X_test)
print(type(y_pred))


#RMSE ve R^2 Hesaplanması
#%%
import matplotlib.pyplot as plt

c= [i for i in range(1,61,1)]
fig= plt.figure()

#mavi çizgi gerçek değerler
plt.plot(c,
         y_test,
         color="blue",
         linewidth=2.5,
         linestyle="-")

#kırmızı çizgi tahmin edilen değerler
plt.plot(c,
         y_pred,
         color="red",
         linewidth=2.5,
         linestyle="-")

fig.suptitle("Actual and Preticted", fontsize=20)
plt.xlabel("Index", fontsize=18)
plt.ylabel("Sales", fontsize=16)


#Hata terimleri
#%%
c= [i for i in range(1,61,1)]
fig= plt.figure()

plt.plot(c,
         y_test-y_pred,
         color="blue",
         linewidth=2.5,
         linestyle="-")

fig.suptitle("Hata Terimleri", fontsize=20)
plt.xlabel("Index", fontsize=18)
plt.ylabel("y_test-y_pred", fontsize=16)
plt.show()

#R squared için gerekli kütüphane ve methodlar import edilir
from sklearn.metrics import  mean_squared_error
from sklearn.metrics import r2_score

mse= mean_squared_error(y_test, y_pred)
r_squared= r2_score(y_test, y_pred)

print("Mean_Squared_Error:", mse)
print("R_square_value:", r_squared)

#scatter plot ile tahminmiş ve gerçek değerlerin ilişkisini görebiliriz
plt.scatter(y_test, y_pred)
plt.xlabel("Y Test")
plt.ylabel("Predicted Y")

