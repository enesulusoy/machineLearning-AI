# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

#Veri setinin içeri alınması
housing= pd.read_csv('Housing.csv')

#İlk 5 satıra bakalım
print(housing.head())

#Veri setinde kolonların veri tiplerine bakalım
print(housing.info())


#Data Preprocessing (Veri Ön İşleme)
#%%
#Birçok kolonda yes ve no değerleri olduğu gözüküyor.
#Bunları Dummy variable dönüşümü yapabiliriz

#Aşağıdaki kod bloğu ile dönüşüm işlemi yapabiliriz
housing['mainroad']= housing['mainroad'].map({'yes':1, 'no':0})
housing['guestroom']= housing['guestroom'].map({'yes':1, 'no':0})
housing['basement']= housing['basement'].map({'yes':1, 'no':0})
housing['hotwaterheating']= housing['hotwaterheating'].map({'yes':1, 'no':0})
housing['airconditioning']= housing['airconditioning'].map({'yes':1, 'no':0})
housing['prefarea']= housing['prefarea'].map({'yes':1, 'no':0})


#Tekrardan veriye bakalım
print(housing.head())

#'furnishingstatus' kolunu 3 ayrı kategoriye sahip olduğu için 3 farklı kolon oluşturalım
status= pd.get_dummies(housing['furnishingstatus'])

#3 kategori olarak ayırdık ama biz 2 adet değişkenle aynı sonuca ulaşabiliriz.
print(status.head())

#Dataframe den ilk kolonu atabiliriz. (drop_first=True)
status= pd.get_dummies(housing["furnishingstatus"],drop_first=True) 
print(status.head())

#Ortaya çıkan dummy variable' ları veri setine ekleyelim.
housing= pd.concat([housing,status], axis=1)

#Tekrardan veri setinin ilk 5 satırına bakalım
print(housing.head())

#'furnishingstatus' kolunu için dummy variable oluşturduğumuz için eski olanı atabiliriz
housing.drop(['furnishingstatus'], axis=1, inplace=True)
print(housing.head())


#Yeni değişkenler oluşturalım
#%%

#Regresyon modelini zenginleştirmek için iki adet interaction terim ekleyelim
housing['areaperbedroom']= housing['area']/housing['bedrooms']
housing['bbratio']= housing['bathrooms']/housing['bedrooms']
print(housing.head())


#Rescaling the Features
#%%
#Veri önişlemede scaling den bahsettik. Area ve diğer sayısal kolonlar arasında çok büyük
#değer farkı bulunuyor. Bunu gidermek için:
    #1. Normalizasyon (min-max scaling)
    #2. Standartlıştırma (mean-o, sigma-1)

#Normalizasyon fonksiyonu oluşturalım
def normalize (x):
    return ((x-np.min(x))/ (max(x) - min(x)))

#Normalizasyonu tüm kolonlara uygulayalım.
housing= housing.apply(normalize)


#Veriyi trainig ve test olarak bölmek
#%%

print(housing.columns)

#Bağımsız değişkenleri X değişkenine atayalım
X= housing[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
            'guestroom','basement', 'hotwaterheating', 'airconditioning',
            'parking', 'prefarea', 'semi-furnished','unfurnished',
            'areaperbedroom', 'bbratio']]

#Bağımlı değişkenleri y değişkenine atayalım
y= housing['price']

#Random state rassal sayı üretme çekirdeği olarak kullanılır ve herhangi bir int sayı olabilir
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,
                                                   y,
                                                   train_size=0.7,
                                                   test_size=0.3,
                                                   random_state=100)

#Model
#%%
import statsmodels.api as sm

#Regresyondaki sabit katsayıyı bu kütüphane için bizim eklememiz gerekiyor
X_train= sm.add_constant(X_train)

#fit leme işlemi
lm= sm.OLS(y_train, X_train).fit()

#Regresyon modelinin özet bilgilerini imceleyelim.
print(lm.summary())


#coeff: regresyon denklemindeki katsayılardır. const bias(interceptor) terimidir
#std err: katsayının değişimini gösterir. Küçük tutulması istenir.
#t: coeff/std error' dan gelen istatistik değeri 
#P value değeri eğer 0.05' ten küçükse o değişkenin modelde tutulması gerekir. Büyükse modelden çıkarılır


#VIF değerinin kontrol edilmesi
#%%
#VIF (Variance inflation factor) bir bağımsız değişkenin diğer bağımsız değişkenlerle bağımlılığını ölçer
#Büyük değere sahip olan bağımsız değişkenler modelden atılır.
#1/1-rsquare formülü ile hesaplanır
#Ancak buradaki r square farklı şekilde hesaplanır
#örneğin bağımsız değişken olan x1 artık bağımlı değişken gibi kabul edilir
#x1 değişkeni x2, x3 ve xn gibi diğer değişkenler ile tahminlenmeye çalışılır ve R square değeri hesaplanır
#Her bi bağımsız değişken bağımlı değişken gibi kabul edilip modellenir ve R square hesaplanır
#R square yüksek ise o anki kabul edilen bağımlı değişkenin diğer değişkenler ile fazla ilişkili olduğu kabul edilir.
#örneğin aşağıdaki örnekte area per bedroom değişkeninin VIF değeri çok yüksek olduğu için modelden atılır.

def vif_cal(input_data, dependent_col):
    
    vif_df= pd.DataFrame(columns=['Var', 'Vif'])
    x_vars= input_data.drop([dependent_col], axis=1)
    xvar_names= x_vars.columns
    
    for i in range(0,xvar_names.shape[0]):
        y= x_vars[xvar_names[i]]
        x= x_vars[xvar_names.drop(xvar_names[i])]
        
        rsq= sm.OLS(y,x).fit().rsquared
        vif= round(1/(1-rsq),2)
        vif_df.loc[i]= [xvar_names[i], vif]
    return vif_df.sort_values(by='Vif', axis=0, ascending=False, inplace=False)

#Vif değerinin hesaplanması
#vif_sonuc= vif_cal(input_data=housing, dependent_col='price')
print(vif_cal(input_data=housing, dependent_col='price'))


#Korelasyon Matrisi (Correlation Matrix)
#%%

import matplotlib.pyplot as plt
import seaborn as sns

#Korelasyon matrisin oluşturulması
plt.figure(figsize=(16,10))
sns.heatmap(housing.corr(), annot= True)


#Özellik Çıkarımı ve Modelin Update Edilmesi
#%%
#Training setten bbratio özelliğini çıkaralım
X_train= X_train.drop('bbratio', 1)

#ve yeni linear model ile devam edelim
lm_2= sm.OLS(y_train, X_train).fit()

#ve modelin özet bilgilerine tekrardan bakalım. R squared değerine herhangi bir değişim olmadı
#Buradan 'bbratio' nun modelin tuturlılığında herhangi bir etkisi yoktur.
print(lm_2.summary())

#Tekrardan VIF değerini hesaplayalım
print(vif_cal(input_data=housing.drop(['bbratio'], axis=1), dependent_col='price')) 

#Bedroom özelliğini verisetinden çıkaralım
X_train= X_train.drop('bedrooms', 1)

#ve yeniden model oluşturalım
lm_3= sm.OLS(y_train, X_train).fit()
print(lm_3.summary())

#Tekrardan VIF değerini hesaplayalım
print(vif_cal(input_data=housing.drop(['bedrooms','bbratio'], axis=1), dependent_col='price')) 

#Tekrardan önemsiz olduğu düşünülen bir değişkeni modelden çıkaralım
X_train= X_train.drop('areaperbedroom', 1)

#ve yeniden model oluşturalım
lm_4= sm.OLS(y_train, X_train).fit()
print(lm_4.summary())

#Tekrardan VIF değerini hesaplayalım
print(vif_cal(input_data=housing.drop(['bedrooms','bbratio','areaperbedroom'], axis=1), dependent_col='price')) 

#Semi furnised kolununu modelden çıkaralım
X_train= X_train.drop('semi-furnished', 1)

#ve yeniden model oluşturalım
lm_5= sm.OLS(y_train, X_train).fit()

#Özete baktığımızda R square'in değişmediğini görürüz
print(lm_5.summary())

#Tekrardan VIF değerini hesaplayalım
print(vif_cal(input_data=housing.drop(['bedrooms','bbratio','areaperbedroom','semi-furnished'], axis=1), dependent_col='price')) 

#Basement kolununu modelden çıkaralım
X_train= X_train.drop('basement', 1)

#ve yeniden model oluşturalım
lm_6= sm.OLS(y_train, X_train).fit()

#Özete baktığımızda R square'in değişmediğini görürüz
print(lm_6.summary())

#Tekrardan VIF değerini hesaplayalım
print(vif_cal(input_data=housing.drop(['bedrooms','bbratio','areaperbedroom','semi-furnished','basement'], axis=1), dependent_col='price')) 


#Son model ile tahmin yapmak
#%%

#Test sete constant terimi içeren kolonun eklenmesi. (interceptor)
X_test_m6= sm.add_constant(X_test)

#Test setten daha önce train setten çıkardığımız kolonların çıkarılması
X_test_m6= X_test_m6.drop(['bedrooms','bbratio','areaperbedroom','semi-furnished','basement'], axis=1)

#Tahminin yapılması
y_pred_m6= lm_6.predict(X_test_m6)


#Model Evaluation (ModelDeğerlendirme)
#%%

#Gerçekleşen ve Tahminlenen değerlerin plotlanması. Mavi çizgi actual, kırmızı çizgi tahminler
c= [i for i in range(1,165,1)]
fig= plt.figure()

#Plotting actual
plt.plot(c,
         y_test,
         color="blue",
         linewidth=2.5,
         linestyle="-")

#Plotting predicted
plt.plot(c,
         y_pred_m6,
         color="red",
         linewidth=2.5,
         linestyle="-")

fig.suptitle("Actual and Predicted", fontsize=20)
plt.xlabel('Index', fontsize=18)
plt.ylabel('Housing Price', fontsize=16)

#Y test ve Y pred in plotlanarak gerçek ve tahmin edilen değerlerin dağılımını göstermek
fig= plt.figure()
plt.scatter(y_test,y_pred_m6)
plt.suptitle('y_test vs y_pred', fontsize=20)
plt.xlabel('y_test', fontsize=18)
plt.ylabel('y_pred', fontsize=16)


#Hata Terimleri
#%%

#Hata terimlerinin ekrana basatırılması
fig= plt.figure()
c= [i for i in range(1,165,1)]
plt.plot(c,
         y_test-y_pred_m6,
         color="blue",
         linewidth=2.5,
         linestyle="-")

plt.suptitle('Error Terms', fontsize=20)
plt.xlabel('Index', fontsize=18)
plt.ylabel('y_test-y_pred', fontsize=16)

#Hata terimlerinin dağılımına bakılması. Normal dağılıma benzediği için kabul edilebilir
fig= plt.figure()
sns.distplot((y_test-y_pred_m6), bins=50)
plt.suptitle('Error Terms', fontsize=20)
plt.xlabel('y_test-y_pred', fontsize=18)
plt.ylabel('Index', fontsize=16)


#Test Hatasını Ölçmek
#%%

import numpy as np
from sklearn import metrics
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_m6)))


