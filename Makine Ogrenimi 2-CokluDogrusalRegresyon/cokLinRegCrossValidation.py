# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings('ignore')

#Housing veri setinin içeri alınması ve ilk 5 satıra bakılması
housing= pd.read_csv('Housing.csv')
print(housing.head())

#Verideki satır sayısı
print(len(housing.index))

#Bu deney için sadece bağımsız değişken 'area' ve bağımlı değişken 'price' kullanacağız
df= housing.loc[:, ['area', 'price']]
print(df.head())

#Değişkenlerin min-max ile scale edilmesi. Tüm veriler 0-1 arasında olacaktır
df_columns= df.columns
scaler= MinMaxScaler()
df= scaler.fit_transform(df)

#Kolonların yeniden adlandırılması
df= pd.DataFrame(df)
df.columns= df_columns
print(df.head())

#area-price ilişkisinin görselleştirilmesi
sns.regplot(x='area',
            y='price',
            data=df,
            fit_reg=False)

#Verinin train ve test olarak ayrılması
df_train, df_test= train_test_split(df,
                                    train_size= 0.7,
                                    test_size= 0.3,
                                    random_state= 10)

print(len(df_train))
print(len(df_test))

#X ve y değişkenlerini hem training hem de test kısımlarına ayırıyoruz
X_train= df_train['area']
X_train= X_train.values.reshape(-1, 1)
y_train = df_train['price']

X_test= df_test['area']
X_test= X_test.values.reshape(-1, 1)
y_test= df_test['price']

#Polinomal Model
#%%

#Verilerin üssü için alternatif değerler
degrees= [1, 2, 3, 6, 10, 20]

#y_train_pred ve y_test_pred matrislerini tahmin sonuçlarını oluşturmak için kullanalım
y_train_pred= np.zeros((len(X_train), len(degrees))) #train predictionları tutacak sıfırlar matrisi
y_test_pred= np.zeros((len(X_test), len(degrees))) #test predictionları tutacak sıfırlar matrisi


for i, degree in enumerate(degrees):
    
    #PolynominalFeatures classına degree dizisini verdiğimizde
    #verilerin dizideki sayılara göre üslerini alır ve oluşturur
    model= make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    
    
    #train ve test verileri üzerinde tahmin yapılması ve değişkenlere storelanması
    #her bir üslü değer için gerekli kolonlara tahminlerin yazılması
    y_train_pred[:, i]= model.predict(X_train)
    y_test_pred[:, i]= model.predict(X_test)
    

#train ve test tahminlerinin görselleştirilmesi
#y ekseni log scale' de dir.
plt.figure(figsize=(16, 8))

#train data
plt.subplot(121)
plt.scatter(X_train, y_train)
plt.yscale('log')
plt.title('Train Data')

for i, degree in enumerate(degrees):
    plt.scatter(X_train, y_train_pred[:, i], s=15, label=str(degree))
    plt.legend(loc='upper left')


#test data
plt.subplot(122)
plt.scatter(X_test, y_test)
plt.yscale('log')
plt.title('Test Data')

for i, degree in enumerate(degrees):
    plt.scatter(X_test, y_test_pred[:, i], s=15, label=str(degree))
    plt.legend(loc='upper left')


#Tüm polinomal modellerin R squared değerlerinin ekrana bastırılması
#Train sette başarı arttıkça test setteki r squared başarısı düştüğü için
#yüksek üssü alınmış verilerle işlem yaptığımızda overfit olmaya doğru gidildiği söylenebilir
#bias-variance trade-off inceleyiniz
print('R-squared değerleri: \n')

for i, degree in enumerate(degrees):
    train_r2= round(sklearn.metrics.r2_score(y_train, y_train_pred[:, i]), 2)
    test_r2= round(sklearn.metrics.r2_score(y_test, y_test_pred[:, i]), 2)
    print('Polynomial degree {0}: train score={1}, test score={2}'.format(degree,
                                                                          train_r2,
                                                                          test_r2))
# 1. Cross Validation olmadan model oluşturma
#%%

#yes ve no lardan oluşan verileri 1 ve 0 olarak değiştirelim
binary_vars_list= ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

def binary_map(x):
    return x.map({'yes': 1, 'no': 0})
    
#binary map fonksiyonu gerekli alanlara uyguluyoruz    
housing[binary_vars_list]= housing[binary_vars_list].apply(binary_map)
print(housing.head())

#'furnishingstatus' için dummy variable ların oluşturulması
#kategori sayısı - 1 adet kolon yeterli olacaktır
status= pd.get_dummies(housing['furnishingstatus'], drop_first= True)
print(status.head())

#dummy variable ların ana dataframe ile birleştirilmesi
housing= pd.concat([housing, status], axis= 1)
print(housing.head())

#'furnishingstatus' için hali hazırda dummy variable kolonları oluşturduğumuz için veri setinden çıkartıyoruz
housing.drop(['furnishingstatus'], axis= 1, inplace= True)
print(housing.head())


#Veriyi Train ve Test sete bölme
#%%

#train-test 70-30 split
df_train, df_test= train_test_split(housing,
                                    train_size= 0.7,
                                    test_size= 0.3,
                                    random_state= 100)

#Verileri scale etmek
scaler= MinMaxScaler()

#Scaling işlemini tüm kolonlara yazdırmak.
numeric_vars= ["area","bedrooms","bathrooms","stories","parking","price"]
df_train[numeric_vars]= scaler.fit_transform(df_train[numeric_vars])
print(df_train.head())

#scaling işleminin test sette uygulanması
df_test[numeric_vars]= scaler.fit_transform(df_test[numeric_vars])
print(df_test.head())

y_train= df_train.pop('price')
X_train= df_train

y_test = df_test.pop('price')
X_test = df_test


#RFE, bizim yerimize istatiksel olarak önemli bağımsız değişkenleri seçen ve işimizi kolaylaştıran bir araçtır
#İçerisine parametre olarak kolon sayısı alır.

#Maksimum kolon sayısı için
print(len(X_train.columns))

#RFE ile 10 adet önemli bağımsız değişken seçilmesi
lm= LinearRegression()
lm.fit(X_train, y_train)

rfe= RFE(lm, n_features_to_select= 10)
rfe= rfe.fit(X_train, y_train)

#Toplamda 13 bağımsız değişken vardu. Seçilmeyenler false olarak işaretlenmiştir
print(list(zip(X_train.columns, rfe.ranking_)))


#X_test verileri ile tahmin yapılması
#%%

y_pred= rfe.predict(X_test)

#test sette r squared hesaplanması
r2= sklearn.metrics.r2_score(y_test, y_pred)
print(r2)

#başka bir değer ile RFE çalıştırma
lm= LinearRegression()
lm.fit(X_train, y_train)

rfe= RFE(lm, n_features_to_select= 6)
rfe= rfe.fit(X_train, y_train)

#bu sefer görüldüğü üzere R squared düştü
y_pred= rfe.predict(X_test)
r2= sklearn.metrics.r2_score(y_test, y_pred)
print(r2)

#Şuanki yaklaşımda bazı problemler var:
    #Verinin hep aynı kısmını train ve test olarak kullanmak
    #Manuel olarak RFE kullanmanın getirdiği gereksiz iş yükü
    #Modelin hep aynı kısmı ile öğrenme yapıldığı için, modelin genelleştirildiği söylenemez


#Cross-Validation
#%%

#13 değişken ile cross validation
#model çağırlıyor ve score metriği belirleniyor. Ve verinin kaç bölüme ayrılacağı set ediliyor
lm= LinearRegression()
scores= cross_val_score(lm,
                        X_train,
                        y_train,
                        scoring= 'r2',
                        cv= 5)
print(scores)

#diğer bir yolla aşağıdaki gibi de yapılabilir
folds= KFold(n_splits= 5,
             shuffle= True,
             random_state= 100)

scores= cross_val_score(lm,
                        X_train,
                        y_train,
                        scoring= 'r2',
                        cv= folds)
print(scores)


#Grid Search Cross Validation ile Hyper Parametre Optimizasyonu
#%%

#adım-1: öncelikle K-fold cv yapısı oluşturulur
folds= KFold(n_splits= 5,
             shuffle= True,
             random_state= 100)

#adım-2: hiper parametremizin aralığını belirliyoruz. 13 bağımsız değişken için bir 1 den 13 e integer içeren
#kadar liste oluşturuyoruz
hyper_params= [{'n_features_to_select': list(range(1, 14))}]

#adım-3: grid search ün çalıştırılması
#3.1 modelin belirlenmesi
lm= LinearRegression()
lm.fit(X_train, y_train)
rfe= RFE(lm)

#3.2 gridSearc ün parametrelerinin oluşturulması
model_cv= GridSearchCV(estimator= rfe,
                       param_grid= hyper_params,
                       scoring= 'r2',
                       cv= folds,
                       verbose= 1,
                       return_train_score= True)

#Fit the model
model_cv.fit(X_train, y_train)

#Cross Validation Sonuçları
cv_results= pd.DataFrame(model_cv.cv_results_)
print(cv_results)

#params kullanılan bağımsız değişken sayısı
#split test score her bir iterasyonda gerçekleşen r squared değeri
#mean test score ortalama r squared değeri 


#eğer plotlarsak daha iyi anlayacağız
#görüldüğü üzere aşağı yukarı 10 bağımsız değişkenden sonra modelin r squared
#değerinde bir değişme gözlemleniyor
plt.figure(figsize= (16, 6))

plt.plot(cv_results['param_n_features_to_select'], cv_results['mean_test_score'])
plt.plot(cv_results['param_n_features_to_select'], cv_results['mean_train_score'])
plt.xlabel('number of features')
plt.ylabel('r_squared')
plt.title('Optimal Number of Features')
plt.legend(['test score', 'train score'], loc= 'upper left')
plt.show()

#Grafiğe bakarak Optimal feature sayısı olarak 10 seçer ve en iyi 10 feature için RFE kullanırsak..
#Final Model
#%%

n_features_optimal= 10

lm= LinearRegression()
lm.fit(X_train, y_train)

rfe= RFE(lm, n_features_to_select= n_features_optimal)
rfe= rfe.fit(X_train, y_train)

#predict prices of X_test
y_pred= lm.predict(X_test)
r2= sklearn.metrics.r2_score(y_test, y_pred)
print(r2)


#Car Price Prediction
#%%

#Verinin okunması
cars= pd.read_csv('CarPrice_Assignment.csv')
print(cars)

#aşağıdaki kod bloğunda tamamen veri önişleme yapılmıştır.
cars['symboling']= cars['symboling'].astype('object')

p= re.compile(r'\w+-?\w+')
cars['car_company']= cars['CarName'].apply(lambda x: re.findall(p, x)[0])

cars.loc[(cars['car_company'] == 'vw') | 
         (cars['car_company'] == 'vokswagen'),
         'car_company'] = 'volkswagen'

cars.loc[cars['car_company'] == 'porcshce', 'car_company'] = 'porsche'
cars.loc[cars['car_company'] == 'toyouta', 'car_company'] = 'toyota'
cars.loc[cars['car_company'] == 'Nissan', 'car_company'] = 'nissan'
cars.loc[cars['car_company'] == 'maxda', 'car_company'] = 'mazda'

cars= cars.drop('CarName', axis= 1)

X= cars.loc[:, ['symboling', 'fueltype', 'aspiration', 'doornumber',
       'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength',
       'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber',
       'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'compressionratio',
       'horsepower', 'peakrpm', 'citympg', 'highwaympg',
       'car_company']]

y= cars['price']

cars_categorical= X.select_dtypes(include=['object'])
print(cars_categorical.head())

cars_dummies= pd.get_dummies(cars_categorical, drop_first= True)
print(cars_dummies.head())

X= X.drop(list(cars_categorical.columns), axis= 1)

X= pd.concat([X, cars_dummies], axis= 1)

cols= X.columns
X= pd.DataFrame(scale(X))
X.columns = cols

X_train, X_test, y_train, y_test= train_test_split(X,
                                                   y,
                                                   train_size= 0.7,
                                                   test_size= 0.3,
                                                   random_state= 40)

print(len(X_train.columns))


#Model Fiting
#%%

folds= KFold(n_splits= 5,
             shuffle= True,
             random_state= 100)

hyper_params= [{'n_features_to_select': list(range(2, 40))}]

lm= LinearRegression()
lm.fit(X_train, y_train)
rfe= RFE(lm)

model_cv= GridSearchCV(estimator= rfe,
                       param_grid= hyper_params,
                       scoring= 'r2',
                       cv= folds,
                       verbose= 1,
                       return_train_score= True)

model_cv.fit(X_train, y_train)
cv_results= pd.DataFrame(model_cv.cv_results_)
print(cv_results)


#Aşağıdaki kod bloğunda oluşan görselde kullanılan bağımsız değişken sayısı arttıkça 
#train set ve test set başarıları 17-18 feature a kadar beraber artıyor. Sonrasında ise 
#test set başarısı artmazken train set başarısı artmaya devam ediyor. Bu da verinin 17-18
#feature kulllnımından sonra overfittin e uğradığı anlamına gelir   
plt.figure(figsize= (16, 6))

plt.plot(cv_results['param_n_features_to_select'], cv_results['mean_test_score'])
plt.plot(cv_results['param_n_features_to_select'], cv_results['mean_train_score'])
plt.xlabel('number of features')
plt.ylabel('r_squared')
plt.title('Optimal Number of Features')
plt.legend(['test score', 'train score'], loc= 'upper left')
plt.show()