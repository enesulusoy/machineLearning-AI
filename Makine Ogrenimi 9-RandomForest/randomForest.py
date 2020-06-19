# -*- coding: utf-8 -*-

#Kredi Temerrut Uygulaması (borcunu zamanında ödeyip ödeyemeyeceği durumu)

import numpy as np
import pandas as pd  
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

#Veriyi Anlama ve Temizleme
#%%

#verinin okunup data frame e dönüştürülemesi
df= pd.read_csv('credit-card-default.csv')
print(df.head())

#kolonların veri tiplerini anlamak
print(df.info())


#Veri Önişleme
#%%

#modeli import etmek için train_test_split
from sklearn.model_selection import train_test_split

#X değişkenine etiket hariç tüm kolonlaru atayalım
X= df.drop('defaulted', axis= 1)

#etiketi y değişkenine atamak
y= df['defaulted']

#veriyi train ve test olarak bölünmesi
X_train, X_test, y_train, y_test= train_test_split(X, y,
                                                   test_size= 0.30,
                                                   random_state= 101)

#Default parametreler ile modeli öğrenelim
#%%

#random forest classifier import edelim
from sklearn.ensemble import RandomForestClassifier

#default parametreler ile modeli oluşturalım
rfc= RandomForestClassifier()

#modeli fit edelim
rfc.fit(X_train, y_train)

#tahmin yapılması
predictions= rfc.predict(X_test)

#sınıflama metriklerinin import edilmesi
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#metrikler
print(classification_report(y_test, predictions))

#confusion matrix
print(confusion_matrix(y_test, predictions))

#accuracy score
print(accuracy_score(y_test, predictions))


#Hyperparameter Tuning
#%%

#Hiperparametreler:
    #n_estimators: Modelin kaç adet karar ağacına bölüneceğini belirtilir.
    #criterion: gini ya da entropy seçilebilir
    #max_features: karar ağacını bölerken kullanılacak özellik sayısıdır.
    #max_depth: ağaçların derinliğini ayarlar
    #min_samples_split: ağacın bölünmesi için gerekli minimum sample sayısı
    #min_samples_leaf: Leaf node oluşturabilmek için gerekli saple sayısı
    
#max_depth parametresini GridSearch ile optimize edelim
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#k-fold un ayarlanması
n_folds= 5

#denenecek parametrelerin oluşturulması
parameters= {'max_depth': range(2, 20 ,5)}

#randomForest modelinin çağırılması
rf= RandomForestClassifier()

#modelin training set ile eğitilmesi
rf= GridSearchCV(rf,
                 parameters,
                 cv= n_folds,
                 scoring= 'accuracy',
                 return_train_score= True,
                 n_jobs= -1,
                 verbose=2)
rf.fit(X_train, y_train)

#GridSearch CV sonuçları
scores= rf.cv_results_
print(pd.DataFrame(scores).head())

#accuracylerin max_depth e göre plotlanması
plt.figure()
plt.plot(scores['param_max_depth'],
         scores['mean_train_score'],
         label='training accuracy')
plt.plot(scores['param_max_depth'],
         scores['mean_test_score'],
         label='test accuracy')

plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#max_depth değeri arttıkça, hem train hem de test değerlerinin bir noktaya kadar 
#arttığını, ancak burdan sonra test puanının düşmeye başladığını görebiliyoruz. 
#Max_depth in artması bir süre sonra overfitting e sebep olmuştur.


#Tuning n_estimators
#modelde kullanılacak karar ağacı sayısının optimize edilmesi
#%%

#GridSearchCV ile optimal n-estimators u bulalım
from sklearn.model_selection import  KFold
from sklearn.model_selection import GridSearchCV

#k-fold un set edilmesi
n_folds= 5

#alternatif hiperparametrelerin belirlenmesi
parameters= {'n_estimators': range(100, 1500, 400)}

#max_depth in set edilmesi
rf= RandomForestClassifier(max_depth= 4)

#training set ile modelin öğrenilmesi
rf= GridSearchCV(rf,
                 parameters,
                 cv= n_folds,
                 scoring= 'accuracy',
                 return_train_score= True,
                 n_jobs= -1,
                 verbose=2)
rf.fit(X_train, y_train)

#GridSearch CV sonuçları
scores= rf.cv_results_
print(pd.DataFrame(scores).head())

#n_estimators ile accuracylerin plotlanması
plt.figure()
plt.plot(scores['param_n_estimators'],
         scores['mean_train_score'],
         label='training accuracy')
plt.plot(scores['param_n_estimators'],
         scores['mean_test_score'],
         label='test accuracy')

plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#n_estimators grafiği 500 civarında optimal olduğunu görebildik
#train ve test accuracy si max oluduğu nokta


#Tuning max_features
#%%

#GridSearchCV ile max features
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#k-fold un set edilmesi
n_folds= 5

#alternatif max_features
parameters= {'max_features': [4, 8, 14, 20, 24]}

#Random Forest Classifier ın çağrılması
rf= RandomForestClassifier(max_depth= 4)

#training setin oluşturulması
rf= GridSearchCV(rf,
                 parameters,
                 cv= n_folds,
                 scoring= 'accuracy',
                 return_train_score= True,
                 n_jobs= -1,
                 verbose=2)
rf.fit(X_train, y_train)

#GridSearchCV sonuçları
scores= rf.cv_results_
print(pd.DataFrame(scores).head())

#max_features ile accuracylerin plotlanması
plt.figure()
plt.plot(scores['param_max_features'],
         scores['mean_train_score'],
         label='training accuracy')
plt.plot(scores['param_max_features'],
         scores['mean_test_score'],
         label='test accuracy')

plt.xlabel('max_features')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#Tuning min_samples_leaf
#%%

#GridSearchCV ile min_samples_leaf
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#k-fold un set edilmesi
n_folds= 5

#alternatif max_features
parameters= {'min_samples_leaf': range(100, 400, 50)}

#Random Forest Classifier ın çağrılması
rf= RandomForestClassifier()

#training setin oluşturulması
rf= GridSearchCV(rf,
                 parameters,
                 cv= n_folds,
                 scoring= 'accuracy',
                 return_train_score= True,
                 n_jobs= -1,
                 verbose=2)
rf.fit(X_train, y_train)

#GridSearchCV sonuçları
scores= rf.cv_results_
print(pd.DataFrame(scores).head())

#min_samples_leaf ile accuracylerin plotlanması
plt.figure()
plt.plot(scores['param_min_samples_leaf'],
         scores['mean_train_score'],
         label='training accuracy')
plt.plot(scores['param_min_samples_leaf'],
         scores['mean_test_score'],
         label='test accuracy')

plt.xlabel('min_samples_leaf')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#min_samples_leaf değeri düşdükçe overfitting yada underfitting olduğunu görebiliriz


#Tuning min_samples_split
#%%

#GridSearchCV ile min_samples_leaf
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#k-fold un set edilmesi
n_folds= 5

#alternatif max_features
parameters= {'min_samples_split': range(200, 500, 50)}

#Random Forest Classifier ın çağrılması
rf= RandomForestClassifier()

#training setin oluşturulması
rf= GridSearchCV(rf,
                 parameters,
                 cv= n_folds,
                 scoring= 'accuracy',
                 return_train_score= True,
                 n_jobs= -1,
                 verbose=2)
rf.fit(X_train, y_train)

#GridSearchCV sonuçları
scores= rf.cv_results_
print(pd.DataFrame(scores).head())

#min_samples_leaf ile accuracylerin plotlanması
plt.figure()
plt.plot(scores['param_min_samples_split'],
         scores['mean_train_score'],
         label='training accuracy')
plt.plot(scores['param_min_samples_split'],
         scores['mean_test_score'],
         label='test accuracy')

plt.xlabel('min_samples_leaf')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#min_samples_split arttıkça training set accuracy nin düştüğünü söyleyebiliriz


#Tüm Parametrelerin Optimize Edilmesi
#%%

param_grid= {
    'max_depth': [4, 8, 10],
    'min_samples_leaf': range(100, 400, 200),
    'min_samples_split': range(200, 500, 200),
    'n_estimators': [100, 200, 300],
    'max_features': [5, 10]
    }

rf= RandomForestClassifier()

grid_search= GridSearchCV(estimator= rf,
                          param_grid= param_grid,
                          cv= 3,
                          n_jobs= -1,
                          verbose= 2,
                          return_train_score= True)
grid_search.fit(X_train, y_train)

#GridSearch ile en iyi parametrelerin ekrana bastırılması
print('We can get accuracy of', grid_search.best_score_, 'using', grid_search.best_params_)

#En iyi parametreler: We can get accuracy of 0.8183809523809523 using 
#{'max_depth': 4, 'max_features': 10, 'min_samples_leaf': 100, 
#'min_samples_split': 400, 'n_estimators': 200}


#En Optimum Parametreler ile Modelin Tekrar Oluşturulması
#%%

#best hiperparametreler ile Model
from sklearn.ensemble import RandomForestClassifier

#Model oluşturma
rfc= RandomForestClassifier(bootstrap= True,
                            max_depth= 10,
                            min_samples_leaf= 100,
                            min_samples_split= 200,
                            max_features= 10,
                            n_estimators= 100)
#fit
rfc.fit(X_train, y_train)

#predict
y_pred= rfc.predict(X_test)

#sınıflama metrikleri
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#accuracy %: TP+TN/ALL
print((6753+692)/(6753+692+305+1250))
print(accuracy_score(y_test, y_pred))









