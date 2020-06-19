# -*- coding: utf-8 -*-

#Gelir Düzeyi Sınıflama

import numpy as np
import pandas as pd  
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

#Veriyi Anlama ve Temizleme
#%%

#verinin okunup data frame e dönüştürülemesi
df= pd.read_csv('adult_dataset.csv')

#veri tiplerini incelemek
print(df.info())

#veri de ? karakterli bir çok kayıp değer bulunuyor
print(df.head())

#verinin workclass özelliğinde kaç adet ? olduğunu gösterelim
df_1= df[df.workclass == '?']
print(df_1)
print(df_1.info())

#workclass taki ? karakterli kayıp değerlerin veriden çıkartılması
df= df[df['workclass'] != '?']
print(df.head())

#tüm kategorik değişkenleri seçmek
df_categorical= df.select_dtypes(include= ['object'])

#kategorik veriler de ? karakteri oluo olmadığını kontrol edelim
df_nan= df_categorical.apply(lambda x: x== '?', axis=0).sum()
print(df_nan)
#Native country ve occupation alanlarında ? karakteri bulunuyor

#veri setinden bu satırları da çıkartalım
df= df[df['occupation'] != '?']
df= df[df['native.country'] != '?']

#artık kayıp değer içermeyen veri seti ile çalışabiliriz
print(df.info())


#Veri Önişleme (Data Preparation)
#%%

#Önceki algoritmalarda dummy variable lara çevirmeden categorik değişkenler ile  çalışma yapamazdık. 
#Ama decision tree categorik veriler ile çalışabilir. Yine de standart bir formatta kodlamamız gerekiyor.
from sklearn import preprocessing

#Tüm kategorik verilerin seçilmesi
df_categorical= df.select_dtypes(include= ['object']) 
print(df_categorical.head())

#Label encoder ın bu kolonlara uygulanması. Sonuçta tekil kategorik değişken sayısı kadar encoded sayısal değişken oluşturulur.
le= preprocessing.LabelEncoder()
df_categorical= df_categorical.apply(le.fit_transform)
print(df_categorical.head())

#orjinal veri seti ile encoded veri setinin birleştirilmesi.
df= df.drop(df_categorical.columns, axis= 1)
df= pd.concat([df, df_categorical], axis= 1)
print(df.head())

#tekrar verinin özelliklerine bakarsak hepsinin integer olduğu görülür.
print(df.info())

#income bağımlı değişkenini kategorik değişken olarak değiştiriyoruz.
df['income']= df['income'].astype('category')


#Model Oluşturma ve Değerlendirme
#%%

#İlk olarak default hyperparametreler ile ağaç oluşturalım

#Verinin train ve test olarak bölünmesi için kütüphanenin import edilmesi
from sklearn.model_selection import train_test_split

#bağımsız ve bağımlı değişkenlerin atanması
X= df.drop('income', axis= 1)
y= df['income']

#train ve test setlerin oluşturulması.
X_train, X_test, y_train, y_test= train_test_split(X, y,
                                                   test_size= 0.30,
                                                   random_state= 99)
print(X_train.head())

#karar ağacı sınıflayıcısının import edilmesi
from sklearn.tree import DecisionTreeClassifier

#max_depth parametresini 5 olarak seçiyoruz
dt_default= DecisionTreeClassifier(max_depth= 5)
dt_default.fit(X_train, y_train)

#♣performans metriklerinin import edilmesi
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#test set üzerinden tahmin yapılması
y_pred_default= dt_default.predict(X_test)

#sınıflama performansının ekrana bastırılması
print(classification_report(y_test, y_pred_default))

#Confusion matrix ve accuracy
print(confusion_matrix(y_test,y_pred_default))
print(accuracy_score(y_test, y_pred_default))


#Decision Tree Görselleştirme
#%%

#Decision tree görselleyebilmek için graphviz kütüphanesine ihtiyaç vardır
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
import graphviz

#feature ların bir liste olarak alınması
features= list(df.columns[1:])
print(features)

#Python karar ağacını görselleştirmek için pydot ve graphviz kütüphanelerine ihtiyaç duyar.
import os
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

#max_depth= 5 ile tree oluşturalım
dot_data= StringIO()
export_graphviz(dt_default,
                out_file= dot_data,
                feature_names= features,
                filled= True,
                rounded= True)

graph= pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


#Tuning max_depth
#Max Depth karar ağacının aşağı doğru ne kadar dallara kırılabileceğini belirledimiz hiper parametredir. Max Depth in 
#optimal değerini bulmak ve model üzerindeki etkisini görebilmek için GridSearch CV kullanalım

#GridSearch ile optimum ağaç deinliğini bulmak
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#k-fold için verinin bölüneceği parça sayısını belirlemek
n_folds= 5

#max depth 1 ile 40 arasında
parameters= {'max_depth': range(1, 40)}

#îmodelin oluşturulması. Kriter olarak gini seçilmiştir
dtree= DecisionTreeClassifier(criterion= 'gini',
                              random_state= 100)

#GridSearch ün çalıştırılması
tree= GridSearchCV(dtree,
                   parameters,
                   cv= n_folds,
                   scoring= 'accuracy',
                   return_train_score=True)

tree.fit(X_train, y_train)

#GridSearch sonuçları
scores= tree.cv_results_
print(pd.DataFrame(scores).head())

#çeşitli max depth değerlerine göre accuracy nin plotlanması
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

#Görüldüğü üzere max_depth= 10 dan sonra test accuracy düşmüş ve overfitting başlamıştır


#Tuning min_samples_leaf
#hiperparametre min_samples_leaf ağacın dallarında olması gereken min. sample sayısını gösterir
#%%

#GridSearchCV ve min_sample_leaf in optimal değerini bulmak
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#K-fold parametresinin belirlenmesi
n_folds= 5

#bir karar ağacı dalında bulunması gereken sample miktarı için 20 20 artan bir liste kullanalım
parameters= {'min_samples_leaf': range(5, 200, 20)} 

#modeli oluşturalım
dtree= DecisionTreeClassifier(criterion= 'gini',
                              random_state= 100)

#modeli training setten öğrenelim
tree= GridSearchCV(dtree,
                   parameters,
                   cv= n_folds,
                   scoring= 'accuracy',
                   return_train_score=True)
tree.fit(X_train, y_train)

#GridSearch CV scorelarını ekrana basalım
scores= tree.cv_results_
print(pd.DataFrame(scores).head())

#çeşitli min leaf değerlerine göre training ve test accuracy değerleri
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

#Graiğe bakarak min_samples_leaf ın düşük değerlerinde ağacın biraz overfit olduğunu göreceksiniz.
#Ancak min_samples_leaf > 100 değerlerinde, model daha kararlı hale gelir ve eğitim test doğruluğu birleşmeye başlar.


#Tuning min_samples_split
#%%

#hiperparametre min_samples_split bir node bölmek için gerekli olan sample sayısını gösterir. Varsayılan değeri
#2 dir, yani bir düğüm 2 örneğe sahip olsa bile, yaprak düğümlerine bölünebilir
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

n_folds= 5

#5-200 arasında 20 ser artan listede min_samples_split in altermatif değerleri tutulur.
parameters= {'min_samples_split': range(5, 200, 20)}

dtree= DecisionTreeClassifier(criterion= 'gini',
                              random_state= 100)

tree= GridSearchCV(dtree,
                   parameters,
                   cv= n_folds,
                   scoring= 'accuracy',
                   return_train_score= True)
tree.fit(X_train, y_train)

#GridSearch CV scorelarını ekrana basalım
scores= tree.cv_results_
print(pd.DataFrame(scores).head())

#çeşitli min leaf değerlerine göre training ve test accuracy değerleri
plt.figure()
plt.plot(scores['param_min_samples_split'],
         scores['mean_train_score'],
         label='training accuracy')
plt.plot(scores['param_min_samples_split'],
         scores['mean_test_score'],
         label='test accuracy')

plt.xlabel('min_samples_split')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#min_samples_split in ufak değerleri için overfitting durumunun azaldığını görebiliriz. min_samples_split değeri
#arttıkça training ve test hataları dengeli bir hale geliyor


#Optimal Hiperparametreleri Bulmak Grid Search (Grid Search to Find Optimal Hyperparameters)
#Tüm hiperparametreleri kullanarak en iyi modeli bulmaya çalışalım
#%%

#parametreleri tutan obje
param_grid= {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
    'criterion': ['entropy', 'gini']}

n_folds= 5

dtree= DecisionTreeClassifier()
grid_search= GridSearchCV(estimator= dtree,
                          param_grid= param_grid,
                          cv= n_folds,
                          verbose= 1,
                          return_train_score= True)
grid_search.fit(X_train,y_train)

#cv sonuçları
cv_results= pd.DataFrame(grid_search.cv_results_)
print(cv_results)

#en iyi sonucu veren modeli ve accuracy ekrana basalım
print('best accuracy', grid_search.best_score_)
print(grid_search.best_estimator_)


#En iyi parametrelerle modeli yeniden çalıştıralım.
#%%

#optimal hyperparameters ile model oluşturalım
clf_gini= DecisionTreeClassifier(criterion= 'gini',
                                 random_state= 100,
                                 max_depth= 10,
                                 min_samples_leaf= 50,
                                 min_samples_split= 50)
clf_gini.fit(X_train, y_train)

#accuracy score
print(clf_gini.score(X_test, y_test))

#plot
dot_data= StringIO()
export_graphviz(clf_gini,
                out_file= dot_data,
                feature_names= features,
                filled= True,
                rounded= True)

graph= pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

#max_depth= 3
clf_gini= DecisionTreeClassifier(criterion= 'gini',
                                 random_state= 100,
                                 max_depth= 3,
                                 min_samples_leaf= 50,
                                 min_samples_split= 50)
clf_gini.fit(X_train, y_train)

#accuracy score
print(clf_gini.score(X_test, y_test))

#plot
dot_data= StringIO()
export_graphviz(clf_gini,
                out_file= dot_data,
                feature_names= features,
                filled= True,
                rounded= True)

graph= pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

#classification metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

y_pred= clf_gini.predict(X_test)
print(classification_report(y_test, y_pred))

#confusion matrix
print(confusion_matrix(y_test, y_pred))


