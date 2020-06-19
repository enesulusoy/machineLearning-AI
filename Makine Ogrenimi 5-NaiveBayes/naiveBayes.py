# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

docs= pd.read_csv('example_train1.csv')

#document kolonu cümleleri, class kolonu ise cümlelerin ait olduğu sınıfları gösterir.
print(docs)

#sınıfları 1 ve 0 olarak binary şekilde tutalım.
docs['Class']= docs.Class.map({'cinema':0, 'education':1})
print(docs)

numpy_array= docs.to_numpy()
#tüm satırlardaki cümleleri bir dizi içerisine koyuyoruz. Tek elemanlı bir dizi gibi 
#davranacaktır. Buna 'Bag of Words' denmektedir

X= numpy_array[:, 0]
Y= numpy_array[:, 1]
Y= Y.astype('int')
print('X')
print(X)
print('Y')
print(Y)

#Daha sonra CountVectorizer() sınıfıyla bag of words dizisini kelimelere parçalayacağız
from sklearn.feature_extraction.text import CountVectorizer
vec= CountVectorizer()

#CountVectorizer() classındaki fit methodu bag of words u unique (tekil) kelimelere parçalar.
#CountVectorizer() dökümanları kelimelere parçalar ve onları alfabetik sıraya göre sıralayıp indexler
vec.fit(X) 
print(vec.vocabulary_)

#and, is ve of a benzer stop word lerin silinmesi
vec= CountVectorizer(stop_words='english')
vec.fit(X)
print(vec.vocabulary_)

#feature isimlerini yazdırma ve feature sayısını ekrana bastırma
print(vec.get_feature_names())
print(len(vec.get_feature_names()))

#feature ları vector e çevirelim
X_transformed= vec.transform(X)
print(X_transformed)
#Bunun sonucunda oluşan veriyi anlayalım: 0 index numaralı cümle 2,4,6 ve 10 index numaralı kelimeleri
#birer kez içermiştir. 


#Daha anlamlı hale getirmek için kolonlara index numaralarına göre kelimeleri yerleştirelim ve
#satırlar ise dokümanları yani cümleleri temsil etmektedir.
X= X_transformed.toarray()
print(X)

#Daha da iyi anlayabilmek için teorik dersteki gibi bir dataframe oluşturalım
print(pd.DataFrame(X, columns= vec.get_feature_names()))

test_docs= pd.read_csv('example_train1.csv')
#text in column 1, classifier in column 2
print(test_docs)

#Şimdi tahmin işlemine geçelim. Aşağıdaki kodların manalarını yukarıdan biliyorsunuz.
test_docs['Class']= test_docs.Class.map({'cinema':0, 'education':1})
print(test_docs)


test_numpy_array= test_docs.to_numpy()
X_test = test_numpy_array[:, 0]
Y_test = test_numpy_array[:, 1]
Y_test = Y_test.astype('int')

print('X_test')
print(X_test)
print('Y_test')
print(Y_test)

X_test_transformed= vec.transform(X_test)
print(X_test_transformed)

X_test= X_test_transformed.toarray()
print(X_test)


#Naive Bayes
#%%

#çok terimli bir NB modeli oluşturma
from sklearn.naive_bayes import MultinomialNB

mnb= MultinomialNB()

#modelin eğitilmesi
mnb.fit(X, Y)

#olasılıkların ekrana bastırılmaı
#cinema için kelime olasılıkları
#education için kelime olasılıkları kullanılarak test verisinde kullanma
pred= mnb.predict_proba(X_test)
print(pred)

proba= mnb.predict_proba(X_test)
print('probability of test document belonging to class CINEMA', proba[:,0])
print('probability of test document belonging to class EDUCATION', proba[:,1])

df= pd.DataFrame(proba, columns= ['Cinema', 'Education'])
print(df)










