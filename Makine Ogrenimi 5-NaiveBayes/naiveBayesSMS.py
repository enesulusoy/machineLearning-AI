# -*- coding: utf-8 -*-

import pandas as pd

#docs= pd.read_excel('SMSSpamCollection.xls', header= None, names= ['Class', 'SMS'])
docs= pd.read_table('SMSSpamCollection+(1)', header= None, names= ['Class', 'sms'])

#classifier in column 1, sms in column 2.
print(docs.head())

#ham ve spam sms lerin sayılması
ham_spam= docs.Class.value_counts()
print(ham_spam)

#Spamın yüzdesel olasılığının bulunması
print('Spam % is ', (ham_spam[1]/float(ham_spam[0]+ham_spam[1]))* 100)

#sınıfları 0 ve 1 olarak değiştiriyoruz
docs['label']= docs.Class.map({'ham':0, 'spam': 1})
print(docs.head())

#sms ve sınıfları ayrı değişkenlere atıyoruz
X= docs.sms
y= docs.label

X= docs.sms
y= docs.label
print(X.shape)  #değişkenlerin boyutlarını kontrol edelim
print(y.shape)


#Veri setini train ve test sete bölelim
#%%

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state= 1)

print(X_train.head())

#stop wordlari silerek kelimeleri vektörize ediyoruz
from sklearn.feature_extraction.text import CountVectorizer
vect= CountVectorizer(stop_words='english')

vect.fit(X_train)
#X_train_dtm= vect.transform(X_train)

#oluşan sözlüğü ekrana basalım. Kelimeler alfabetik olarak sıralanıyor ve sıraya göre index değeri alıyorlar
print(vect.vocabulary_)

#vektörleri sparse matrislere çeviriyoruz
X_train_transformed= vect.transform(X_train) 
X_test_transformed= vect.transform(X_test)

#bize sınıf olarak matrix objesi dönmektedir
print(type(X_train_transformed))
print(X_train_transformed)


#Model Oluşturma
#%%

#Native Bayes modelini çağıralım
from sklearn.naive_bayes import  MultinomialNB
mnb= MultinomialNB()

#modeli training setten öğretelim
mnb.fit(X_train_transformed, y_train)

#test setten sınıfları tahmin edelim
y_pred_class= mnb.predict(X_train_transformed)

#tahmin olasılıkları
y_pred_proba= mnb.predict_proba(X_test_transformed)

#tüm accuracy metrikleri ekrana bas
from sklearn import metrics
dogruluk= metrics.accuracy_score(y_test, y_pred_class)
print(dogruluk)
print(y_pred_class)

#alfa parametreleri laplacian smoothing için kullanılmaktadır. Olmayan kelimelerin olasılığını 0 yapmamak için
print(mnb)  

#confusion matrix
print(metrics.confusion_matrix(y_test, y_pred_class))

confusion= metrics.confusion_matrix(y_test, y_pred_class)
print(confusion)

#[row, column]
TN= confusion[0, 0]
FP= confusion[0, 1]
FN= confusion[1, 0]
TP= confusion[1, 1]

sensitivity= TP / float(FN + TP)
print('sensitivity', sensitivity)

specificity= TN / float(TN + FP)
print('specificity', specificity)

precision= TP / float(TP + FP)
print('precision', precision)
print(metrics.precision_score(y_test, y_pred_class))


print('precision', precision)
print('PRECISION SCORE :', metrics.precision_score(y_test, y_pred_class))
print('RECALL SCORE :', metrics.recall_score(y_test, y_pred_class))
print('F1 SCORE', metrics.f1_score(y_test, y_pred_class))

print(y_pred_proba)


#ROC curve oluşturma
#%%

from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt

false_positive_rate, true_positive_rate, thresholds= roc_curve(y_test, y_pred_proba[:, 1])
roc_auc= auc(false_positive_rate, true_positive_rate)

#area under the curve
print(roc_auc)
print(true_positive_rate)
print(false_positive_rate)
print(thresholds)

#TPR ve FPR ve threshold değerleri için data frame
pd.DataFrame({'Threshold': thresholds,
              'TPR': true_positive_rate,
              'FPR': false_positive_rate})


#ROC Curve' un görselleştirilmesi
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate)




