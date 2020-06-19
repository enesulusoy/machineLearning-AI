# -*- coding: utf-8 -*-

#Email Spam Classifier

import numpy as np
import pandas as pd  
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


#Verinin yüklenmesi
email_rec= pd.read_csv('Spam.txt', sep= ',', header= None)
print(email_rec.head())

#kolon isimleri sayı olarak geldiği için sütün adlarını verelim
email_rec.columns = ["word_freq_make","word_freq_address","word_freq_all",
                     "word_freq_3d","word_freq_our","word_freq_over",
                     "word_freq_remove","word_freq_internet","word_freq_order",
                     "word_freq_mail","word_freq_receive","word_freq_will",
                     "word_freq_people","word_freq_report","word_freq_addresses",
                     "word_freq_free","word_freq_business","word_freq_email",
                     "word_freq_you","word_freq_credit","word_freq_your",
                     "word_freq_font","word_freq_000","word_freq_money",
                     "word_freq_hp","word_freq_hpl","word_freq_george",
                     "word_freq_650","word_freq_lab","word_freq_labs",
                     "word_freq_telnet","word_freq_857","word_freq_data",
                     "word_freq_415","word_freq_85","word_freq_technology",
                     "word_freq_1999","word_freq_parts","word_freq_pm",
                     "word_freq_direct","word_freq_cs","word_freq_meeting",
                     "word_freq_original","word_freq_project","word_freq_re",
                     "word_freq_edu","word_freq_table","word_freq_conference",
                     "char_freq_;","char_freq_(","char_freq_[","char_freq_!",
                     "char_freq_$","char_freq_hash","capital_run_length_average",
                     "capital_run_length_longest","capital_run_length_total","spam"]

print(email_rec.head())

#verinin boyutları
print(email_rec.shape)

#verinin kolonlardaki veri tipleri
print(email_rec.info())

#veri setini boş değerler için kontrol edelim
#hiç noş değer bulunmuyor
print(email_rec.isnull().sum())

#verinin %39.4 spamdır
print(email_rec['spam'].describe())


#Veri Hazırlama (Data Preparation)
#%%

#Verinin scale edilip edilmeme kararını verebilmek için bağımsız değişkenerin ortalamasına bakılır
#Örneğin 'capital_run_length_longest' ın ortalaması 52.172 dir. Ancak 'word_frq_make' ortalaması sadece 0.104 dür
#Arada farkın olması 0-1 arasında olmaması nedeniyle scale edilmesii gereklidir
print(email_rec.describe())

#X bağımsız değişkenlerin tutulduğu değişken
#y bağımlı değişkenin integer a dönüştürülmesi
X= email_rec.drop("spam", axis=1)
y= email_rec.spam.values.astype(int)
print(y)

#veriyi standardizasyon uygulanması
from sklearn.preprocessing import scale
X= scale(X)

#verinin train ve test olarak bölünmesi
X_train, X_test, y_train, y_test= train_test_split(X,
                                                   y,
                                                   test_size= 0.3,
                                                   random_state= 4)

#hem y train hem de y test in spam non_spam dağılımının yakın olmasına dikkat edilir
print(y_train.mean())
print(y_test.mean())
#y_train ve y_test in spam ve non_spam dağılımının yakın olması istenir

#Model Oluşturma (Model Building)
#%%

#SCV class ı ve C=1 parametresi ile ilk modelin oluşturulması
model= SVC(C= 1)

#fit
model.fit(X_train, y_train)

#predict
y_pred= model.predict(X_test)
print(y_pred)

#confusion matrix oluşturmak
from sklearn import metrics

confusion= metrics.confusion_matrix(y_true= y_test, y_pred= y_pred)
print(confusion)

#Diğer metrikler

#accuracy
print('accuracy', metrics.accuracy_score(y_test,y_pred))

#precision
print('precision', metrics.precision_score(y_test, y_pred))

#recall/sensitivity
print('recall', metrics.recall_score(y_test, y_pred))

#specificity (%95 of hams correctly classified)
print('specificity', 811/(811+38))


#Sonuçların Yorumlanması:
    #92% başarıyla mailler doğru sınaflanmıştır.(accuracy)
    #88.5% spam mailler doğru olarak sınıflanmıştır.(recall)
    #specifity %95 oranda ham yani non spam mailler doğru sınıflanmıştır


#Hyperparameter Tuning (K-Fold Cross Validation)
#%%

#veriyi 5 folda bölmek
folds= KFold(n_splits= 5, shuffle= True, random_state= 4)

#C parametresini 1 set edelim
model= SVC(C= 1)

#cross-validation sonuçlarının hesaplanması
#not, cv bağımsız değişkeninin 'folds' nesnesini aldığını 
#ve metrik olarak 'accuracy' belirledik
cv_results= cross_val_score(model,
                            X_train,
                            y_train,
                            cv= folds,
                            scoring= 'accuracy')

#verinin 5 parçası içinde modelin çalıştırılması
print(cv_results)
print('mean accuracy= {}'.format(cv_results.mean()))


#Optimal C prametresini bulmak için GridSearch kullanalım
#%%

#C parametresinin farklı değerlerini bir obje de tutalım
params= {'C': [0.1, 1, 10, 100, 1000]}
model= SVC()

#GridSearch parametrelerinin doldurulması
model_cv= GridSearchCV(estimator= model,
                       param_grid= params,
                       scoring= 'accuracy',
                       cv= folds,
                       verbose= 1,
                       return_train_score= True)

#fit the model - it will fit 5 folds across all values of C
model_cv.fit(X_train, y_train)

#GridSearch sonuçlarının ekrana bastıralım
cv_results= pd.DataFrame(model_cv.cv_results_)
print(cv_results) #mean_test_score oranlarına bakılır ve en yüksek olan seçilebilir

#C parametresinin farklı değerleri için training ve test accuracy plotunu çizdirelim
plt.figure(figsize=(8, 6))
plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
plt.plot(cv_results['param_C'], cv_results['mean_train_score'])

plt.xlabel('C')
plt.ylabel('Accuracy')
plt.legend(['test accuracy', 'train accuracy'], loc= 'upper left')
plt.xscale('log')

#Grafikte görüleceği üzere c= 10^1 değerinden sonra test accuracy si düşmeye ancak train accuracysi
#artmaya devam etmiştir. Bu da bize modelin C= 10 değerinden sonra overfit olduğunu gösterir.
#C parametresi yüksek değerleri için zaten SVM de sınıflama hatalarının küçük olmasını sağlıyorduk 

#Sonuç olarak bulduğumuz optimal C değeri için modeli tekrar çalıştıralım.
best_score= model_cv.best_score_
best_C= model_cv.best_params_['C']

print('En yüksek accuracy değeri {0} at C= {1}'.format(best_score, best_C))

#C= 10 değeri için başarı metriklerine bakalım
model= SVC(C= best_C)

#fit
model.fit(X_train, y_train)

#predict
y_pred= model.predict(X_test)
print(y_pred)

#metrics
#print other metrics

#accuracy
print('accuracy', metrics.accuracy_score(y_test, y_pred))

#precision
print('precision', metrics.precision_score(y_test, y_pred))

#recall/sensitivity
print('recall', metrics.recall_score(y_test, y_pred))




