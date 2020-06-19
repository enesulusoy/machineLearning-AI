# -*- coding: utf-8 -*-

#Email Spam Classifier

import numpy as np
import pandas as pd  
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale

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

#X bağımsız değişkenlerin tutulduğu değişken
#y bağımlı değişkenin integer a dönüştürülmesi
X= email_rec.drop("spam", axis=1)
y= email_rec.spam.values.astype(int)
print(y)

#veriyi standardizasyon uygulanması
X_scaled= scale(X)

#verinin train ve test olarak bölünmesi
X_train, X_test, y_train, y_test= train_test_split(X_scaled,
                                                   y,
                                                   test_size= 0.3,
                                                   random_state= 4)


#Model Oluşturma
#%%

#kernel olarak RBF seçiyoruz
model= SVC(C= 1, kernel= 'rbf')
model.fit(X_train, y_train)

#tahmin ettirme
y_pred= model.predict(X_test)

#Model Evaluation Metrics
#%%

#confusion matrix
print(confusion_matrix(y_true= y_test, y_pred= y_pred))

#accuracy
print('accuracy', metrics.accuracy_score(y_test,y_pred))

#precision
print('precision', metrics.precision_score(y_test, y_pred))

#recall/sensitivity
print('recall', metrics.recall_score(y_test, y_pred))


#Hyperparameter Tuning
#%

#GridSearch te C ve radial basis fonksiyonun (rbf) gamma parametrelerini deneyeceğiz.
#Yüksek gamma değerleri modeli non linear hale getirir
folds= KFold(n_splits= 5,
             shuffle= True,
             random_state= 4)

#Hyper parametrelerin set edilmesi
hyper_params= [{'gamma': [1e-2, 1e-3, 1e-4],
                'C': [1, 10, 100, 1000]}]

#modelin seçilmesi
model= SVC(kernel= 'rbf')

#GridSearch
model_cv= GridSearchCV(estimator= model,
                       param_grid= hyper_params,
                       scoring= 'accuracy',
                       cv= folds,
                       verbose= 1,
                       return_train_score= True)

#modelin fit edilmesi
model_cv.fit(X_train, y_train)

#GridSearch Cross Validation sonuçları
cv_results= pd.DataFrame(model_cv.cv_results_)
print(cv_results)

#
cv_results['param_C']= cv_results['param_C'].astype('int')


#plotting
#%%

plt.figure(figsize=(16, 6))

#subplot 1/3
plt.subplot(131)
gamma_01= cv_results[cv_results['param_gamma']==0.01]

plt.plot(gamma_01['param_C'], gamma_01['mean_test_score'])
plt.plot(gamma_01['param_C'], gamma_01['mean_train_score'])
plt.xlabel('X')
plt.ylabel('Accuracy')
plt.title('Gamma= 0.01')
plt.ylim([0.80, 1])
plt.legend(['test accuracy', 'train accuracy'], loc= 'upper left')
plt.xscale('log')

#subplot 2/3
plt.subplot(132)
gamma_001= cv_results[cv_results['param_gamma']==0.001]

plt.plot(gamma_001['param_C'], gamma_001['mean_test_score'])
plt.plot(gamma_001['param_C'], gamma_001['mean_train_score'])
plt.xlabel('X')
plt.ylabel('Accuracy')
plt.title('Gamma= 0.01')
plt.ylim([0.80, 1])
plt.legend(['test accuracy', 'train accuracy'], loc= 'upper left')
plt.xscale('log')

#subplot 3/3
plt.subplot(133)
gamma_0001= cv_results[cv_results['param_gamma']==0.0001]

plt.plot(gamma_0001['param_C'], gamma_0001['mean_test_score'])
plt.plot(gamma_0001['param_C'], gamma_0001['mean_train_score'])
plt.xlabel('X')
plt.ylabel('Accuracy')
plt.title('Gamma= 0.01')
plt.ylim([0.80, 1])
plt.legend(['test accuracy', 'train accuracy'], loc= 'upper left')
plt.xscale('log')

#Plotlara baktığımızda gamma değerleri düştükçe yani non linearity azaldıkça train ve test
#başarılarının daha paralel şekilde arttığı görülür. Bu nedenle non linear model kullanmamak
#daha iyidir denilebilir.

#Optimum accuracy score ve hyperparameters yazdırma
best_score= model_cv.best_score_
best_hyperparams= model_cv.best_params_

print('The best test score is {0} corresponding to heyperparameters {1}'.format(best_score, best_hyperparams))
#The best test score is 0.9338509316770185 corresponding to heyperparameters {'C': 100, 'gamma': 0.001}

#biz {'C': 100, 'gamma': 0.0001} kullanacağız. Çünkü non linearity azaldıkça train ve test
#başarılarının daha paralel şekilde arttığı görmüştük.


#Final Model Oluşturma ve Değerlendirme
#%%

#optimal parametrelerin belirlenmesi
best_params= {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'}

#model
final_model= SVC(C= 100,
                 gamma= 0.0001,
                 kernel= 'rbf') 

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

#metrics
print(metrics.confusion_matrix(y_test, y_pred), '\n')

#accuracy
print('accuracy', metrics.accuracy_score(y_test, y_pred))

#precision
print('precision', metrics.precision_score(y_test, y_pred))

#recall/sensitivity
print('recall', metrics.recall_score(y_test, y_pred))

#Conclusion
#Bu problem için non linear kernel kullanımına gerek yoktur diyebiliriz.
#Linear SVM ile de neredeyse aynı sonuçlara ulaştığımızı söyleyebiliriz.

#**Kategorik veriyle çalışırken SVM kullanmak mantıklı olmayacaktır.
#Bunun yerine Decision Trees algoritması kullanılabilir. 