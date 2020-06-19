# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import neighbors

from matplotlib.colors import ListedColormap

#Gerekli veri setlerinin içeriye alınması
churn_data= pd.read_csv('churn_data.csv')
customer_data= pd.read_csv('customer_data.csv')
internet_data= pd.read_csv('internet_data.csv')

#Veri setlerinin customerID den inner join ile birleştirilmesi
df_1= pd.merge(churn_data, customer_data, how= 'inner', on= 'customerID')
telecom= pd.merge(df_1, internet_data, how= 'inner', on= 'customerID')


#Veri Setinin İncelenmesi
#%%

#ilk 5 satır, Churn kolunu bizim tahminlememiz gereken bağımlı değişkendir
print(telecom.head())

#veri setinin istatiksel özeti
print(telecom.describe())

#veri setinin değişken bazında veri tipleri ve özellikleri
print(telecom.info())


#Veri Hazırlama (Data Preparation)
#%%

#yes ve no değerlerinini 1 ve 0 lara dönüştürülmesi
telecom['PhoneService']= telecom['PhoneService'].map({'Yes':1, 'No':0})
telecom['PaperlessBilling']= telecom['PaperlessBilling'].map({'Yes':1, 'No':0})
telecom['Churn']= telecom['Churn'].map({'Yes':1, 'No':0})
telecom['Partner']= telecom['Partner'].map({'Yes':1, 'No':0})
telecom['Dependents']= telecom['Dependents'].map({'Yes':1, 'No':0})

#Dummy Variable oluşturma
#%%

#kategorik değişkenler için dummy variable oluşturulması ve 1 tanesinin veriden çıkartılması
cont= pd.get_dummies(telecom['Contract'], prefix='Contract', drop_first= True)
telecom= pd.concat([telecom, cont], axis= 1)

pm= pd.get_dummies(telecom['PaymentMethod'], prefix='PaymentMethod', drop_first= True)
telecom= pd.concat([telecom, pm], axis= 1)

gen= pd.get_dummies(telecom['gender'], prefix='gender', drop_first= True)
telecom= pd.concat([telecom, gen], axis= 1)

ml= pd.get_dummies(telecom['MultipleLines'], prefix='MultipleLines')
ml1= ml.drop(['MultipleLines_No phone service'], 1)
telecom= pd.concat([telecom, ml1], axis= 1)

iser= pd.get_dummies(telecom['InternetService'], prefix='InternetService', drop_first= True)
telecom= pd.concat([telecom, iser], axis= 1)

os= pd.get_dummies(telecom['OnlineSecurity'], prefix='OnlineSecurity')
os1= os.drop(['OnlineSecurity_No internet service'], 1)
telecom= pd.concat([telecom, os1], axis= 1)

ob= pd.get_dummies(telecom['OnlineBackup'], prefix='OnlineBackup')
ob1= ob.drop(['OnlineBackup_No internet service'], 1)
telecom= pd.concat([telecom, ob1], axis= 1)

dp= pd.get_dummies(telecom['DeviceProtection'], prefix='DeviceProtection')
dp1= dp.drop(['DeviceProtection_No internet service'], 1)
telecom= pd.concat([telecom, dp1], axis= 1)

ts= pd.get_dummies(telecom['TechSupport'], prefix='TechSupport')
ts1= ts.drop(['TechSupport_No internet service'], 1)
telecom= pd.concat([telecom, ts1], axis= 1)

st= pd.get_dummies(telecom['StreamingTV'], prefix='StreamingTV')
st1= st.drop(['StreamingTV_No internet service'], 1)
telecom= pd.concat([telecom, st1], axis= 1)

sm= pd.get_dummies(telecom['StreamingMovies'], prefix='StreamingMovies')
sm1= sm.drop(['StreamingMovies_No internet service'], 1)
telecom= pd.concat([telecom, sm1], axis= 1)


#Tekrarlanan değişkenleri silelim
#%%
print(telecom.columns)
#dummy variable ları oluşturduğumuz orjinal kolonları çıkartabiliriz
telecom= telecom.drop(['Contract','PaymentMethod','gender','MultipleLines',
                       'InternetService','OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection','TechSupport', 'StreamingTV',
                       'StreamingMovies'], 1)

#total charges alanını stringden float a çeviriyoruz
#telecom['TotalCharges'] = telecom['TotalCharges'].convert_objects(convert_numeric=True)
telecom['TotalCharges']= pd.to_numeric(telecom['TotalCharges'], errors='coerce')
print(telecom.info())


#Aykırı Değerleri Kontrol Etme (Checking for Outliers)
num_telecom= telecom[['tenure','MonthlyCharges','SeniorCitizen','TotalCharges']]

#Outlier ları görebilmek için verinin quartil değerlerini kullanıyoruz
#görüldüğü üzere sürekli değişkenler gayet normal biçimde artış gösteriyor
#ve aykırı ya da uç değerler gözükmüyor.
#%99 quartil değerinden sonraki max değeri birbirine çok yakın bu da quartil olmadığını gösteriyor
print(num_telecom.describe(percentiles= [.25,.5,.75,.90,.95,.99]))

#boş değerleri kontrol edelim
print(telecom.isnull().sum())

#yüzeysel olarak değişkenşerin null ya da nan miktarının gösterilmesi
print(round(100*(telecom.isnull().sum()/len(telecom.index)), 2))

#'TotalCharges' kolunundan kayıp değerlerin silinmesi
telecom= telecom[~np.isnan(telecom['TotalCharges'])]

#kayıp değerler silindikten sonra verinin tekrar kayıp değer durumunun ekrana bastırılması
print(round(100*(telecom.isnull().sum()/len(telecom.index)), 2))


#Değişkenlerin Normalizasyonu (Feature Standardisation)
#%%

#sürekli değişkenlerin normalize edilmesi (Regression olduğu için)
df= telecom[['tenure','MonthlyCharges','TotalCharges']]
normalize_df= (df-df.mean()) / df.std()

telecom= telecom.drop(['tenure', 'MonthlyCharges','TotalCharges'], 1)
telecom= pd.concat([telecom, normalize_df], axis= 1)
print(telecom)

#Churn rate kontrol etme
churn= (sum(telecom['Churn'])/len(telecom['Churn'].index))*100
print(churn)
#churn değerleri 1 ve 0 olduğu için toplayabilir ve toplam satır sayısına bölünebilir
#Toplamda %27 churn rate olduğunu görebiliyoruz


#Model Oluşturma
#%%

#Verinin training ve test sete bölünmesi
from sklearn.model_selection import train_test_split

#customerID bir bilgi içermediği için siliir ve X değişkenine bağımısız değişkenler atanır
X= telecom.drop(['Churn','customerID'], axis=1)

#bağımlı değişken ise y değişkenine atanır
y= telecom['Churn']

print(y.head())

#trainig ve sete bölme
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    train_size= 0.7,
                                                    test_size= 0.3,
                                                    random_state= 100)

#ilk training model oluşturulması
import statsmodels.api as sm

#logistic regresyon model
logm1= sm.GLM(y_train, (sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()
print(logm1.fit().summary())

#Correlation Matrix (korelasyon matrisi)
#%%

import matplotlib.pyplot as plt
import seaborn as sns

#Correlation matrix oluşturulmasu
plt.figure(figsize=(20,10))
sns.heatmap(telecom.corr(), annot= True)
#çok açıklayıcı görünmüyor

#yüksek korelasyona sahip bağımsız değişkenlerin hem test hem train setlerden çıkartılması
print(X_test.columns)
X_test2= X_test.drop(['MultipleLines_No','OnlineSecurity_No',
                      'OnlineBackup_No','DeviceProtection_No',
                      'TechSupport_No','StreamingTV_No','StreamingMovies_No'], 1)

X_train2= X_train.drop(['MultipleLines_No','OnlineSecurity_No',
                      'OnlineBackup_No','DeviceProtection_No',
                      'TechSupport_No','StreamingTV_No','StreamingMovies_No'], 1)

#Tekrar Correlation Matrix

#yüksek korelasyon değişkenlerinin çıkartılmasında sonra tekrar bakalım
plt.figure(figsize= (20,10))
sns.heatmap(X_train2.corr(), annot= True)

#modeli tekrardan çalıştıralım
logm2= sm.GLM(y_train, (sm.add_constant(X_train2)), family=sm.families.Binomial())
logm2.fit().summary()
print(logm2.fit().summary())


#RFE ile feature seçimi
#%%

from sklearn.linear_model import LogisticRegression
logreg= LogisticRegression()

from sklearn.feature_selection import RFE
rfe= RFE(logreg, 13) #en önemli 13 değişkenin seçilmesi
rfe= rfe.fit(X, y)

print(rfe.support_) #RFE sonuçlarının ekrana bastırılması
print(rfe.ranking_)
print(telecom.columns)

#RFE tarafından seçilmiş bağımsız değişkenler 
col= ['PhoneService','PaperlessBilling','Contract_One year',
      'Contract_Two year','PaymentMethod_Electronic check',
      'MultipleLines_No','InternetService_Fiber optic',
      'InternetService_No','OnlineSecurity_Yes','TechSupport_Yes',
      'StreamingMovies_No','tenure','TotalCharges']

#RFE tarafından seçilen bağımsız değişkenler ile modelin tekrar çalıştırılması
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logsk= LogisticRegression()
logsk.fit(X_train[col], y_train)

#Modelin tekrar incelenmesi, İstatistiksel olarak önemli yani p-value su 
#0.05 den düşük bağımsız değişkenleri elde edebildik
logm4= sm.GLM(y_train, (sm.add_constant(X_train[col])), family= sm.families.Binomial())
logm4.fit().summary()
print(logm4.fit().summary())


#VIF değerleri ile yine de multi collinearity i önlemek için bağımsız değişkenleri kontrol edeceğiz
def vif_cal(input_data, dependent_col):
    vif_df= pd.DataFrame(columns= ['Var', 'Vif'])
    x_vars= input_data.drop([dependent_col], axis=1)
    xvar_names= x_vars.columns

    for i in range(0, xvar_names.shape[0]):
        y= x_vars[xvar_names[i]]
        x= x_vars[xvar_names.drop(xvar_names[i])]
        
        rsq= sm.OLS(y, x).fit().rsquared
        vif= round(1/(1-rsq),2)
        vif_df.loc[i]= [xvar_names[i], vif]
        
    return vif_df.sort_values(by= 'Vif', axis= 0, ascending= False, inplace= False)    
      
vif_deger= vif_cal(input_data= telecom.drop(['customerID','SeniorCitizen','Partner',
                                  'Dependents','PaymentMethod_Credit card (automatic)',
                                  'PaymentMethod_Mailed check','gender_Male','MultipleLines_Yes',
                                  'OnlineSecurity_No', 'OnlineBackup_No','OnlineBackup_Yes',
                                  'DeviceProtection_No','DeviceProtection_Yes','TechSupport_No',
                                  'StreamingTV_No','StreamingTV_Yes','StreamingMovies_Yes',
                                  'MonthlyCharges'], axis= 1), dependent_col= 'Churn')
        
print(vif_deger)        
        
#phone service değişkenin modelden çıkartılması
col= ['PaperlessBilling','Contract_One year',
      'Contract_Two year','PaymentMethod_Electronic check',
      'MultipleLines_No','InternetService_Fiber optic',
      'InternetService_No','OnlineSecurity_Yes','TechSupport_Yes',
      'StreamingMovies_No','tenure','TotalCharges']        
        
logm5= sm.GLM(y_train, (sm.add_constant(X_train[col])), family= sm.families.Binomial())        
logm5.fit().summary()        
print(logm5.fit().summary())        
        
        
#Tekrar VİF değerlerini hesaplayalım
vif_deger1= vif_cal(input_data= telecom.drop(['customerID','PhoneService','SeniorCitizen','Partner',
                                  'Dependents','PaymentMethod_Credit card (automatic)',
                                  'PaymentMethod_Mailed check','gender_Male','MultipleLines_Yes',
                                  'OnlineSecurity_No', 'OnlineBackup_No','OnlineBackup_Yes',
                                  'DeviceProtection_No','DeviceProtection_Yes','TechSupport_No',
                                  'StreamingTV_No','StreamingTV_Yes','StreamingMovies_Yes',
                                  'MonthlyCharges'], axis= 1), dependent_col= 'Churn')
        
print(vif_deger1)       
        
#şuanki değişkenler ile tekrar model oluşturalım
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logsk= LogisticRegression()
logsk.fit(X_train[col], y_train)      


#Tahmin Yapma (Making Predictions)
#%%

#tahminlenmiş değerlerin olasılıkları train set için
y_pred= logsk.predict_proba(X_test[col])

#array olan y pred i dataframe e dönüştürmek
y_pred_df= pd.DataFrame(y_pred)

#sadece 1 kolonlu dataframe e çevirmek
y_pred_1= y_pred_df.iloc[:, [1]]

#ilk 5 satırı görmek
print(y_pred_1.head())

#y test kısmını dataframe çevirmek
y_test_df= pd.DataFrame(y_test)

#CustId yi index olarak tutmak
y_test_df['CustID']= y_test_df.index

#hem y_pred_1 hem de y test için index kolonlarını silmek
y_pred_1.reset_index(drop= True, inplace= True)
y_test_df.reset_index(drop= True, inplace= True)

#y test ve y pred 1 i birleştirmek
y_pred_final= pd.concat([y_test_df, y_pred_1], axis= 1)

#kolonu yeniden adlandırma
y_pred_final= y_pred_final.rename(columns= {1: 'Churn_Prob'})

#kolonların sırasını değiştirmek
y_pred_final= y_pred_final.reindex(['CustID','Churn','Churn_Prob'], axis= 1)

#lets see the head of y_pred_final
print(y_pred_final.head())

#Churn Prob > 0.5 sie 1 değilse 0 döndüren hesaplanmış kolonu oluşturmak
y_pred_final['predicted']= y_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)

#ilk 5 satıra bakalım
print(y_pred_final.head())


#Model Değerlendirme (Model Evaluation)
#%%

from sklearn import metrics
#confusion matrix
confusion= metrics.confusion_matrix(y_pred_final.Churn, y_pred_final.predicted)
print(confusion)        
        
#Accuracy
dogruluk= metrics.accuracy_score(y_pred_final.Churn, y_pred_final.predicted)
print(dogruluk)

TP= confusion[1,1] #true positive
TN= confusion[0,0] #true negative
FP= confusion[0,1] #false positive
FN= confusion[1,0] #false negative

#Sensitivity
sen= TP / float(TP+FN)
print(sen)

#Specifitity
spe= TN / float(TN+FP)
print(spe)

#False Positive Rate
fpr= FP / float(TN+FP)
print(fpr)

#True Positive Rate
tpr= TP / float(TP+FP)
print(tpr)


#ROC Curve
#%%

def draw_roc(actual, probs):
    fpr, tpr, thresholds= metrics.roc_curve(actual, probs,
                                            drop_intermediate= False)
    auc_score= metrics.roc_auc_score(actual, probs)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label= 'ROC Curve (area = %0.2f' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right')
    plt.show()

    return fpr, tpr, thresholds

sonuc= draw_roc(y_pred_final.Churn, y_pred_final.predicted)
print(sonuc)

#Optimal Threshold Degerinin Bulunması
#Optimal nokta: spesificity ve sensitivity nin birbirini kestiği nokta
#%%

#farklı  threshold değerleri içeren bir dataframe in oluşturulması
numbers= [float(x)/10 for x in range(10)] 
for i in numbers:
    y_pred_final[i]= y_pred_final.Churn_Prob.map(lambda  x: 1 if x > i else 0)

print(y_pred_final.head())

 
#şimdi de farklı threshold değerleir için spesificity ve sensitivity değerlerini hesaplayalım
cutoff_df= pd.DataFrame(columns= ['prob', 'accuracy', 'sensi', 'speci'])
from sklearn.metrics import confusion_matrix

#TP= confusion[1,1] #true positive
#TN= confusion[0,0] #true negative
#FP= confusion[0,1] #false positive
#FN= confusion[1,0] #false negative

num= [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1= metrics.confusion_matrix(y_pred_final.Churn, y_pred_final[i])
    total1= sum(sum(cm1))
    accuracy= (cm1[0,0]+cm1[1,1])/total1

    speci= cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi= cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i]= [i, accuracy, sensi, speci]

print(cutoff_df)

#farklı olasılık ve threshold değerleri için sens ve specificity i gösteren grafiğin oluşturulmasu
cutoff_df.plot.line(x='prob', y=['accuracy', 'sensi', 'speci'])

#oluşturduğumuz grafiğe göre en uygun threshold değerinin 0.3 olduğunu görebiliriz (kesişim noktası)
y_pred_final['final_predicted']= y_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.3 else 0)
print(y_pred_final.head())

#tekrardan accuracy e bakalım
dogruluk= metrics.accuracy_score(y_pred_final.Churn, y_pred_final.final_predicted)
print(dogruluk)

confusion= metrics.confusion_matrix(y_pred_final.Churn, y_pred_final.final_predicted)
print(confusion)