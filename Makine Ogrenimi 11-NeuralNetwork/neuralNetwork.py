# -*- coding: utf-8 -*-

#Titanic Dataseti ile Yaş Tahmini

import sklearn
import numpy as np
import pandas as pd  
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import  cross_val_score
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits

import os
import warnings
warnings.filterwarnings('ignore')

#Veriyi Anlama ve Temizleme
#%%

#verinin okunup data frame e dönüştürülemesi
train_df= pd.read_csv('train.csv')
test_df= pd.read_csv('test.csv')

#verileri birleştirme
full_df= pd.concat([train_df, test_df], ignore_index= True)

#verilerin ilk 5 satırını gözlemleyelim
print(train_df.head())
print(test_df.head())

#veri setleri hakkında bilgi edinelim
print(train_df.info())
print(test_df.info())

#dataframe çevirme
train_df= pd.DataFrame()
test_df = pd.DataFrame()

#full set için veri tiplerine bakalım
print(full_df.info())

#get train_df and test_df from full_df
def extract_df():
    tr_df= full_df.loc[full_df['Survived'].notnull()]
    print(tr_df.info)
    
    te_df= full_df.loc[full_df['Survived'].isnull()]
    print(te_df.info())
    
    return tr_df, te_df
    
train_df, test_df = extract_df()

title_sr= full_df.Name.str.extract(' ([A-Za-z]+)\.', expand= False)
full_df['Title']= title_sr

print(pd.crosstab(full_df['Title'], full_df['Sex']))
print(title_sr.value_counts())


#Simplify tilte feature
#%%

title_list=set(title_sr)

#title_list bakalım
print(title_list)

map_title_dic= {'Mlle': 'Miss',
                'Ms': 'Miss',
                'Nme': 'Mrs'}

working_dic= {}
for key in ['Lady', 'Countress', 'Capt', 'Col', 'Don',
            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
    working_dic[key]= 'Rare'

map_title_dic.update(working_dic)

full_df['Title']= full_df['Title'].replace(map_title_dic)

print(set(list(full_df['Title'])))


#Drop unwanted column
SubCol01= test_df.PassengerId
try:
    full_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis= 1, inplace= True)

except :
    print('except')
    
train_df, test_df = extract_df()

print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index= False).mean())

feature_list= list(full_df)
print(feature_list)

for feature in feature_list:
    print('----------------')
    print(feature + ' ' + str(len(full_df[feature].value_counts())))
    

#Plot
#%%

train_df.hist(bins= 'auto',
              figsize= (9, 7),
              grid= False)

print(train_df.isnull().sum())
print(test_df.isnull().sum())


#Fill NA
print(full_df['Embarked'].value_counts())

full_df['Embarked'].fillna('S', inplace=True)

full_df['Fare'].median()
full_df['Fare'].fillna(test_df['Fare'].median(), inplace= True)

train_df, test_df = extract_df()


#One Hot Encoding
#%%

full_df['Sex']= full_df['Sex'].map({'female': 0, 'male': 1}).astype(int)

def onehot(df, feature_list):
    print(df.shape)
    try:
        df= pd.get_dummies(df, columns= feature_list)
        print(df.shape)
        return df
        
    except:
        print('except')
        
    
onehot_list= ['Title', 'Pclass', 'Embarked']
full_df= onehot(full_df, onehot_list)
print(full_df)

train_df, test_df = extract_df()

print(train_df.head())
print(test_df.head())


#'Age' kolunu için ilgili işlemler
#%%

X_train_age= full_df[[x for x in list(train_df) if not x in ['Survived']]]

#train için veri bölünmesi
X_predict_age= X_train_age.loc[X_train_age['Age'].isnull()]

#boş olmayan değerleri alalım
X_train_age= X_train_age.loc[X_train_age['Age'].notnull()]

y_train_age= X_train_age.Age

try:
    X_train_age.drop('Age', axis= 1, inplace= True)
    X_predict_age.drop('Age', axis= 1, inplace= True)

except:
    print('except')

print(X_predict_age.head())
print(X_train_age.head())

#verileri scale etme
from sklearn import preprocessing
scaler2= preprocessing.StandardScaler().fit(X_train_age)

X_train_age= scaler2.transform(X_train_age)
X_predict_age= scaler2.transform(X_predict_age)

age_none_list= full_df[full_df['Age'].isnull()].index.tolist()

print(X_train_age[1])


#Model Oluşturma ve Değerlendirme
#%%

#model import etme
from sklearn.neural_network import MLPRegressor

#model oluşturma
mlr= MLPRegressor(solver= 'lbfgs',
                  alpha= 1e-5,
                  hidden_layer_sizes=(50, 50),
                  random_state= 1)

#fit
mlr.fit(X_train_age, y_train_age)

#doğruluk
print(mlr.score(X_train_age, y_train_age))

#plot
plt.figure(figsize= (20, 10))
plt.rc('font', size= 16)
plt.scatter(mlr.predict(X_train_age), y_train_age)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.show()

for a, b in zip(np.array(y_train_age), mlr.predict(X_train_age)):
    print(a, ' ', b)
    
pred= mlr.predict(X_predict_age)
print(pred)

#'Age' kolonunu dolduralım
full_df['Age'][age_none_list]= mlr.predict(X_predict_age).tolist()

#tekrar train ve test set ayrımı yapalım
train_df, test_df= extract_df()

print(full_df)


#'Survived' için sınıflama işlemi
#%%

X_train= full_df[full_df['Survived'].notnull()]
print(X_train.head())

y_train = full_df['Survived'][full_df['Survived'].notnull()]
print(y_train.head())

X_predict= full_df[full_df['Survived'].isnull()]
print(X_predict.head())

try:
    X_train.drop('Survived', axis= 1, inplace= True)
    X_predict.drop('Survived', axis= 1, inplace= True)

except:
    print('except')

#scale etme    
from sklearn import preprocessing
scaler= preprocessing.StandardScaler().fit(X_train)

X_train= scaler.transform(X_train)
X_predict= scaler.transform(X_predict)


#Model eğitme
#%%

from sklearn.neural_network import MLPClassifier

#model oluşturalım
clf= MLPClassifier(solver= 'lbfgs',
                   alpha= 1e-5,
                   hidden_layer_sizes= (50, 50),
                   random_state= 1)

#fit
clf.fit(X_train, y_train)

#doğruluk
print(clf.score(X_train, y_train))

print(clf.predict(X_train))

SubCol02= clf.predict(X_predict).astype(int)
print(SubCol02)



