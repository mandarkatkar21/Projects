# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:00:14 2024

@author: manda
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,precision_recall_fscore_support
import warnings
import os


a1=pd.read_excel('C:\\Data_Science\\Projects\\Credit Modeling\\case_study1.xlsx')
a2=pd.read_excel('C:\\Data_Science\\Projects\\Credit Modeling\\case_study2.xlsx')

df1=a1.copy()
df2=a2.copy()



# remove null values


df1=df1.loc[df1['Age_Oldest_TL']!=-99999]   # we just removed null values i.e -99999 from df1

# columns which have null values that too above 20% of the data then we will remove that column
# from data if null values below 10k then will remove that rows

column_to_be_removed=[]

for i in df2.columns:
    if df2.loc[df2[i]==-99999].shape[0]>10000:
        column_to_be_removed.append(i)


''' now we are removed that columns because they 10000+ values are null .....
 to impute that values is quit dangerous because we are just assuming 
 that values are mean/median which is not good so better we remove that 
 particular column'''
 
 
df2=df2.drop(column_to_be_removed,axis=1)  # dropped that columns which have more 10k null values

# and also removing that rows which have -99999 value
for i in df2.columns:
    df2=df2.loc[df2[i]!=-99999]
    
# to check null value is present orr not
df1.isna().sum()
# no null values are present

# checking common columns. to merging table on the basis of columns
for i in df1.columns:
    if i in df2.columns:
        print(i)
        
# merge two different dataframes
df=pd.merge(df1,df2,how='inner',left_on=['PROSPECTID'],right_on=['PROSPECTID'])

# checking the null values
df.info()
df.isna().sum() 


# check how many columns are categorical
cat_column=[]
for i in df.columns:
    if df[i].dtype=='object':
        cat_column.append(i)
print(cat_column)

# lets check what are the unique values are present in that particular column

for i in cat_column:
    print('column name:',i)
    print(df[i].value_counts())
    print('________________________________________________')



# chi2 test
for i in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']:
    chi2,p_val, _, _=chi2_contingency(pd.crosstab(df[i],df['Approved_Flag']))
    print(i,'---',p_val)

# since all the categorical feature have p value <=0.05 then will accept all



# VIF for numerical column
numeric_columns = []
for i in df.columns:
    if df[i].dtype != 'object' and i not in ['PROSPECTID','Approved_flag']:
        numeric_columns.append(i)
    
    
    
# VIF sequentially check
vif_data = df[numeric_columns]
total_columns = vif_data.shape[1]
columns_to_be_kept=[]
column_index = 0

for i in range(0,total_columns):
    
    vif_value= variance_inflation_factor(vif_data,column_index)
    print(column_index,'----',vif_value)
    
    if vif_value <=6:
        columns_to_be_kept.append(numeric_columns[i])
        column_index = column_index + 1
        
    else:
        vif_data = vif_data.drop([numeric_columns[i]], axis=1)
    
    


# check ANOVA for columns to be kept

from scipy.stats import f_oneway

columns_to_be_kept_numerical=[]

for i in columns_to_be_kept:
    a=list(df[i])
    b=list(df['Approved_Flag'])
    
    group_p1 = [value for value, group in zip(a,b) if group == 'P1']
    group_p2 = [value for value, group in zip(a,b) if group == 'P2']
    group_p3 = [value for value, group in zip(a,b) if group == 'P3']
    group_p4 = [value for value, group in zip(a,b) if group == 'P4']
    
    f_statistic,p_value = f_oneway(group_p1,group_p2,group_p3,group_p4)
    
    if p_value<=0.05:
        columns_to_be_kept_numerical.append(i)
    
    
    
# listing all the final feature
features = columns_to_be_kept_numerical + ['MARITALSTATUS','EDUCATION','GENDER','last_prod_enq2','first_prod_enq2']
df = df[features + ['Approved_Flag']]
    
# label and one hot encoding for the categorical features
['MARITALSTATUS','EDUCATION','GENDER','last_prod_enq2','first_prod_enq2']
    

df['MARITALSTATUS'].unique()
df['EDUCATION'].unique()
df['GENDER'].unique()
df['last_prod_enq2'].unique()
df['first_prod_enq2'].unique()
    

# ordinal feature -- EDUCATION
# SSC                        :1
# 12th                       :2
# Graduate                   :3
# under graduate             :3
# post graduate              :4
# others                     :1
# Professional               :3


df.loc[df['EDUCATION']== 'SSC',['EDUCATION']]            = 1
df.loc[df['EDUCATION']== '12TH',['EDUCATION']]           = 2
df.loc[df['EDUCATION']== 'GRADUATE',['EDUCATION']]       = 3
df.loc[df['EDUCATION']== 'UNDER GRADUATE',['EDUCATION']] = 3
df.loc[df['EDUCATION']== 'POST-GRADUATE',['EDUCATION']]  = 4
df.loc[df['EDUCATION']== 'OTHERS',['EDUCATION']]         = 1
df.loc[df['EDUCATION']== 'PROFESSIONAL',['EDUCATION']]   = 3

     
    
df['EDUCATION'].value_counts()
df['EDUCATION']= df['EDUCATION'].astype(int)



df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS','GENDER','last_prod_enq2','first_prod_enq2'])


df_encoded.info()

k=df_encoded.describe()
k
# machine learing model fitting




# Data processing
# 1. Random Forest
y = df_encoded [ 'Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators = 200, random_state=42)
rf_classifier.fit(x_train, y_train)
y_pred = rf_classifier.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)
print ()
print(f'Accuracy: {accuracy}')
print ()
precision, recall, f1_score, _ = precision_recall_fscore_support (y_test, y_pred)
for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print("Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print("F1 Score: {f1_score[i]}")
    print()



# 2. xgboost
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4)


y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

xgb_classifier.fit(x_train, y_train)
y_pred = xgb_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print ()
print(f'Accuracy: {accuracy:.2f}')
print ()


precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print("Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print("F1 Score: {f1_score[i]}")
    print()    




# Decision Tree

from sklearn.tree import DecisionTreeClassifier

y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10)

dt_model.fit(x_train, y_train)

y_pred = dt_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print ()
print (f"Accuracy: {accuracy:.2f}")
print ()

precision, recall, f1_score,_ = precision_recall_fscore_support (y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print("Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print("F1 Score: {f1_score[i]}")
    print()



# XGboost is giving highest accuracy so we will do hyperparameter tuning on XGboost
y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)


# Define the hyperparameter grid

param_grid = {
    'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9],
    'learning_rate' : [0.001, 0.01, 0.1, 1],
    'max_depth' : [3, 5, 8, 10],
    'alpha' :[1, 10, 100], 
    'n_estimators' : [10,50,100]
 }



from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost import XGBClassifier
estimator=XGBClassifier(objective='multi:softmax', num_class=4)


random_search = RandomizedSearchCV(
    estimator=estimator,
    param_distributions=param_grid,
    n_iter=100,                
    scoring='accuracy',       
    cv=3,                     
    verbose=2,
    random_state=42,         
    n_jobs=-1                
)

random_search.fit(x_train,y_train)
random_search.best_params_


params_for_grid_ = {
    'colsample_bytree': [0.7],
    'learning_rate' : [0.1],
    'max_depth' : [8],
    'alpha' :[10,15,20] ,
    'n_estimators' : [100,120,150,180,220]
 }
grid=GridSearchCV(estimator=estimator, param_grid=params_for_grid_,n_jobs=-1,cv=5,verbose=2,scoring='accuracy')

grid.fit(x_train,y_train)


grid.best_params_


# now we will build XGB classifier with best parameters

xgb_final=XGBClassifier(objective='multi:softmax', num_class=4,n_estimators=150,max_depth=8,
              learning_rate=0.1,colsample_bytree=0.7,alpha=10)


xgb_final.fit(x_train,y_train)


y_pred=xgb_final.predict(x_test)
y_pred_training=xgb_final.predict(x_train)

from sklearn.metrics import classification_report,accuracy_score
print("training Accuracy")
print(classification_report(y_train, y_pred_training))
print(accuracy_score(y_train, y_pred_training))

print("testing Accuracy")
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


























