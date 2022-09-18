#!/usr/bin/env python
# coding: utf-8

# In[290]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# # 1. Download the dataset: Churn_Modelling_dataset.csv
# 
# 

# # 2. Load the dataset

# In[234]:


df=pd.read_csv(r'C:\Users\vigna\Downloads\Churn_Modelling_dataset.csv',index_col=0)


# In[235]:


df.head()


# In[236]:


df.columns


# In[237]:


df.shape


# In[238]:


np.unique(df['Exited'])


# # 5. Handle the Missing values

# # There  is no null values

# In[239]:


df.isnull().sum()


# # 4. Perform descriptive statistics on the dataset.

# In[240]:


df. describe(include='all')


# # 3. Perform Below Visualizations.
# ● Univariate Analysis
# ● Bi - Variate Analysis
# ● Multi - Variate Analysis

# In[241]:


plt.scatter(df.index,df['Exited'])


# In[242]:


sns.displot(df['Balance'])


# In[243]:


sns.rugplot(df['CreditScore'])


# In[244]:


sns.scatterplot(df['CreditScore'], df['Balance'])


# In[245]:


sns.boxplot(x='IsActiveMember', y='Exited', data=df)


# In[246]:


sns.lmplot('Tenure', 'CreditScore', df, hue="Exited", fit_reg=False);


# # 7. Check for Categorical columns and perform encoding.

# In[247]:


df['Gender'].unique()


# In[248]:


df['Geography'].unique()


# In[249]:


df['Geography']=preprocessing.LabelEncoder().fit_transform(df['Geography'])


# In[250]:


df


# In[251]:


df['Gender']=df['Gender'].map({'Male':0,'Female':1})


# In[252]:


df


# # 6. Find the outliers and replace the outliers

# In[253]:


df.describe()['Balance']


# In[220]:


def impute_outliers_IQR(df):

   q1=df.quantile(0.25)

   q3=df.quantile(0.75)

   IQR=q3-q1

   upper = df[~(df>(q3+1.5*IQR))].max()

   lower = df[~(df<(q1-1.5*IQR))].min()

   df = np.where(df > upper,

       df.mean(),

       np.where(

           df < lower,

           df.mean(),

           df

           )

       )

   return df


# In[221]:


df['Balance'] = impute_outliers_IQR(df['Balance'])


# In[222]:


df.describe()['Balance']


# In[223]:


df.describe()['EstimatedSalary']


# In[224]:


df['EstimatedSalary'] = impute_outliers_IQR(df['EstimatedSalary'])


# In[225]:


df.describe()['EstimatedSalary']


# In[254]:


def find_outliers_IQR(df):

   q1=df.quantile(0.25)

   q3=df.quantile(0.75)

   IQR=q3-q1

   outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]

   return outliers


# In[255]:


outliers = find_outliers_IQR(df['EstimatedSalary'])


# In[256]:


print(outliers)


# In[259]:


print(len(outliers))
print(outliers.max())
print(outliers.min())


# In[260]:


outliers = find_outliers_IQR(df['Balance'])


# In[261]:


print(len(outliers))
print(outliers.max())
print(outliers.min())


# In[262]:


sns.boxplot(y=df['Balance'])


# In[263]:


sns.boxplot(y=df['EstimatedSalary'])


# In[264]:


sns.boxplot(y=df['CreditScore'])


# In[265]:


outliers = find_outliers_IQR(df['CreditScore'])


# In[266]:


print(len(outliers))
print(outliers.max())
print(outliers.min())


# In[267]:


df['CreditScore'] = impute_outliers_IQR(df['CreditScore'])


# In[268]:


df.describe()['CreditScore']


# In[269]:


sns.boxplot(y=df['CreditScore'])


# # 8. Split the data into dependent and independent variables.

# In[271]:


features=list(set(df)-set(['Exited']))


# In[272]:


features


# In[273]:


x=df[features].values


# In[274]:


x


# In[275]:


y=df['Exited'].values


# In[276]:


y


# In[277]:


df[features]


# # 9. Scale the independent variables

# In[296]:


scale = StandardScaler()
x = scale.fit_transform(df[['EstimatedSalary','Balance','CreditScore']])
x


# # 10. Split the data into training and testing

# In[297]:


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)


# In[298]:


train_x,test_x,train_y,test_y


# In[ ]:




