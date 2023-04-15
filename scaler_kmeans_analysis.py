#!/usr/bin/env python
# coding: utf-8

# <div style="font-family:verdana; word-spacing:1.5px;">
# <p style="background-color:#33e0ff;color:white;text-align:center;font-size:175%;padding: 10px;"> Introduction</p>
# </div>    

# <div style="font-family:verdana; word-spacing:1.5px;">
#     
# <b>Objective</b><br>
#     
#     
# We are tasked to cluster them on the basis of their job profile, company, and other features.
#     We are focused on profiling the best companies and job positions to work.
# <br>   
#     
# <b>About the Dataset</b>   <br>
# We are provided with the information for a segment of learners by Scaler, an online tech-versity. 
#     <br>
#     Working as a data scientist with the analytics vertical of Scaler, we got this dataset from the Scaler database.<br><br>
#       The dataset contains following features:
#     <ul>
#      <li> ‘Unnamed 0’- Index of the dataset
#  <li>   Email_hash- Anonymised Personal Identifiable Information (PII)
#  <li>   Company- Current employer of the learner
#  <li>   orgyear- Employment start date
#  <li>   CTC- Current CTC
#  <li>   Job_position- Job profile in the company
#  <li>   CTC_updated_year: Year in which CTC got updated (Yearly increments, Promotions)
# 
# </ul><br>
# There are 206923 data points and 8 features.
# <br><br>
#     
# <b>Concept Used :</b><br>
# <ul>
#     <li> Manual Clustering
#     <li> Unsupervised Clustering - K- means, Hierarchical Clustering
# </ul>
# 
# </div>  

# In[183]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[184]:


import re
import seaborn as sns
from matplotlib import pyplot as plt


# In[185]:


#Reading csv data
data = pd.read_csv("scaler_clustering.csv")


# In[186]:


print("Dimensions of dataset ",data.shape)

data.head()


# In[187]:


data.info()


# <b> Checking for Null Values in dataset</b>

# In[188]:


data.isna().sum()


# <div style="display:fill;
#            border-radius:5px;
#            background-color:#BDE6ED;
#            font-size:110%;
#            font-family:verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:black;"> 
#     Data contains null values in 4 columns [company, normalized_company_name, orgyear, job_position].
#     </p>
#     </div>

# <b> Checking for Duplicate Rows in dataset</b>

# In[189]:


len(data[data.duplicated()])


# <div style="display:fill;
#            border-radius:5px;
#            background-color:#BDE6ED;
#            font-size:110%;
#            font-family:verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:black;"> 
#     Data doesn't contain any duplicate rows.
#     </p>
#     </div>

# <div style="font-family:verdana; word-spacing:1.5px;">
# <p style="background-color:#33e0ff;color:white;text-align:center;font-size:175%;padding: 10px;"> Data Preprocessing</p>
# </div>    

# <b>Checking duplicated PII ids in column email_hash</b>

# In[190]:


data['email_hash'].value_counts().head(10)


# In[191]:


display(data[data['email_hash'] == 'bbace3cc586400bbc65765bc6a16b77d8913836cfc98b77c05488f02f5714a4b'])
display(data[data['email_hash'] == '6842660273f70e9aa239026ba33bfe82275d6ab0d20124021b952b5bc3d07e6c'])


# <div style="display:fill;
#            border-radius:5px;
#            background-color:#BDE6ED;
#            font-size:110%;
#            font-family:verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:black;"> 
#     Apparently for single Anonymised Personal Identifiable Information (PII) id there exists multiple rows with same joining dates and company but different job positions, this couldn't be possible.<br><bbr>
#     We will take the first row in case of duplicated PII ids.
#     </p>
#     </div>

# In[192]:


data = data.groupby('email_hash').first().reset_index()


# <b>Creating null value indicator columns (Feature Engineering)</b>

# In[193]:


for i in ['orgyear','ctc_updated_year','company_hash','job_position']:
    data[i+'_na'] = data[i].isna()


# <b>Cleaning text columns</b>

# In[194]:


data['YoE'] = data['ctc_updated_year'] - data['orgyear']


# <b>Frequency mean encoding</b>

# In[195]:


feat = 'company_hash'
data[feat] = data[feat].fillna('na')
enc_nom = (data.groupby(feat).size()) / len(data)
data[feat+'_encode'] = data[feat].apply(lambda x : enc_nom[x])

feat = 'job_position'
data[feat] = data[feat].fillna('na')
enc_nom = (data.groupby(feat).size()) / len(data)*10000
data[feat+'_encode'] = data[feat].apply(lambda x : enc_nom[x])


# <b>Reemoving Outliers from Orgyear column</b>

# In[196]:


sorted(data['orgyear'].fillna(0).astype(int).unique())


# Removing future years, as this case is impossible to happen, also removing single digit years.

# In[197]:


data = data[~data['orgyear'].isin([0,
 1,
 2,
 3,
 4,
 5,
 6,
 38,
 83,
 91,
 200,
 201,
 206,
 208,
 209,
 1900, 2023,
 2024,
 2025,
 2026,
 2027,
 2028,
 2029,
 2031,
 2101,
 2106,
 2107,
 2204,
 20165])]


# In[198]:


data = data[~(data['YoE']<0)]


# <div style="font-family:verdana; word-spacing:1.5px;">
# <p style="background-color:#33e0ff;color:white;text-align:center;font-size:175%;padding: 10px;">EDA</p>
# </div>    

# <div style="font-family:verdana; word-spacing:1.5px;">
#     <p style="text-align:center;font-size:125%;padding: 10px;"><b>Univariate Analysis</b></p>
#     </div>

# <b>Plotting Categorical Features</b>

# In[199]:


categroical_columns = [ 'company_hash','job_position','orgyear','ctc_updated_year']


# In[200]:


for i in categroical_columns:
    tmp = data.copy()
    tmp['count'] = 1
    tmp = tmp.groupby(i).sum()['count'].reset_index().sort_values('count',ascending=False).head(15)
    plt.figure(figsize=(25,8))
    sns.barplot(data=tmp,y='count',x=i).set(title=i)
    
    plt.show()
    


# <b>Plotting Continuous Features</b>

# In[201]:


sns.displot(data['ctc'],kde=True,bins=50)
plt.show()


# The plot seems to be having large range of values, let's try to scale column for visualizing.

# In[202]:


v = data['ctc']
#v = (v-v.mean())/v.std()
sns.boxplot(v)
plt.show()


# In[203]:


data.sort_values(['ctc']).iloc[1000:1020,:]


# In[204]:


data = data[data['ctc'] >702475]


# <b> Outlier Removal using IQR</b> 

# In[205]:


dftmp = data.copy()
print(dftmp.shape)
cols = ['ctc'] # one or more

Q1 = dftmp[cols].quantile(0.25)
Q3 = dftmp[cols].quantile(0.75)
IQR = Q3 - Q1

dftmp = dftmp[~((dftmp[cols] < (Q1 - 1.5 * IQR)) |(dftmp[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
print(dftmp.shape)


# In[206]:


#dftmp = dftmp[dftmp['ctc']>300000]


# In[207]:


v = dftmp['ctc']
sns.displot(v,kde=True,bins=50)
plt.show()


# In[208]:


v = dftmp['ctc']/100000
#v = (v-v.mean())/v.std()
sns.boxplot(v)
plt.show()


# In[209]:


v = np.log2(dftmp['ctc'])
sns.displot(v,kde=True,bins=20)
plt.show()


# In[210]:


dateda = dftmp.copy()


# In[ ]:





# <div style="font-family:verdana; word-spacing:1.5px;">
#     <p style="text-align:center;font-size:125%;padding: 10px;"><b>Multivariate Analysis</b></p>
#     </div>

# In[211]:


tmp = dftmp.copy()
tmp = tmp.groupby(['job_position']).max()['ctc'].reset_index().sort_values('ctc',ascending=False).head(50)
plt.figure(figsize=(20,30))
sns.barplot(data=tmp,x='ctc',y='job_position').set(title="Top Paying Jobs")
plt.show()
list(tmp['job_position'])


# In[212]:


tmp = dftmp.copy()
tmp = tmp.groupby(['company_hash']).max()['ctc'].reset_index().sort_values('ctc',ascending=False).head(50)
plt.figure(figsize=(20,30))
sns.barplot(data=tmp,x='ctc',y='company_hash').set(title="Top Paying Companies")
plt.show()

list(tmp['company_hash'])


# In[213]:


tmp = dftmp.copy()
tmp = tmp[tmp['company_hash'].isin(['vbvkgz',
 'xzntr ntwyzgrgsj xzaxv ucn rna',
 'ktgnvu',
 'ovrnoxat ntwyzgrgsj',
 'fvrbvqn rvmo',
 'sgrabvz ovwyo',
 'tqxwoogz qa mvzsvrgqt',
 'vruyvsqtu otwhqxnxto',
 'owyztxatq trtwnqxw xzaxv',
 'ozvuatvr',
 'qhmqxp xzw',
 'ojbvzntw ogenfvqt ogrhnxgzo',
 'uyxrxuo',
 'ovbohzs trtwnqgzxwo',
 'wxowg',
 'ojbvzntw',
 'bgqsvz onvzrtj',
 'amo mvzp',
 'ottpxej',
 'bxwqgogen',
 'nvnv wgzohrnvzwj otqcxwto',
 'zcxaxv',
 'bgtzsvst',
 'cbfvqt',
 'st',
 'vpvbvx ntwyzgrgsxto',
 'rxzptaxz',
 'otqcxwtzgf',
 'qvphntz',
 'zvsqv cxoxgz xzaxv uqxcvnt rxbxnta',
 'otvqo ygraxzso wgqugqvnxgz',
 'zhnvzxd',
 'bwvett',
 'onhatzn',
 'fttduvz wgzohrnvzn',
 'zvz',
 'ojzguojo xzw',
 'nhqmgyxqt',
 'yxsy eqtihtzwj nqvaxzs exqb',
 'nhqcg xzw',
 'uvjnb',
 'rgctmgzxng',
 'xpohrv',
 'mrxuuvq',
 'aqtvb11',
 'ktnv srgmvr',
 'onvzkv rxcxzs',
 'pgwy xzahonqxto',
 'xaew mvzp rna',
 'bvxaovet'])]
tmp = tmp[tmp['orgyear'] >= 2016]
tmp = tmp.groupby(['company_hash','orgyear']).max()['ctc'].reset_index().sort_values('ctc',ascending=False)
plt.figure(figsize=(15,40))
sns.barplot(data=tmp,x='ctc',y='company_hash',hue='orgyear').set(title="Top Paying Companies Change in avg pay yearwise")
plt.show()


# In[214]:


tmp = dftmp.copy()
tmp = tmp[tmp['company_hash'].isin(['vbvkgz',
 'xzntr ntwyzgrgsj xzaxv ucn rna',
 'ktgnvu',
 'ovrnoxat ntwyzgrgsj',
 'fvrbvqn rvmo',
 'sgrabvz ovwyo',
 'tqxwoogz qa mvzsvrgqt',
 'vruyvsqtu otwhqxnxto',
 'owyztxatq trtwnqxw xzaxv',
 'ozvuatvr',
 'qhmqxp xzw',
 'ojbvzntw ogenfvqt ogrhnxgzo',
 'uyxrxuo',
 'ovbohzs trtwnqgzxwo',
 'wxowg',
 'ojbvzntw',
 'bgqsvz onvzrtj',
 'amo mvzp',
 'ottpxej',
 'bxwqgogen',
 'nvnv wgzohrnvzwj otqcxwto',
 'zcxaxv',
 'bgtzsvst',
 'cbfvqt',
 'st',
 'vpvbvx ntwyzgrgsxto',
 'rxzptaxz',
 'otqcxwtzgf',
 'qvphntz',
 'zvsqv cxoxgz xzaxv uqxcvnt rxbxnta',
 'otvqo ygraxzso wgqugqvnxgz',
 'zhnvzxd',
 'bwvett',
 'onhatzn',
 'fttduvz wgzohrnvzn',
 'zvz',
 'ojzguojo xzw',
 'nhqmgyxqt',
 'yxsy eqtihtzwj nqvaxzs exqb',
 'nhqcg xzw',
 'uvjnb',
 'rgctmgzxng',
 'xpohrv',
 'mrxuuvq',
 'aqtvb11',
 'ktnv srgmvr',
 'onvzkv rxcxzs',
 'pgwy xzahonqxto',
 'xaew mvzp rna',
 'bvxaovet'])]
tmp = tmp[tmp['orgyear'] >= 2016]
tmp = tmp.groupby(['job_position','orgyear']).max()['ctc'].reset_index().sort_values('ctc',ascending=False)
plt.figure(figsize=(15,40))
sns.barplot(data=tmp,x='ctc',y='job_position',hue='orgyear').set(title="Top Paying Companies Change in avg pay yearwise")
plt.show()


# In[215]:


tmp = dftmp.copy()

tmp = tmp[tmp['orgyear'] >= 2016]
tmp = tmp.groupby(['orgyear']).mean()['ctc'].reset_index().sort_values('ctc',ascending=False).head(50)
plt.figure(figsize=(20,10))
sns.barplot(data=tmp,y='ctc',x='orgyear').set(title="Mean CTC yearwise Comparision")
plt.show()


# <div style="font-family:verdana; word-spacing:1.5px;">
#     <p style="text-align:center;font-size:175%;padding: 10px;"><b> Manual Clustering</b></p>
#     </div>

# In[216]:


grp = ['company_hash','job_position','YoE']
data_tmp1 = dateda.groupby(grp).agg({'ctc':['mean','median','min','max','count']}).reset_index()
data_tmp1.columns  = ["{} {}".format(b_, a_) if a_ not in grp else "{}".format(a_) for a_, b_ in zip(data_tmp1.columns.droplevel(1), data_tmp1.columns.droplevel(0))  ]
data_tmp1.head(100).tail(50)

datatmp = dateda.merge(data_tmp1[['company_hash', 'job_position', 'YoE', 'mean ctc']],on=['company_hash', 'job_position', 'YoE'],how='left')



col1 = 'ctc'
col2 = 'mean ctc' 
conditions  = [ datatmp[col1] > datatmp[col2], datatmp[col1] == datatmp[col2], datatmp[col1] < datatmp[col2] ]
choices     = [ 1, 2, 3 ]
    
datatmp['Designation'] = np.select(conditions, choices, default=np.nan)


# In[217]:


grp = ['company_hash','job_position']
data_tmp1 = datatmp.groupby(grp).agg({'ctc':[('mean2','mean'),'median','min','max','count']}).reset_index()
data_tmp1.columns  = ["{} {}".format(b_, a_) if a_ not in grp else "{}".format(a_) for a_, b_ in zip(data_tmp1.columns.droplevel(1), data_tmp1.columns.droplevel(0))  ]
data_tmp1.head(100).tail(50)


datatmp = datatmp.merge(data_tmp1[grp + ['mean2 ctc']],on=grp,how='left')


col1 = 'ctc'
col2 = 'mean2 ctc' 
conditions  = [ datatmp[col1] > datatmp[col2], datatmp[col1] == datatmp[col2], datatmp[col1] < datatmp[col2] ]
choices     = [ 1, 2, 3 ]
    
datatmp['Class'] = np.select(conditions, choices, default=np.nan)


# In[218]:


grp = ['company_hash']
data_tmp1 = datatmp.groupby(grp).agg({'ctc':[('mean3','mean'),'median','min','max','count']}).reset_index()
data_tmp1.columns  = ["{} {}".format(b_, a_) if a_ not in grp else "{}".format(a_) for a_, b_ in zip(data_tmp1.columns.droplevel(1), data_tmp1.columns.droplevel(0))  ]
data_tmp1.head(100).tail(50)


datatmp = datatmp.merge(data_tmp1[grp + ['mean3 ctc']],on=grp,how='left')


col1 = 'ctc'
col2 = 'mean3 ctc' 
conditions  = [ datatmp[col1] > datatmp[col2], datatmp[col1] == datatmp[col2], datatmp[col1] < datatmp[col2] ]
choices     = [ 1, 2, 3 ]
    
datatmp['Tier'] = np.select(conditions, choices, default=np.nan)


# In[219]:


datatmp['diff_desig'] = datatmp['ctc'] - datatmp['mean ctc']
datatmp['diff_class'] = datatmp['ctc'] - datatmp['mean2 ctc']
datatmp['diff_tier'] = datatmp['ctc'] - datatmp['mean3 ctc']


# <div style="font-family:verdana; word-spacing:1.5px;">
#     <p style="text-align:center;font-size:175%;padding: 10px;"><b>Answering question based on manual clustering</b></p>
#     </div>
#     

# <b>Top 10 employees (earning more than most of the employees in the company) - Tier 1 </b>

# In[220]:


datatmp[datatmp['Tier'] == 1].sort_values('diff_tier',ascending=False).head(10)[['email_hash','ctc','mean3 ctc']]


# <b>Top 10 employees of data science in Amazon / TCS etc earning more than their peers - Class 1</b>

# In[221]:


datatmp[(datatmp['Tier'] == 1)&(datatmp['Class'] == 1)&(datatmp['job_position'].isin(['Data Science Analyst','Data Scientist','Data Scientist II','Associate Data Scientist','Senior Data Scientist']))].sort_values('diff_class',ascending=False).head(10)[['email_hash','ctc','mean2 ctc']]


# <b> Bottom 10 employees of data science in Amazon / TCS etc earning less than their peers - Class 3</b>

# In[222]:


datatmp[(datatmp['Tier'] == 1)&(datatmp['Class'] == 3)&(datatmp['job_position'].isin(['Data Science Analyst','Data Scientist','Data Scientist II','Associate Data Scientist','Senior Data Scientist']))].sort_values('diff_class',ascending=True).head(10)[['email_hash','ctc','mean2 ctc']]


# <b> Bottom 10 employees (earning less than most of the employees in the company)- Tier 3</b>

# In[223]:


datatmp[datatmp['Tier'] == 3].sort_values('diff_tier',ascending=True).head(10)[['email_hash','ctc','mean3 ctc']]


# <b>Top 10 employees in ktgnvu- X department - having 5/6/7 years of experience earning more than their peers - Tier X</b>

# In[224]:


datatmp[(datatmp['YoE'].isin([5,6,7]))&(datatmp['company_hash'].isin(['ktgnvu']))].sort_values('diff_desig',ascending=False).head(10)[['email_hash','ctc','mean ctc']]


# <b> Top 10 companies (based on their CTC)</b>

# In[225]:


datatmp.groupby('company_hash').mean()['ctc'].reset_index().sort_values('ctc',ascending=False).head(10)[['company_hash','ctc']]


# <b> Top 2 positions in every company (based on their CTC)</b>

# In[226]:


tmp = datatmp[datatmp['job_position'] != 'na']
tmp = tmp.groupby(['company_hash','job_position']).mean().sort_values(['company_hash','ctc']).reset_index()
tmp = tmp.groupby('company_hash').head(2)[['company_hash','job_position']]
tmp


# <b> Top 2 positions in top Paying companies</b>

# In[227]:


tmp[tmp['company_hash'].isin(['vbvkgz',
 'xzntr ntwyzgrgsj xzaxv ucn rna',
 'ktgnvu',
 'ovrnoxat ntwyzgrgsj',
 'fvrbvqn rvmo',
 'sgrabvz ovwyo',
 'tqxwoogz qa mvzsvrgqt',
 'vruyvsqtu otwhqxnxto',
 'owyztxatq trtwnqxw xzaxv',
 'ozvuatvr',
 'qhmqxp xzw',
 'ojbvzntw ogenfvqt ogrhnxgzo',
 'uyxrxuo',
 'ovbohzs trtwnqgzxwo',
 'wxowg',
 'ojbvzntw',
 'bgqsvz onvzrtj',
 'amo mvzp',
 'ottpxej',
 'bxwqgogen',
 'nvnv wgzohrnvzwj otqcxwto',
 'zcxaxv',
 'bgtzsvst',])]


# <div style="font-family:verdana; word-spacing:1.5px;">
# <p style="background-color:#33e0ff;color:white;text-align:center;font-size:175%;padding: 10px;">Preparing data for training model(Imputation/Scaling)</p>
# </div>    

# In[228]:


data = dateda.copy()
data


# <b>Transforming ctc feature using log function</b>

# In[229]:


data['ctc_log'] = np.log2(data['ctc'])


# <b> Columns like ['normalized_company_name','job_position','email_hash','Unnamed: 0','company'] are text.<br> We can't use them during imputation, so we'll remove these columns</b>

# In[230]:


drop_cols = ['job_position','email_hash','Unnamed: 0','company_hash']
for i in drop_cols:
    try:
        data.drop([i],axis=1,inplace=True)
    except:
        print('no')


# In[231]:


data.columns


# In[232]:


data.info()


# <b>Summary Statistics</b>

# In[233]:


data.describe()


# In[234]:


data.isna().sum()


# <div style="font-family:verdana; word-spacing:1.5px;">
# <p style="background-color:#33e0ff;color:white;text-align:center;font-size:175%;padding: 10px;">Training Model</p>
# </div>    

# In[235]:


from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# <div style="font-family:verdana; word-spacing:1.5px;">
#     <p style="text-align:center;font-size:175%;padding: 10px;"><b>Kmeans clustering</b></p>
#     </div>
#     

# <div style="font-family:verdana; word-spacing:1.5px;">
#     Standardizing data before applyting unsupervised algorithm can have consequences as stated in question, 
#     <br><br>
# <p>
#      <i>"Should the observations or features first be standardized in some way?"</i>
#     </p>
#    
# </div>    
# 
#  
#  -[Page 399, Introduction to Statistiical Learning](http://www.statlearning.com/s/ISLR-Seventh-Printing.pdf)
# 

# So we will be training a model with unscaled features too.

# In[236]:


pipe_knn = Pipeline([('scaler', StandardScaler()), ('knn_imputer',  KNNImputer(n_neighbors=2, weights="uniform"))])
pipe_knn_5 = Pipeline([('scaler', StandardScaler()), ('knn_imputer',  KNNImputer(n_neighbors=5, weights="uniform"))])
pipe = Pipeline([('scaler', StandardScaler()), ('simple_imputer',  SimpleImputer(missing_values=np.nan, strategy='mean'))])
pipe_knn_pca = Pipeline([('scaler', StandardScaler()), ('knn_imputer',  KNNImputer(n_neighbors=2, weights="uniform")),('pca',PCA(n_components=8))])
pipe_unscaled = Pipeline([('knn_imputer',  KNNImputer(n_neighbors=5, weights="uniform"))])


# <b> Finding optimal num of clusters using Elbow method</b>

# In[237]:


for name,pipeline in [('KNN Immputation',pipe_knn),('KNN Imputation with (default) 5 neighbours',pipe_knn_5),('Mean Imputation ',pipe),
                      ('KNN Immputation + PCA', pipe_knn_pca),('KNN Imputation Unscaled data',pipe_unscaled )]:

    X = pipeline.fit_transform(data)
    X = pd.DataFrame(X)
    if "PCA" not in name :
        X.columns= data.columns

    sse = {}
    #sil_score = {}
    print("Running for ",name)
    for k in range(1, 30):
        #print('K :',k)
        kmeans = MiniBatchKMeans(init="k-means++",n_clusters=k,
                              random_state=0).fit(X)
        label = kmeans.labels_
        data["clusters"] = label
        #print(data["clusters"])
        sse[k] = kmeans.inertia_ 

        #sil_score[k] = silhouette_score(X, label, metric='euclidean')

    plt.figure(figsize=(14,7))
    plt.plot(list(sse.keys()), list(sse.values()),'b-',label='Sum of squared error')
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.title("Plot for "+name)
    plt.show()



# <h2><b>Insights</b></h2>
# 
# <table>
# <tbody>
# <tr style="height: 23px;">
# <td style="height: 23px;">Model</td>
# <td style="height: 23px;">n_clusters</td>
# </tr>
# <tr style="height: 23px;">
# <td style="height: 23px;">KNN Immputation</td>
# <td style="height: 23px;">&nbsp;16</td>
# </tr>
# <tr style="height: 23px;">
# <td style="height: 23px;">KNN Imputation with (default) 5 neighbours</td>
# <td style="height: 23px;">&nbsp;20</td>
# </tr>
# <tr style="height: 23px;">
# <td style="height: 23px;">Mean Imputation</td>
# <td style="height: 23px;">&nbsp;25</td>
# </tr>
# <tr style="height: 23px;">
# <td style="height: 23px;">KNN Immputation + PCA</td>
# <td style="height: 23px;">&nbsp;21</td>
# </tr>
# <tr style="height: 23.5px;">
# <td style="height: 23.5px;">KNN Imputation Unscaled data</td>
# <td style="height: 23.5px;">&nbsp;5</td>
# </tr>
# </tbody>
# </table>
# <br>
# <br>
# <b> Number of clusters is around 16-20 for scaled data, while around 5 for unscaled data</b>

# In[238]:


from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram


# <div style="font-family:verdana; word-spacing:1.5px;">
#     <p style="text-align:center;font-size:175%;padding: 10px;"><b>Agglomerative Clustering</b></p>
#     </div>
#     

# In[239]:


tmp = X.sample(frac=0.2)

tmp.shape


# In[240]:


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    
    dendrogram(linkage_matrix, **kwargs)


# In[241]:



model = AgglomerativeClustering(distance_threshold =0, n_clusters=None, compute_distances=True,linkage='average').fit(tmp)

plt.figure(figsize=(25,20))
plt.title("Hierarchical Clustering Dendrogram (Avg Linkage)")
plot_dendrogram(model, truncate_mode="level", p=6)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


# In[242]:



model = AgglomerativeClustering(distance_threshold =0, n_clusters=None, compute_distances=True,linkage='complete').fit(tmp)

plt.figure(figsize=(25,20))
plt.title("Hierarchical Clustering Dendrogram (Complete Linkage)")
plot_dendrogram(model, truncate_mode="level", p=6)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


# In[243]:



model = AgglomerativeClustering(n_clusters=17, compute_distances=True,linkage='average').fit(tmp)

plt.figure(figsize=(25,20))
plt.title("Hierarchical Clustering Dendrogram (Avg Linkage)")
plot_dendrogram(model, truncate_mode="level", p=6)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


# In[244]:



model = AgglomerativeClustering(n_clusters=17, compute_distances=True,linkage='complete').fit(tmp)

plt.figure(figsize=(25,20))
plt.title("Hierarchical Clustering Dendrogram (Complete Linkage)")
plot_dendrogram(model, truncate_mode="level", p=6)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


# In[245]:



model = AgglomerativeClustering(n_clusters=5, compute_distances=True,linkage='single').fit(tmp)

plt.figure(figsize=(25,20))
plt.title("Hierarchical Clustering Dendrogram (Single Linkage)")
plot_dendrogram(model, truncate_mode="level", p=6)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


# <h2><b>Insights</b></h2>
# 
# 
# <br><br>
# <b> Number of clusters around 2 seems optimal in most cases, while in last plot(with single linkage) number of clusters around 16 is optimal</b>

# <div style="font-family:verdana; word-spacing:1.5px;">
# <p style="background-color:#33e0ff;color:white;text-align:center;font-size:175%;padding: 10px;">Insights/ Recommendations</p>
# </div>    

# <b>Insights</b>
# 
# <ul>
#     <li> Top Paying job titles include 'Engineering Leadership',
#  'Backend Engineer',
#  'Product Manager',
#  'Program Manager',
#  'SDET',
#  'QA Engineer',
#  'Data Scientist',
#  'Android Engineer' and
#  'FullStack Engineer'.
#   <li> Top paying companies include ‘vbvkgz',
#  'xzntr ntwyzgrgsj xzaxv ucn rna',
#  'ktgnvu',
#  'ovrnoxat ntwyzgrgsj',
#  'fvrbvqn rvmo',
#  'sgrabvz ovwyo',
#  'tqxwoogz qa mvzsvrgqt',
#  'vruyvsqtu otwhqxnxto',
#  'owyztxatq trtwnqxw xzaxv',
#  'ozvuatvr',.
#   <li>  Among top paying companies, salary for these is getting lesser in recent years, G'ottpxej',
#  'bxwqgogen',
#  'nvnv wgzohrnvzwj otqcxwto',
#  'zcxaxv',
#  'bgtzsvst',.
#       <li> Among Top paying companies mean salary for these company is increasing every year, 'ktgnvu',
#  'ovrnoxat ntwyzgrgsj',
#  'fvrbvqn rvmo'
#           <li> Avg CTC seems to be decreasing with year.
# 
#  </ul>

# <b>Recommendations</b>
# 
# <ul>
# <li> Freshers who want to work on technical side should look for roles related to Backend Engineer, SDET, QA engineer, Dataa Scientist, Android Engineer,Full stack engineer to get good salaries as expirience increases.
# <li> Freshers who want best CTC should aim for companies like ‘vbvkgz', 'xzntr ntwyzgrgsj xzaxv ucn rna', 'ktgnvu', 'ovrnoxat ntwyzgrgsj', 'fvrbvqn rvmo', 'sgrabvz ovwyo', 'tqxwoogz qa mvzsvrgqt', 'vruyvsqtu otwhqxnxto', 'owyztxatq trtwnqxw xzaxv', 'ozvuatvr',.
# </ul>
# 

# In[ ]:




