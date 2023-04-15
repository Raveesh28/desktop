#!/usr/bin/env python
# coding: utf-8

# In[842]:


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


# # Business Problem:
# - In recent years, attention has increasingly been paid to human resources (HR), since worker quality and skills represent a growth factor and a real competitive advantage for companies. After proving its mettle in sales and marketing, artificial intelligence is also becoming central to employee-related decisions within HR management. Organizational growth largely depends on staff retention. Losing employees frequently impacts the morale of the organization and hiring new employees is more expensive than retaining existing ones.
# 
# - Recruiting and retaining employees is seen by industry watchers as a tough battle for the insurance company. Churn among employees is high and it’s very easy for employees to stop working for the service on the fly or jump to another company depending on the rates.
# 
# - As the companies get bigger, **the high churn could become a bigger problem**. To find new employees, insurance company is casting a wide net, including people who don’t have cars for jobs. But this acquisition is really costly. Losing employees frequently impacts the morale of the organization and **acquiring new employees is more expensive than retaining existing ones**.
# 
# - As a data scientist with the Analytics Department of a the insurance company, **focused on employee team attrition**, we are provided with the **monthly information for a segment of employees for 2019 and 2020** and tasked **to predict whether a employee will be leaving the company or not** based on their **attributes** like:
#     - Demographics (city, age, gender etc.)
#     - Tenure information (joining date, Last Date)
#     - Historical data regarding the performance of the employee (Quarterly rating, Monthly business acquired, Designation, Salary)

# 

# ### Column Profiling:
# 
# - MMMM-YY : Reporting Date (Monthly)
# - Emp_ID : Unique id for employees
# - Age : Age of the employee
# - Gender : Gender of the employee – Male : 0, Female: 1
# - City : City Code of the employee
# - Education_Level : Education level – 0 for 10+ ,1 for 12+ ,2 for graduate
# - Salary : Monthly average Salary of the employee
# - Date Of Joining : Joining date for the employee
# - LastWorkingDate : Last date of working for the employee
# - Joining Designation : Designation of the employee at the time of joining
# - Designation : Designation of the employee at the time of reporting
# - Total Business Value : The total business value acquired by the employee in a month (negative business indicates - cancellation/refund or car EMI adjustments)
# - Quarterly Rating : Quarterly rating of the employee: 1,2,3,4,5 (higher is better)

# ### Importing required packages:

# In[843]:


import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style='whitegrid')
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
 accuracy_score, confusion_matrix, classification_report,
 roc_auc_score, roc_curve, auc,
 plot_confusion_matrix, plot_roc_curve
)


# ### Loading data into Dataframe:

# In[844]:


company_data = pd.read_csv("ola_driver_scaler.csv")
company_data


# __Summary__:
# 
# - We have 19104 data points, and 14 features.

# ### Identification of variables and data types:

# In[845]:


company_data.shape


# In[846]:


company_data.info()


# In[ ]:





# In[847]:


def feature_names(df):
    
    print(f"Columns with category datatypes (Categorical Features) are :     {list(df.select_dtypes('object').columns)}")
    print('-'*125)
    print('-'*125)
    print(f"Columns with integer and float datatypes (Numerical Features) are:     {list(df.select_dtypes(['int64','float64']).columns)}")


# In[848]:


feature_names(company_data)


# In[849]:


company_data.Gender=company_data["Gender"].astype("object")
company_data.Education_Level=company_data["Education_Level"].astype("object")


# In[850]:


company_data.info()


# ### Analysing the basic metrics:

# In[851]:


company_data.describe(include=[np.number]).transpose()


# In[852]:


company_data.describe(include = [object]).transpose()


# # Missing values:

# In[853]:


# Missing values:

def missingValue(df):
    #Identifying Missing data.
    total_null = df.isnull().sum().sort_values(ascending = False)
    percent = ((df.isnull().sum()/len(df))*100).sort_values(ascending = False)
    print(f"Total records in our data =  {df.shape[0]} where missing values are as follows:")

    missing_data = pd.concat([total_null,percent.round(2)],axis=1,keys=['Total Missing','In Percent'])
    return missing_data


# In[854]:


missing_df = missingValue(company_data)
missing_df[missing_df['Total Missing'] > 0]


# __Summary:__
# - As we can see, 91.54 % data for **`LastWorkingDate`** is missing. Now, we need to know that in order to predict if the employee will leave the company or not can be best predicted if we have last working day info available and hence although the missing percentage is high, it's not because of Null values but the employees are not planning to leave the company and hence have not provided the info.Therefore we should not remove this feature from our data, instead we need to treat this as our **target variable.**

# In[855]:


# Creating a copy of original data before proceeding further:
company_data_1 = company_data.copy()


# # Feature Engineering to prepare the data for actual analysis:

# In[856]:


company_data['Driver_ID'].unique()


# - Total 2788 unique Emp_ID 's

# In[857]:


company_data['Driver_ID'].nunique()


# - Total 2381 rows for unique employee id's and hence (2381 - 2788 ) = -407 employee ids are either repeated or not present. We will check this...

# ### Feature creation : quarterly_performance
# **Creating a column (quarterly_performance) which tells whether the quarterly rating has increased for that employee** 
# - for those whose quarterly rating has increased we assign the value 1

# In[858]:


temp_rating = company_data[['Driver_ID','Quarterly Rating']].groupby('Driver_ID').first().reset_index()


# In[859]:


temp_rating['Quarterly_Rating_first'] = temp_rating['Quarterly Rating']
temp_rating['Quarterly_Rating_last'] =  company_data[['Driver_ID','Quarterly Rating']].groupby('Driver_ID').last().reset_index()['Quarterly Rating']
temp_rating['quarterly_performance'] = np.where(temp_rating['Quarterly_Rating_last'] - temp_rating['Quarterly_Rating_first'] > 0, 1,0)
temp_rating


# In[860]:


temp_rating.drop(['Quarterly Rating','Quarterly_Rating_first','Quarterly_Rating_last'], axis= 1, inplace = True)
temp_rating


# ### Feature creation : Salary_increment
# **Creating a column (Salary_increment) which tells whether the monthly Salary has increased for that employee** 
# - for those whose monthly Salary has increased we assign the value 1

# In[861]:


temp_Income = company_data[['Driver_ID','Income']].groupby('Driver_ID').first().reset_index()
temp_Income['last'] = company_data[['Driver_ID','Income']].groupby('Driver_ID').last().reset_index()['Income']
temp_Income['Income_increment'] = np.where(temp_Income['last'] - temp_Income['Income'] > 0, 1,0)
temp_Income


# In[862]:


temp_Income.drop(['Income','last'], axis= 1, inplace = True)
temp_Income


# ### Target variable creation: 
# **Creating a column (target) which tells whether the employee has left the company.**
# - for employee whose last working day is present will have the value 1.
# - aggregatiing on **last** value of **age** for a particular drive as it will be at the end of 2020
# - **mean** aggregation is used on **Quarterly Rating**
# - **sum** aggregation is used on **Total Business Value**
# - aggregatiing on **first** value for all other features.

# In[863]:


# Creating a dictionary named "Emp_ID_dict" so that we can apply the aggregate function on the new dataset 
# with feature engineered columns and discarding old columns and columns with  Unknown field

Driver_ID_dict = {
    'MMM-YY' :'first',
    'Driver_ID' :'first',
    'Age' :'last',
    'City' : 'first',
    'Gender' :'first',
    'Education_Level' :'first',
    'Income':'first',
    'Dateofjoining' :'first',
    'LastWorkingDate' : 'last',
    'Joining Designation' :'first',
    'Grade' :'first',
    'Quarterly Rating' :'mean',
    'Total Business Value' :'sum'   
}


# In[864]:


Driver_ID_dict_df = company_data.groupby('Driver_ID').agg(Driver_ID_dict).reset_index(drop = True)
Driver_ID_dict_df


# In[865]:


# Checking if we have not dropped some of the data from target feature mistankenly:

Driver_ID_dict_df['LastWorkingDate'].nunique() == company_data['LastWorkingDate'].nunique()


# In[866]:


unique_Driver_array = Driver_ID_dict_df['Driver_ID'].unique()
unique_Driver_array


# In[867]:


cnt = 0
Driver_IDs_not_present = []
for i in range(1,2789):
    if i not in unique_Driver_array:
        Driver_IDs_not_present.append(i)
        cnt+=1
print(cnt)


# In[868]:


unique_Driver_array_org = company_data['Driver_ID'].unique()
unique_Driver_array_org


# In[869]:


cnt = 0
Driver_IDs_not_present_org = []
for i in range(1,2789):
    if i not in unique_Driver_array_org:
        Driver_IDs_not_present_org.append(i)
        cnt+=1
print(cnt)


# In[870]:


# To cross verify if we are getting all the same Driver_IDs before and after the feature engineering aggregation step.

Driver_IDs_not_present == Driver_IDs_not_present_org


# In[871]:


Driver_ID_dict_df.columns


# In[872]:


missing_df_new = missingValue(Driver_ID_dict_df)
missing_df_new[missing_df_new['Total Missing'] > 0]


# In[873]:


# Target variable creation

Driver_ID_dict_df['target'] = Driver_ID_dict_df['LastWorkingDate'].apply(lambda x: 0 if x == None else 1)


# In[874]:


Driver_ID_dict_df


# ### Merging dataframes:
# 
# Merging **temp_rating** and **temp_Salary** into **Emp_ID_dict_df** and creating final dataframe for further analysis.

# In[875]:


company_df_1 = pd.merge(Driver_ID_dict_df,temp_rating ,how='inner', on = 'Driver_ID')
company_data_final = pd.merge(company_df_1,temp_Income ,how='inner', on = 'Driver_ID')
company_data_final


# # Exploratory Data Analysis:

# In[876]:


company_data_final.info()


# In[ ]:





# In[877]:


feature_names(company_data_final)


# ## Analysing `MMM-YY`

# In[878]:


company_data_final['MMM-YY'].nunique()


# In[879]:


# Coverting to To datetime:

company_data_final['MMM-YY']=pd.to_datetime(company_data_final['MMM-YY'])
company_data_final['MMM-YY'].value_counts()


# In[880]:


company_data_2 = company_data_final.copy()

company_data_2['reporting_year'] = company_data_2['MMM-YY'].dt.year
#2
company_data_2['reporting_month'] = company_data_2['MMM-YY'].dt.month
#3
company_data_2['reporting_day_of_week'] = company_data_2['MMM-YY'].dt.day_of_week


# In[881]:


company_data_2


# In[882]:


company_data_2['reporting_year'].value_counts()


# In[883]:


company_data_2['reporting_month'].value_counts()


# ## Analysing `Dateofjoining`

# In[884]:


company_data_final['Dateofjoining'].nunique()


# In[885]:


# Coverting company_data_final to datetime:

company_data_final['Dateofjoining']=pd.to_datetime(company_data_final['Dateofjoining'])
company_data_final['Dateofjoining'].value_counts()


# In[886]:


# Coverting company_data_2 to datetime:

company_data_2['Dateofjoining']=pd.to_datetime(company_data_2['Dateofjoining'])


# In[887]:


company_data_2 = company_data_final.copy()

#1
company_data_2['joining_year'] = company_data_2['Dateofjoining'].dt.year
#2
company_data_2['joining_month'] = company_data_2['Dateofjoining'].dt.month
#3
company_data_2['joining_day_of_week'] = company_data_2['Dateofjoining'].dt.day_of_week


# ## Analysing `LastWorkingDate`

# In[888]:


company_data_final['LastWorkingDate'].nunique()


# In[889]:


# Coverting company_data_final to datetime:

company_data_final['LastWorkingDate']=pd.to_datetime(company_data_final['LastWorkingDate'])
company_data_final['LastWorkingDate'].value_counts()


# In[890]:


company_data_final.info()


# In[891]:


company_data_final.Gender=company_data_final["Gender"].astype("object")
company_data_final.Education_Level=company_data_final["Education_Level"].astype("object")


# In[892]:


company_data_final.info()


# In[893]:


company_data_2 = company_data_final.copy()
company_data_2


# In[894]:


feature_names(company_data_final)


# ## Analysing `City`

# In[895]:


company_data_final['City'].value_counts()


# ## Analysing `Driver_ID`:

# In[896]:


company_data_final['Driver_ID'].value_counts()


# In[897]:


company_data['Driver_ID'].nunique()


# ## Analysing `Age` and `Income`:

# In[898]:


company_data_final['Age'].value_counts().sort_values().head()


# In[899]:


company_data_final['Age'].max(),company_data_final['Age'].min()


# In[900]:


company_data_final['Income'].max(),company_data_final['Income'].min()


# In[901]:


bins_age=[18,25,35,45,55,65,100]  
bins_Income = [10000, 25000, 50000, 75000,100000,188418]
label1=['18-25','25-35','35-45','45-55','55-65','65-100']
label2 = ['Low Salary','Moderate Income','High Income','Very High Income','Extremely High Income']
company_data_2['Age Groups']=pd.cut(company_data_2['Age'],bins_age,labels = label1)
company_data_2['Income Groups'] = pd.cut(company_data_2['Income'],bins_Income,labels = label2)
company_data_2.head()


# ## Analysis of `Total Business Value`

# In[902]:


company_data_final['Total Business Value'].value_counts().sort_values()


# In[903]:


# COnverting to int64
company_data_final['quarterly_performance'] = company_data_final['quarterly_performance'].astype('int64')
company_data_final['Income_increment'] = company_data_final['Income_increment'].astype('int64')


# In[904]:


company_data_final.info()


# In[905]:


feature_names(company_data_final)


# In[906]:


# Checking correlation among independent variables and how they interact with each other.

fig, ax = plt.subplots(figsize=(12, 10))
fig.subplots_adjust(top=.94)

ax.set_yticklabels(ax.get_yticklabels(), ha="right", fontsize=12, weight='bold')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight='bold')

sns.heatmap(company_data_final.corr(), annot = True, fmt='.2f', linewidths=.3, ax = ax ,cmap='RdPu')
plt.show()


# - `Quarterly Rating` is higly corelated with `Total Business Value` (0.71)
# - `Joining Designation` is higly corelated with `Designation` (0.73)
# - `Income` is higly corelated with `Designation` (0.74)
# - `Joining Designation` is moderately corelated with `Income` (0.48)
# - `Income_increment` is moderately corelated with `Quarterly Rating` (0.32)
# - `quarterly_performance` is very weakly correlated with `target` (-0.41)
# - `Total Business Value` is very weakly correlated with `target` (-0.38)

# In[907]:


company_data_final


# In[908]:


def numerical_feat(df,colname,nrows=2,mcols=2,width=15,height=15):
    fig , ax = plt.subplots(nrows,mcols,figsize=(width,height))
    fig.set_facecolor("lightgrey")
    rows = 0
    for var in colname:        
        ax[rows][0].set_title("Boxplot for Outlier Detection ", fontweight="bold")
        plt.ylabel(var, fontsize=12)
        sns.boxplot(y = df[var],color='crimson',ax=ax[rows][0])
        
        # plt.subplot(nrows,mcols,pltcounter+1)
        sns.distplot(df[var],color='purple',ax=ax[rows][1])
        ax[rows][1].axvline(df[var].mean(), color='r', linestyle='--', label="Mean")
        ax[rows][1].axvline(df[var].median(), color='m', linestyle='-', label="Median")
        ax[rows][1].axvline(df[var].mode()[0], color='royalblue', linestyle='-', label="Mode")
        ax[rows][1].set_title("Outlier Detection ", fontweight="bold")
        ax[rows][1].legend({'Mean':df[var].mean(),'Median':df[var].median(),'Mode':df[var].mode()})
        rows += 1
    plt.show()


# In[909]:


numerical_cols = ['Total Business Value', 'Income', 'Age']


# In[910]:


numerical_feat(company_data_final,numerical_cols,len(numerical_cols),2,15,15)


# __Summary__:
# - We can see tons of outliers in `Total Business Value`.Also, it'sdistribution is left skewed, which tells that a minority chunk of driver contributes to vast business value aquired by employees.
# - However, Income and Age are more or less normally distributed with minimum outliers.

# In[911]:


# Frequency of each feature in percentage.
def categorical_feat(df, colnames, nrows=2,mcols=2,width=15,height=70, sortbyindex=False):
    fig , ax = plt.subplots(nrows,mcols,figsize=(width,height))  
    fig.set_facecolor(color = 'lightgrey')
    string = "Frequency of "
    rows = 0                          
    for colname in colnames:
        count = (df[colname].value_counts(normalize=True)*100)
        string += colname + ' in (%)'
        if sortbyindex:
                count = count.sort_index()
        count.plot.bar(color=sns.color_palette("flare"),ax=ax[rows][0])
        ax[rows][0].tick_params(axis='x', rotation=30)
        ax[rows][0].set_ylabel(string, fontsize=14)
        ax[rows][0].set_xlabel(colname, fontsize=14)
        
        count.plot.pie(colors = sns.color_palette("flare"),autopct='%0.0f%%',
                       textprops={'fontsize': 14},shadow = True, ax=ax[rows][1])#explode=[0.2 if colname[i] == min(colname) else 0])        
        ax[rows][0].set_title("Frequency wise " + colname, fontweight="bold")
        string = "Frequency of "
        rows += 1 


# In[912]:


categorical_cols = ['Education_Level', 'Joining Designation', 'Grade', 'target', 'quarterly_performance','Income_increment','Age Groups','Gender','Income Groups','City']


# In[913]:


for i in categorical_cols:
    print(f" Unique values in {i} are {company_data_2[i].nunique()}")


# In[914]:


categorical_feat(company_data_2,categorical_cols,len(categorical_cols),2)


# __Summary__:
# 
# - **City code with C20** has highest no.of employees
# - Majority number of employees are from high to moderate Salary groups.
# - The female employees are 41% whereas males dominates the population with 59%.
# - Most of the employees are from **25 to 35 age groups.**
# - In these two years 2019 and 2020, **only 2%** of them have got some amount of **increment in their Salary.**
# - **15% of employees** saw quarterly **performance increase** in their ratings.
# - For **32% of the employees, we have the last working day not present** whereas as **68% of them will be leaving the insurance company soon** as they have reported their last working day.
# - Designation and joining desination shows similar behaiviour.
# - The education level is in same ration for Primary school passouts, secondary school passouts and graduates.
# 

# 
# ### Datetime feature creation for model training:
# 
# - Dropping the rows for datetime object and converting them to ordered **year, month and day of the week** to fetch some **meaningful numerical input** for model training.

# In[915]:


#1
company_data_final['reporting_year'] = company_data_final['MMM-YY'].dt.year
#2
company_data_final['reporting_month'] = company_data_final['MMM-YY'].dt.month
#3
company_data_final['reporting_day_of_week'] = company_data_final['MMM-YY'].dt.day_of_week
#1
company_data_final['joining_year'] = company_data_final['Dateofjoining'].dt.year
#2
company_data_final['joining_month'] = company_data_final['Dateofjoining'].dt.month
#3
company_data_final['joining_day_of_week'] = company_data_final['Dateofjoining'].dt.day_of_week


# In[916]:


company_data_final.info()


# In[917]:


# dropping unwanted features

company_data_final.drop(['Driver_ID','MMM-YY', 'Dateofjoining', 'LastWorkingDate'], axis=1, inplace=True)


# In[918]:


# Checking for feature with non-numerical values:

from pandas.api.types import is_numeric_dtype
company_col = list(company_data_final.columns)
for col in company_col:
    if is_numeric_dtype(company_data_final[col])== False:
        print(col)


# In[919]:


company_data_final['Gender'].value_counts()


# In[920]:


company_data_final['Gender'] = company_data_final['Gender'].apply(lambda x: 0 if x == 'Male' else 1)


# In[921]:


company_data_final['Gender'].value_counts()


# In[922]:


company_data_final['Education_Level'].value_counts()


# In[923]:


#company_data_final['Education_Level'] = company_data_final['Education_Level'].map({'College':0,'Bachelor':1,'Master':2})
#company_data_final['Education_Level'].value_counts()


# #### Note: We will do target encoding for `City` after train-test split.

# In[924]:


company_data_final


# In[925]:


# target -> 1:known LWD 0: Unknown LWD

company_data_final["target"].value_counts(normalize = True)


# In[926]:


# Assigning the featurs as X and target as y
# target -> 1:known LWD 0: Unknown LWD

X= company_data_final.drop(["target"],axis =1)
y= company_data_final["target"]


# ### Splitting data into train , validation and test

# In[927]:


# Train, CV, test split
from sklearn.model_selection import train_test_split
#0.6, 0.2, 0.2 split

X_tr_cv, X_test, y_tr_cv, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_tr_cv, y_tr_cv, test_size=0.2, random_state=42)


# In[928]:


print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_CV dataset: ", X_val.shape)
print("Number transactions y_CV dataset: ", y_val.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# In[929]:


#conda install -c conda-forge category_encoders


# In[930]:


#! pip install --upDesignation category_encoders


# In[931]:


#Converting city from category to numerical via target encoding for Train data:

from category_encoders import TargetEncoder
encoder = TargetEncoder()
X_train['City'] = encoder.fit_transform(X_train['City'], y_train)


# In[932]:


#Converting city from category to numerical via target encoding for cv data:

X_val['City'] = encoder.transform(X_val['City'], y_val)


# In[933]:


#Converting city from category to numerical via target encoding for test data:

X_test['City'] = encoder.transform(X_test['City'], y_test)


# #### We have used target encoding for `City` and used the encoder object to fit_transform which fits the encoders on train set and then transforms as well but transform only transforms the encoders learnt from train set onto test and val sets.

# In[934]:


X_train


# # Model 1 - Decision Trees
# 
# ### Simple Decision Tree Implementation:
# - Using f1_score as the data is imbalanced
# - Hyper-parameter tuning with **max_depth** and **class_weight**

# In[935]:


# Hyper-pram tuning + DT model
# target(1) -> 1:known LWD (0.68)
# target(0) -> 0:Unknown LWD (0.32)


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

train_scores = []
val_scores = []

l=1
u=20
d=1
w=2.32

for depth in np.arange(l,u,d):
    clf = DecisionTreeClassifier(random_state=0, max_depth=depth, class_weight={ 0:0.68, 1:w } )
    clf.fit(X_train, y_train)
    train_y_pred = clf.predict(X_train)
    val_y_pred = clf.predict(X_val)
    train_score = f1_score(y_train, train_y_pred)
    val_score = f1_score(y_val, val_y_pred)
    train_scores.append(train_score)
    val_scores.append(val_score)


# In[936]:


import matplotlib.pyplot as plt

plt.figure()
plt.plot(list(np.arange(l,u,d)), train_scores, label="train")
plt.plot(list(np.arange(l,u,d)), val_scores, label="val")
plt.legend(loc='lower right')
plt.xlabel("Depth")
plt.ylabel("F1-Score")
plt.grid()
plt.show()


# In[937]:


best_idx = np.argmax(val_scores)
best_idx


# In[938]:


val_scores


# In[939]:


l+d*best_idx


# In[940]:


# Model with depth_best
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve

best_idx = np.argmax(val_scores)
l_best = l+d*best_idx
print(l_best)
clf = DecisionTreeClassifier(random_state=0, max_depth=l_best, class_weight={ 0:0.68, 1:w } )
clf.fit(X_train, y_train)

y_pred_val = clf.predict(X_val)
val_score = f1_score(y_val, y_pred_val)

print(val_score)

confusion_matrix(y_val, y_pred_val)


# In[941]:


test_score = clf.score(X_test, y_test) # Bydefault -> accuracy score
print(test_score)

y_pred = clf.predict(X_test)


# In[942]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix

print(f"Accuracy : {accuracy_score(y_test, y_pred)*100}%")
print(f"recall_score : {recall_score(y_test, y_pred)*100}%")
print(f"precision_score : {precision_score(y_test, y_pred)*100}%")
print(f"f1_score : {f1_score(y_test, y_pred)*100}%")
print(f"confusion_matrix :")
print(confusion_matrix(y_test, y_pred))


# In[943]:


# Predicted        Not leaving the insurance company   leaving the insurance company
# Actual
# Not leaving the insurance company  100  -TN           50   -FP
# leaving the insurance company      17   -FN           310  -TP
confusion = confusion_matrix(y_test, y_pred)
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[944]:



# Calculate the sensitivity

TP/(TP+FN)
# Calculate the specificity

TN/(TN+FP)
from sklearn.metrics import classification_report

print(f"{classification_report(y_test, y_pred, target_names=['Not leaving the insurance company','leaving the insurance company'])}")


# In[945]:


# AUC- ROC

y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.grid()
plt.title("AU-ROC Curve")
plt.show()
print(f"AUC SCORE :{auc}" )


# In[ ]:





# In[946]:


# Feature importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1] # Sort feature importances in descending order
names = [X_test.columns[i] for i in indices] # Rearrange feature names so they match the sorted feature importances
plt.figure(figsize=(15, 7)) # Create plot
plt.title("Feature Importance") # Create plot title
plt.bar(range(X_test.shape[1]), importances[indices],color=sns.color_palette("flare")) # Add bars
plt.xticks(range(X_test.shape[1]), names, rotation=90) # Add feature names as x-axis labels
plt.show() 


# # Observations from Model 1:
# 
# - The **depth best = 4** and **class_weght ratio 0 == 0.68 and 1 == 2.32** .Even on more depth it's giving same performace but the diff between train and cv predictions is high and hence we will prefer depth == 4.
# - The **f1 score** for predicting **leaving the insurance company is 0.90** 
# - The recall score for predicting **leaving the insurance company is 0.94**
# - The precision score for predicting **leaving the insurance company is 0.85**
# - The AUC score for predicting **leaving the insurance company is 0.89**
# - The **most important features** according to model 1 are :
#     - **`reporting_year`,`Total Business Value`,`reporting_month`,`quarterly_performance`**
#     - `Education_Level` and `Age` are have comparatively less/ negligible importance.
#     - All the other features can be neglected completly.

# # Model 2 - Decision Trees (GridSearchCV)
# 
# ### Another approach for Decision trees implementaion:
# - Using k-fold CV and Grid Search

# In[947]:


# Simple DT
# 5-fold CV
# Grid Search for best hyper-param
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import tree
from sklearn.model_selection import GridSearchCV

params = {
    "max_depth" : [3, 5, 7, 9],
    "max_leaf_nodes" : [15, 20, 25, 30]
}

model1 = DTC()
clf = GridSearchCV(model1, params, scoring = "f1", cv=5)

clf.fit(X_train, y_train)


# In[948]:


res = clf.cv_results_

for i in range(len(res["params"])):
    print(f"Parameters:{res['params'][i]} Mean_score: {res['mean_test_score'][i]} Rank: {res['rank_test_score'][i]}")


# In[949]:


print(clf.best_estimator_)


# In[950]:


# from sklearn.model_selection import learning_curve


# In[951]:


# Learning Curves
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, title):

    train_sizes, train_scores, test_scores,fit_times,score_times= learning_curve(estimator,X,y,return_times=True)

    fig, axes = plt.subplots(1, 1, figsize = (10, 5))

    axes.set_title(title)
    axes.plot
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")


    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(
      train_sizes,
      train_scores_mean - train_scores_std,
      train_scores_mean + train_scores_std,
      alpha=0.1,
      color="r",
    )
    axes.fill_between(
      train_sizes,
      test_scores_mean - test_scores_std,
      test_scores_mean + test_scores_std,
      alpha=0.1,
      color="g",
    )
    axes.plot(
      train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes.plot(
      train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes.legend(loc="best")

    plt.show()


# In[952]:


model_kcv = clf.best_estimator_

model_kcv.fit(X_train, y_train)

plot_learning_curve(model_kcv, X_train, y_train, "Decision Trees")

print(model_kcv.score(X_train, y_train))

# more data could help as CV-score is improving as datset size increases.


# In[953]:


# plot the decision tree
from sklearn import tree

plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(model_kcv, fontsize=10)
plt.show()


# In[954]:


#Testing on test data

test_score = model_kcv.score(X_test, y_test) # Bydefault -> accuracy score
print(test_score)

y_pred = model_kcv.predict(X_test)


# In[955]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix

print(f"Accuracy : {accuracy_score(y_test, y_pred)*100}%")
print(f"recall_score : {recall_score(y_test, y_pred)*100}%")
print(f"precision_score : {precision_score(y_test, y_pred)*100}%")
print(f"f1_score : {f1_score(y_test, y_pred)*100}%")
print(f"confusion_matrix :")
print(confusion_matrix(y_test, y_pred))


# In[956]:


# Predicted        Not leaving the insurance company   leaving the insurance company
# Actual
# Not leaving the insurance company  127  -TN           23   -FP
# leaving the insurance company      31   -FN           296  -TP


# In[957]:


from sklearn.metrics import classification_report

print(f"{classification_report(y_test, y_pred, target_names=['Not leaving the insurance company','leaving the insurance company'])}")


# In[958]:


# AUC- ROC

y_pred_proba = model_kcv.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.grid()
plt.title("AU-ROC Curve")
plt.show()


# In[959]:


# Feature importance
importances = model_kcv.feature_importances_
indices = np.argsort(importances)[::-1] # Sort feature importances in descending order
names = [X_test.columns[i] for i in indices] # Rearrange feature names so they match the sorted feature importances
plt.figure(figsize=(15, 7)) # Create plot
plt.title("Feature Importance") # Create plot title
plt.bar(range(X_test.shape[1]), importances[indices],color=sns.color_palette("flare")) # Add bars
plt.xticks(range(X_test.shape[1]), names, rotation=90) # Add feature names as x-axis labels
plt.show()


# # Observations from Model 2:
# 
# - The **max_depth=7, max_leaf_nodes=20** .Even on more depth it's giving same performace and hence we will prefer most optimal hyperparams
# - The **f1 score** for predicting **leaving the insurance company is 0.92** 
# - The recall score for predicting **leaving the insurance company is 0.91**
# - The precision score for predicting **leaving the insurance company is 0.93**
# - The AUC score for predicting **leaving the insurance company is 0.91**
# - The **most important features** according to model 1 are :
#     - **`Total Business Value`,`reporting_year`,`reporting_month`,`quarterly_performance`**
#     - `Salary` ,`joining_month` and `Age` are have comparatively less/ negligible importance.
#     - All the other features can be neglected completly.
# -  As compared to previous Model 1, we are **getting more better scores.**

# # Model 3 -  Random Forest:
# 
# ### Using example of an ensemble learning algorithm called bagging.
# 
# - We will use the Random Forest model which uses Bagging, where decision tree models with higher variance are present.
# - Using f1_score as the data is imbalanced
# - Hyper-parameter tuning with **max_depth** , **class_weight**, **max_samples**, **num of learners**.

# In[960]:


# Hyper-pram tuning + DT model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

train_scores = []
val_scores = []

l=1
u=20
d=1
w=2.32
num_learners=100
row_sampling_rate = 0.75


for depth in np.arange(l,u,d):
    clf = RandomForestClassifier(max_depth=depth, max_samples=row_sampling_rate,                                 n_estimators=num_learners, random_state=42, oob_score=True, class_weight={ 0:0.68, 1:w } )
    clf.fit(X_train, y_train)
    train_y_pred = clf.predict(X_train)
    val_y_pred = clf.predict(X_val)
    train_score = f1_score(y_train, train_y_pred)
    val_score = f1_score(y_val, val_y_pred)
    train_scores.append(train_score)
    val_scores.append(val_score)


# In[961]:


import matplotlib.pyplot as plt

plt.figure()
plt.plot(list(np.arange(l,u,d)), train_scores, label="train")
plt.plot(list(np.arange(l,u,d)), val_scores, label="val")
plt.legend(loc='lower right')
plt.xlabel("depth")
plt.ylabel("F1-Score")
plt.grid()
plt.show()


# In[962]:


# Model with depth_best
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve

best_idx = np.argmax(val_scores)
l_best = l+d*best_idx
print(f"l_best:{l_best}")
clf = RandomForestClassifier(max_depth=l_best, max_samples = row_sampling_rate,                             n_estimators=num_learners, random_state=42, class_weight={ 0:0.68, 1:w } )
clf.fit(X_train, y_train)

y_pred_val = clf.predict(X_val)
val_score = f1_score(y_val, y_pred_val)

print(f"val_score:{val_score}")

confusion_matrix(y_val, y_pred_val)


# In[963]:


#Testing on test data

test_score = clf.score(X_test, y_test) # Bydefault -> accuracy score
print(test_score)

y_pred = clf.predict(X_test)


# In[964]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix

print(f"Accuracy : {accuracy_score(y_test, y_pred)*100}%")
print(f"recall_score : {recall_score(y_test, y_pred)*100}%")
print(f"precision_score : {precision_score(y_test, y_pred)*100}%")
print(f"f1_score : {f1_score(y_test, y_pred)*100}%")
print(f"confusion_matrix :")
print(confusion_matrix(y_test, y_pred))


# In[965]:


# Predicted        Not leaving the insurance company   leaving the insurance company
# Actual
# Not leaving the insurance company  95  -TN           55   -FP
# leaving the insurance company      7   -FN           320  -TP

confusion = confusion_matrix(y_test, y_pred)
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Calculate the sensitivity

print(f"sensitivity: {np.round((TP/(TP+FN)),2)}")
# Calculate the specificity

print(f"specificity: {np.round((TN/(TN+FP)),2)}")

from sklearn.metrics import classification_report

print(f"{classification_report(y_test, y_pred, target_names=['Not leaving the insurance company','leaving the insurance company'])}")


# In[966]:


# AUC- ROC

y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.grid()
plt.title("AU-ROC Curve")
plt.show()
print(f"AUC SCORE :{auc}" )


# In[967]:


X_train.columns


# In[968]:


# Feature importance

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1] # Sort feature importances in descending order
names = [X_test.columns[i] for i in indices] # Rearrange feature names so they match the sorted feature importances
plt.figure(figsize=(15, 7)) # Create plot
plt.title("Feature Importance") # Create plot title
plt.bar(range(X_test.shape[1]), importances[indices],color=sns.color_palette("flare")) # Add bars
plt.xticks(range(X_test.shape[1]), names, rotation=90) # Add feature names as x-axis labels
plt.show()


# # Observations for Model 3:
# 
# - The **max_depth=7, max_leaf_nodes=20** .Even on more depth it's giving same performace and hence we will prefer most optimal hyperparams
# - The **f1 score** for predicting **leaving the insurance company is 0.91** 
# - The recall score for predicting **leaving the insurance company is 0.98**
# - The precision score for predicting **leaving the insurance company is 0.85**
# - The AUC score for predicting **leaving the insurance company is 0.93**
# - The **most important features** according to model 1 are :
#     - **`Total BusinessValue`,`quarterly_performance`,`reporting_month`,`QuarterlyRating`,`joining_year`,`reporting_year`,,**
#     - `Salary` ,`joining_month` ,`City`,and `Age` are have comparatively less/ negligible importance.
#     - All the other features can be neglected completly.
# -  As compared to previous Model 1 and 2, we are **getting comparatively better scores.**
# - Here using Bagging we tried to solve over-fitting problem while we will use Boosting which will be used to reduce bias.
# 

# # Actionable Insights & Recommendations:
# 
# - The model performance comparisons are as follows : 
#     -  **Model 2** (Decision Trees with k-fold CV and gridsearch) > **Model 3** (Random Forest > **Model 1** (Decision Trees).
#     
# - **`Total BusinessValue`** is one of the most important feature **in predicting if a employee is going to leave the company** or not.As the total business value acquired by the employee in a month is helping the company in generating revenue and it's obvious that it is **highly correlated with the `quarterly_performance`** (quarterly rating) which is also an important feature.Hence, the company should **focus on such top performers** and should give some kind of **felicitation/awards to keep the morale high** of employees. Also for those, **whose ratings are not good**, the company should conduct bimonthly meets (online/offline) **to educate employees about ethical/moral lessons** so that the customers wouldn't give them bad ratings as negative business indicates in cancellation/refund or car EMI adjustments which is **hampering the company in long run.**
# 
# - **`reporting_year` , `joining_year`,  and `reporting_month`** also are important features in chrning predictions as it can tell very clearly in how much time in general, a employee is leaving the company. Also, **if the employees are not regular in reporting their updates/status** , it's clear indication that **they are not interested in staying**. The company should identify such employees should aks them about the such irregularities reasons and thus can focus on **undersatnding their problems** and should resolve their queries as much as possible. 
# 
# - Feature **`Salary_increment`** stands out with Model 5 as one of the decent features. As it's eveident from the EDA, In these two years 2019 and 2020, **only 2%** of the employees have got some amount of **increment in their Salary.** Which means, if the employees are not satisfied with their progress or they are not getting the expected outcome and hence they are looking out for some other options (another company). So the **company should focus on covering the employees basic necessities** such as insurance policies, health checkups, permanent emplyoment status, etc which will avoid the employees in recognising themselves as part of finacially **unreliable gig economy** and also will **loosen some financial burden** which inturn would **motivates them to stay with insurance company for long time.**
