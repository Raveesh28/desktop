#!/usr/bin/env python
# coding: utf-8

# # Overview of the Notebook
# 
# - **Loading and inspecting the Dataset**
#     - Checking Shape of the Dateset , Meaningful Column names
#     - Validating Duplicate Records, Checking Missing values
#     - Unique values (counts & names) for each Feature
#     - Data & Datatype validation
# - **Target variable Analysis**
#      - Checking Imbalance
# - **EDA & Pre-processing**
#      - Numerical variable
#      - Categorical variable
# - **Feature Engineering**
#      - Derving New features.
# - **Model Building**
#      - Correlation Analysis
#      - Handling Categorical variables using dummies
#      - Train, Cross validation & Test Split
#      - Imputation - HAndling missing values
#      - Rescaling features
#      - Pipeline creation
#      - Train Model using Logistic Regression
#          - Basic Model
#          - Advanced Model using Hyper Parmater optimization
#          - Advanced Model using Hyper Parmater optimization & class weights
# - **Model Performance Evaluation**
#      - AUC ROC curve
#      - Recall vs Precision
#      - F1 score
#      - Optimal cut-off using Precision-Recall Trade off
#      - Comparision between Modes on performance measures.
# - **Business Insights**

# In[ ]:





# In[357]:


#Importing packages
import numpy as np
import pandas as pd

# Importing matplotlib and seaborn for graphs
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import kstest
import statsmodels.api as sm

# Importing Date & Time util modules
from dateutil.parser import parse

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Missing Value - Calculator

# In[358]:


def missingValue(df):
    #Identifying Missing data. Already verified above. To be sure again checking.
    total_null = df.isnull().sum().sort_values(ascending = False)
    percent = ((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending = False)
    print("Total records = ", df.shape[0])

    md = pd.concat([total_null,percent.round(2)],axis=1,keys=['Total Missing','In Percent'])
    return md


# ### Numerical Variable Analysis
#  - box plot
#  - distplot

# In[359]:


def plot_num_var(df,colname,name):    
    # Visualizing our dependent variable and Skewness
    fig , (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
    fig.set_facecolor("lightgrey")

    sns.boxplot(y= colname,x='loan_status',data=df,ax=ax1)
    ax1.set_ylabel(name, fontsize=14,family = "Comic Sans MS")
    ax1.set_xlabel('Count', fontsize=14,family = "Comic Sans MS")
    ax1.set_title(name + ' by Loan Status', fontweight="bold",fontsize=15,family = "Comic Sans MS")

    sns.distplot(df[colname],color='y',ax=ax2,kde=True)
    
    mean = df[colname].mean()
    median = df[colname].median()
    mode = df[colname].mode()[0]
    
    label_mean= ("Mean :  {:.2f}".format(mean))
    label_median = ("Median :  {:.2f}".format(median))
    label_mode = ("Mode :  {:.2f}".format(mode))
    
    ax2.set_title("Distribution of " + name, fontweight="bold",fontsize=15,family = "Comic Sans MS")
    ax2.set_ylabel('Density', fontsize=12,family = "Comic Sans MS")
    ax2.set_xlabel(name, fontsize=12,family = "Comic Sans MS")
    ax2.axvline(mean,color="g",label=label_mean)
    ax2.axvline(median,color="b",label=label_median)
    ax2.axvline(mode,color="r",label=label_mode)
    ax2.legend()
    plt.show()


# ### Categorical variables
#  - Count plot
#  - Stack bar plot

# In[360]:


# Frequency of each feature in percentage.
def count_plt(df, colname, name,width=14,height=14,rotation=0):
    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor("lightgrey")
    string = "Frequency of " + name
    ax = sns.countplot(df[colname], order=sorted(df[colname].unique()), color='#56B4E9',saturation=1)

    plt.xticks(rotation = rotation,fontsize=16,family="Comic Sans MS")
    plt.yticks(fontsize=16,family="Comic Sans MS")
    plt.ylabel(string, fontsize=18,family = "Comic Sans MS")
    plt.xlabel(name, fontsize=18,family = "Comic Sans MS")
    for p in ax.patches:
        ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))


# In[361]:


def stack_bar(df,colname,name):
    cross_tab_pct = pd.crosstab(index=df[colname],
                            columns=df['loan_status'],normalize="index")
    cross_tab = pd.crosstab(index=df[colname],columns=df['loan_status'])
    
    cross_tab_pct.plot(kind='bar', stacked=True, colormap='Wistia', figsize=(10, 6))

    plt.legend(loc="upper right", ncol=2)
    plt.xlabel(name,fontsize=14,family = "Comic Sans MS")
    plt.ylabel("Loan Status",fontsize=14,family = "Comic Sans MS")
    plt.xticks(rotation=0)

    for n, x in enumerate([*cross_tab.index.values]):
        for (proportion, count, y_loc) in zip(cross_tab_pct.loc[x],
                                              cross_tab.loc[x],
                                              cross_tab_pct.loc[x].cumsum()):

            plt.text(x=n - 0.17,y=(y_loc - proportion) + (proportion / 2),
                     s=f'{count}\n({np.round(proportion * 100, 1)}%)', 
                     color="black",fontsize=12,fontweight="bold")

    plt.show()


# In[362]:


def stack_bar_h(df,colname,name):
    cross_tab_pct = pd.crosstab(index=df[colname],
                            columns=df['loan_status'],normalize="index")
    cross_tab = pd.crosstab(index=df[colname],columns=df['loan_status'])
    
    cross_tab_pct.plot(kind='barh',stacked=True, colormap='Wistia', figsize=(10, 18))

    plt.legend(loc="lower right", ncol=2)
    plt.xlabel(name,fontsize=14,family = "Comic Sans MS")
    plt.ylabel("Loan Status",fontsize=14,family = "Comic Sans MS")
    plt.xticks(rotation=0)

    for n, x in enumerate([*cross_tab.index.values]):
        for (proportion, count, y_loc) in zip(cross_tab_pct.loc[x],cross_tab.loc[x],
                                              cross_tab_pct.loc[x].cumsum()):

            plt.text(x=(y_loc - proportion) + (proportion / 2),y=n - 0.11,
                     s=f'{count}\n({np.round(proportion * 100, 1)}%)', 
                     color="black", fontsize=10,)

    plt.show()


# In[363]:


loan_data = pd.read_csv("jai_kisan_logistic_regression.csv")
loan_data.head()


# #### Validating Duplicate Records

# In[364]:


loan_data.shape


# In[365]:


loan_data.columns


# #### Validating Duplicate Records

# In[366]:


loan_data.duplicated().sum()


# In[367]:


missingValue(loan_data).head(7)


# ### Inferences
#  - There are missing values. We will handled same during EDA and Pre-Processing the data

# In[368]:


loan_data['loan_status'].unique()


# In[369]:


loan_data.info()


# ## Target variable Analysis

# In[370]:


fig, ax = plt.subplots()

labels = ['Fully Paid','Charged Off']
explode=(0.1,0)

loan_status = loan_data["loan_status"].value_counts()

df = pd.DataFrame({'labels': loan_status.index,
                   'values': loan_status.values
                  })
ax.pie(loan_status.values, explode=explode, labels=labels,  
       colors=['b','#56B4E9'], autopct='%1.4f%%', 
       shadow=True, startangle=-20,   
       pctdistance=1.3,labeldistance=1.6)

ax.axis('equal')
ax.set_title("Fully Paid vs Charges Off")
ax.legend(frameon=False, bbox_to_anchor=(1.2,0.8))


# ### Inference
#  - There are approximately 80.5% of unpaid loans, while 19% have been charged off, resulting in an imbalance in classification.

# ## Pre-Processing & EDA

# ### Numerical Variables

# #### loan_amnt

# In[371]:


loan_data[['loan_amnt']].describe().T


# In[372]:


loan_data.groupby(['loan_status'])['loan_amnt'].describe()


# In[373]:


plot_num_var(loan_data,'loan_amnt','Loan Amount')


# ### Inference
#  - Medain Loan Amount is 14113
#  - Charged-offs have a higher loan amount than fully paid with a mean loan amount of 13866 & 15126, respectively.

# ### Interest Rate

# In[374]:


loan_data[['int_rate']].describe().T


# In[375]:


loan_data.groupby(['loan_status'])['int_rate'].describe()


# In[376]:


plot_num_var(loan_data,'int_rate','Interest Rate')


# ### Inference
#  - Medain interest rate of 13%, Interest rates range from 5.32% to 30.99%.
#  - Charged-offs have a higher interest rate than fully paid with a mean interest rate of 15.88% & 13.09%, respectively.

# #### Installement

# In[377]:


loan_data[['installment']].describe().T


# In[378]:


loan_data.groupby(['loan_status'])['installment'].describe()


# In[379]:


plot_num_var(loan_data,'installment','Installment Amount')


# ### Inference
#  - Charged-offs have a slighty higher installemnt amount than Fully paid.
#  - The mean and median installation amounts for **charge-off are 452 and 399** respectively
#  - The mean and median installation amounts for **Fully Paid are 426 and 369** respectively

# #### Annual Income

# In[380]:


loan_data[['annual_inc']].describe().T


# In[381]:


loan_data.groupby(['loan_status'])['annual_inc'].describe()


# In[382]:


plot_num_var(loan_data,'annual_inc','Annual Income')


# ### Inferences
#  - Based on the above graph and table, the annual income range is very wide. We should perform some transformations, like log, to get a better picture.

# In[383]:


## trainsforming target variable using numpy.log1p, 
loan_data["annual_inc_ln"] = np.log1p(loan_data["annual_inc"])


# In[384]:


loan_data[['annual_inc_ln']].describe().T


# In[385]:


plot_num_var(loan_data,'annual_inc_ln','Annual Income')


# In[386]:


loan_data.groupby(['loan_status'])['annual_inc_ln'].describe()


# In[387]:


77673/loan_data.shape[0]


# In[388]:


318357/loan_data.shape[0]


# ### Inference
#  - In terms of individual annual income, the distribution of charged off loans is similar to that of fully paid loans, except individual with salary 0
#  - Logistic Regression models are not much impacted due to the presence of outliers because the sigmoid function tapers the outliers. But the presence of extreme outliers may somehow affect the performance of the model and lowering the performance.
#  
# ##### Note -   To improve the performance of the model we will be removing the outliers using the repetitive process of 
#   - training model and detecting and removing outliers.

# In[389]:


loan_data.drop('annual_inc_ln', axis=1, inplace=True)


# #### dti
#  -  A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested Lending club loan, divided by the borrower’s self-reported monthly income.

# In[390]:


loan_data[['dti']].describe().T


# In[391]:


loan_data.groupby(['loan_status'])['dti'].describe()


# In[392]:


plot_num_var(loan_data,'dti','Debt-To-Income-Ratio')


# In[393]:


loan_data.loc[loan_data['dti']>=50, 'loan_status'].value_counts()


# In[394]:


9/35


# In[395]:


loan_data.loc[loan_data['dti']<=10, 'loan_status'].value_counts()


# In[396]:


10850/(68242+10850)


# ### Inferences
#  - The likelihood of a loan getting charged-off increases as DTI values increase

# #### Open Credit lines
#  - The number of open credit lines in the borrower's credit file.

# In[397]:


loan_data[['open_acc']].describe().T


# In[398]:


loan_data['open_acc'].nunique()


# In[399]:


plt.figure(figsize=(10,3),dpi=100)
fig.set_facecolor("lightgrey")
sns.countplot(loan_data['open_acc'], order=sorted(loan_data['open_acc'].unique()), color='#56B4E9')
a, b = plt.xticks(np.arange(0, 90, 5), np.arange(0, 90, 5))
plt.title('Number of Open Credit Lines')
plt.show()


# #### Public record (pub_rec )
#  - Number of derogatory public records

# In[400]:


loan_data[['pub_rec']].describe().T


# In[401]:


loan_data['pub_rec'].value_counts().head(7)


# In[402]:


loan_data.loc[loan_data['pub_rec']>=1, 'loan_status'].value_counts()


# In[403]:


12334/(12334+45424)


# In[404]:


loan_data.loc[loan_data['pub_rec']>2, 'loan_status'].value_counts()


# In[405]:


611/(611+1932)


# ### Inferences
#  - As we can see that for derogatory public record have high probability of loan getting charged-off 

# #### Revolving Balance
#  - Total credit revolving balance

# In[406]:


loan_data['revol_bal'].nunique()


# In[407]:


loan_data[['revol_bal']].describe().T


# In[408]:


plot_num_var(loan_data,'revol_bal','Revolving Credit Balance')


# ### Inferences.
#  - Based on the above graph and table, the annual income range is very wide. We should perform some transformations, like log, to get a better picture.
#  - We will handle the outliers later on.

# In[409]:


## trainsforming target variable using numpy.log1p, 
loan_data["revol_bal_ln"] = np.log1p(loan_data["revol_bal"])


# In[410]:


plot_num_var(loan_data,'revol_bal_ln','Revolving Credit Balance(ln)')


# In[411]:


loan_data.groupby(['loan_status'])['revol_bal'].describe()


# In[412]:


loan_data.drop('revol_bal_ln', axis=1, inplace=True)


# #### revol_util
#  - Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.

# In[413]:


loan_data[['revol_util']].describe().T


# In[414]:


plot_num_var(loan_data,'revol_util','Revolving line utilization rate')


# ### Inferences
#  - Some outliers observered. We will remove later.

# #### total_acc 
#  - The total number of credit lines currently in the borrower's credit file

# In[415]:


loan_data[['total_acc']].describe().T


# In[416]:


loan_data.groupby(['loan_status'])['total_acc'].describe()


# In[417]:


plot_num_var(loan_data,'total_acc','Total No. of Credit lines')


# #### Inferences 
#  - Mean difference between Charged-off and Fully paid for total number of credit lines are not much.

# #### mort_acc 
#  - Number of mortgage accounts.

# In[418]:


loan_data[['mort_acc']].describe().T


# In[419]:


loan_data.groupby(['loan_status'])['mort_acc'].describe()


# In[420]:


loan_data['mort_acc'].value_counts().head(10)


# In[421]:


loan_data.loc[loan_data['mort_acc']>=10, 'loan_status'].value_counts()


# In[422]:


269/(1797+269)


# ### Inferences
#  - According to the above analysis, people with 0 Mortgage accounts have a high risk of defaulting on their loans

# #### pub_rec_bankruptcies
#  - Number of public record bankruptcies

# In[423]:


loan_data['pub_rec_bankruptcies'].value_counts().sort_index()


# In[424]:


loan_data.loc[loan_data['pub_rec_bankruptcies']>=1, 'loan_status'].value_counts()


# In[425]:


9265/(9265+35850)


# ### Inferences
#  - According to the above analysis, people with 1 or more number of public record bankruptcies have a high risk of defaulting on their loans

# ### Categorical variables

# #### Grade & Sub-Grade

# In[426]:


print(sorted(loan_data['grade'].unique()))


# In[427]:


stack_bar(loan_data,'grade',"Grade")


# In[428]:


print(sorted(loan_data['sub_grade'].unique()))


# In[429]:


stack_bar_h(loan_data,'sub_grade',"Sub Grade")


# In[430]:


count_plt(loan_data,'sub_grade','Sub Grade',width=24,height=16)


# ### Inferences
#  - Since the subgrade is implicit in the subgrade, we can ignore it.
#  - The Loan Status is directly impacted by Sub-Grade. It is likely that a sub-grade will lead to a charge-off if the grade is not good

# In[431]:


loan_data.drop('grade',axis=1,inplace=True)


# #### term

# In[432]:


loan_data['term'].value_counts()


# In[433]:


count_plt(loan_data,'term','Loan Term',width=8,height=8)


# In[434]:


stack_bar(loan_data,'term',"Loan Term")


# #### Converting to integrer value

# In[435]:


loan_data['term'] = loan_data['term'].apply(lambda term: np.int8(term.split()[0]))


# ### Inferences
#  - In comparison to **36-month (3 years) loans, 60-month (5 years)** loans have a **2x higher rate of charge-offs**.
#  - A five-year loan has a probability of **charged-off of 32%**, which is much higher than a three-year loan.

# #### emp_title

# In[436]:


loan_data['emp_title'].nunique()


# In[437]:


loan_data['emp_title'].value_counts()


# ### Inferences
#  - The two top job titles that take most loans are teacher and manager.

# In[438]:


loan_data.loc[loan_data['emp_title'] == 'Manager', 'loan_status'].value_counts()


# In[439]:


929/(3321+929)


# In[440]:


loan_data.loc[loan_data['emp_title'] == 'Technition', 'loan_status'].value_counts()


# In[441]:


(loan_data['emp_title'].nunique()/loan_data.shape[0])*100


# ### Inferences
#  -  In total, 43% of the total records has a different employee title. However, this feature is not very useful without creating categories. Thus, it has been removed.

# In[442]:


loan_data.drop('emp_title',axis=1,inplace=True)


# #### issue_d

# In[443]:


loan_data['issue_d'].value_counts(dropna=False)


# In[444]:


loan_data["issue_d"] = pd.to_datetime(loan_data['issue_d'])


# In[445]:


loan_data['issue_d'] = loan_data['issue_d'].dt.year


# In[446]:


loan_data['issue_d'].value_counts(dropna=False)


# In[447]:


stack_bar(loan_data,'issue_d',"Issue Month")


# ### Inferences
#  - Based on the issue month from year 2013 to 2015, a slight increase was noted for loan getting charged-off .
#  - Data for 2016 shows less charged off than previous years, which could be due to not being full year data.

# #### emp_length

# In[448]:


loan_data['emp_length'].value_counts(dropna=False)


# In[449]:


stack_bar(loan_data,'emp_length',"Employee Length")


# ### Inference
#  - Loan status is constant with the length of the employee. We therefore removed this feature.

# In[450]:


loan_data.drop('emp_length',axis=1,inplace=True)


# #### Home Ownership

# In[451]:


loan_data['home_ownership'].value_counts()


# ### Inferences
#  - Home Ownership Category - OTHER will be combined with NONE & ANY

# In[452]:


loan_data['home_ownership'].replace(['NONE', 'ANY'], 'OTHER', inplace=True)


# In[453]:


count_plt(loan_data,'home_ownership','Home Ownership',width=8,height=7)


# In[454]:


stack_bar(loan_data,'home_ownership','Home Ownership')


# ### Inferences
#  - We can see from the above graph that there is a high risk of Charge-off for owners and rented homes

# #### Verfication Status

# In[455]:


count_plt(loan_data,'verification_status','Verification Status',width=8,height=7)


# In[456]:


stack_bar(loan_data,'verification_status',"Verification Status")


# ### Inferences
# - Although income is verified, the charge-off rate is higher.

# #### Purpose of the loan

# In[457]:


loan_data['purpose'].value_counts()


# In[458]:


count_plt(loan_data,'purpose','Loan Purpose',width=14,height=8,rotation=60)


# In[459]:


stack_bar_h(loan_data,'purpose',"Loan Purpose")


# ### Inference
#  - When the aim of the business is to start or to invest in a small business, there is a 30% chance of getting charged-off

# #### Title

# In[460]:


loan_data['title'].nunique()


# In[461]:


loan_data['title'].value_counts().head(5)


# In[462]:


loan_data['title'].value_counts().head(5)


# In[463]:


loan_data.drop('title',axis=1,inplace=True)


# ### Inferences
#  - It appears the title is a subcategory of loan purpose. With 48K+ different sub-purposes and already capturing all the information in the purpose variable, we can remove this variable.

# #### Initial list Status 
#  - Whole loan vs Fraction purpose
#  - Initial list status indicates the initial listing status of the loan. Possible values are W, F. W stands for whole loans, that is, available to investors to be purchased in their entirety (Borrowers benefit from getting ‘instant funding’). 
#  - Lending club provides a randomized subset of loans by grade available to purchase as a whole loan for a brief period of time (12 hours). The rest are available for fractional purchase.

# In[464]:


count_plt(loan_data,'initial_list_status','Whole loan vs Fraction purchase',width=8,height=6)


# In[465]:


stack_bar(loan_data,'initial_list_status',"Whole vs Fraction")


# #### Application Type

# In[466]:


loan_data['application_type'].value_counts()


# In[467]:


count_plt(loan_data,'application_type','Application Type',width=8,height=6)


# In[468]:


stack_bar(loan_data,'application_type',"Application Type")


# ### Inference
#   - The Direct Pay Application Type has a high chance of getting charged-off. Meanwhile, joint pay has a slighty lower chance of being charged off than individual pay

# #### Address

# In[469]:


loan_data['address'].nunique()


# In[470]:


(loan_data['address'].nunique()/loan_data.shape[0])*100


# ### Inference
# - We can group the data by zipcode, which might provide us with more insights.
# - In 99% of cases, the values are different. It would be helpful if the data based on state was provided. Hence Fropping the column

# In[471]:


loan_data.shape


# #### earliest_cr_line
#  - The month the borrower's earliest reported credit line was opened

# In[472]:


loan_data['earliest_cr_line'].nunique()


# In[473]:


loan_data["earliest_cr_line"] = pd.to_datetime(loan_data['earliest_cr_line'])


# In[474]:


loan_data['earliest_cr_line'] = loan_data['earliest_cr_line'].dt.year


# In[475]:


loan_data['earliest_cr_line'].value_counts()


# ## Feature Engineering

# #### address
#  - Extracting the Zipcode from the address

# In[476]:


loan_data['zipcode'] = loan_data['address'].apply(lambda address:address[-5:])


# In[477]:


stack_bar(loan_data,'zipcode',"Area")


# ### Inference
# - Based on the above graph, we can see that zip codes 11650,86630, and 93700 have a 100% probability of getting charged-off.

# In[478]:


loan_data.drop('address',axis=1,inplace=True)


# ### Inference
#  - Important information is already captured as part of zipcode. Hence dropping the column

# #### dti
# - According to our previous analysis, dti greater than 50 has 35% of the loan to be charged-off, whereas dti less than 10 has only 13% of the loan to be charged-off.
# - Lets divide the dti value into bins to understand the impact on the loan_status

# In[479]:


bins = [0,10,20,30,1000]
labels =["0-10","10-20","20-30","Above 30"]
loan_data['dti_cat'] = pd.cut(loan_data['dti'], bins,labels=labels)


# In[480]:


loan_data['dti_cat'].head()


# In[481]:


stack_bar(loan_data,'dti_cat',"Dti Category")


# In[482]:


loan_data.drop('dti',axis=1,inplace=True)


# ### Inferences
#  - It is clear that as the dti value increases, so does the probability of being charged off.

# #### pub_rec

# In[483]:


def pub_rec(num):
    if num <= 2:
        return 0
    elif num >= 0:
        return 1
    else:
        return num


# In[484]:


loan_data['pub_rec_cat'] = loan_data.pub_rec.apply(pub_rec)


# In[485]:


loan_data["pub_rec_cat"] = loan_data["pub_rec_cat"].astype("category")


# In[486]:


stack_bar(loan_data,'pub_rec_cat',"Public Record")


# In[487]:


loan_data.drop('pub_rec',axis=1,inplace=True)


# ### Inference
#  - If Public record having derogatory value more than 2 then we can see loan getting charged-off by 24%

# #### mort_acc

# In[488]:


def mort_acc(num):
    if num == 0.0:
        return 0
    elif num >= 1.0:
        return 1
    else:
        return num


# In[489]:


loan_data['mort_acc_cat'] = loan_data.mort_acc.apply(mort_acc)
loan_data["mort_acc_cat"] = loan_data["mort_acc_cat"].astype("category")


# In[490]:


stack_bar(loan_data,'mort_acc_cat',"Number of Mortage accounts")


# In[491]:


loan_data.drop('mort_acc',axis=1,inplace=True)


# ### Inference
#  - The probability of the loan getting charged off is 24% if the borrower does not have a mortgage account

# #### pub_rec_bankruptcies

# In[492]:


def pub_rec_bankruptcies(num):
    if num == 0.0:
        return 0
    elif num >= 1.0:
        return 1
    else:
        return num


# In[493]:


loan_data['pub_rec_bankruptcies_cat'] = loan_data.pub_rec_bankruptcies.apply(pub_rec_bankruptcies)
loan_data["pub_rec_bankruptcies_cat"] = loan_data["pub_rec_bankruptcies_cat"].astype("category")


# In[494]:


stack_bar(loan_data,'pub_rec_bankruptcies_cat',"Public Record for Bankruptcies")


# In[495]:


loan_data.drop('pub_rec_bankruptcies',axis=1,inplace=True)


# ### Inference
#  - If there are more bankruptcies on public records than 1 then we can see the loan getting charged off by 20%

# #### loan_stats (Target Variable)

# In[496]:


loan_data['loan_status'].unique()


# In[497]:


def loan_status(str_):
    if str_ == 'Charged Off':
        return 1
    else:
        return 0


# In[498]:


loan_data['loan_status'] = loan_data.loan_status.apply(loan_status)


# In[499]:


loan_data['loan_status'].unique()


# In[500]:


loan_data.shape


# ### Inferences
#  - Overall we have 23 features which shows some relations w.r.t. target variable.
#  - After EDA we have removed few features
#       - emp_length
#       - emp_title
#       - grade
#       - title
#  - Few new features are derived from existing features 
#      - pub_rec_bankruptcies_cat
#      - dti_cat
#      - zipcode
#      - mort_acc_cat
#      - pub_rec_cat

# ## Checking Correlation

# In[501]:


plt.figure(figsize = (16, 10))
ax = sns.heatmap(loan_data.corr(),
            annot=True,cmap='YlGnBu',square=True)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=40,fontsize=16,family = "Comic Sans MS",
    horizontalalignment='right')

ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,fontsize=16,family = "Comic Sans MS",
    horizontalalignment='right')
    
plt.show()


# ### Inferences
#  - Loan Amount ad installment is highly corelated with 95%.
#  - Not much correlation between other variables can be observed. open_acc and total_acc are most co-related features with 68%

# ## Handling Categorical variable 
#  - **Categorical to Numerical** - Our training data more useful and expressive, and it can be rescaled easily. By using numeric values, we more easily determine a probability for our values. In particular, one hot encoding is used for our output values, since it provides more nuanced predictions than single labels

# ### One Hot Encoding
# 
# We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# In[502]:


loan_data.columns


# In[503]:


loan_data.info()


# In[504]:


loan_data.shape


# In[505]:


cat_columns = ['sub_grade', 'home_ownership','verification_status', 'issue_d', 
               'purpose',  'initial_list_status', 'application_type','zipcode',
       'dti_cat', 'pub_rec_cat', 'mort_acc_cat', 'pub_rec_bankruptcies_cat']


# In[506]:


dummyVar = pd.get_dummies(loan_data[cat_columns],drop_first=True)
dummyVar.shape


# In[507]:


dummyVar.head()


# In[508]:


# Merging the dummy variable to significant variable dataframe.
loan_data_encoded = pd.concat([loan_data,dummyVar],axis=1)
loan_data_encoded.shape


# In[509]:


# Dropping origincal Categorical variables as no need. Already added them as numerical.
loan_data_encoded.drop(cat_columns,axis=1,inplace=True)
loan_data_encoded.shape


# In[510]:


loan_data_encoded.columns


# ## train, validation & test split
#  - Train 60%
#  - Cross validation - 20%
#  - Test validation - 20%

# ### train & test Split

# In[511]:


# Train & Test data split
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


# In[512]:


#putting features variables in X
X = loan_data_encoded.drop(['loan_status'], axis=1)

#putting response variables in Y
y = loan_data_encoded['loan_status']    

# Splitting the data into train and test
X_tr_cv, X_test, y_tr_cv, y_test = train_test_split(X,y, train_size=0.8,test_size=0.2,random_state=100)


# ### Train & Cross validation split 

# In[513]:


# Splitting the data into train and test
X_train, X_val, y_train, y_val = train_test_split(X_tr_cv,y_tr_cv,test_size=0.25,random_state=1)


# #### Libraries used for Model creation

# In[514]:


# For imputation to NAN values.
from sklearn.impute import SimpleImputer

# For rescaling we are using Standarad scaler
from sklearn.preprocessing import StandardScaler

# For logistic regression model
from sklearn.linear_model import LogisticRegression

# For feature selection
from sklearn.feature_selection import RFE

# For pipeline creation
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

# For collecting different metrics.
from sklearn.metrics import f1_score
from sklearn import metrics


# ### Utility function draw ROC curve
#  - True Positve rate vs False Positive rate

# In[515]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 6))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, thresholds


# ## Handlng Missing values
#  - Data is not complete without handling missing values and many machine learning algorithms do not allow missing values.
#  - it is essential to address any missing data before feeding it to your model.
#  - In the case study we are using SimpleImputer with median

# In[516]:


imputer = SimpleImputer(strategy='median', missing_values=np.nan)


# ## Rescaling the Features
# 
# As per above table, features are varying in different ranges. This will be problem. It is important that we rescale the feature such that thay have a comparable scales. This can lead us time consuming during model evaluation.
# 
# So it is advices to Standardization and normalization so that units of coefficients obtained are in same scale. Two common ways of rescaling are
# 
# - Standardization (mean-0, sigma-1)
# - Min-Max scaling (Normization)
# 
# In this case we are using Standardizationscaling

# In[517]:


scaler = StandardScaler()


# ## Build Pipeline 
#  - Imputation
#  - Rescaling
#  - Building the model

# ### 1.  Basic Model creation

# In[518]:


pl_basic_logreg = Pipeline(steps=[('imputer',imputer),
                              ('scaler',scaler),
                              ('logistic_model',LogisticRegression())
                             ])  


# In[519]:


pl_basic_logreg.fit(X_train,y_train)


# In[520]:


train_y_pred = pl_basic_logreg.predict(X_train)
train_score = f1_score(y_train, train_y_pred)


# In[521]:


print(" F1 Score for Basic Model (Train) ", train_score)


# In[522]:


X_test['revol_util'] = X_test['revol_util'].fillna(X_test['revol_util'].median())


# In[523]:


y_pred_test = pl_basic_logreg.predict(X_test)
test_score = f1_score(y_test, y_pred_test)

print("F1 Score for Basic Model (Test) ",test_score)


# ### 2. Using Hyper-parmeter Optimization

# In[524]:


train_scores = []
val_scores = []

la_low = 0.01
la_upp = 100
la_diff = 5

for lambda_ in np.arange(la_low,la_upp,la_diff):
    hp_logreg = Pipeline(steps=[('imputer',imputer),
                              ('scaler',scaler),
                              ('logistic_model',LogisticRegression(C=1/lambda_))
                             ])  
    hp_logreg.fit(X_train, y_train)
    train_y_pred = hp_logreg.predict(X_train)
    val_y_pred = hp_logreg.predict(X_val)
    train_score = f1_score(y_train, train_y_pred)
    val_score = f1_score(y_val, val_y_pred)
    train_scores.append(train_score)
    val_scores.append(val_score)


# In[525]:


plt.figure()
plt.plot(list(np.arange(la_low,la_upp,la_diff)), train_scores, label="train")
plt.plot(list(np.arange(la_low,la_upp,la_diff)), val_scores, label="val")
plt.legend(loc='lower right')
plt.xlabel("Hyper Parameter - Lambda")
plt.ylabel("F1-Score")
plt.grid()
plt.show()


# In[526]:


# Model with lambda_best
best_hp_model = np.argmax(val_scores)
print(val_scores[best_hp_model])


# In[527]:


X_test['revol_util'] = X_test['revol_util'].fillna(X_test['revol_util'].median())


# In[528]:


l_best = la_low+la_diff*best_hp_model
best_hp_logreg = Pipeline(steps=[('imputer',imputer),
                              ('scaler',scaler),
                              ('logistic_model',LogisticRegression(C=1/l_best))
                             ])   
best_hp_logreg.fit(X_train, y_train)

y_pred_hp_test = best_hp_logreg.predict(X_test)
test_score = f1_score(y_test, y_pred_hp_test)

print('F1 Score for Best Hyper-Parmeter Model (Test) ',test_score)


# In[529]:


print(f"Accuracy : {metrics.accuracy_score(y_test, y_pred_hp_test)*100}%")
print(f"recall_score : {metrics.recall_score(y_test, y_pred_hp_test)*100}%")
print(f"precision_score : {metrics.precision_score(y_test, y_pred_hp_test)*100}%")
print(f"f1_score : {metrics.f1_score(y_test, y_pred_hp_test)*100}%")
print(f"AUC score : {metrics.roc_auc_score( y_test, y_pred_hp_test)*100}%")
print(f"confusion_matrix :")
print(metrics.confusion_matrix(y_test, y_pred_hp_test))


# In[530]:


print(metrics.classification_report(y_test,y_pred_hp_test))


# In[531]:


draw_roc(y_test, y_pred_hp_test)


# ### Recall vs Precision

# In[532]:


fig = plt.figure(figsize = (8,6))
fig.set_facecolor("lightgrey")

# Precision Recall Curve
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_hp_test)
plt.plot(thresholds, precision[:-1], "b",label='Precision')
plt.plot(thresholds, recall[:-1], "g",label='Recall')
plt.vlines(x=0.62,ymax=1,ymin=0.0,color="purple",linestyles="--")
plt.hlines(y=0.66,xmax=1,xmin=0.0,color="grey",linestyles="--")
plt.title('Precision - Recall Curve',fontsize=18,family = "Comic Sans MS")
plt.legend()
plt.show()


# ### 3. Advanced Model with Hyper-parameter, and balancing the data using class weights

# In[533]:


train_scores = []
val_scores = []

la_low = 0.01
la_upp = 10000
la_diff = 500

for lambda_ in np.arange(la_low,la_upp,la_diff):
    hp__clwg_logreg = Pipeline(steps=[('imputer',imputer),
                              ('scaler',scaler),
                              ('logistic_model',LogisticRegression(C=1/lambda_,class_weight={ 0:0.25, 1:0.75 }))])  
    hp__clwg_logreg.fit(X_train, y_train)
    train_y_pred = hp__clwg_logreg.predict(X_train)
    val_y_pred = hp__clwg_logreg.predict(X_val)
    train_score = f1_score(y_train, train_y_pred)
    val_score = f1_score(y_val, val_y_pred)
    train_scores.append(train_score)
    val_scores.append(val_score)


# In[534]:


plt.figure()
plt.plot(list(np.arange(la_low,la_upp,la_diff)), train_scores, label="train")
plt.plot(list(np.arange(la_low,la_upp,la_diff)), val_scores, label="val")
plt.legend(loc='lower right')
plt.xlabel("Hyper Parameter - Lambda")
plt.ylabel("F1-Score")
plt.grid()
plt.show()


# In[535]:


# Model with lambda_best
best_hp_clwg_model = np.argmax(val_scores)
print(val_scores[best_hp_clwg_model])


# In[536]:


l_best = la_low+la_diff*best_hp_clwg_model
best_hp_clwg_logreg = Pipeline(steps=[('imputer',imputer),
                              ('scaler',scaler),
                              ('logistic_model',LogisticRegression(C=1/l_best,class_weight={0:0.25, 1:0.75 }))
                             ])   
best_hp_clwg_logreg.fit(X_train, y_train)

y_pred_test = best_hp_clwg_logreg.predict(X_test)
test_score = f1_score(y_test, y_pred_test)

print('F1 Score for Best Hyper-Parmeter with class weight Model (Test) ',test_score)


# In[537]:


print(f"Accuracy : {metrics.accuracy_score(y_test, y_pred_test)*100}%")
print(f"recall_score : {metrics.recall_score(y_test, y_pred_test)*100}%")
print(f"precision_score : {metrics.precision_score(y_test, y_pred_test)*100}%")
print(f"f1_score : {metrics.f1_score(y_test, y_pred_test)*100}%")
print(f"AUC score : {metrics.roc_auc_score( y_test, y_pred_test)*100}%")
print(f"confusion_matrix :")
print(metrics.confusion_matrix(y_test, y_pred_test))


# In[538]:


print(metrics.classification_report(y_test,y_pred_test))


# In[539]:


draw_roc(y_test, y_pred_test)


# ### Recall vs Precision

# In[540]:


fig = plt.figure(figsize = (8,6))
fig.set_facecolor("lightgrey")

# Precision Recall Curve
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_test)
plt.plot(thresholds, precision[:-1], "b",label='Precision')
plt.plot(thresholds, recall[:-1], "g",label='Recall')
plt.vlines(x=0.98,ymax=1,ymin=0.0,color="purple",linestyles="--")
plt.hlines(y=0.65,xmax=1,xmin=0.0,color="r",linestyles="--")
plt.title('Precision - Recall Curve',fontsize=18,family = "Comic Sans MS")
plt.legend()
plt.show()


# ### Top 5 features that played key role in getting charged-off or not.
#  - Used RFE technique

# In[541]:


X_train['revol_util'] = X_train['revol_util'].fillna(X_train['revol_util'].median())    


# In[ ]:


rfe = RFE(best_hp_clwg_logreg['logistic_model'], n_features_to_select=15)             
rfe = rfe.fit(X_train, y_train)


# In[ ]:


cols=X_train.columns[rfe.support_]
cols


# In[ ]:


#Function to fit the logistic regression model from the statmodel package
def fit_LogRegModel(X_train):
    # Adding a constant variable  
    X_train = sm.add_constant(X_train)
    lm = sm.GLM(y_train,X_train,family = sm.families.Binomial()).fit() 
    print(lm.summary())
    return lm


# In[ ]:


# Calculate the VIFs for the new model
def getVIF(X_train):
    vif = pd.DataFrame()
    X = X_train
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)


# ## Assessing the Model using StatsModels

# In[ ]:


# Creating X_test dataframe with RFE selected variables
X_train_GM = X_train[cols]
lm = fit_LogRegModel(X_train_GM)


# In[ ]:


X_train_GM = X_train_GM.drop(['zipcode_05113','zipcode_86630','zipcode_93700','zipcode_11650','zipcode_29597'], axis=1)


# In[ ]:


lm = fit_LogRegModel(X_train_GM)


# In[ ]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train_GM)), family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[ ]:


# Make a VIF dataframe for all the variables present
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Features'] = X_train_GM.columns
vif['VIF'] = [variance_inflation_factor(X_train_GM.values, i) for i in range(X_train_GM.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ### Inferences 
#  - Key features that heavily affected the outcome are -
#     - dti, mort_acc, verification_status, sub_grade & int_rate

# ### Confusion Metrics w.r.t. Lending Club Loan
# 
# ![image.png](attachment:bd6ccb4b-4736-439a-91c6-d2a5d7bd85da.png)

# ### Which metrics we should select for our model will depend on the Business use case.
# 
# ##### Case 1 - When the bank does not want to lose the money as well as the customers. we make sure that our model can detect real defaulters and there are less false positives? This is important as we can lose out on an opportunity to finance more supply chains and earn interest on it.
# - In case of low recall, Lending Club loan might lose money. Low precision means even if the borrower is not a defaulter or charged off, he will not be approved for a loan. That means lost business for the banks. **It is important to have a balance between recall and precision, so a good F1-score will make sure that balance is maintained.**
# 
# ##### Case 2 - The bank does not want to lose the money but can grow slowly with genuine customers. Since NPA (non-performing asset) is a real problem in this industry, it’s important we play safe and shouldn’t disburse loans to anyone with NPA.
# - In this case, when predicting whether or not a loan will default - it would be **better to have a high recall because the banks don't want to lose money**, so it would be a good idea to alert the bank even if there is a slight doubt about the borrower.Low precision, in this case, might be okay.
# 
# ##### Case 3: When a bank wants to grow faster and get more customers at the expense of losing some money in some cases.
# - In this case, it would be ok to have a **slight higher precision compare the recall**. 
# 

# ### Comparison between Model 3 & Model 2
# ![image.png](attachment:5b78cfbd-b5f9-4757-a714-e09c6e43f127.png)

# ### Inferences
#  - From the above metrics it is clearly shows **Model 3 is much better than Model 2 as balance between recall and precision is maintained.**
#  - A low recall or precision (one or both inputs) makes the **F1-score more sensitive, which is great if you want to balance the two.The higher the F1-score the better the model for case 1**
#  - Model 3 has **F1-score as 65** where as **Model 2 has F-score as 62**only.
#  - Moreover, we can clearly see that **recall is very high for models with balanced data.** In our case it it Model 3.

# ## Inferences and Recommendations

# ### Inferences based on EDA.
#  - Eighty-five percent of loan balances are fully paid, while 19 percent have been charged off
#  - There is a strong correlation between loan amount and installment (with 0.95)
#  - Mortgages are the most common form of home ownership
#  - 94% of people who have grades 'A' pay their loans on time.
#  - The two top job titles that take most loans are teacher and manager.
#  - zip codes 11650,86630, and 93700 have a 100% probability of getting charged-off. Location plays imprtant role for loan getting charged-off.
#  
# ### Inferences based on the Model
#  - From the above metrics it is clearly shows **Model 3 is much better than Model 2 as balance between recall and precision is maintained.**
#  - A low recall or precision (one or both inputs) makes the **F1-score more sensitive, which is great if you want to balance the two.The higher the F1-score the better the model for case 1**
#  - Model 3 has **F1-score as 65** where as **Model 2 has F-score as 62**only.
#  - Moreover, we can clearly see that **recall is very high for models with balanced data.** In our case it it Model 3.

# ### Recommendations
#  - Model 3 is recommended as it can detect real defaulters and ensure that the bank will not lose the opportunity to finance more supply chains and earn interest.
#  - One way to make sure we have fewer defaulters is to get customers with high grades.
#  - zip codes 11650,86630, and 93700 have a 100% probability of getting charged-off. Banks should refrain from lending to these areas until they understand why. As well, setup a team to analyze, as this is a common trend for getting charged-off at those locations.
#  - Key features that heavily affected the outcome are - **dti, mort_acc, verification_status, sub_grade & int_rate**

# #### I would appreciate an upvote **if you liked the Notebook** and feel free to share feedback!!!

# In[ ]:




