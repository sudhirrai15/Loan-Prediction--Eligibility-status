#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd 
import numpy as np        # For mathematical calculations 
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns                  # For data visualization 
import warnings                        # To ignore any warnings 
warnings.filterwarnings("ignore")


# In[50]:


train=pd.read_csv('train.csv') 


# In[51]:


test=pd.read_csv("test.csv")


# In[52]:


#copy of orginal dataset
train_original=train.copy() 
test_original=test.copy()


# In[53]:


#Features present in our data and  their data types
train.columns


# In[54]:


test.columns


# In[55]:


#data types for each variable
train.dtypes


# In[56]:


#shape of our dataset
train.shape , test.shape


# In[57]:


#we will do univariate analysis. It is the simplest form of analyzing data where we examine 
#each variable individually. For categorical features we can use frequency table or bar plots 
#which will calculate the number of each category in a particular variable. 
#For numerical features, probability density plots can be used to look at the distribution of the variable.

train['Loan_Status'].value_counts()


# In[58]:


# Normalize can be set to True to print proportions instead of number
train['Loan_Status'].value_counts(normalize=True)


# In[59]:


train['Loan_Status'].value_counts().plot.bar()


# In[60]:


#Categorical features: These features have categories (Gender, Married, Self_Employed, Credit_History, Loan_Status)
#Ordinal features: Variables in categorical features having some order involved (Dependents, Education, Property_Area)
#Numerical features: These features have numerical values (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term)

plt.figure(1) 

plt.subplot(221) 
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 

plt.subplot(222) 
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 

plt.subplot(223) 
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 

plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 

plt.show()


# In[61]:


plt.figure(1) 

plt.subplot(131) 
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents') 

plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title= 'Education') 

plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area') 

plt.show()


# In[62]:


plt.figure(1)

plt.subplot(121) 
sns.distplot(train['ApplicantIncome']); 

plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5)) 

plt.show()


# In[63]:


train.boxplot(column='ApplicantIncome', by = 'Education') 
plt.suptitle("")


# In[64]:


plt.figure(1) 
plt.subplot(121) 
sns.distplot(train['CoapplicantIncome']); 

plt.subplot(122) 
train['CoapplicantIncome'].plot.box(figsize=(16,5)) 

plt.show()


# In[65]:


plt.figure(1) 

plt.subplot(121) 

df=train.dropna()

sns.distplot(df['LoanAmount']); 

plt.subplot(122) 
train['LoanAmount'].plot.box(figsize=(16,5)) 

plt.show()


# In[66]:


#bivariate analysis
#Applicants with high income should have more chances of loan approval.
#Applicants who have repaid their previous debts should have higher chances of loan approval.
#Loan approval should also depend on the loan amount. If the loan amount is less, chances of loan approval should be high.
#Lesser the amount to be paid monthly to repay the loan, higher the chances of loan approval.


# In[67]:


Gender=pd.crosstab(train['Gender'],train['Loan_Status']) 
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


# In[68]:


Married=pd.crosstab(train['Married'],train['Loan_Status']) 

Dependents=pd.crosstab(train['Dependents'],train['Loan_Status']) 

Education=pd.crosstab(train['Education'],train['Loan_Status']) 

Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status']) 

Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show() 

Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.show() 

Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show() 

Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show()


# In[69]:


Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status']) 

Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 

plt.show() 


# In[70]:


Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.show()


# In[71]:


train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()


# In[72]:


bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high'] 

train['Income_bin']=pd.cut(df['ApplicantIncome'],bins,labels=group)

Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status']) 

Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('ApplicantIncome') 

plt.ylabel('Percentage')


# In[73]:


bins=[0,1000,3000,42000] 

group=['Low','Average','High'] 

train['Coapplicant_Income_bin']=pd.cut(df['CoapplicantIncome'],bins,labels=group)

Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status']) 

Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 

plt.xlabel('CoapplicantIncome') 

plt.ylabel('Percentage')


# In[74]:


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']

bins=[0,2500,4000,6000,81000] 

group=['Low','Average','High', 'Very high'] 

train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)

Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status']) 

Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 

plt.xlabel('Total_Income') 

plt.ylabel('Percentage')


# In[75]:


bins=[0,100,200,700] 

group=['Low','Average','High'] 

train['LoanAmount_bin']=pd.cut(df['LoanAmount'],bins,labels=group)

LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status']) 

LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 

plt.xlabel('LoanAmount')

plt.ylabel('Percentage')


# In[76]:


#drop the bins which we created for our visualization

train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)


# In[77]:


train['Dependents'].replace('3+', 3,inplace=True)   #replace 3+ with 3
test['Dependents'].replace('3+', 3,inplace=True)    #replace 3+ with 3
train['Loan_Status'].replace('N', 0,inplace=True)   #replace N with 0  
train['Loan_Status'].replace('Y', 1,inplace=True)   #replace Y with 1


# In[78]:


#heat map to visualize the correlation

matrix = train.corr() 

f, ax = plt.subplots(figsize=(9, 6)) 

sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");


# In[79]:


#Missing value imputation
#ist out feature-wise count of missing values

train.isnull().sum()


# In[80]:


# to fill the missing values we follow these methods:
#For numerical variables: imputation using mean or median
#For categorical variables: imputation using mode


# In[81]:


train['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 

train['Married'].fillna(train['Married'].mode()[0], inplace=True) 

train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 

train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 

train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)


# In[82]:


train['Loan_Amount_Term'].value_counts()


# In[83]:


#loan amount term 360 is repeated for most of the time we will use mode for loan amount
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)


# In[84]:


# we have lot of outlier so we will fill the missing value with median
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# In[85]:


train.isnull().sum()


# In[86]:


test['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True) 
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True) 
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# In[87]:


#we can remove the outlier with log transformation. It does not affect the smaller values much, but reduces the larger values

train['LoanAmount_log'] = np.log(train['LoanAmount']) 
train['LoanAmount_log'].hist(bins=20) 

test['LoanAmount_log'] = np.log(test['LoanAmount'])


# In[88]:


#we can drop the loan_id variable as it does not play a vital role in logistic regression
train=train.drop('Loan_ID',axis=1) 
test=test.drop('Loan_ID',axis=1)


# In[89]:


#Sklearn requires the target variable in a separate dataset.
X = train.drop('Loan_Status',1) 
y = train.Loan_Status


# In[90]:


#making dummies variavle for categorical variable as the logistic regression wor on numerical variable.

X=pd.get_dummies(X) 
train=pd.get_dummies(train) 
test=pd.get_dummies(test)


# In[91]:


#split our train dataset into two part: train and validation dataset.
from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)


# In[92]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score


# In[93]:


model = LogisticRegression() 


# In[94]:


model.fit(x_train, y_train)


# In[95]:


#Here the C parameter represents inverse of regularization strength. 
#Regularization is applying a penalty to increasing the magnitude of parameter values in order to reduce overfitting. Smaller values of C specify stronger regularization
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, 
                   penalty='l2', random_state=1, solver='liblinear', tol=0.0001,verbose=0, warm_start=False)


# In[96]:


#predict the Loan_Status for validation set and calculate its accuracy
pred_cv = model.predict(x_cv)


# In[97]:


accuracy_score(y_cv,pred_cv)


# In[98]:


#predict on test data set
pred_test = model.predict(test)


# In[100]:


submission=pd.read_csv("sample_submission.csv")


# In[101]:


submission['Loan_Status']=pred_test 


# In[102]:


submission['Loan_ID']=test_original['Loan_ID']


# In[103]:


submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)


# In[104]:


pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')


# In[106]:


#cross validation logistic model with stratified 5 folds and make predictions for test dataset.
from sklearn.model_selection import StratifiedKFold


# In[107]:


i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         
    model = LogisticRegression(random_state=1)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score)     
    i+=1 
    pred_test = model.predict(test) 
    pred=model.predict_proba(xvl)[:,1]


# In[110]:


#The mean validation accuracy for this model turns out to be 0.81.
#visualize the roc curve.
from sklearn import metrics 
fpr,tpr, _ = metrics.roc_curve(yvl,  pred) 
auc = metrics.roc_auc_score(yvl, pred) 
plt.figure(figsize=(12,8)) 
plt.plot(fpr,tpr,label="validation, auc="+str(auc)) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.legend(loc=4) 
plt.show()


# In[112]:


submission['Loan_Status']=pred_test 
submission['Loan_ID']=test_original['Loan_ID']


# In[ ]:


submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)


# In[114]:


pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('Logistic.csv')


# In[115]:


#add new feature which help in the prediction of loan approval
#creat one new column with Ttal income which is combined of applicants and co-aaplicants


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome'] 
test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']


# In[116]:


sns.distplot(train['Total_Income']);


# In[117]:


train['Total_Income_log'] = np.log(train['Total_Income']) 
sns.distplot(train['Total_Income_log']); 
test['Total_Income_log'] = np.log(test['Total_Income'])


# In[118]:


#create an EMI feature which can also be a good factor to predict the loan approval; Low EMI high chance to repay the loan
train['EMI']=train['LoanAmount']/train['Loan_Amount_Term'] 
test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']


# In[119]:


sns.distplot(train['EMI']);


# In[120]:


#creat one more variable Balance Income: income left after the EMI has been paid.
#If this value is high, the chances are high that a person will repay the loan and 
#hence increasing the chances of loan approval.

# Multiply with 1000 to make the units equal test['Balance Income']=test['Total_Income']-(test['EMI']*1000)

train['Balance Income']=train['Total_Income']-(train['EMI']*1000) 
test['Balance Income']=test['Total_Income']-(test['EMI']*1000)


# In[121]:


sns.distplot(train['Balance Income']);


# In[122]:


#drop the variables which we used to create these new features. Reason for doing this is, 
#the correlation between those old features and these new features will be very high and logistic regression assumes
#that the variables are not highly correlated

train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1) 

test=test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)


# In[123]:


X = train.drop('Loan_Status',1) 
y = train.Loan_Status                # Save target variable in separate dataset


# In[124]:


#Logistic Regression
i=1 
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         

    model = LogisticRegression(random_state=1)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score)     
    i+=1 
    pred_test = model.predict(test) 
    pred=model.predict_proba(xvl)[:,1]


# In[125]:


submission['Loan_Status']=pred_test            # filling Loan_Status with predictions submission['Loan_ID']=test_original['Loan_ID'] # filling Loan_ID with test Loan_ID
# replacing 0 and 1 with N and Y 
submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)
# Converting submission file to .csv format 
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('Log2.csv')


# In[126]:


#Decision Tree
from sklearn import tree


# In[127]:


i=1 
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         
    model = tree.DecisionTreeClassifier(random_state=1)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score)     
    i+=1 
    pred_test = model.predict(test)


# In[128]:


submission['Loan_Status']=pred_test            # filling Loan_Status with predictions submission['Loan_ID']=test_original['Loan_ID'] # filling Loan_ID with test Loan_ID
# replacing 0 and 1 with N and Y 
submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)
# Converting submission file to .csv format 
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('Decision Tree.csv')


# In[129]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         
    model = RandomForestClassifier(random_state=1, max_depth=10)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score)     
    i+=1 

    pred_test = model.predict(test)


# In[131]:


#improve the accuracy by tuning the hyperparameters for this model
#grid search to get the optimized values of hyper parameters. Grid-search is a way to select the best of a family of hyper parameters,
#parametrized by a grid of parameters

from sklearn.model_selection import GridSearchCV
# Provide range for max_depth from 1 to 20 with an interval of 2 and from 1 to 200 with an interval of 20 for n_estimators 

paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}

grid_search=GridSearchCV(RandomForestClassifier(random_state=1),paramgrid)

from sklearn.model_selection import train_test_split 

x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3, random_state=1)

# Fit the grid search model 

grid_search.fit(x_train,y_train)

GridSearchCV(cv=None, error_score='raise',estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
              max_depth=None, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False,
            random_state=1, verbose=0, warm_start=False),fit_params=None, iid=True, n_jobs=1,       
param_grid={'max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 'n_estimators': [1, 21, 41, 61, 81, 101, 121, 141, 161, 181]},       
pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',       
scoring=None, verbose=0)

# Estimating the optimized value 
grid_search.best_estimator = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=3, max_features='auto', 
                       max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, 
                       min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=41, n_jobs=1,  
                       oob_score=False, random_state=1, verbose=0, warm_start=False)


# In[133]:


i=1 
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         
    model = RandomForestClassifier(random_state=1, max_depth=3, n_estimators=41)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score)     
    i+=1 
    pred_test = model.predict(test) 
    pred2=model.predict_proba(test)[:,1]


# In[134]:


submission['Loan_Status']=pred_test            # filling Loan_Status with predictions submission['Loan_ID']=test_original['Loan_ID'] # filling Loan_ID with test Loan_ID
# replacing 0 and 1 with N and Y 
submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)
# Converting submission file to .csv format 
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('Random Forest.csv')


# In[135]:


# feature importance 
#features are most important for this problem.
#We will use feature_importances_ attribute of sklearn to do so

importances=pd.Series(model.feature_importances_, index=X.columns) 
importances.plot(kind='barh', figsize=(12,8))


# In[137]:


pip install xgboost


# In[138]:


#XGBOOST
#XGBoost works only with numeric variables and we have already replaced the categorical variables with numeric variables
#Parameter used in XGBOOST:
#n_estimator: This specifies the number of trees for the model.
#max_depth: We can specify maximum depth of a tree using this parameter.

from xgboost import XGBClassifier
i=1 
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         
    model = XGBClassifier(n_estimators=50, max_depth=4)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score)     
    i+=1 
    pred_test = model.predict(test) 
    pred3=model.predict_proba(test)[:,1]


# In[139]:


submission['Loan_Status']=pred_test 
submission['Loan_ID']=test_original['Loan_ID']
submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('XGBoost.csv')


# In[ ]:




