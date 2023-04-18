#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""correlation
one way anova
regression 
multiple linear regression"""


# In[3]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[35]:


data = pd.read_excel('Dataset.xlsx')
data.head()


# In[36]:


data.isnull().sum()


# In[37]:


data.shape


# In[38]:


data.value_counts()


# In[39]:


data.columns


# In[40]:


data["# appli. rec'd"]=data["# appli. rec'd"].fillna(data["# appli. rec'd"].mean())
data["# appl. accepted"]=data["# appl. accepted"].fillna(data["# appl. accepted"].mean())
data["# new stud. enrolled"]=data["# new stud. enrolled"].fillna(data["# new stud. enrolled"].mean())
data["% new stud. from top 10%"]=data["% new stud. from top 10%"].fillna(data["% new stud. from top 10%"].mean())
data["% new stud. from top 25%"]=data["% new stud. from top 25%"].fillna(data["% new stud. from top 25%"].mean())
data["# FT undergrad"]=data["# FT undergrad"].fillna(data["# FT undergrad"].mean())
data["# PT undergrad"]=data["# PT undergrad"].fillna(data["# PT undergrad"].mean())
data["in-state tuition"]=data["in-state tuition"].fillna(data["in-state tuition"].mean())
data["out-of-state tuition"]=data["out-of-state tuition"].fillna(data["out-of-state tuition"].mean())
data["room"]=data["room"].fillna(data["room"].mean())
data["board"]=data["board"].fillna(data["board"].mean())
data["add. fees"]=data["add. fees"].fillna(data["add. fees"].mean())
data["estim. book costs"]=data["estim. book costs"].fillna(data["estim. book costs"].mean())
data["estim. personal $"]=data["estim. personal $"].fillna(data["estim. personal $"].mean())
data["% fac. w/PHD"]=data["% fac. w/PHD"].fillna(data["% fac. w/PHD"].mean())
data["stud./fac. ratio"]=data["stud./fac. ratio"].fillna(data["stud./fac. ratio"].mean())
data["Graduation rate"]=data["Graduation rate"].fillna(data["Graduation rate"].mean())


# In[41]:


data.describe()


# In[42]:


import matplotlib.pyplot as plt
plt.hist(data["Graduation rate"])


# In[43]:


plt.boxplot(data["Graduation rate"])


# In[11]:


Graduation_rate=data["Graduation rate"].values
public_private=data["Public (1)/ Private (2)"].values
plotdata=pd.DataFrame({ 'public_private': public_private,'Graduation_rate': Graduation_rate})
plotdata.plot(kind='barh', stacked=True)


# In[12]:


plt.plot(data["Graduation rate"])

# show the graph
plt.show()


# In[ ]:





# In[13]:


#correlation
data.corr()


# In[14]:


import seaborn as sns
sns.heatmap(data.corr())


# In[15]:


import seaborn as sns
sns.pairplot(data)


# In[16]:


data.columns


# In[31]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
data['State']= label_encoder.fit_transform(data['State'])


# In[ ]:





# In[32]:


#regression
#linear regression
y=data['Public (1)/ Private (2)']
x=data['Graduation rate']


# In[33]:


import statsmodels.formula.api as smf
model=smf.ols("y~x",data=data).fit()


# In[20]:


model.params


# In[21]:


model.summary()


# In[28]:


#multi linear regression
x=data.drop(['Public (1)/ Private (2)','College Name'],axis=1)
y=data['Public (1)/ Private (2)']


# In[34]:


import statsmodels.formula.api as smf
model=smf.ols("y~x",data=data).fit()


# In[24]:


model.summary()


# In[25]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 56)


# In[26]:


from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_absolute_error as mae
lr = LR(normalize=True)
lr.fit(train_x, train_y)


# In[27]:


train_predict = lr.predict(train_x)
k = mae(train_predict, train_y)
print('Training Mean Absolute Error', k )


# In[28]:


test_predict = lr.predict(test_x)
k = mae(test_predict, test_y)
print('Test Mean Absolute Error    ', k )


# In[29]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
k = mean_absolute_percentage_error(train_predict, train_y)
print('Training Mean Absolute Percentage Error', k )
k = mean_absolute_percentage_error(test_predict, test_y)
print('Test Mean Absolute Percentage Error    ', k )


# In[30]:


#ANOVA
#One way Anova
from statsmodels.formula.api import ols
import scipy 
from scipy import stats
import statsmodels.api as sm
#Normality test 
data_pub_priv=stats.shapiro(data['Public (1)/ Private (2)'])    #Shapiro Test
data_pub_priv_Value=data_pub_priv[1]
print("p-value is: "+str(data_pub_priv_Value))


# In[ ]:




