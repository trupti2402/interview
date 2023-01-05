#!/usr/bin/env python
# coding: utf-8

# Name : Truptirani Mahapatra
# 
# Introduction:
#     Given is a dataset in which we have to analyze and predict the demand of different products over different cities.
# 
# My Objectives:
# 1. Builing a time series model of the dataset and evaluating it using Cross validation
# 2. Implementing the nested time series cross validation strategy for grouped forecasting.
# 3. Applying Exploratory Data Analysis and finding some necessary insights from the dataset
# 
#     

# In[1]:


# We need to import Libraries to perform different functions. Numpy is a library for the Python programming language, to perform high-level mathematical function
# Pandas is Python library that has functions for analyzing, cleaning, exploring, and manipulating data.
# Matplotlib and seaborn are  comprehensive library for creating statical,  and interactive visualizations in Python.
#The datetime module supplies classes for manipulating dates and times.
#The warnings filter controls whether warnings are ignored, displayed, or turned into errors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# In[2]:


originalTrain = pd.read_csv("RaniPython/train.csv")
originalTest = pd.read_csv("RaniPython/test.csv")


# In[3]:


originalTrain.head()


# In[4]:


originalTrain.shape


# In[5]:


originalTrain.describe()


# In[6]:


originalTrain.info()


# In[7]:


originalTrain.isnull().sum()


# In[8]:


originalTrain['city'].value_counts()


# In[9]:


originalTrain['shop'].value_counts()


# In[10]:


originalTrain['brand'].unique()


# In[11]:


originalTrain['container'].nunique()


# In[12]:


originalTrain['container'].unique()


# # Description:
#     The train dataset contains 7560 rows and 12 columns.The above dataset contains many null values for each columns.
#     --The  id column is float type which has particular id numbers,
#     --date column is object type providing the dates, 
#     --city column is object gives the name of the city it is a categorical column having 4 cities namely Athens,Larisa,Irakleion,Patra,Thessaloniki 
#     --lat means latitude and long means longitude providing the geographical data of the place, 
#     --shop column is also is object type and containing 6 categorical values i.e shop_1 to shop_6 
#     --brands column is categorical having 5 different types of brand namely 'kinder-cola', 'adult-cola', 'orange-power', 'gazoza','lemon-boost' 
#     --container gives different container names 'glass', 'plastic', 'can'
#     --price says about the cost of the container and 
#     --quantity gives information about the quantity of the sold products.
#     

# In[13]:


originalTest.head()


# In[14]:


originalTest.shape


# In[15]:


originalTest.describe()


# In[16]:


originalTest.info()


# In[17]:


originalTest.isnull().sum()


# #Description of test data
# The test data here has 1080 rows and 12 columns. It also contain null values in columns lat,long,container and capacity.
# The distribution of train-test data: percentage of train data in the dataset is 7560/(7560+1080)*100=87.5% and  
# percentage of test data in the dataset is 100-87.5=12.5%

# In[18]:


#Handling Null Values


# In[19]:


originalTrain.tail(20)


# In[20]:


originalTrain[6479:]


# Here we can see from id no.6480 all the values of the column are nan,
# hence we can drop the nan values from the dataset.

# In[21]:


originalTrain=originalTrain.dropna()


# In[22]:


originalTest.dropna()


# After removing null values the shape of originaltrain set is  6376 rows × 12 columns and 
# original test set is 1058 rows × 12 columns

# In[23]:


#We can see here the id column has no use in future analysis so we can drop it.
train_data = originalTrain.copy()
test_data = originalTest.copy()

#Dropping unnecessary Id column.

train_data.drop('id', axis=1, inplace=True)
test_data.drop('id', axis=1, inplace=True)


# In[24]:


originalTrain.head()


# # Explaratory Data Analysis

# In[25]:


sns.heatmap(train_data.corr(),annot=True,cmap='coolwarm')


# Through the heatmap we can conclude that as price and quantity are negatively corelated to eachother, as the price increases quantity tends to decrease. 

# In[26]:


sns.distplot(train_data['price'],color='red')


# In[27]:


sns.histplot(train_data['quantity'],color='red')


# In[28]:


sns.barplot(x=train_data['brand'], y=train_data['price'])
plt.title('price of different brands')


# In[29]:


sns.set_context("poster", font_scale=.7)
plt.figure(figsize=(7,7))
sns.set_palette('RdYlBu')
sns.countplot(train_data['city'])


# In[30]:


sns.set_palette('PiYG')
plt.figure(figsize=(10,10))
sns.set_context("poster", font_scale=0.7)
sns.countplot(train_data['shop'])


# # Time Series Analysis

# In[31]:


#IN this dataset we are going to predict the demand using time series Analysis. Hence we are going to take the date
#column as the index column and drop other columns keeping the Quantity column intact


# In[32]:


train_data.set_index('date',inplace=True)
train_data.index


# In[33]:


train_data.head()


# In[34]:


train_data=train_data.drop(columns=['city','lat','long','pop','shop','brand','container','capacity','price'])


# In[35]:


train_data.tail()


# In[36]:


train_data.shape


# In[37]:


train_data.describe()


# In[38]:


train_data.plot()


# *Lets us now check the stationarity of data using Dickey Fuller Test as the trends or seasonality in the dataset
# affects the data *Here we need to take Null Hypothesis in ADF test is the Time Series is non-Stationary and 
# Aternate Hypothesis as dataset is stationary *If the ADF Statistic < Critical Value - Reject the Null Hypothesis 
# *If the ADF Statistic > Critical Value - Failed to Reject the Null Hypothesis

# In[39]:


### Testing For Stationarity

from statsmodels.tsa.stattools import adfuller


# In[40]:


result=adfuller(train_data['quantity'])


# In[41]:


#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(quantity):
    result=adfuller(quantity)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    


# In[42]:


adfuller_test(train_data['quantity'])


# In[43]:


#We can check the trend and seasonality using the  seasonal_decompose() method.
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_result = seasonal_decompose(train_data,period=1)

decompose_result.plot()


# In[44]:


#Trend is a pattern in data that shows the movement of a series to relatively
#higher or lower values over a long period of time

#removing trend using the rolling mean differencing 

rolling_mean = train_data.rolling(window=12).mean()

#Rolling means creating a rolling window with
# a specified size and perform calculations on
# the data in this window which, of course, rolls through the data.

#Rolling averages are useful for finding long-term trends otherwise
# disguised by occasional fluctuations. 
rolling_mean_detrended = train_data - rolling_mean

ax1 = plt.subplot(121)
rolling_mean_detrended.plot(figsize=(12,4),color="tab:red", 
                            title="Differenced With Rolling Mean over 12 months", 
                            ax=ax1)
ax2 = plt.subplot(122)
train_data.plot(figsize=(12,4), 
          color="tab:red", 
          title="Original Values", 
          ax=ax2)


# In[45]:


from statsmodels.tsa.seasonal import seasonal_decompose
decompose_result = seasonal_decompose(rolling_mean_detrended.dropna(),period=1)
#We can check the trend and seasonality using the
# seasonal_decompose() method.
decompose_result.plot()


#Here in the  chart, we can see the decomposed
# structure of data and the structure of the components 
#in the data set which were affecting it.


# In[46]:


rolling_mean_detrended


# In[47]:


rolling_mean_detrended.shift(2).head(24)


# In[48]:


#removing seasonality from the time series
rolling_mean_detrended_diff = rolling_mean_detrended - rolling_mean_detrended.shift()

ax1 = plt.subplot(121)
rolling_mean_detrended_diff.plot(figsize=(12,4),
                                 color="tab:red", 
                                 title="rolled and differenced time series", 
                                 ax=ax1)
ax2 = plt.subplot(122)
train_data.plot(figsize=(12,4), 
          color="tab:red", 
          title="Original Values",
          ax=ax2)


# In[49]:


rolling_mean_detrended_diff = rolling_mean_detrended_diff.dropna()


# In[50]:


rolling_mean_detrended_diff


# In[51]:


result = adfuller(rolling_mean_detrended_diff['quantity'].dropna())
#print(result)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# # Final Thoughts on Autocorrelation and Partial Autocorrelation
# Identification of an AR model is often best done with the PACF.
# For an AR model, the theoretical PACF “shuts off” past the order of the model. The phrase “shuts off” means that in theory the partial autocorrelations are equal to 0 beyond that point. Put another way, the number of non-zero partial autocorrelations gives the order of the AR model. By the “order of the model” we mean the most extreme lag of x that is used as a predictor.
# Identification of an MA model is often best done with the ACF rather than the PACF.
# 
# For an MA model, the theoretical PACF does not shut off, but instead tapers toward 0 in some manner. A clearer pattern for an MA model is in the ACF. The ACF will have non-zero autocorrelations only at lags involved in the model.
# 
# p,d,q p AR model lags d differencing q MA lags

# In[52]:


from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
#To check for the stationarity of the time series,
# statsmodels provides a plot_acf method to plot an autocorrelation plot.
plot_acf(train_data['quantity'])


# In[53]:


from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(train_data['quantity'])


# In[54]:


get_ipython().system('pip install pmdarima')
from pmdarima import auto_arima


# In[55]:


ar=auto_arima(rolling_mean_detrended_diff['quantity'],trace=True)
ar.summary()


# In[ ]:





# In[69]:


from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

model = SARIMAX(train_data, order=(5,0,0), seasonal_order=(5,0,0,12))
model = model.fit()


# In[70]:


forecast = model.forecast(steps=60)
rolling_mean_detrended_diff.plot()
forecast.plot()


# In[ ]:





# In[63]:





# In[65]:


pip install statsmodels


# In[88]:


import numpy as np
from sklearn.model_selection import TimeSeriesSplit
X = rolling_mean_detrended_diff.iloc[:4454]['quantity']
y = rolling_mean_detrended_diff.iloc[4455:]['quantity']
tscv = TimeSeriesSplit()
print(tscv)
TimeSeriesSplit(max_train_size=None, n_splits=3)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index) 
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[96]:


from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
sm.tsa.ARIMA(rolling_mean_detrended_diff['quantity'],order=(5,0,5))
tscv = TimeSeriesSplit(n_splits = 4)
rmse = []
for train_index, test_index in tscv.split(cross_validation):
    cv_train, cv_test = cross_validation.iloc[train_index], cross_validation.iloc[test_index]
    
    arima = sm.tsa.ARIMA(cv_train, (2,2)).fit(disp=False)
    
    predictions = arima.predict(cv_test.index.values[0], cv_test.index.values[-1])
    true_values = cv_test.values
    rmse.append(sqrt(mean_squared_error(true_values, predictions)))
    
print("RMSE: {}".format(np.mean(rmse)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




