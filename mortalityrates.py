
# coding: utf-8

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# read in csv data
regions = pd.read_csv('regions.csv')


# In[3]:


# Read in csv data
df = pd.read_csv('train.csv', parse_dates=['date'], index_col='Id')


# In[4]:


# get info about regions
regions.head()


# In[59]:


# get info about data
df.tail()


# In[54]:


df.info()


# In[55]:


df.describe()


# In[56]:


# get column headers
df.columns


# In[57]:


# Clean data!


# In[4]:


# Remove nan data and give it a new name
df_clean=df.dropna()
df_clean.describe()


# In[59]:


# A quick overview of the mortality rate, ie what we are trying to predict
df_clean['mortality_rate'].plot.hist(bins=50)


# In[60]:


# To get an overall view of the relationships between the data. Y is mortality rate (the dependent variable) and X are O3, PM10, PM25,
# NO2 and T2M (independent variables)

# From the plot it can be seen that there is a general overall agreement between each region for each relationship, apart from E12*07
# Clear mulitcollinearity between PM10 and PM25 and possibly NO2. VIF for PM10 and PM25 is close to 7, whilst PM10 and NO2 are about 2


g=sns.pairplot(df_clean, vars=['mortality_rate', 'O3', 'PM10', 'PM25', 'NO2','T2M'], hue='region' ,kind='reg')


# In[245]:


#df_clean = df_clean.loc[df['region'] == 'E12000009']
#sns.pairplot(df_clean, vars=['mortality_rate', 'O3', 'PM10', 'PM25', 'NO2','T2M'] ,kind='reg')


# In[103]:


#df_clean.loc[df_clean['region'] == 'E12000007','O3']=np.nan
#df_clean=df_clean.dropna()
#df_clean.describe()


# In[338]:


# Possible multi-collinearity between some of the independent variavles. This makes sense considering there are chemical relationships between them
# PM10 and PM25 show a strong correlation and hence co-linearity. Also

sns.jointplot(x='PM25',y='PM10', kind='hex',data=df_clean)
sns.jointplot(x='PM10',y='NO2', kind='hex', data=df_clean)


# Correlation >90%, hence strong multi-collinearity between two independent variables. 


# In[18]:


# Possible multi-collinearity between some of the independent variavles. This makes sense considering there are chemical relationships between them
# PM10 and PM25 show a strong correlation and hence co-linearity. Also


sns.jointplot(x='O3',y='NO2', kind='hex', data=df_clean)
sns.jointplot(x='T2M',y='mortality_rate', kind='hex', data=df_clean)

# Correlation >90%, hence strong multi-collinearity between two independent variables. 


# In[17]:


# Load in libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import statsmodels.formula.api as sm

# set variables in model
o3=[]
pm10=[]
t2m=[]
pv=[]

mae=[]
mse=[]
rmse=[]

r2=[]

site = ['E12000001','E12000002','E12000003','E12000004','E12000005','E12000006','E12000007','E12000008','E12000009']

df_clean=df.dropna()

# Loop over each site, each time producing a multiple linear regression result. REsults are stored and also plotted for comparison
# for each site. 
for i in range(0,len(site)):
    
    # find data associated with the site "i"
    df_clean_site = df_clean.loc[df_clean['region'] == site[i]]
    
    # Define x and y data
    y = df_clean_site['mortality_rate']
    X = df_clean_site[['O3','PM10','T2M']]
    
    
    # Create training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Initialise the linear regression
    lm = LinearRegression()
    
    # Fit data
    lm.fit(X_train,y_train)
    
    ############################
    
    # Here I was info from the statsmodels library such as P values.
    
    # Create a dummy dataframe for testing
    y_df = pd.DataFrame(data=y_train, columns=['mortality_rate'])
    dummyset = pd.concat([y_df,X_train],axis=1)
    
    # Use ols to predict  r2 values
    model1 = sm.ols(formula='mortality_rate ~ O3 + PM10 + T2M', data=dummyset)
    fitted1 = model1.fit()
    pv.append(fitted1.pvalues)
    fitted1.summary()
    
    ############################
    
    # Append coefficient and intercept data to each variable for each site "i"  
    o3.append([lm.coef_[0],lm.intercept_])
    pm10.append([lm.coef_[1],lm.intercept_])
    t2m.append([lm.coef_[2],lm.intercept_])
    

    # Plot coefficients
    f=plt.figure(1,figsize=[12,3])
    plt.plot(site[i],lm.coef_[0],'ro')
    plt.plot(site[i],lm.coef_[1],'bo')
    plt.plot(site[i],lm.coef_[2],'go')
    
    plt.xlabel('Site name')
    plt.ylabel('Coefficient value')
    f.legend(['O3','PM10','T2M'])
    
    # Now make predictions based on fit and X_test
    predictions = lm.predict( X_test)
    
    # Output in terms of errors to see how good our model is
    mae.append(metrics.mean_absolute_error(y_test, predictions))
    mse.append(metrics.mean_squared_error(y_test, predictions))
    rmse.append(np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    r2.append(metrics.r2_score(y_test, predictions))
       
    
    # Plot RMSE and R2
    f=plt.figure(2,figsize=[12,3])
    plt.plot(site[i],rmse[i],'ro',site[i],r2[i],'bo')
    plt.xlabel('Site name')
    f.legend(['RMSE','R2'])
    
    
    # Plot predicited vs Y_test
    f=plt.figure(3,figsize=[12,3])
    plt.scatter(y_test,predictions)
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')
    f.legend(['E12000001','E12000002','E12000003','E12000004','E12000005','E12000006','E12000007','E12000008','E12000009'])
    
    # A KDE plot of the distribution of residuals at each site
    f=plt.figure(4,figsize=[12,6])
    sns.distplot((y_test-predictions),bins=50,rug=False, hist=False);
    f.legend(['E12000001','E12000002','E12000003','E12000004','E12000005','E12000006','E12000007','E12000008','E12000009'])

    


# In[18]:


print(pv)


# In[19]:


# Mean coefficient values

o3bar = sum(np.array(o3))/9
pm10bar = sum(np.array(pm10))/9
t2mbar = sum(np.array(t2m))/9
rmsebar = ((sum(np.array(rmse)**2))/9)**0.5
print(str(rmsebar))

df_average = pd.DataFrame({'O3':[o3bar[0],o3bar[1]],'PM10':[pm10bar[0],pm10bar[1]],'T2M':[t2mbar[0],t2mbar[1]]},index='M C'.split())

df_average.head()



# In[20]:


# So what do the coefficients mean on average?

# Holding all other features fixed, a 1 unit increase in O3 is associated with a dencrease of 0.001 in mortality rate.
# Holding all other features fixed, a 1 unit increase in PM10 is associated with an increase of 0.001 in mortality rate.
# Holding all other features fixed, a 1 unit increase in 2M Temperature is associated with an decrease of 0.024 in mortality rate. .

# Both O3 and PM10 show opposite effects, although a decrease in O3 means an increase in NO2. The impact of temperature is most renounced.

# P-values show Temperature to reject null hypotheses consistently, whilst O3 and PM10 show only a few occasions where a null hypothesis 
# can be rejected. Maybe that both O3 and PM10 are just adding noise are not significant contributors here.


# Perhaps temperature alone is the best variable to use?
# Also have not considered time, is there a seasonal effect?


# In[21]:



fig, ax1 = plt.subplots(figsize=(12,3))
ax1.set_xticks([2007,2008,2009,2010,2011,2012])

ax1.plot(df_clean_site['date'],df_clean_site['mortality_rate'],'b',label="mortality rate")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax2 = ax1.twinx()
ax2.plot(df_clean_site['date'],df_clean_site['T2M'],'r',label="T2M")
plt.legend(bbox_to_anchor=(1.05, 1.12), loc=2, borderaxespad=0.)
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Mortality rate ): ')
ax2.set_ylabel('Temp @ 2m (K) ')



# In[22]:


# Now consider using all data to produce a model, but only considering temperature as it is the varialbe which rejects the null hypothesis
# Using ideas from https://machinelearningmastery.com/time-series-seasonality-with-python/
# https://pandas.pydata.org/pandas-docs/version/0.18/generated/pandas.Series.apply.html
# Attempt to fit T2M, all stations, and seasonal cycle

# Create dummy variables for regions. 1: when region, 0: when not
df_clean1 = df_clean
regionuk = pd.get_dummies(df_clean['region'])
# Combine together
df_clean1 = pd.concat([df_clean,regionuk],axis=1)

# Now fit seasonal cycle using idea of fitting a curve using analogy of y = x^4*b1 + x^3*b2 + x^2*b3 + x^1*b4 + b5y
# "The seasonal component in a given time series is likely a sine wave over a generally fixed period and amplitude. 
# This can be approximated easily using a curve-fitting method."

df_clean1['dnum1'] = (df_clean1['date'].apply(lambda i: i.dayofyear if i else ''))**4
df_clean1['dnum2'] = (df_clean1['date'].apply(lambda i: i.dayofyear if i else ''))**3
df_clean1['dnum3'] = (df_clean1['date'].apply(lambda i: i.dayofyear if i else ''))**2
df_clean1['dnum4'] = (df_clean1['date'].apply(lambda i: i.dayofyear if i else ''))**1

df_clean1.head(366)


# In[23]:


# Define x and y data
y = df_clean1['mortality_rate']
X = df_clean1[['T2M','E12000001','E12000002','E12000003', 'E12000004','E12000005', 'E12000006','E12000007', 'E12000008','E12000009','dnum1','dnum2','dnum3','dnum4']]


# Create training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Initialise the linear regression
lm = LinearRegression()

# Fit data
lm.fit(X_train,y_train)

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[24]:


############################

# Here I use info from the statsmodels library such as r2 values. Possibility also for P-values as well

# Create a dummy dataframe for testing
y_df = pd.DataFrame(data=y_train, columns=['mortality_rate'])
dummyset = pd.concat([y_df,X_train],axis=1)

# Use ols to predict  r2 values
model1 = sm.ols(formula='mortality_rate ~ T2M + E12000001 + E12000002 + E12000003 + E12000004 + E12000005 + E12000006 + E12000007 + E12000008 + E12000009 + dnum1 + dnum2 + dnum3 + dnum4', data=dummyset)
fitted1 = model1.fit()
r2.append(fitted1.rsquared)
pv.append(fitted1.pvalues)
fitted1.summary()

############################


# In[25]:


# Now make predictions based on fit and X_test
predictions = lm.predict( X_test)


# In[26]:


# Output in terms of errors to see how good our model is
# https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/

mae=metrics.mean_absolute_error(y_test, predictions)
mse=metrics.mean_squared_error(y_test, predictions)
rmse=np.sqrt(metrics.mean_squared_error(y_test, predictions))
r2 = metrics.r2_score(y_test, predictions)

print('MAE:'+str(mae))
print('MSE:'+str(mse))
print('RMSE:'+str(rmse))
print('R^2:' + str(r2))


# In[27]:


plt.plot(predictions,y_test,'o',[0.5,2.5],[0.5,2.5])
plt.xlabel('Predictions')
plt.ylabel('Mortality rates')
plt.title('Predictions vs mortality rates')


# In[28]:


# Differences in statmodels and scikit-learn outputs can be realted different styles of approach. Scikit-learn better for ML, 
# but statmodels better for statisitics

# https://stats.stackexchange.com/questions/6/the-two-cultures-statistics-vs-machine-learning

