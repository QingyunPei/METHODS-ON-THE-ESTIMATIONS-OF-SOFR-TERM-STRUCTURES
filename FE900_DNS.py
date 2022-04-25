# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 00:13:54 2021

@author: Qingyun Pei
"""
## DNS model
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
# import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error


#%%
tao = [i+0.001 for i in range(0,121)]
fixed_lambda = 0.0609
beta_1_loading = [1.0]*len(tao)
beta_2_loading = [0.0]*len(tao)
beta_3_loading = [0.0]*len(tao)
for i in range(len(tao)):
    beta_2_loading[i] = (1-np.exp(-fixed_lambda*tao[i]))/(fixed_lambda*tao[i])
    beta_3_loading[i] = (1-np.exp(-fixed_lambda*tao[i]))/(fixed_lambda*tao[i])-np.exp(-fixed_lambda*tao[i])


fig0=plt.figure(figsize=(20,10),dpi=80,num=4)
plt.plot(tao, beta_1_loading, '-',label='Level')
plt.plot(tao, beta_2_loading, ':',label='Slpoe')
plt.plot(tao, beta_3_loading, '-.',label='Curvature')

plt.xlabel('Months',fontsize = 20)
plt.ylabel('Value',fontsize = 20)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(loc='upper right',fontsize = 20) 
# plt.title('Factor loading plot',fontsize = 15)
plt.show()

#%%
# import data
data = pd.read_csv('SOFR.csv')

#%%
## normal regression
## Finxed number
lambda_0 = 0.0518
#tao = np.array([1/21, 1/4, 1, 2, 3, 6, 12])
tao = np.array([1/52, 1/26, 1/12, 1/6, 1/4, 1/2, 1])
level = np.array([1]*len(tao))
slope = (1 - np.exp(-lambda_0*tao))/(lambda_0*tao)
curv = slope - np.exp(-lambda_0*tao)
y_sample = np.array(data.iloc[118,1:8])
# y_sample = np.array([0.0155,0.0155,0.015616,0.015551, 0.0156914, 0.0121467, 0.0062416, 0.0035846])
fixed_df = pd.DataFrame([y_sample,level,slope,curv], index=['values', 'level', 'slope','curvature'])
df2 = pd.DataFrame(fixed_df.values.T, columns=fixed_df.index)
## Normal nelspon seigel
lm = ols('values ~ level + slope + curvature -1', df2).fit()
print(lm.summary())
# lm.resid ##residules

## Curve Plot
plt.plot(tao, y_sample, '*', label='Date:2020/09/04', color='black')
y_sample_1 = np.array(data.iloc[119,1:8])
plt.plot(tao, y_sample_1, '*', label='Date:2020/09/11', color='red')

tao_plot = [0]*200
y_series = [0]*200
for i in range(0,200):
    tao_plot[i] = 0.1*(i+1)
    level_plot = 1
    slope_plot = (1 - np.exp(-lambda_0*tao_plot[i]))/(lambda_0*tao_plot[i])
    curv_plot = slope_plot - np.exp(-lambda_0*tao_plot[i])
    y_series[i] = lm.params[0]*level_plot+lm.params[1]*slope_plot+lm.params[2]*curv_plot
    
plt.plot(tao_plot, y_series)
plt.xlabel('Months',fontsize = 15)
plt.ylabel('Yield(%)',fontsize = 15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title('Yield Curve with statistic Nelson-seigel Model')
plt.legend()
plt.show() 

#%% Dynamic Nelson Seigel model
# Index series
beta_0 = []
beta_1 = []
beta_2 = []

# tao = np.array([1/21, 1/4, 1, 2, 3, 6, 12])
tao = np.array([1/52, 1/26, 1/12, 1/6, 1/4, 1/2, 1])
def cancel_navalue_regression(tao,y_value):
    n = len(y_value)
    y = []
    tao_select = []
    for i in range(0,n):
        if np.isnan(y_value[i]):
            continue
        else:
            y.append(y_value[i])
            tao_select.append(tao[i])
    tao_select = np.array(tao_select)
    y = np.array(y)
    level_select = np.array([1]*len(tao_select))
    slope_select = (1 - np.exp(-lambda_0*tao_select))/(lambda_0*tao_select)
    curv_select = slope_select - np.exp(-lambda_0*tao_select)
    fixed_df_select = pd.DataFrame([y,level_select,slope_select,curv_select], index=['values', 'level', 'slope','curvature'])
    df2_select = pd.DataFrame(fixed_df_select.values.T, columns=fixed_df_select.index)
    lm_select = ols('values ~ level + slope + curvature -1', df2_select).fit()
    coeff = [lm_select.params[0],lm_select.params[1],lm_select.params[2]]
    return(coeff)

data_rows = data.shape[0]
for i in range(0,data_rows):
    coeff = cancel_navalue_regression(tao,data.iloc[i,1:8]/100)
    beta_0.append(coeff[0])
    beta_1.append(coeff[1])
    beta_2.append(coeff[2])

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(3, 1, 1)    
plt.plot(data['Date'],beta_0)
plt.ylabel('Beta_1')
# plt.xlabel('Time to maturity')
data.Date = pd.to_datetime(data.Date)
xfmt = mdates.DateFormatter('%m/%d/%y')
ax.xaxis.set_major_formatter(xfmt)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100))

ax = fig.add_subplot(3, 1, 2)    
plt.plot(data['Date'],beta_1)
plt.ylabel('Beta_2')
# plt.xlabel('Time to maturity')
data.Date = pd.to_datetime(data.Date)
xfmt = mdates.DateFormatter('%m/%d/%y')
ax.xaxis.set_major_formatter(xfmt)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100))

ax = fig.add_subplot(3, 1, 3)    
plt.plot(data['Date'],beta_2)
plt.ylabel('Beta_3')
plt.xlabel('Time')
data.Date = pd.to_datetime(data.Date)
xfmt = mdates.DateFormatter('%m/%d/%y')
ax.xaxis.set_major_formatter(xfmt)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100))

#%%
## Creat the data frame series
Coefficient_beta_trans = pd.DataFrame([np.array(beta_0),np.array(beta_1),np.array(beta_2)], index=['Beta_0', 'Beta_1', 'Beta_2'], columns=data.Date)
Coefficient_beta = pd.DataFrame(Coefficient_beta_trans.T, columns=Coefficient_beta_trans.index, index=Coefficient_beta_trans.columns)

#%%
Size = int(Coefficient_beta.shape[0] * 0.33)
Train, Test = Coefficient_beta.iloc[0:Size,:], Coefficient_beta.iloc[Size:Coefficient_beta.shape[0]+1,:]
## Time series analysis of the indexes
Coeff_diff = Train.diff()
# Coeff_diff_2 = Coeff_diff.diff()
Coeff_diff = Coeff_diff.dropna()
# Coeff_diff_2 = Coeff_diff.dropna()

plt.figure(figsize=(15, 9))
plt.plot(Coeff_diff.iloc[:,0])
plt.xlabel('Time',fontsize = 13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
# plt.title('First order differnce')

acf = plot_acf(Coeff_diff.iloc[:,2], lags=20)
plt.xlabel('Lag')
plt.title("ACF")

pacf = plot_pacf(Coeff_diff.iloc[:,2], lags=20)
plt.xlabel('Lag')
plt.title("PACF")
pacf.show()

#%%
model = ARIMA(Coeff_diff.iloc[:,2], order=(0, 1, 1))
result = model.fit()
print(result.summary())

# line plot of residuals
residuals = pd.DataFrame(result.resid)
residuals.plot(figsize=(15, 9))
plt.show()
# density plot of residuals
residuals.plot(kind='kde',figsize=(15, 9))
plt.show()
# summary stats of residuals
print(residuals.describe())

#%% Beta_0 expected
predictions_beta_0 = []
number_of_coeff = 0
history = [x for x in Train.iloc[:,number_of_coeff]]
## Rolling down
for t in range(Test.shape[0]):
	model = ARIMA(history, order=(0,1,1))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions_beta_0.append(yhat)
	obs = Test.iloc[t,number_of_coeff]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))

# evaluate forecasts
rmse = np.sqrt(mean_squared_error(Test.iloc[:,0], predictions_beta_0))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
plt.figure(figsize=(15, 9))
plt.plot(Train.iloc[:,number_of_coeff],color='Green',label = 'Training data')
plt.plot(Test.iloc[:,number_of_coeff],label = 'Testing data')
plt.plot(data.iloc[Size-1:Coefficient_beta.shape[0]-1,0],predictions_beta_0, color='red',label = 'Estimated data')
plt.legend()
plt.show()

#%% Beta_1 expected
predictions_beta_1 = []
number_of_coeff = 1
history = [x for x in Train.iloc[:,number_of_coeff]]
## Rolling down
for t in range(Test.shape[0]):
	model = ARIMA(history, order=(0,1,1))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions_beta_1.append(yhat)
	obs = Test.iloc[t,number_of_coeff]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))

# evaluate forecasts
rmse = np.sqrt(mean_squared_error(Test.iloc[:,0], predictions_beta_1))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
plt.figure(figsize=(15, 9))
plt.plot(Train.iloc[:,number_of_coeff],color='Green',label = 'Training data')
plt.plot(Test.iloc[:,number_of_coeff],label = 'Testing data')
plt.plot(data.iloc[Size-1:Coefficient_beta.shape[0]-1,0],predictions_beta_1, color='red',label = 'Estimated data')
plt.legend()
plt.show()

#%% Beta_2 expected
predictions_beta_2 = []
number_of_coeff = 2
history = [x for x in Train.iloc[:,number_of_coeff]]
## Rolling down
for t in range(Test.shape[0]):
	model = ARIMA(history, order=(0,1,1))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions_beta_2.append(yhat)
	obs = Test.iloc[t,number_of_coeff]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))

# evaluate forecasts
rmse = np.sqrt(mean_squared_error(Test.iloc[:,0], predictions_beta_2))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
plt.figure(figsize=(15, 9))
plt.plot(Train.iloc[:,number_of_coeff],color='Green',label = 'Training data')
plt.plot(Test.iloc[:,number_of_coeff],label = 'Testing data')
plt.plot(data.iloc[Size-1:Coefficient_beta.shape[0]-1,0],predictions_beta_2, color='red',label = 'Estimated data')
plt.legend()
plt.show()

#%% Compare the expected rate with real rate in dofferent tenors
predict_rate = []
for i in range(0,len(tao)):
    for j in range(0,len(predictions_beta_0)):
        rate_temp = predictions_beta_0[j]*fixed_df.iloc[1,i]+predictions_beta_1[j]*fixed_df.iloc[2,i]+predictions_beta_2[j]*fixed_df.iloc[3,i]
        predict_rate.append(rate_temp)

rate_pred = np.array(predict_rate).reshape(len(tao),-1)*100

## Plot rate
for i in range(0,len(tao)):
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(data.iloc[0:Size-1,0],data.iloc[0:Size-1,i+1],color='Green',label = 'Training data')
    plt.plot(data.iloc[Size-1:Coefficient_beta.shape[0]-1,0],data.iloc[Size-1:Coefficient_beta.shape[0]-1,i+1],label = 'Testing data')
    plt.plot(data.iloc[Size-1:Coefficient_beta.shape[0]-1,0],rate_pred[i,:], color='red',label = 'Estimated data')
    plt.legend()
    plt.show()

#%%
