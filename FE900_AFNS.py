# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 17:08:50 2021

@author: Qingyun Pei
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.formula.api import ols
    

#%% Data
data = pd.read_csv('SOFR.csv')
y_real = np.mat(data.iloc[:,1:8])
tao = [1/52, 1/26, 1/12, 1/6, 1/4, 1/2, 1]
lambda_0 = 0.0609

## We need some trainning historical regressison data
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

beta_0 = []
beta_1 = []
beta_2 = []

data_rows = data.shape[0]
for i in range(0,data_rows):
    coeff = cancel_navalue_regression(tao,data.iloc[i,1:8]/100)
    beta_0.append(coeff[0])
    beta_1.append(coeff[1])
    beta_2.append(coeff[2])

#%%
## Define Functions
## Define the Control matrix
def Control(Sigma_mat, lambda_0, T, t):
    A = Sigma_mat[0,0]**2+Sigma_mat[0,1]**2+Sigma_mat[0,2]**2
    B = Sigma_mat[1,0]**2+Sigma_mat[1,1]**2+Sigma_mat[1,2]**2
    C = Sigma_mat[2,0]**2+Sigma_mat[2,1]**2+Sigma_mat[2,2]**2
    D = Sigma_mat[0,0]*Sigma_mat[1,0]+Sigma_mat[0,1]*Sigma_mat[1,1]+Sigma_mat[0,2]*Sigma_mat[1,2]
    E = Sigma_mat[0,0]*Sigma_mat[2,0]+Sigma_mat[0,1]*Sigma_mat[2,1]+Sigma_mat[0,2]*Sigma_mat[2,2]
    F = Sigma_mat[1,0]*Sigma_mat[2,0]+Sigma_mat[1,1]*Sigma_mat[2,1]+Sigma_mat[1,2]*Sigma_mat[2,2]
    I_1 = A/6*(T-t)**2
    I_2 = B*(1/(2*lambda_0**2)- 1/lambda_0**3*(1-np.exp(-lambda_0*(T-t)))/(T-t)+ 1/(4*lambda_0**3)*(1-np.exp(-2*lambda_0*(T-t)))/(T-t))
    I_3 = C*(1/(2*lambda_0**2)+ 1/lambda_0**2*np.exp(-lambda_0*(T-t))- 1/(4*lambda_0)*(T-t)*np.exp(-2*lambda_0*(T-t))-\
             3/(4*lambda_0**2)*np.exp(-2*lambda_0*(T-t))- 2/lambda_0**3*(1-np.exp(-lambda_0*(T-t)))/(T-t)+\
             5/(8*lambda_0**3)*(1-np.exp(-2*lambda_0*(T-t)))/(T-t))
    I_4 = D*(1/(2*lambda_0)*(T-t)+ 1/lambda_0**2*np.exp(-lambda_0*(T-t))- 1/lambda_0**3*(1-np.exp(-lambda_0*(T-t)))/(T-t))
    I_5 = E*(3/lambda_0**2*np.exp(-lambda_0*(T-t))+ 1/(2*lambda_0)*(T-t)+ 1/lambda_0*(T-t)*np.exp(-lambda_0*(T-t))- 3/lambda_0**3*(1-np.exp(-lambda_0*(T-t)))/(T-t))
    I_6 = F*(1/lambda_0**2+ 1/lambda_0**2*np.exp(-lambda_0*(T-t))- 1/(2*lambda_0**2)*np.exp(-2*lambda_0*(T-t))-\
             3/lambda_0**3*(1-np.exp(-lambda_0*(T-t)))/(T-t)+ 3/(4*lambda_0**3)*(1-np.exp(-2*lambda_0*(T-t)))/(T-t))
    sum_I = I_1+I_2+I_3+I_4+I_5+I_6
    return(sum_I)

def fun_sigma_0(Sigma_mat, K, s):
    return np.exp(-K*s)@Sigma_mat@Sigma_mat.T@np.exp(-K.T*s)

def integral_Q(Sigma_mat,K_P,a,b):
    n=10
    interval = (b-a)/n
    sum = 0
    x = a
    for i in range(0,n+1):
        x = x+i*interval
        fun_value = fun_sigma_0(Sigma_mat, K_P, x)*interval
        sum = sum+fun_value
    return(sum)

def B(t,T,lambda_0):
    B = np.mat([[0.0],[0.0],[0.0]])
    B[0,0] = -(T-t)
    B[1,0] = -(1-np.exp(-lambda_0*(T-t)))/lambda_0
    B[2,0] = (T-t)*np.exp(-lambda_0*(T-t))+B[1,0]
    return(B)

def mean_coeff(X_test):
    mean = []
    for i in range(0,len(X_test)):
        mean_temp = np.mean(X_test[i])
        mean.append(mean_temp)
    mean = np.array([mean])
    return(mean.T)

def Kalman_likelihood(tao_i, Kappa_mat, theta, Sigma_mat, Trans_mat, Control_vec, y_real, number):
    X_list = [theta]
    for i in range(0,len(tao_i)-1):
        X_t_t1 = (np.identity(3)-np.exp(-Kappa_mat*(tao_i[i+1]-tao_i[i])))@theta + np.exp(-Kappa_mat*(tao_i[i+1]-tao_i[i]))@X_list[i]
        X_list.append(X_t_t1)
        Var_t_t1 =  np.exp(-Kappa_mat*(tao_i[i+1]-tao_i[i]))@ Sigma_mat @ np.exp(-Kappa_mat*(tao_i[i+1]-tao_i[i])).T + integral_Q(Sigma_mat,Kappa_mat,0,(tao_i[i+1]-tao_i[i]))
        F_t_Iv = (Trans_mat[i,:] @ Var_t_t1 @ Trans_mat[i,:].T + (tao_i[i+1]-tao_i[i])).I
#       Kalmen = Var_t_t1 @ Trans_mat[i,:].T @ F_t_Iv
        y_estimated =  Control_vec[i] + Trans_mat[i,:] @ X_t_t1
        v_t = y_real[number-1][0,i]/100-y_estimated
        X_t = X_t_t1 + Var_t_t1 @ Trans_mat[i,:].T @ F_t_Iv @ v_t
        Sigma_mat = Var_t_t1 - Var_t_t1 @ Trans_mat[i,:].T @ F_t_Iv @ Trans_mat[i,:] @ Var_t_t1.T
    return(X_t)

#%%
## Initial data, first 50 rows
## beta_0_test = beta_0[0:50]
## beta_1_test = beta_1[0:50]
## beta_2_test = beta_2[0:50]

## X_test = np.array([beta_0_test, beta_1_test, beta_2_test])
## Cov_test = np.cov(X_test)
## mean_test = mean_coeff(X_test)



#%%
## Define the Transoformation matrix
Trans = []
Kappa_mat = -np.mat([[0,0,0],
                   [0,lambda_0,-lambda_0],
                   [0,0,lambda_0]])

Sigma_tog = []
for i in tao:
    T = i
    t = 0
    coef_1 = 1
#   coef_1 = -(T-t)
    Trans.append(coef_1)
    coef_2 = (1 - np.exp(-lambda_0*i))/(lambda_0*i)
#   coef_2 = -(1-np.exp(-lambda_0*(T-t)))/lambda_0
    Trans.append(coef_2)
    coef_3 = coef_2 - np.exp(-lambda_0*i)
#   coef_3 = (T-t)*np.exp(-lambda_0*(T-t))+coef_2
    Trans.append(coef_3)
Trans_mat = np.mat(Trans).reshape(-1,3)


## Kalmen Filter
beta_0_afns = []
beta_1_afns = []
beta_2_afns = []
Compansate = []
error = []

tao_i = [0, 1/52, 1/26, 1/12, 1/6, 1/4, 1/2, 1]
## testing set number is 50, the remainning is 162-50 = 112
test_number = 4
number_remain = 162-test_number
for j in range(0,number_remain):
    # Mean and Covariance
    beta_0_test = beta_0[j:test_number+j]
    beta_1_test = beta_1[j:test_number+j]
    beta_2_test = beta_2[j:test_number+j]   
    X_test = np.array([beta_0_test, beta_1_test, beta_2_test])
    Cov_test = np.cov(X_test)
    mean_test = mean_coeff(X_test)
    
    # define theta and sigma
    Sigma_mat = Cov_test
    theta = mean_test
    
    ## Define Control matrix
    Control_list = []
    for i in tao:
    #   Control_list_temp = Control(Sigma_mat, lambda_0, i, 0)
        Control_list_temp = -Control(Sigma_mat, lambda_0, i, 0)/i
        Control_list.append(Control_list_temp)
    Compansate.append(Control_list)
    Control_vec = np.mat([Control_list]).T
  
    
    ## Define Sigma update
    Sigma_mat_0 = integral_Q(Sigma_mat,Kappa_mat,0,6)   
    
    ## H: Variance of white noice
#   H = np.identity(1)
    
    ## delete NA
    tao_i_na = [0]
    for k in range(0,len(tao)):
        if np.isnan(y_real[j+test_number,k]):
            continue
        else:
            tao_i_na.append(tao[k])
    
    num_without_na = len(tao_i_na)-1
    

    ## Start Kalman
    X_estimated = Kalman_likelihood(tao_i_na, Kappa_mat, theta, Sigma_mat_0, Trans_mat, Control_vec, y_real, j+test_number)
    beta_0_afns.append(X_estimated[0,0])
    beta_1_afns.append(X_estimated[1,0])
    beta_2_afns.append(X_estimated[2,0])
    error_estimated = sum(y_real[j+test_number,0:num_without_na].T/100-Trans_mat[0:num_without_na,:] @ X_estimated)
    error.append(error_estimated[0,0])

# error = y_real[1,:].T/100-Trans_mat @ X_t


#   y_estimated = Control_vec + Trans_mat@X_list[i]
#   print(y_estimated)
#   Cov_mat = np.exp(-Kappa_mat*1/52)@ Sigma_mat @ np.exp(-Kappa_mat*1/52).T + integral_Q(Sigma_mat,Kappa_mat,0,1/52)
#   print(Cov_mat)
#   Kalman = Cov_mat@Trans_mat.T@(Trans_mat @ Cov_mat @ Trans_mat.T + H).I
#   print(Kalman)
#   X_t1 = (np.identity(3)-np.exp(-Kappa_mat*1/52))@theta + np.exp(-Kappa_mat*1/52)@X_list[i] + Kalman@(y_real[i+1].T-y_estimated)
#   X_list.append(X_t1)
#   print(X_t1)
#   Sigma_mat = (np.identity(3)-Kalman@Trans_mat)@Cov_mat
#   print(i)

#%% Coefficient Plot
## Beta_0 AFNS
plt.figure(figsize=(15, 9))
plt.plot(data['Date'][0:test_number], beta_0[0:test_number],color='Green',label = 'Training data')
plt.plot(data['Date'][test_number:162], beta_0[test_number:162],label = 'Testing data')
plt.plot(data['Date'][test_number:162],beta_0_afns, color='red',label = 'Estimated data')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
plt.legend()
plt.show()

## Beta_1 AFNS
plt.figure(figsize=(15, 9))
plt.plot(data['Date'][0:test_number], beta_1[0:test_number],color='Green',label = 'Training data')
plt.plot(data['Date'][test_number:162], beta_1[test_number:162],label = 'Testing data')
plt.plot(data['Date'][test_number:162],beta_1_afns, color='red',label = 'Estimated data')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
plt.legend()
plt.show()

## Beta_2 AFNS
plt.figure(figsize=(15, 9))
plt.plot(data['Date'][0:test_number], beta_2[0:test_number],color='Green',label = 'Training data')
plt.plot(data['Date'][test_number:162], beta_2[test_number:162],label = 'Testing data')
plt.plot(data['Date'][test_number:162],beta_2_afns, color='red',label = 'Estimated data')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
plt.legend()
plt.show()


#%% Compare the expected rate with real rate in dofferent tenors
predict_rate = []
for i in range(0,len(tao)):
    for j in range(0,len(beta_0_afns)):
        rate_temp = beta_0_afns[j]*Trans_mat[i,0]+beta_1_afns[j]*Trans_mat[i,1]+beta_2_afns[j]*Trans_mat[i,2]
        predict_rate.append(rate_temp)

rate_pred_1 = np.array(predict_rate).reshape(len(tao),-1)*100

## Plot rate
for i in range(0,len(tao)):
    plt.figure(figsize=(15, 9))
    plt.plot(data.iloc[0:test_number,0],data.iloc[0:test_number,i+1],color='Green',label = 'Training data')
    plt.plot(data.iloc[test_number:162,0],data.iloc[test_number:162,i+1],label = 'Testing data')
    plt.plot(data.iloc[test_number:162,0],rate_pred_1[i,:], color='red',label = 'Estimated data via AFNS')
#   plt.plot(data.iloc[53:162,0],rate_pred[i,:], color='orange',label = 'Estimated data via AFNS')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100))
    plt.legend()
    plt.show()

#%%
## Error plot
plt.figure(figsize=(15, 9))
plt.plot(data['Date'][test_number:162],error, color='red',label = 'Estimated data')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
plt.legend()
plt.show()

#%%
## loading plot
Coeff_dns = [0.4236, -0.4228, -0.4506]
Coeff_afns = [0.1611, -0.1603, -0.1792]
data_norm = [0.08, 0.07429, 0.07742, 0.0746, 0.07246, 0.05385, 0.04576]
tao_i = [1/252, 1/52, 1/12, 1/6, 1/4, 1/2, 1]
lambda_0 = 0.0518
pred_coeff_dns = []
pred_coeff_afns = []
arr = []
for j in range(0,53):
    i = j/52
    arr.append(i)
    level = 1
    slope = (1 - np.exp(-lambda_0*i))/(lambda_0*i)
    curv = slope - np.exp(-lambda_0*i)
    pred_coeff_dns_temp = level*Coeff_dns[0]+slope*Coeff_dns[1]+curv*Coeff_dns[2]
    pred_coeff_dns.append(pred_coeff_dns_temp*100)
    pred_coeff_afns_temp = level*Coeff_afns[0]+slope*Coeff_afns[1]+curv*Coeff_afns[2]
    pred_coeff_afns.append(pred_coeff_afns_temp*100)

plt.figure(figsize=(15, 9))
plt.plot(tao_i,data_norm, 'x', color='black',label = 'Real data')
plt.plot(arr,pred_coeff_dns, '--', color='black',label = 'Dynamic data')
plt.plot(arr,pred_coeff_afns, '-.', color='black',label = 'Arbitrage-free data')
plt.legend()
plt.show()
