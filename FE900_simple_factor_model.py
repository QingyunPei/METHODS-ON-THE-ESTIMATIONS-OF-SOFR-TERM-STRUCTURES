# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 23:48:32 2021

@author: Qingyun Pei
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

plot_3_1 = pd.read_csv('EFFR.csv')
#%%
fig = plt.figure(figsize=(16, 12)) 
plt.plot(plot_3_1['Date'],plot_3_1['SOFRRATE'],color='Black',label='SOFR')
plt.plot(plot_3_1['Date'],plot_3_1['FEDL01'],color='Red',linestyle="--",label='EFFR')
plt.ylabel('Rate')
plt.xlabel('Time')
plt.legend()
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100))
plot_3_1.Date = pd.to_datetime(plot_3_1.Date)
xfmt = mdates.DateFormatter('%m/%d/%y')
# fig.xaxis.set_major_formatter(xfmt)

#%%
## Estimate the term-structure of SOFR with simple idea
data = pd.read_csv('SOFR_Futures_Mar31_3.csv')
futures = pd.read_csv('Futures.csv')
# Theta = np.zeros((1,10))
Month = [4,5,6,7,8,9,10,11,12,1,2,3]

#%%
def Forward_rate(data,Theta):
    n = len(data.iloc[:,0])
    Forward = np.zeros((n,1))
    for i in range(0,n):
        count = data.iloc[i,5]
        for j in range(0,count):
            Forward[i,0] = Forward[i,0]+Theta[0,j]
    return(Forward)


def One_month_futures(data,Forward,price_real):
    Months_date = 21
    n = price_real.dropna().shape[0]
    if(n >= 21):
        One_month_estimate = 100-Forward[(n-Months_date):n,0].sum()/Months_date
        One_month = One_month_estimate
    else:
        One_month_estimate = Forward[0:n,0].sum()
        One_month_real = data.iloc[0:(Months_date-n),9].sum()
        One_month =100-(One_month_estimate + One_month_real)/Months_date
    return(One_month)

     
def product_inner(inputs):
    inputs = np.array(inputs)
    n = inputs.shape[0]
    product = 1
    for i in range(0,n):
        product = product*inputs[i] 
    return(product) 

def Three_month_futures(data,Forward,price_real):
    Months_date = 63
    n = price_real.dropna().shape[0]
    if(n >=63):
        Three_month_estimate = 100-252/Months_date*(product_inner(1+Forward[(n-Months_date):n,0]/360)-1)
        Three_month = Three_month_estimate
    else:
        Three_month_real = product_inner(1+data.iloc[0:(Months_date-n),9]/360)
        Three_month_estimate = product_inner(1+Forward[0:n,0]/360)
        Three_month = 100-252/Months_date*(Three_month_real*Three_month_estimate-1)    
    return(Three_month)

def Loss_one_mon(one_mon_real, one_month_estimate):
    Loss_one = (one_mon_real-one_month_estimate)**2
    return(Loss_one)

def Loss_three_mon(three_mon_real, three_month_estimate):
    Loss_three = (three_mon_real-three_month_estimate)**2
    return(Loss_three)

def Loss_function(Loss_one,Loss_three, Theta):
    Theta_MSE = Theta @ Theta.T
    Weight_index = 0.01
    weight_one = 1
    weight_three = 1
    Loss = np.sqrt(weight_one*sum(Loss_one) + weight_three*sum(Loss_three))+Weight_index*np.sqrt(Theta_MSE)
    return(Loss)

def United_Loss(data,Theta,futures,number_one):
    Loss_one = []
    Loss_three = []
    number_futures = futures.shape[1]-1
    number_three = number_futures - number_one
    Theta = np.array(Theta)
    Forward = Forward_rate(data,Theta)
    for i in range(0,number_one):
        one_month_estimate = One_month_futures(data,Forward,futures.iloc[:,i+1])
        one_month_Loss = Loss_one_mon(futures.iloc[0,i+1], one_month_estimate)
        Loss_one.append(one_month_Loss)
    for j in range(0,number_three):
        three_month_estimate = Three_month_futures(data,Forward,futures.iloc[:,number_one+j+1])
        three_month_Loss = Loss_three_mon(futures.iloc[0,number_one+j+1], three_month_estimate)
        Loss_three.append(three_month_Loss)
    Loss = Loss_function(Loss_one, Loss_three, Theta)
    return(Loss)

#%%
## The method of optimizating the miniumn value and Theta sets
# import random
number_one = 5
DNA_SIZE = 24
POP_SIZE = 50
CROSSOVER_RATE = 1
MUTATION_RATE = 0.8
N_GENERATIONS = 3000

X0_BOUND = [-0.1, 0.1]  # x0
X1_BOUND = [-0.1, 0.1]  # x1
X2_BOUND = [-0.1, 0.1]  # x2
X3_BOUND = [-0.1, 0.1]  # x3
X4_BOUND = [-0.1, 0.1]  # x4
X5_BOUND = [-0.1, 0.1]  # x5
X6_BOUND = [-0.1, 0.1]  # x6
X7_BOUND = [-0.1, 0.1]  # x7
X8_BOUND = [-0.1, 0.1]  # x8
X9_BOUND = [-0.1, 0.1]  # x9

def get_fitness(data, pop):
    pred = []
    x0, x1, x2, x3, x4, x5, x6, x7, x8 = translateDNA(pop)
    for i in range(0, len(x0)):
        Theta = np.array([[x0[i], x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i], x8[i]]])
        pred_value = United_Loss(data,Theta,futures,number_one)[0,0]
        pred.append(pred_value)
    return pred, -(pred - np.max(pred)) + 1e-3


def translateDNA(pop):  
    x0_pop = pop[:, 0:DNA_SIZE]  
    x1_pop = pop[:, DNA_SIZE:DNA_SIZE * 2]  
    x2_pop = pop[:, DNA_SIZE * 2:DNA_SIZE*3]  
    x3_pop = pop[:, DNA_SIZE * 3:DNA_SIZE*4]
    x4_pop = pop[:, DNA_SIZE * 4:DNA_SIZE*5]
    x5_pop = pop[:, DNA_SIZE * 5:DNA_SIZE*6]
    x6_pop = pop[:, DNA_SIZE * 6:DNA_SIZE*7]
    x7_pop = pop[:, DNA_SIZE * 7:DNA_SIZE*8]
    x8_pop = pop[:, DNA_SIZE * 8:]
#   x9_pop = pop[:, DNA_SIZE * 9:]
    
    # print(x0_pop)
    # print(x1_pop)
    # print(x2_pop)
    # print(x3_pop)
    # print(x4_pop)
    # print(x5_pop)
    # print(x6_pop)
    # print(x7_pop)
    # print(x8_pop)
    # print(x9_pop)
    # print("\n")

    '''pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)'''  
    x0 = x0_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X0_BOUND[1] - X0_BOUND[0]) + X0_BOUND[0]
    x1 = x1_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X1_BOUND[1] - X1_BOUND[0]) + X1_BOUND[0]
    x2 = x2_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X2_BOUND[1] - X2_BOUND[0]) + X2_BOUND[0]
    x3 = x3_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X3_BOUND[1] - X3_BOUND[0]) + X3_BOUND[0]
    x4 = x4_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X4_BOUND[1] - X4_BOUND[0]) + X4_BOUND[0]
    x5 = x5_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X5_BOUND[1] - X5_BOUND[0]) + X5_BOUND[0]
    x6 = x6_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X6_BOUND[1] - X6_BOUND[0]) + X6_BOUND[0]
    x7 = x7_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X7_BOUND[1] - X7_BOUND[0]) + X7_BOUND[0]
    x8 = x8_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X8_BOUND[1] - X8_BOUND[0]) + X8_BOUND[0]
#   x9 = x9_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X9_BOUND[1] - X9_BOUND[0]) + X9_BOUND[0]
    
    # print(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)
    return x0, x1, x2, x3, x4, x5, x6, x7, x8


def mutation(child, MUTATION_RATE):
    if np.random.rand() < MUTATION_RATE:  
        mutate_point = np.random.randint(0, DNA_SIZE * 8)  
        child[mutate_point] = child[mutate_point] ^ 1 


def crossover_and_mutation(pop, CROSSOVER_RATE):   
    new_pop = []
    for father in pop:  
        child = father  
        if np.random.rand() < CROSSOVER_RATE:  
            mother = pop[np.random.randint(POP_SIZE)]  
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 8)  
            child[cross_points:] = mother[cross_points:]  
        mutation(child, MUTATION_RATE)  
        new_pop.append(child)

    return new_pop


def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness)/(fitness.sum()) )
    return pop[idx]


def print_info(pop):
    fitness = get_fitness(data, pop)[1]
    min_fitness_index = np.argmin(fitness)
    print("min_fitness:", fitness[min_fitness_index])
    x0, x1, x2, x3, x4, x5, x6, x7, x8= translateDNA(pop)
    print("Best genetype：", pop[min_fitness_index])
    print("(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):", (x0[min_fitness_index], x1[min_fitness_index],\
                                                        x2[min_fitness_index], x3[min_fitness_index],\
                                                        x4[min_fitness_index], x5[min_fitness_index],\
                                                        x6[min_fitness_index], x7[min_fitness_index],\
                                                        x8[min_fitness_index]))

def Theta_min(pop):
    fitness = get_fitness(data, pop)[1]
    max_fitness_index = np.argmax(fitness)
    x0, x1, x2, x3, x4, x5, x6, x7, x8= translateDNA(pop)
    value_min = [[x0[max_fitness_index], x1[max_fitness_index],\
                 x2[max_fitness_index], x3[max_fitness_index],\
                 x4[max_fitness_index], x5[max_fitness_index],\
                 x6[max_fitness_index], x7[max_fitness_index],\
                 x8[max_fitness_index]]]
    value_min = np.array(value_min)
    return value_min
#%%
from tqdm import tqdm

min_error = []
iteration = []

if __name__ == "__main__":
    pop = np.random.randint(0, 2, size=(POP_SIZE, DNA_SIZE * 9))  # matrix (POP_SIZE, DNA_SIZE)
    
    for i in tqdm(range(N_GENERATIONS)):  
        x0, x1, x2, x3, x4, x5, x6, x7, x8= translateDNA(pop)
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
        fitness = get_fitness(data,pop)[1]
        pred_min = min(get_fitness(data,pop)[0])
        
        min_error.append(pred_min)
        iteration.append(i)
        
        pop = select(pop, fitness)

    min_pop = pop
    print_info(pop)

#%%
## Error Plot
import matplotlib.ticker as ticker


plt.figure(figsize=(16, 12))
plt.title('Error of the optimization problem with GA method', fontsize = 15)
plt.xlabel('Iteration times')
plt.ylabel('Error')
plt.plot(iteration, min_error, '-', label = 'Error of iterations')
plt.legend()
plt.show()

#%%    
## Plot the one month-term
Theta_mini = Theta_min(min_pop)
Forward_min = Forward_rate(data,Theta_mini)

plt.figure(figsize=(16, 12))
plt.title('Show the One month futures rate', fontsize = 15)
plt.xlabel('Time')
plt.ylabel('Rate')
plt.plot(data.Date, Forward_min, '--', label = 'One month futures estimated')
plt.plot(data.Date, data.iloc[:,7], '-', label = 'EFFR')
# plt.plot(data.Date, one_mon_min_estimate, '--', label = 'One month futures estimated')
# plt.plot(data.Date, 100-np.array([data.iloc[:,1]]).T, '-', label = 'One month futures realed')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(64))
plt.legend()
plt.show()

#%%
Term_rate = []
Term_SOFR = []
Term_EFFR = []
n = len(Forward_min)
for i in range(0,n):
    term = 1
    term_SOFR = 1
    term_EFFR = 1
    for j in range(0,i+1):
        term = term*(1+Forward_min[j]/252)
        term_SOFR = term_SOFR*(1+data.iloc[j,3]/252)
        term_EFFR = term_EFFR*(1+data.iloc[j,7]/252)
    term_rate_sing = (term)**(252/(i+1))-1
    term_SOFR_sing = (term_SOFR)**(252/(i+1))-1
    term_EFFR_sing = (term_EFFR)**(252/(i+1))-1
    Term_rate.append(term_rate_sing)
    Term_SOFR.append(term_SOFR_sing)
    Term_EFFR.append(term_EFFR_sing)

plt.figure(figsize=(16, 12))
plt.title('Estimated term rate', fontsize = 15)
plt.xlabel('Time')
plt.ylabel('Rate')
plt.plot(data.Date, Term_rate, '--', label = 'Estimated Term', color = 'blue')
plt.plot(data.Date, Term_SOFR, '-', label = 'SOFR Term', color = 'green')
plt.plot(data.Date, Term_EFFR, '-.', label = 'EFFR Term', color = 'red')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(64))
plt.legend()
plt.show()

#%%
n = len(Forward_min)
Term_rate_dis = []
Term_SOFR_dis = []
Term_EFFR_dis = []  
for i in range(0,n):
    a = 1/(1+Term_rate[i])**(i/252)
    Term_rate_dis.append(a)
    b = 1/(1+Term_SOFR[i])**(i/252)
    Term_SOFR_dis.append(b)
    c = 1/(1+Term_EFFR[i])**(i/252)
    Term_EFFR_dis.append(c)
    
plt.figure(figsize=(16, 12))
plt.title('Estimated discounded term rate', fontsize = 15)
plt.xlabel('Time')
plt.ylabel('Rate')
plt.plot(data.Date, Term_rate_dis, '--', label = 'Estimated discounded Term', color = 'blue')
plt.plot(data.Date, Term_SOFR_dis, '-', label = 'SOFR discounded Term', color = 'green')
plt.plot(data.Date, Term_EFFR_dis, '-.', label = 'EFFR discounded Term', color = 'red')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(64))
plt.legend()
plt.show()
