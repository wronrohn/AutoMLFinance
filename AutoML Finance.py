import quandl
import random as rand
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#initialization of parameters
dow30=["MMM","AXP","AAPL","BA","CAT","CVX","CSCO"]
action=[(1,0,0,0,0,0,0),(0,1,0,0,0,0,0),(0,0,1,0,0,0,0),(0,0,0,1,0,0,0),(0,0,0,0,1,0,0),(0,0,0,0,0,1,0),(0,0,0,0,0,0,1)]
epsilon=0.1
lmd=1e-4
gamma=0.8
alpha=0.1
terminal=50000
state_actions={}

# get data from Quandl
quandl.ApiConfig.api_key = 'MYKEY'

data = quandl.get_table('WIKI/PRICES', ticker = dow30,
                        qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                        date = { 'gte': '2014-1-1', 'lte': '2016-12-31' }, paginate=True)

# data cleaning
clean = data.set_index('date')
table = clean.pivot(columns='ticker')

# calculate daily and annual returns of the stocks
returns_daily = table.pct_change()
returns_annual = returns_daily.mean() * 250

# get daily and covariance of returns of the stock
cov_daily = returns_daily.cov()
cov_annual = cov_daily * 250


# set the number of combinations for imaginary portfolios
num_assets = len(dow30)

#calculation of sharpe
def sharpe_cal(weights,returns_annual,cov_annual):
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    sharpe = returns / volatility
    return sharpe

#initialize actions from an unseen state
def qinit(state,actions):
    start=sharpe_cal(state,returns_annual,cov_annual)
    qreward=[]

    #calibrating rewards
    for i in actions:
        state=state+i
        change=sharpe_cal(state/sum(state),returns_annual,cov_annual)-start
        qreward.append(change)
    
    choice=np.argmax(qreward)
    return choice

#finding rewards from the state-actions data
def qfind(state_action,state):
    reward=[]
    actions=[]
    group=list(state_action)
    for i in group:
        if i[0]==state:
            reward.append(state_action[i])
            actions.append(i[1])
    if not actions:
        choice=qinit(state,action)
    else:
        choice=np.argmax(reward)
    return actions[choice]

#array to string
def tostring(array):
    a=str(array).replace("[","").replace("]","")
    return a

#string to array
def toarray(string):
    a=string.split(",")
    a=[int(x) for x in a]
    return a


#Value-iterative learning
def iterative():
    #initializing
    weight=np.zeros(num_assets)
    values=np.zeros(num_assets)
    sharpe_port=0
    
    while(True):
        #random to execute epsilon-greedy
        random=rand.randint(0,1)

        if random>epsilon:
            #greedy-choice
            choice=qinit(weight,action)
        else:
            #non-greedy choice
            choice=rand.randint(0,num_assets-1)
         
        weight=weight+action[choice]
        value=sharpe_cal(weight/sum(weight),returns_annual,cov_annual)-sharpe_port
       
        values[choice]=(1-alpha)*values[choice]+gamma*value

        new_sharpe_port=sharpe_cal(weight/sum(weight),returns_annual,cov_annual)
        if abs(new_sharpe_port-sharpe_port)>lmd:
            sharpe_port=new_sharpe_port
        else:
            break
    weight=np.round(weight/sum(weight),2)
    return weight,sharpe_port


#Q-learning method
def qlearning():
    #initialize weight and rewards array
    weight=np.ones(num_assets)
    rewards=np.zeros(num_assets)

    #calculate the starting state of having each stock equally
    sharpe_port=sharpe_cal(weight/sum(weight),returns_annual,cov_annual)

    while(True):
        while(True):
            #random to execute epsilon-greedy
             random=rand.randint(0,1)

             if random<=epsilon or not state_actions:
                 #non-greedy-choice
                 choice=rand.randint(0,num_assets-1)
             else:
                 #greedy choice
                 choice=np.argmax(qfind(state_actions,weight))
           
             #new state and reward after taking action of choice
             newweight=weight+action[choice]
             reward=sharpe_cal(weight,returns_annual,cov_annual)
            
             #transformation for hashing
             weight=tostring(weight)
             newweight=tostring(newweight)
             
             #Q-learning function
             state_actions[(weight,choice)]=state_actions[(weight,choice)]+alpha*(reward+gamma*qfind(state_actions,newweight)-state_actions[(weight,choice)])
             
             #retransformation for state_actions pairing
             weight=newweight
             weight=toarray(weight)
            
            #basically here, the terminal state is set to be after 50000 trades
             if sum(weight)==terminal:
                 break
        
        #covergence calculation
        result=sharpe_cal(weight,returns_annual,cov_annual)
        if abs(result-sharpe_port)>lmd:
            weight=np.ones(num_assets)
            sharpe_port=result
        else:
            break
        
    weight=np.round(weight/sum(weight),2)
    return weight,sharpe_port



stock_weights,sharpe_ratio=iterative()
port_returns=stock_weights*returns_annual
print(sharpe_ratio)
print(stock_weights)




     

