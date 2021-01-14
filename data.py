import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd

## Functions generating training and test data
def training_data(pricing_model,S_min,S_max,N_points=100,halton = False,virtual = False,mc = False,nsim = 10000):
    n = int(np.sqrt(N_points))
    T = pricing_model.T
    dt = T/n
    if halton:
        hltn = tfp.mcmc.sample_halton_sequence(2, num_results=N_points,dtype=tf.float64)
        spot = S_min+(S_max-S_min)*hltn[:,0]     
        tau = dt+(T-dt)*hltn[:,1]
    else:
        spot = np.tile(np.linspace(S_min,S_max,n,dtype=np.float64),n)
        tau = np.repeat(np.linspace(dt,T,n,dtype=np.float64),n)
    if mc : 
        prices = pricing_model.mc_price(spot,tau,nsim)
    else : 
        prices = pricing_model.price(spot,tau)
    deltas = pricing_model.delta(spot,tau)
    
    if virtual:
        maxS = S_max*np.ones(10)
        minS = S_min*np.ones(10)
        S_linspace = np.linspace(S_min+2,S_max-2,10,dtype=np.float64)
        
        spot = np.concatenate((spot,S_linspace,minS-1,minS-2,maxS+2,maxS+1))
        tau = np.concatenate((tau,np.zeros(10),np.tile(np.linspace(dt,T,10,dtype=np.float64),4)))
        
        price_at_maturity = np.maximum(S_linspace-pricing_model.K,0)
        price_itm = spot[-20:]-np.exp(-pricing_model.r*tau[-20:])*pricing_model.K
        prices = np.concatenate((prices,price_at_maturity,np.zeros(20),price_itm))
        deltas = np.concatenate((deltas,np.zeros(50)))
  
    df_train = pd.DataFrame(data=[tau,spot,prices,deltas]).T
    df_train.columns=['tau','spot','price','delta']
    
    X_ = tf.constant(df_train.iloc[:,:2],dtype =tf.float64)
    y_ = tf.constant(df_train.iloc[:,2],dtype=tf.float64)
    return X_, y_, df_train

def test_data(pricing_model,tau,stock_):
    tau_slice = tau*np.ones(len(stock_))
    x_test = np.vstack([tau_slice,stock_]).T
    y_test = pricing_model.price(stock_,tau_slice)
    return x_test, y_test
