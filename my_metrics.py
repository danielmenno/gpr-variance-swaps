import numpy as np
from gp_models import GPR_model

def PnL(spot,price_model,trained_kernel,X,y,mean_f,l = 2):
    #Define the inital wealth as the option price at t=0 and S=S0
    W = price_model.price(spot[0,:],price_model.T)
    #Created n_paths repititions of the time discretisation
    dt = price_model.T/spot.shape[0]
    taus = np.tile(price_model.T-np.arange(0,price_model.T,dt),spot.shape[1])
    # Flatten the 2D paths array to a 1D vector in order to feed it to the GPR model
    flattenedStocks = np.ndarray.flatten(spot)
    X_hedge = np.vstack([taus,flattenedStocks]).T
    #Initialse a GPR model instance with the trained kernel, then compute deltas
    gpr = GPR_model(trained_kernel,X_hedge,X,y,mean_fn=mean_f)
    flattenedDeltas = gpr.delta()
    deltas = np.reshape(flattenedDeltas,spot.shape)
    #Compute the discrete delta hedging strategy
    for i in range(spot.shape[0]-1):
        W = spot[i+1,:]*deltas[i,:]+ (W-spot[i,:]*deltas[i,:])*np.exp(price_model.r*dt)
    payoff = np.maximum((spot[-1,:]-price_model.K),0)
    
    if l == 1: return np.mean(np.abs(W-payoff))
    else : return np.mean((W-payoff)**2)
