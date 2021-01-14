import scipy.stats as si
import numpy as np

class Black_Scholes():
    def __init__(self,interest= 0.04,volatility = 0.22, maturity = 0.4, strike = 50):
        self.r = interest
        self.sigma = volatility
        self.T = maturity
        self.K = strike
    def price(self,S,tau_):
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma ** 2) * tau_) / (self.sigma * np.sqrt(tau_))
        d2 = (np.log(S / self.K) + (self.r - 0.5 * self.sigma ** 2) * tau_) / (self.sigma * np.sqrt(tau_))
        call = (S * si.norm.cdf(d1, 0.0, 1.0) - self.K * np.exp(-self.r * tau_) * si.norm.cdf(d2, 0.0, 1.0))

        return call
    
    def mc_price(self,S,tau_,nsim = 2000):
        prices_ = []
        for i in range(len(S)):
            t = tau_[i]
            Z = np.random.normal(size =nsim)
            W_T = np.sqrt(t)*Z
            S_T = S[i]*np.exp((self.r-0.5*self.sigma**2)*t+self.sigma*W_T)
            sim_payoff = np.exp(-self.r*t)*np.maximum(S_T-self.K,0)
            prices_.append(np.mean(sim_payoff))
        return np.array(prices_)
    
    def delta(self,S,tau_):    
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma ** 2) * tau_) / (self.sigma * np.sqrt(tau_))  
        delta_call = si.norm.cdf(d1, 0.0, 1.0)
        
        return delta_call
    def theta(self,S,tau_):    
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma ** 2) * tau_) / (self.sigma * np.sqrt(tau_))
        d2 = (np.log(S / self.K) + (self.r - 0.5 * self.sigma ** 2) * tau_) / (self.sigma * np.sqrt(tau_))
        prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
        theta_call = (-self.sigma * S * prob_density) / (2 * np.sqrt(tau_)) - self.r * self.K * np.exp(-self.r * tau_)\
            * si.norm.cdf(d2, 0.0, 1.0)

        return theta_call
    def generate_paths(self,n_paths,mu=0.06,N=20,S0=50,normal = True,var = 2):
        R = n_paths
        logS= np.zeros((N,R))
        if normal: 
            logS[0,] = np.log(np.random.normal(S0,var,R))
        else:
            logS[0,]=np.log(S0)*np.ones((1,R))
        for i in range(R):
            for j in range(N-1):
                increment = (mu-0.5*self.sigma**2)*self.T/N+self.sigma*np.random.normal(0,np.sqrt(self.T)/np.sqrt(N))
                logS[j+1,i] =logS[j,i]+increment

        S=np.exp(logS)
        return S
class LVM():
    def __init__(self,interest= 0.05, maturity = 0.4, strike = 50):
        self.r = interest
        self.T = maturity
        self.K = strike
    def sigma(self,t,S,S_star):
        if np.abs(np.log(S,S_star))< 0.4:
            sigma_ = 0.4 - 0.16 * np.exp(-0.5*(self.T-t))*np.cos(1.25*np.pi * np.log(S,S_star))
        else:
            sigma_ = 0.4
        return sigma_ 