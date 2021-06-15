import numpy as np
import scipy.stats as si

"""

Classes that generate an implied volatility surface depending on different parametrisations
The Smile class is the abstract parent class and contains methods to compute put and call prices

Additionally the K_var method computes the variance strike of the variance swap based on a discrete number of option prices through a linear approximation method

The main two child classes used are the SVI and SSVI parametrisations
"""

class Smile():
    def __init__(self,interest = 0.05, S0 = 100):
        self.r = interest
        self.S0 = S0

    def d12(self,S,K,T,r,sigma):
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return d1,d2
    
    def call(self,S,tau,K):
        sigma = self.sigma(tau,K)
        d1,d2 = self.d12(S,K,tau,self.r,sigma)
        call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-self.r * tau) * si.norm.cdf(d2, 0.0, 1.0))

        return call
    def put(self,S,tau,K):
        sigma = self.sigma(tau,K)
        d1,d2 = self.d12(S,K,tau,self.r,sigma)
        put = (K * np.exp(-self.r * tau) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    
        return put  
    def K_var(self,T=[1],N=20,s0=None,bound=20):
        ks=[]
        if s0==None: s0 = self.S0
        if bound > s0:  bound = s0-1
        put_strikes = np.linspace(s0-bound,s0,N)
        call_strikes = np.linspace(s0,s0+bound,N)

        for t in T:
            put_weights = self.put_weights(put_strikes,s0,t)
            call_weights = self.call_weights(call_strikes,s0,t)
#             print(put_weights)
#             print(call_weights)
            term1 = 2/t*(self.r*t-(np.exp(self.r*t)-1))
            put_prices = self.put(s0,t,put_strikes)
            call_prices = self.call(s0,t,call_strikes)

            int1 = np.exp(self.r*t)*np.sum(put_prices*put_weights)
            int2 = np.exp(self.r*t)*np.sum(call_prices*call_weights)
            #print(term1,int1,int2)
            ks.append(term1+int1+int2)
            
        return np.array(ks)
    def log_payoff(self,S_T,S_Star,T):
        return 2/T*((S_T/S_Star-1)-np.log(S_T/S_Star))
    
    def call_weights(self,strikes,S_Star,T):
        ws = []
        delta = strikes[-1]-strikes[-2]
        strikes = np.append(strikes,strikes[-1]+delta)
        for i in range(len(strikes)-1):
            wi = (self.log_payoff(strikes[i+1],S_Star,T)-self.log_payoff(strikes[i],S_Star,T))/(strikes[i+1]-strikes[i])
            wi-= np.sum(ws)
            ws.append(wi)
        return np.array(ws)
    def put_weights(self,strikes,S_Star,T):
        ws = []
        delta = strikes[-1]-strikes[-2]
        strikes = np.flip(np.insert(strikes,0,strikes[0]-delta))
        for i in range(len(strikes)-1):
            wi = (self.log_payoff(strikes[i+1],S_Star,T)-self.log_payoff(strikes[i],S_Star,T))/(strikes[i]-strikes[i+1])
            wi-= np.sum(ws)
            ws.append(wi)
        return np.flip(np.array(ws))
class Quadratic(Smile):
    def __init__(self,interest = 0.05, S0 = 100,sigma0 = 0.2,sigma_max=0.36,slope = 1e-4):
        super().__init__(interest,S0)
        self.slope = slope
        self.sigma0 = sigma0
        self.sigma_max = sigma_max
    def sigma(self,K):
        s = self.sigma0+self.slope*(self.S0-K)**2
        if not self.slope == 0:
            conditions = 1.*((self.S0-K)**2>(self.sigma_max-self.sigma0)/self.slope)
        else:
            conditions = np.zeros(len(K))
        return conditions*self.sigma_max+(1-conditions)*s
class Cosine(Smile):
    def __init__(self,interest = 0.05, S0 = 100,sigma0 = 0.2,sigma_max=0.36,K_star=100):
        super().__init__(interest,S0)
        self.K_star = K_star
        self.sigma0 = sigma0
        self.sigma_max = sigma_max
    def sigma(self,tau,K):

        condition = np.abs(np.log(K/self.K_star))<self.sigma_max
        sigma_ = self.sigma_max - (self.sigma_max-self.sigma0) * np.exp(-0.5*tau)*np.cos(1/(2*self.sigma_max)*np.pi * np.log(K/self.K_star)) * condition
        return sigma_ 
class SVI(Smile):
    def __init__(self,interest = 0.05, S0 = 100,a=0.04,b=0.4,rho=-0.4,m=0,sigma0=0.1):
        super().__init__(interest,S0)
        self.a = a
        self.b = b
        self.rho = rho
        self.m = m
        self.sigma0 = sigma0

    def get_parameters(self):
        return (self.a,self.b,self.rho,self.m,self.sigma0)
    def sigma(self,tau,K):
        moneyness = np.log(K/(np.exp(self.r*tau)*self.S0))
        
        v = self.a+self.b*(self.rho*(moneyness-self.m)+np.sqrt((moneyness-self.m)**2+self.sigma0**2))
       
        return np.sqrt(v)
class SSVI(Smile):
    def __init__(self,interest = 0.05,S0=100,rho = -0.4,eta = 1, lamb = 0.3,sigma0=0.22):
        super().__init__(interest,S0)
        self.rho = rho
        self.eta = eta
        self.lamb = lamb
        self.sigma0 = sigma0
        
    def sigma(self,tau,K):
        theta = self.theta(tau)
        phi = self.phi(theta)
        k = np.log(K/(np.exp(self.r*tau)*self.S0))
        w = theta/2*(1+self.rho*phi*k+np.sqrt((phi*k+self.rho)**2+(1-self.rho**2)))
        return np.sqrt(w/tau)
    def theta(self,tau):
        return self.sigma0**2*tau
    def phi(self,theta):
            return self.eta*theta**(-self.lamb)