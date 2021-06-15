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
    
    def d12(self,S,K,T,r,sigma):
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return d1,d2
    
    def call(self,S, K, T, r, sigma):
        d1,d2 = self.d12(S,K,T,r,sigma)
        call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))

        return call
    
    def put(self,S, K, T, r, sigma):
        d1,d2 = self.d12(S,K,T,r,sigma)
        put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    
        return put
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
    
    def call_mc(self,S, K, T, r, sigma,nsim = 1000):
        Z = np.random.normal(size=[nsim,len(K)])
        W_T = np.sqrt(T)*Z
        S_T = S*np.exp((r-0.5*sigma**2)*T+sigma*W_T)
        sim_payoff = np.exp(-r*T)*np.maximum(S_T-K,0)
        prices = np.mean(sim_payoff,axis=0)
        return prices
    
    
    def RV(self,paths):
        ti = paths[:-1,:]
        tii = paths[1:,:]
        return np.sum(np.log(tii/ti)**2,axis=0)
    
    def var_swaption_price(self,K,n_paths=2000,n_steps=20):
        paths = self.generate_paths(n_paths,N=n_steps)
        rvs = self.RV(paths)
        payoffs = []
        for k in K:
            payoff_i = np.maximum((rvs-k),0)
            payoffs.append(np.mean(payoff_i))
        return np.array(payoffs)
    
    def mc_RV(self,n_paths=2000,T=[1],N=20):
        ks = []  
        for t in T:
            paths  = self.generate_paths(n_paths,t,N)
            ks.append(np.mean(self.RV(paths)))
        return np.array(ks)
    def K_var(self,T=[1],N=20,s0=None,bound=20):
        ks=[]
        if s0==None: s0 = self.K
        if bound > s0:  bound = s0-1
        put_strikes = np.linspace(s0-bound,s0,N)
        call_strikes = np.linspace(s0,s0+bound,N)

        for t in T:
            put_weights = self.put_weights(put_strikes,s0,t)
            call_weights = self.call_weights(call_strikes,s0,t)
            
            term1 = 2/t*(self.r*t-(np.exp(self.r*t)-1))
            put_prices = self.put(s0,put_strikes,t,self.r,self.sigma)
            call_prices = self.call(s0,call_strikes,t,self.r,self.sigma)

            int1 = np.exp(self.r*t)*np.sum(put_prices*put_weights)
            int2 = np.exp(self.r*t)*np.sum(call_prices*call_weights)
            print(term1,int1,int2)
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
    def variance_vega(self,S,tau_,K=None):
        if K==None: K =self.K
        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma ** 2) * tau_) / (self.sigma * np.sqrt(tau_))
        var_vega = np.exp(-d1**2/2)*S*np.sqrt(tau_)/(np.sqrt(2*np.pi)*2*self.sigma)
        return var_vega
    
    def generate_paths(self,n_paths,T=1,N=20,mu=0.04,S0=50,r_neutral = False,normal = False,var = 2):
        if r_neutral: mu  = self.r
        R = n_paths
        logS= np.zeros((N,R))
        if normal: 
            logS[0,] = np.log(np.random.normal(S0,var,R))
        else:
            logS[0,]=np.log(S0)*np.ones((1,R))
        for i in range(R):
            for j in range(N-1):
                increment = (mu-0.5*self.sigma**2)*T/N+self.sigma*np.random.normal(0,np.sqrt(T)/np.sqrt(N))
                logS[j+1,i] =logS[j,i]+increment

        S=np.exp(logS)
        return S
class LVM():
    def __init__(self,interest= 0.05, maturity = 0.4, strike = 50,S_star = 50):
        self.r = interest
        self.T = maturity
        self.K = strike
        self.S_star = S_star
    def sigma(self,t,S,S_star):
#         if np.abs(np.log(S/S_star))< 0.4:
#             sigma_ = 0.4 - 0.16 * np.exp(-0.5*(self.T-t))*np.cos(1.25*np.pi * np.log(S/S_star))
#         else:
#             sigma_ = 0.4
        condition = np.abs(np.log(S/S_star))<0.3
        sigma_ = 0.3 - 0.06 * np.exp(-0.5*(self.T-t))*np.cos(1.25*np.pi * np.log(S/S_star)) * condition
        return sigma_ 
    def generate_paths(self,n_paths,T=1,N=20,mu=0.04,S0=50,r_neutral=False):
        if r_neutral: mu = self.r
        logS = np.ones([1,n_paths])*np.log(S0)
        for i in range(N-1):
            previous_stock = logS[-1,:]
            sigmas_i = self.sigma((i+1)*T/N,np.exp(previous_stock),self.S_star)
            increments = (mu-0.5*sigmas_i**2)*T/N+sigmas_i*np.random.normal(0,np.sqrt(T)/np.sqrt(N),n_paths)
            logS = np.concatenate((logS,np.reshape(previous_stock+increments,[1,-1])),axis=0)
        S=np.exp(logS)
        return S
    def price(self,S,tau,n_paths=5000,N=100):
        paths = self.generate_paths(n_paths=n_paths,T=tau,N=N,S0=S, r_neutral=True)
        S_T = paths[-1,:]
        payoffs = np.maximum(S_T-self.K,0)
        return np.mean(payoffs)
    def put(self,S,K,tau,n_paths = 10000):
        paths = self.generate_paths(n_paths=n_paths,T=tau,S0=S,r_neutral=True)
        S_T = paths[-1,:]
        S_T = np.broadcast_to(S_T,[len(K),len(S_T)])
        payoffs =  np.maximum(np.reshape(K,[-1,1])- S_T,0)
        return np.mean(payoffs,axis=1)
    def call(self,S,K,tau,n_paths = 10000):
        paths = self.generate_paths(n_paths=n_paths,T=tau,S0=S,r_neutral=True)
        S_T = paths[-1,:]
        S_T = np.broadcast_to(S_T,[len(K),len(S_T)])
        payoffs =  np.maximum(S_T - np.reshape(K,[-1,1]),0)
        return np.mean(payoffs,axis=1)
    def delta(self,S,tau):
        return np.zeros(len(S))

    def K_var(self,T=[1],N=20,s0=None,bound=20):
        ks=[]
        if s0==None: s0 = self.K
        if bound > s0:  bound = s0-1
        put_strikes = np.linspace(s0-bound,s0,N)
        call_strikes = np.linspace(s0,s0+bound,N)

        for t in T:
            put_weights = self.put_weights(put_strikes,s0,t)
            call_weights = self.call_weights(call_strikes,s0,t)
#             print(put_weights)
#             print(call_weights)
            term1 = 2/t*(self.r*t-(np.exp(self.r*t)-1))
            put_prices = self.put(s0,put_strikes,t,n_paths=100000)
            call_prices = self.call(s0,call_strikes,t,n_paths = 100000)
            
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
    def price_and_kvar(self,S,tau,n_paths=10000,N=100):
        paths = self.generate_paths(n_paths=n_paths,T=tau,N=N,S0=S, r_neutral=True)
        S_T = paths[-1,:]
        payoffs = np.maximum(S_T-self.K,0)
        price_ = np.exp(-self.r*tau)*np.mean(payoffs)
        ti = paths[:-1,:]
        tii = paths[1:,:]
        rv = 1/tau*np.mean(np.sum(np.log(tii/ti)**2,axis=0))
        return price_,rv
    
class nonMarkov():
    def __init__(self,interest= 0.05, maturity = 0.4, strike = 50,sigma0 = 0.2,d = 3,alpha = 0.3):
        self.r = interest
        self.T = maturity
        self.K = strike
        self.d = d
        self.s0 = 0.2
        self.alpha = alpha
    def sigma(self,log_history,dt):
        hist1 = np.array(log_history[-self.d:])
        hist2 = np.array(log_history[-self.d-1:-1])
        var_ = 1/(self.d*dt)*np.sum((hist1-hist2)**2)
        var_ = (1-self.alpha)*var_ + self.alpha*self.s0**2
        return np.sqrt(var_)
    def generate_paths(self,n_paths,T=1,N=20,mu=0.04,S0=50,r_neutral=False):
        if r_neutral: mu = self.r
        sigma_t = []
        R = n_paths
        logS= np.zeros((N,R))
        logS[0,]=np.log(S0)*np.ones((1,R))
        for i in range(R):
            for j in range(N-1):
                if j<self.d+1: 
                    sigma_i = self.s0
                else:
                    sigma_i = self.sigma(logS[j-self.d-1:j,i],T/N)
                sigma_t.append(sigma_i)  
                increment = (mu-0.5*sigma_i**2)*T/N+sigma_i*np.random.normal(0,np.sqrt(T)/np.sqrt(N))
                logS[j+1,i] =logS[j,i]+increment
        S=np.exp(logS)
        return S,sigma_t

class Heston():
    def __init__(self,r = 0.05,K=50,kappa=0.3,theta=0.04,xi = 0.1,sigma0=0.2):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.sigma0=sigma0
        self.r = r
        self.K = K
        
    def variance(self,prev,dt):
        incr = self.kappa*(self.theta-prev)*dt+self.xi*np.sqrt(prev)*np.random.normal(0,np.sqrt(dt),len(prev))
        return prev+incr
    
    def generate_paths(self,n_paths,T=1,N=20,mu=0.04,S0=50,r_neutral = False):
        if r_neutral: mu = self.r
        logS = np.ones([1,n_paths])*np.log(S0)
        old_sigmas = np.ones(n_paths)*self.sigma0
        for i in range(N-1):
            previous_stock = logS[-1,:]
            sigmas_i = np.sqrt(self.variance(old_sigmas**2,T/N))
            old_sigmas =sigmas_i
            increments = (mu-0.5*sigmas_i**2)*T/N+sigmas_i*np.random.normal(0,np.sqrt(T)/np.sqrt(N),n_paths)
            logS = np.concatenate((logS,np.reshape(previous_stock+increments,[1,-1])),axis=0)
        S=np.exp(logS)
        return S
    def price_and_kvar(self,S,tau,n_paths=10000,N=200):
        paths = self.generate_paths(n_paths=n_paths,T=tau,N=N,S0=S, r_neutral=True)
        S_tau = paths[-1,:]
        payoffs = np.maximum(S_tau-self.K,0)
        price_ = np.exp(-self.r*tau)*np.mean(payoffs)
        ti = paths[:-1,:]
        tii = paths[1:,:]
        rv = 1/tau*np.mean(np.sum(np.log(tii/ti)**2,axis=0))
        return price_,rv
        
        
class PDV():
    def __init__(self,interest= 0.05, maturity = 0.4, strike = 50,sigma0 = 0.2,s_plus = 0.3,s_min = 0.1,d = 5,kappa = 0.2):
        self.r = interest
        self.T = maturity
        self.K = strike
        self.d = d
        self.s0 = sigma0
        self.kapa = kappa
        self.s_plus = s_plus
        self.s_min = s_min
        
    def sigma(self,log_history,dt):
        log_hist = np.array(log_history[-self.d-1:-1,:])
        hist = np.exp(log_hist)
        running_means = np.mean(hist,axis=0)
        conditions = 1*(np.exp(log_history[-1,:])>running_means)
        #print(conditions)
        return conditions * self.s_min + (1-conditions)*self.s_plus
    
    def generate_paths(self,n_paths,T=1,N=20,mu=0.04,S0=50,r_neutral=False):

        if r_neutral: mu = self.r
        logS = np.ones([1,n_paths])*np.log(S0)
        for i in range(N-1):
            previous_stock = logS[-1,:]
            if i<self.d+1: 
                sigmas_i = self.s0
            else:
                sigmas_i = self.sigma(logS[i-self.d-1:i,:],T/N)
            #print(sigmas_i)
            increments = (mu -0.5*sigmas_i**2)*T/N+sigmas_i*np.random.normal(0,np.sqrt(T)/np.sqrt(N),n_paths)
            logS = np.concatenate((logS,np.reshape(previous_stock+increments,[1,-1])),axis=0)
            #print(np.min(logS))
        S=np.exp(logS)
        return S      
        