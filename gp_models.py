import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si
from tqdm import tqdm
tfk = tfp.math.psd_kernels

"""
This .py file contains the definition of the classes used in order to perform Gaussian Process Regression. They are all based on existing classes in the Tensorflow Probability API, but redefine certain methods and attributes in order satisfy the specific requirements of the code. """

# Custom positive semidefinite kernel: anisotropic RBF and Matern Kernels, as well as  "Hybrid" kernel
# that multiplies a stationary kernel on  the first (d-1) dimensions with a linear kernel in the d dimension
# NB: the Hybrid kernel has not actually been used in this code and might be prone to bugs

# Class inherites from TensorFlow Probability abstract Kernel class
class CustomKernel(tfk.PositiveSemidefiniteKernel):
    def __init__(self,feature_ndims=1, parameters=None,dtype=None, name='Custom_Kernel', validate_args=False):
        super().__init__(feature_ndims, dtype, name, validate_args, parameters)
        self.amplitude = parameters['amp']
        self.length_scales = parameters['scales']
        self.family = parameters['family']
        if self.family == 'Hybrid':
            self.bias = parameters['bias']
            self.slope = parameters['slope']
            self.shift = parameters['shift']
              
    def _apply(self,x,y,example_ndims):
        diff = (x-y)
        #print(x,y)
        if (self.family == 'M52'): return self._apply_M52(diff)
        if (self.family == 'M32'): return self._apply_M32(diff)
        if (self.family == 'Hybrid'): return self._apply_hybrid(diff,x[...,-1],y[...,-1])
        return self._apply_SE(diff)
    #Apply RBF kernel
    def _apply_SE(self,diff):
        scaled_diff = diff/self.length_scales
        return self.amplitude*tf.exp(-tf.math.reduce_sum(scaled_diff**2/2,axis=-1))
    # Apply Matérn 5/2 kernel
    def _apply_M52(self,diff):
        scaled_diff = diff/self.length_scales
        prod_element = (1+np.sqrt(5)*tf.abs(scaled_diff)+5/3*tf.abs(scaled_diff)**2)*tf.exp(-np.sqrt(5)*tf.abs(scaled_diff))
        kappa = self.amplitude*tf.reduce_prod(prod_element,axis=-1)
        return kappa
    # Apply Matérn 3/2 kernel
    def _apply_M32(self,diff):
        scaled_diff = diff/self.length_scales
        prod_element = (1+np.sqrt(3)*tf.abs(scaled_diff))*tf.math.exp(-np.sqrt(3)*tf.abs(scaled_diff))
        kappa = self.amplitude*tf.reduce_prod(prod_element,axis=-1,keepdims=False)
        return kappa
    # Apply product of Matérn and Linear kernel
    def _apply_hybrid(self,diff,t,tt):
        term_1 = self._apply_M52(diff[...,:-1])
        term_2 = self.bias**2 + self.slope**2*(t-self.shift)*(tt-self.shift)

        return term_1*term_2
        
    # Must redefine these 2 methods when subclassing
    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return scalar_shape

    def _batch_shape_tensor(self):
        return tf.convert_to_tensor(self.batch_shape, dtype=tf.int32, name='batch_shape')

    
    
# TF Probability distinguishes between a GaussianProcess and GaussianProcessRegressionModel class 
# First, the GaussianProcess class is used to initialise a GP and train the kernel hyperparameters on the observed data

class GP_model(tfp.distributions.GaussianProcess):
    def __init__(self,kernel, index_points=None, mean_fn=None,beta = (), observation_noise_variance=0.0,\
                jitter=1e-06, validate_args=False, allow_nan_stats=False, name='GP'):
        super().__init__(kernel, index_points, mean_fn, observation_noise_variance,jitter, validate_args, allow_nan_stats, name)
        
        #We add the mean function hyperparameters (stored in 'beta') to the variables used to compute the gradient in the optimisation procedure
        self.hyperparam = self.trainable_variables + beta
    
    # Method that computes the loss (negative log likelihood) and the gradient w.r.t the hyperparameters
    # The hyperparameters are updated by applying the gradients
    @tf.function
    def optimize(self,y_,optimizer):        
        with tf.GradientTape() as tape:
            loss = -self.log_prob(y_)
        grads = tape.gradient(loss, self.hyperparam)
        optimizer.apply_gradients(zip(grads, self.hyperparam))
        del tape
        return loss
    # Run the optimisation step n_iters times
    def fit(self,y_,optimizer = tf.optimizers.Adam(),n_iters = 1000, verbose = True):
        for i in tqdm(range(n_iters),ascii=True,desc='Progress',ncols=80):
            neg_log_likelihood = self.optimize(y_,optimizer)
            if i % 200 == 0 and verbose:
                print("Step {}: NLL = {}".format(i, neg_log_likelihood))
        print("Final NLL = {}".format(neg_log_likelihood))
    
    # We redefine the covariance and variance methods in order to accomodate stochastic Kriging
    def _compute_covariance(self, index_points):
        kernel_matrix = self.kernel.matrix(index_points, index_points)
        if self._is_univariate_marginal(index_points):
          # kernel_matrix thus has shape [..., 1, 1]; squeeze off the last dims and
          # tack on the observation noise variance.
            return (tf.squeeze(kernel_matrix, axis=[-2, -1]) +
                  self.observation_noise_variance)
        else:
            observation_noise_variance = tf.convert_to_tensor(
              self.observation_noise_variance)

            return tf.linalg.set_diag(kernel_matrix, tf.linalg.diag_part(kernel_matrix) + observation_noise_variance)
   
    def _variance(self, index_points=None):
        index_points = self._get_index_points(index_points)       
        kernel_diag = self.kernel.apply(index_points, index_points)
        if self._is_univariate_marginal(index_points):
            return (tf.squeeze(kernel_diag, axis=[-1]) +
                  self.observation_noise_variance)
        else:

            return kernel_diag + self.observation_noise_variance

# We redefine the SchurComplement Kernel class from TF Probability to again allow for Stochastic Kriging
# This is because the GaussianProcessRegressionModel class uses the SchurComplement to construct the posterior covariance matrix
class CustomSchurComplement(tfk.SchurComplement):
    def __init__(self,base_kernel, fixed_inputs, diag_shift=None, validate_args=False,name='SchurComplement'):
        super().__init__(base_kernel,fixed_inputs,diag_shift,validate_args,name)
    
    def _divisor_matrix(self, fixed_inputs=None):
        fixed_inputs = tf.convert_to_tensor(self._fixed_inputs if fixed_inputs is None else fixed_inputs)
        divisor_matrix = self._base_kernel.matrix(fixed_inputs, fixed_inputs)
        if self._diag_shift is not None:
            diag_shift = tf.convert_to_tensor(self._diag_shift)
            divisor_matrix = tf.linalg.set_diag(divisor_matrix, tf.linalg.diag_part(divisor_matrix) + diag_shift)
        return divisor_matrix

# Subclassing of the GaussianProcessRegressionModel method in order to add functionalities such as gradient computations (i.e. compute the derivative of the GP w.r.t one of its inputs), as well as prediction confidence bounds
class GPR_model(tfp.distributions.GaussianProcessRegressionModel):
    def __init__(self,kernel,index_points = None ,observation_index_points = None ,observations = None,\
                observation_noise_variance = 0.0,predictive_noise_variance=None, mean_fn=None,\
                jitter=1e-06, validate_args=False, allow_nan_stats=False,name='GPR_Model'):
        
        super().__init__(kernel, index_points, observation_index_points, observations,\
                        observation_noise_variance, predictive_noise_variance, mean_fn,jitter, validate_args,allow_nan_stats,name)
        self.prior_kernel = kernel
        self.isTimeToMaturity = True
        # Special treatment for Stochastic Kriging (i.e. when the noise variance is vector-valued)
        if tf.shape(observation_noise_variance)>1:
            self._kernel = CustomSchurComplement(kernel,observation_index_points,observation_noise_variance)
            if mean_fn is None: mean_fn = lambda x: tf.zeros([1],dtype=index_points.dtype)
            # The posterior mean needs to be redefined after setting the kernel to the CustomSchurComplement
            # This is because the parent class will initialise the mean function BEFORE, using the wrong kernel
            with tf.name_scope('init'):
                def conditional_mean_fn(x):
                    observations = tf.convert_to_tensor(self._observations)
                    observation_index_points = tf.convert_to_tensor(
                        self._observation_index_points)
                    k_x_obs_linop = tf.linalg.LinearOperatorFullMatrix(
                        kernel.matrix(x, observation_index_points))
                    chol_linop = tf.linalg.LinearOperatorLowerTriangular(
                        self.kernel.divisor_matrix_cholesky(
                            fixed_inputs=observation_index_points))

                    diff = observations - mean_fn(observation_index_points)
                    return mean_fn(x) + k_x_obs_linop.matvec(
                        chol_linop.solvevec(chol_linop.solvevec(diff), adjoint=True))
            self._mean_fn = conditional_mean_fn
    # Redefining the variance and covariance computations for Stochastic Kriging
    # NB: predictive_noise_variance must have dim (m,) where m is the length of index_points
    def _variance(self, index_points=None):
        index_points = self._get_index_points(index_points)
        kernel_diag = self.kernel.apply(index_points, index_points)
        if len(tf.shape(index_points)) == 1:
            index_points = tf.expand_dims(index_points,-1)
        if self._is_univariate_marginal(index_points):
            return (tf.squeeze(kernel_diag, axis=[-1]) + self.predictive_noise_variance)

        else:
            return kernel_diag + self.predictive_noise_variance    
    def _compute_covariance(self, index_points):
        kernel_matrix = self.kernel.matrix(index_points, index_points)
        if self._is_univariate_marginal(index_points):
          # kernel_matrix thus has shape [..., 1, 1]; squeeze off the last dims and
          # tack on the observation noise variance.
            return (tf.squeeze(kernel_matrix, axis=[-2, -1]) +
                  self.predictive_noise_variance)
        else:
            predictive_noise_variance = tf.convert_to_tensor(
              self.predictive_noise_variance)

            return tf.linalg.set_diag(kernel_matrix, tf.linalg.diag_part(kernel_matrix) + predictive_noise_variance)
       
    def _mean(self, index_points=None):
        index_points = self._get_index_points(index_points)
        mean = self._mean_fn(index_points)

        return mean
    # The best estimate of the option price is given by the posterior mean
    def price(self): return self.mean()
    # Compute the gradient of the price w.r.t. to all input parameters (e.g. time to maturity, spot)
    def gradient(self):
        with tf.GradientTape() as tape:
            tape.watch(self.index_points)
            p_hat = self.price()
        grad = tape.gradient(p_hat,self.index_points)
        del tape
        return grad
    # Delta is dP/dS, which is the 2nd input parameter in the code conventions
    def delta(self): return self.gradient()[:,1]    
    # Theta is dP/dt
    def theta(self): 
        # Check is input is absolute time t or time to maturity tau
        if self.isTimeToMaturity: return -self.gradient()[:,0]
        else : return self.gradient()[:,0]
    # Covariance matrix of the derivate of the GP
    def K_g(self,stock = True):
        j = stock + 0
        b = tf.identity(self.index_points)
        rows =  []
        for i in range(len(b)):
            a = self.index_points[i,:]
            with tf.GradientTape() as t2:
                t2.watch(b)
                with tf.GradientTape() as t1:
                    t1.watch(a)
                    kstar = self.kernel.apply(a,b)
                dk = t1.gradient(kstar,a)[j]
            dkk = t2.gradient(dk,b)[:,j]
            rows.append(dkk)
        k_g_temp = tf.constant(np.array(rows),dtype=tf.float64)
        k_g = tf.linalg.set_diag(k_g_temp,self.V_g())
        return k_g
    # Variance of the derivative of the GP
    def V_g(self,stock = True):
        i = stock + 0
        temp1 = tf.identity(self.index_points)
        #Add small amount of jitter to avoid evaluating in zero
        temp2 = tf.identity(self.index_points+self.jitter**2) 
        with tf.GradientTape() as tape2:
            tape2.watch(temp2)
            with tf.GradientTape() as tape1:
                tape1.watch(temp1)
                K_star = self.kernel.apply(temp1, temp2)
            dK_star = tape1.gradient(K_star, temp1)[:,i]
        d2K_star = tape2.gradient(dK_star, temp2)[:,i]
        return d2K_star
    
    def bounds_price(self,confidence):
        z = si.norm.ppf((confidence+1)/2)
        return self.mean() - z*np.sqrt(self.variance()), self.mean() + z*np.sqrt(self.variance())
    
    def bounds_delta(self,confidence):
        z = si.norm.ppf((confidence+1)/2)
        return self.delta() - z*np.sqrt(self.V_g()), self.delta() + z*np.sqrt(self.V_g())

    def bounds_theta(self,confidence):
        z = si.norm.ppf((confidence+1)/2)
        return self.theta() - z*np.sqrt(self.V_g(False)), self.theta() + z*np.sqrt(self.V_g(False))
    
    def plot_price(self,true_price,x_values = None,bounds = True,confidence = 0.95,title = True,fig_name=None):
        lBound , uBound = self.bounds_price(confidence)
        return self.plot(self.price(),true_price,x_values,bounds,confidence,lBound,uBound,'Option Price',title,fig_name)
    
    def plot_delta(self,true_delta,bounds= True,confidence = 0.95,fig_name=None):
        lBound , uBound = self.bounds_delta(confidence)
        return self.plot(self.delta(),true_delta,bounds,confidence,lBound,uBound,'Delta',fig_name)
    
    def plot_theta(self,true_theta,bounds= True,confidence = 0.95,fig_name=None):
        lBound , uBound = self.bounds_theta(confidence)
        return self.plot(self.theta(),true_theta,bounds,confidence,lBound,uBound,'Theta',fig_name)
    
    def plot(self,y_pred,true_values,x_values,bounds,confidence,u_bound,l_bound,ylabel,title,fig_name=None):
        #test_stocks = self.index_points[:,1]
        test_stocks = x_values
        lBound = l_bound
        uBound = u_bound
        rmse = np.sqrt(np.sum((y_pred-true_values)**2)/len(y_pred))
        plt.figure()
        plt.plot(test_stocks,true_values,label = 'Ground Truth')
        plt.scatter(test_stocks,y_pred,color='red',marker='x',s=5,label = 'GP estimate')
        if bounds:
            plt.plot(test_stocks,lBound,'--',color='black',linewidth=1,label = '95% confidence interval')
            plt.plot(test_stocks,uBound,'--',color='black',linewidth=1)
        plt.xlabel('Strike')
        plt.ylabel(ylabel)
        if title:
            plt.title('RMSE = '+str(np.round(rmse,4)))
        plt.legend()
        if fig_name != None: plt.savefig(fig_name)
        plt.show() 

        
# Sanity-check class that implements all the analytical formulas for the TF methods used above
class GPR_analytical():
    def __init__(self,kernel,data,labels,mean_fn,noise,test_data=None):
        self.kernel = kernel
        self.X = data
        self.y = labels
        self.mean_fn = mean_fn
        self.sigma = noise
        self.X_test = test_data
        self.inv_matrix = tf.linalg.inv(self.kernel.tensor(data,data,1,1)+\
                                        noise*tf.eye(len(data),dtype=tf.float64))
        self.K = self.kernel.matrix(test_data,data)
        
    def set_test_data(self,test_data):
        self.X_test = test_data
        self.K = self.kernel.matrix(test_data,self.X)
        
    def price(self):
        term_1 = self.mean_fn(self.X_test)
        term_2 = self.K@self.inv_matrix@tf.reshape((self.y-self.mean_fn(self.X)),[len(self.y),1])
        
        return term_1 + tf.squeeze(term_2)
    
    def covariance(self):
        term_1 = self.kernel.matrix(self.X_test,self.X_test)      
        term_2 = self.K@self.inv_matrix@tf.transpose(self.K)
        
        return term_1-term_2
    
    def variance(self): return tf.linalg.diag_part(self.covariance())
    
    def dk_dS(self,x_,y_,i):
        reshaped = tf.transpose(tf.broadcast_to(y_[:,i],[len(x_),len(y_)]))
        l_ = self.kernel.length_scales[i]
        if self.kernel.family == 'SE': 
            return tf.transpose((reshaped-x_[:,i]))/l_**2\
                *self.kernel.matrix(x_,y_)
        elif self.kernel.family == 'M32':
            return 3*tf.transpose((reshaped-x_[:,i])/l_**2/(1+np.sqrt(3)/l_*tf.abs(reshaped-x_[:,i])))\
                    *self.kernel.matrix(x_,y_)
        elif self.kernel.family == 'M52':
            numerator = 5/3*tf.transpose((reshaped-x_[:,i])/l_**2*(1+np.sqrt(5)/l_*tf.abs(reshaped-x_[:,i])))
            denominator = tf.transpose(1 +  np.sqrt(5)/l_*tf.abs(reshaped-x_[:,i])+5/(3*l_**2)*(reshaped-x_[:,i])**2)
            return numerator/denominator *self.kernel.matrix(x_,y_)
        else : return 0
   
    def delta(self):
        b_1 = self.mean_fn(np.array([[0.,1.]]))-self.mean_fn(np.array([[0.,0.]]))
        term_2 = self.dk_dS(self.X_test,self.X)@self.inv_matrix@\
                tf.reshape((self.y-self.mean_fn(self.X)),[len(y),1])
        
        return tf.squeeze(b_1+term_2)
    
    def theta(self):
        term_2 = self.dk_dt(self.X_test,self.X)@self.inv_matrix@\
                tf.reshape((self.y-self.mean_fn(self.X)),[len(y),1])
        return -tf.squeeze(term_2)
    
    def ddk_dS2(self,x_,y_,i):
        l_ = self.kernel.length_scales[i]
        if self.kernel.family == 'SE':
            mat_x = tf.broadcast_to(x_[:,i],[len(x_),len(x_)])
            mat_y = tf.transpose(tf.broadcast_to(y_[:,i],[len(y_),len(y_)]))
            coef = (1-1/l_**2*(mat_y-mat_x)**2)/l_**2
            return coef*self.kernel.matrix(x_,y_)
        elif self.kernel.family == 'M32':
            return 3/l_**2*self.kernel.amplitude
        elif self.kernel.family == 'M52':
            return 5/(3*l_**2)*self.kernel.amplitude
        else: return 0

    def K_g(self,stock = True):
        i = stock +0
        term_1 = self.ddk_dS2(self.X_test,self.X_test,i)
        term_2 = self.dk_dS(self.X_test,self.X,i)@self.inv_matrix@tf.transpose(self.dk_dS(self.X_test,self.X,i))
        
        return term_1 - term_2
    def V_g(self): return tf.linalg.diag_part(self.K_g())
    
    def upper_bound(self,confidence):
        z = si.norm.ppf((confidence+1)/2)
        return self.delta() + z*np.sqrt(self.V_g())
    def lower_bound(self,confidence):
        z = si.norm.ppf((confidence+1)/2)
        return self.delta() - z*np.sqrt(self.V_g())
    
    def plot_price(self,true_price,confidence=0.95,bounds=True):
        test_stocks = self.X_test[:,-1]
        plt.plot(test_stocks,true_price)
        plt.scatter(test_stocks,self.price(),marker = 'x',color='red',s = 2)
#         if bounds:
#             plt.plot(test_stocks,self.upper_bound(confidence),'--',color='black',linewidth = 1)
#             plt.plot(test_stocks,self.lower_bound(confidence),'--',color='black',linewidth = 1)
        plt.show()
    
    def plot_delta(self,true_delta,confidence=0.95,bounds=True):
        test_stocks = self.X_test[:,-1]
        plt.plot(test_stocks,true_delta)
        plt.scatter(test_stocks,self.delta(),marker = 'o',color='red')
        if bounds:
            plt.plot(test_stocks,self.upper_bound(confidence),'--',color='black',linewidth = 1)
            plt.plot(test_stocks,self.lower_bound(confidence),'--',color='black',linewidth = 1)
        plt.show()