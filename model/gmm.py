import numpy as np
from scipy.special import softmax
from math import sqrt, pi
import matplotlib.pyplot as plt
import os 

import cpnest.model



def gaussian(x, a, mu, sigma):
    return a/(sqrt(2*pi)*sigma) *np.exp(-0.5*((x-mu)/(sigma))**2)

def sum_of_gaussians(x, p):
    assert len(p)%3==0, 'Invalid number of parameters: I expect w, mu, sigma for every component of the mixture'
    num_gauss = len(p)//3
    y = np.zeros(shape=x.shape)
    
    for i in range(num_gauss):
        a     = p['w_'+str(i+1)]
        mu    = p['mu_'+str(i+1)]
        sigma = p['sigma_'+str(i+1)]
        y += gaussian(x, a, mu, sigma)

    return y

class GaussianMixtureModel(cpnest.model.Model):
    """
    An n-dimensional gaussian
    """
    def __init__(self, data, num_gauss, additional_priors = True, gaussian_errors=False):
        self.y  = data['y']
        self.dy = data['dy']
        self.num_gauss = num_gauss
        self.additional_priors = additional_priors
        self.gaussian_errors   = gaussian_errors
        #self.sigma = 3.0
        
        self.names  = []
        self.bounds = []
        
        #define model paramters
        self.names.extend([f'w_{i+1}' for i in range(self.num_gauss-1)])
        self.bounds.extend([[0, 1] for _ in range(self.num_gauss-1)])
        
        self.names.extend([f'mu_{i+1}' for i in range(self.num_gauss)])
        self.bounds.extend([[-2, 6] for _ in range(self.num_gauss)])
        
        self.names.extend([f'sigma_{i+1}' for i in range(self.num_gauss)])
        self.bounds.extend([[0.1, 5] for _ in range(self.num_gauss)])
        
        #self.names.extend(['sigma'])
        #self.bounds.extend([[1, 10]])

    def log_likelihood(self, p):
        '''
        model = sum_of_gaussians(self.data['x'], p, self.num_gauss)
        #L     = gaussian(self.data['y']-model, a = 1, mu = model, sigma = p['sigma'])
        #logL  = np.log(L)
        logL = ((self.data['y']- model)/p['sigma'])**2
        return logL.sum()
        '''

        L = np.zeros(self.y.shape)
        for i in range(self.num_gauss):
            if i < self.num_gauss-1:
                weight   = p[f'w_{i+1}']
            else:
                weight   = 1 - np.sum([p[f'w_{j+1}'] for j in range(self.num_gauss-1) ])
            #weight   = p['w_{0}'.format(i+1)]
            residual = self.y-p[f'mu_{i+1}']
            sigma2    = p[f'sigma_{i+1}']**2  #sigma squared (variance)
            if self.gaussian_errors:
                sigma2 += self.dy**2
                

            logL_i = (np.log(weight)- np.log(np.sqrt(2*np.pi*sigma2)) 
                - 0.5*(residual)**2/sigma2 )
            
            L  += np.exp(logL_i)
            #L += weight/(np.sqrt(2*np.pi)*sigma) * np.exp(-(residual/sigma)**2)

            
            #logL  += logL_i

        logL = np.log(L).sum()
        
        
        return logL
        
        
    
    def log_prior(self, p): 
        
        
        
        ''' CHECK WEIGHTS SUM UP TO 1
        This is the best working option: 
        tried also softmax and normalization but the results are bad
        '''
        sum_weights   = sum([p[f'w_{i+1}'] for i in range(self.num_gauss-1)])
        
        if sum_weights > 1:
            return -np.inf
        
        ''' SOFTMAX
        weights = np.array([p['w_{0}'.format(i+1)] for i in range(self.num_gauss)])
        norm_weights = softmax(weights)
        for i in range(self.num_gauss):
            p['w_{0}'.format(i+1)] = norm_weights[i]
        '''
        
        ''' NORMALIZATION
        for i in range(self.num_gauss):
            p['w_{0}'.format(i+1)]/= sum_weights #normalize weights
        '''
        
        if self.additional_priors:
            mu_params = [p[f'mu_{i+1}'] for i in range(self.num_gauss)]
            if not mu_params == sorted(mu_params) :
                return -np.inf
        
 
        
        logP = super(GaussianMixtureModel, self).log_prior(p)
        return logP

