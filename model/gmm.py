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
    def __init__(self, data, num_gauss, additional_priors = True):
        self.data = data
        self.num_gauss = num_gauss
        self.additional_priors = additional_priors
        #self.sigma = 3.0
        
        self.names  = []
        self.bounds = []
        
        #define model paramters
        self.names.extend(['w_{0}'.format(i+1) for i in range(self.num_gauss)])
        self.bounds.extend([[0, 1] for _ in range(self.num_gauss)])
        
        self.names.extend(['mu_{0}'.format(i+1) for i in range(self.num_gauss)])
        self.bounds.extend([[-2,6] for _ in range(self.num_gauss)])
        
        self.names.extend(['sigma_{0}'.format(i+1) for i in range(self.num_gauss)])
        self.bounds.extend([[0.1,5] for _ in range(self.num_gauss)])
        
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

        L = np.zeros(self.data['y'].shape)
        for i in range(self.num_gauss):

            #weight = 1 - np.sum([p['w_{0}'.format(i+1)] for j in range(self.num_gauss) if i != j  ])
            weight = p['w_{0}'.format(i+1)]

            residual = self.data['y']-p['mu_{0}'.format(i+1)]
            sigma    = p['sigma_{0}'.format(i+1)]

            logL_i = (np.log(weight) - np.log(np.sqrt(2*np.pi) *  p['sigma_{0}'.format(i+1)]) 
                - 0.5*(residual/sigma)**2 )
            
            L  += np.exp(logL_i)
            #L += weight/(np.sqrt(2*np.pi)*sigma) * np.exp(-(residual/sigma)**2)

            
            #logL  += logL_i

        logL = np.log(L).sum()
        
        
        return logL
        
        
    
    def log_prior(self, p): 
        '''
        Computes the log prior. 
        Always checks whether the weights sum up to 1: 
            returns -inf in wrong cases
        If additional_prior = True: checks also that the means are sorted in ascending order (not mixed)
            returns -inf in wrong cases

        Otherwise:
            returns -log_prior (uniform) over the parameters 
        '''


        mu_params = [p['mu_{0}'.format(i+1)] for i in range(self.num_gauss)]
        #mu_diffs  = np.diff(mu_params)
        #min_diff  = abs(np.min(mu_diffs))
        weights = np.array([p['w_{0}'.format(i+1)] for i in range(self.num_gauss)])
        sum_weights   = sum([p['w_{0}'.format(i+1)] for i in range(self.num_gauss)])
        
        ''' CHECK WEIGHTS SUM UP TO 1
        This is the best working option: 
        tried also softmax and normalization but the results are bad
        '''
        if abs(1 - sum_weights) >= 1e-3:
            return -np.inf
        
        ''' SOFTMAX
        norm_weights = softmax(weights)
        for i in range(self.num_gauss):
            p['w_{0}'.format(i+1)] = norm_weights[i]
        '''
        
        ''' NORMALIZATION
        for i in range(self.num_gauss):
            p['w_{0}'.format(i+1)]/= sum_weights #normalize weights
        '''
        
        if self.additional_priors:
            if not mu_params == sorted(mu_params) :
                return -np.inf
        
 
        
        logP = super(GaussianMixtureModel, self).log_prior(p)
        return logP

