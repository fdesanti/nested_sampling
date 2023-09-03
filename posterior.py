import os 
import pandas as pd
import corner
import numpy as np
from tqdm import tqdm
#import arviz as az
#import xarray as xr
from chainconsumer import ChainConsumer

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from model.gmm import gaussian, sum_of_gaussians
from utils import *

blue_color = mcolors.TABLEAU_COLORS['tab:blue']
colours = [mcolors.TABLEAU_COLORS[key] for key in mcolors.TABLEAU_COLORS.keys()]


class Posteriorsamples():
     
     def __init__(self, output_dir, samples_filename=None):
          self.output_dir = output_dir
          if not os.path.exists(output_dir):
               raise NotADirectoryError("The given directory does not exists!")
          
          if samples_filename is None:
              self.samples_filename = 'posterior.dat'
          else:
              self.samples_filename = samples_filename
          
          self.read_samples()
          self.convert_to_latex_label()

    
          return
    
     
     def read_samples(self):
        
            
        header_file = os.path.join(self.output_dir, 'header.txt')
        self.names  = np.genfromtxt(header_file, dtype = 'str')[0:-1] #ignoring last column which is LogL
        samples_path = os.path.join(self.output_dir, self.samples_filename)
        
        samples_df = pd.read_csv(samples_path, sep =' ', names=self.names, skiprows=2, 
                                 usecols = [i for i in range(len(self.names))])
                                 
        
        n_gauss = (len(samples_df.keys())+1)//3
        w_ngauss = 1 - np.sum([samples_df[f'w_{i+1}'].to_numpy() for i in range(n_gauss-1)], axis = 0)

        samples_df.insert(n_gauss-1, f"w_{n_gauss}", w_ngauss)
        

        self.samples = dict()
        for name in samples_df.keys():
            self.samples[name] = samples_df[name].to_numpy()
        
        #update self.names
        self.names = list(self.samples.keys())
        return self.samples
     

     def convert_to_latex_label(self):
        try:
            self.names
        except:
            raise RuntimeError("Header must exists")
      

        latex_names = []
        for name in self.names:
            if not name.startswith('w'):
                latex_name = "$\\"+name +"$"
            else:
                latex_name = "$"+name+"$"
            latex_names.append(latex_name)

        self.latex_names = latex_names
        return 
     
     def get_best_estimate(self, CL='90'):
         
         if CL=='90':
             min_cl = 0.05
             max_cl = 0.95
         elif CL=='68':
             min_cl = 0.16
             max_cl = 0.84
         else:
             raise NotImplementedError
         
         self.CL = CL

         self.best_estimates    = dict()
         self.best_estimates_cl = dict()

         for name in self.samples.keys():
             samples = self.samples[name]
             low    = np.quantile(samples, min_cl)
             median = np.quantile(samples, 0.5)
             high   = np.quantile(samples, max_cl)
             self.best_estimates[name]    = median
             self.best_estimates_cl[name] = [low, high]

         return self.best_estimates, self.best_estimates_cl
     

     def plot_posterior_mixture(self, data = None, nbins = 'fd', 
                                data_label = 'log$T_{90}$', density = True, CL=None):
         
         
        
         if self.best_estimates is None:
             self.get_best_estimate()
         
         x = np.linspace(-5, 7, 1000)
         plt.rcParams['font.size']=15

         plt.figure()
         #plotting variances of reconstructed posterior
         
         if CL is not None:
             percentiles = [(100-CL)/2, 100-(100-CL)/2]
             y = []
             for _ in range(10000):
                 p = dict()
                 for name in self.names:
                     i = np.random.randint(len(self.samples['mu_1']))

                     #p[name] = np.random.uniform(low  = self.best_estimates_cl[name][0], high = self.best_estimates_cl[name][1], size=1)[0]

                     p[name] = self.samples[name][i]
                 #y = sum_of_gaussians(x, p)
                 y.append(sum_of_gaussians(x, p))
             y=np.array(y)
             '''
             y_max = np.zeros(len(x))
             y_min = np.zeros(len(x))
             for i in range(len(x)):
                 y_max[i] = max(y[i])
                 y_min[i] = min(y[i])
             '''
             low = np.percentile(y, percentiles[0], axis = 0)
             high = np.percentile(y, percentiles[1], axis = 0)


             plt.fill_between(x, low, high,color = 'mistyrose', alpha = 1, label = f'{CL}% CI')

         if data is not None:
             plt.hist(data, bins = nbins, density = density, histtype='step', alpha = 1, label = 'data', linewidth=1.5)
             #plt.hist(data, bins = nbins, density = density, histtype='stepfilled', color = blue_color, alpha = 0.5)
             plt.xlabel(data_label)
             plt.xlim(min(data)-0.5, max(data)+0.5)
         plt.ylabel('Normalized density')

        #plotting mean posterior
         
         
         
         if CL is None:
            for i in range(len(self.best_estimates)//3):
                w = self.best_estimates[f'w_{i+1}']
                mu = self.best_estimates[f'mu_{i+1}']
                sigma = self.best_estimates[f'sigma_{i+1}']
                y = gaussian(x, w, mu, sigma)
                plt.plot(x, y, color= colours[i])
                plt.fill_between(x, 0*x, y,color = colours[i], alpha = 0.3)
         
         y = sum_of_gaussians(x, self.best_estimates)
         plt.plot(x, y, label='reconstructed', color='r', linewidth=2)
         
         plt.minorticks_on()
         plt.legend(loc = 'upper left')
         if CL is not None:
             fname = f'{self.output_dir}/posterior_plot_CL_{CL}.png'
         else:
             fname = f'{self.output_dir}/posterior_plot_mixtures.png'
         plt.savefig(fname, dpi=200)
         plt.show()

         return
     
     def corner_plot(self, num_bins = None):
         
         try:
             self.latex_names
         except:
             self.convert_to_latex_label()

         if num_bins is None:
            num_bins = determine_best_num_bins(self.samples)
         
         plt.rcParams['font.size']=15
         

         figure = corner.corner(self.samples, bins = num_bins, labels = self.latex_names, color = blue_color, quantiles=[0.05, 0.5, 0.95],
                                    #truths = best_estimates, truth_color='orange', 
                                    show_titles=True)


         fname = self.output_dir+'/corner_plot.png'
         plt.savefig(fname, dpi = 300)
         plt.show()


         return 
     
     def compute_class_prob(self, logT90, component, verbose = True):
         
         n_gauss = len(self.names)//3
         
         #computing Z
         Z = 0
         for i in range(n_gauss):
             w     = self.best_estimates['w_{0}'.format(i+1)]
             mu    = self.best_estimates['mu_{0}'.format(i+1)]
             sigma = self.best_estimates['sigma_{0}'.format(i+1)]
             Z += gaussian(logT90, w, mu, sigma)
             
             
         
         w     = self.best_estimates['w_{0}'.format(component)]
         mu    = self.best_estimates['mu_{0}'.format(component)]
         sigma = self.best_estimates['sigma_{0}'.format(component)]
        
         P = gaussian(logT90, w, mu, sigma) / Z 
        
         if verbose:
            print('P(k = %d) = %.3f' %(component, P))
            
         return P
     
     def compute_evidence(self, logT90, N = int(1e6), verbose = True):
         """LogT90 is an array"""
         n_gauss = len(self.names)//3
         
         #computing Z
         Z = 0
         
         for _ in tqdm(range(N)):
            Z_i = 0 
            for n in range(n_gauss):
                i,j,k = np.random.randint(3, size=3)
                w     = self.samples['w_{0}'.format(n+1)][i]
                mu    = self.samples['mu_{0}'.format(n+1)][j]
                sigma = self.samples['sigma_{0}'.format(n+1)][k]
                Z_i += gaussian(logT90, w, mu, sigma)
            
            Z += Z_i.sum()
         
         Z /= N
         
         if verbose:
             print(f'Final Evidence = {Z}')
         
         return Z
    
     

