import os 
import timeit

from optparse import OptionParser

#import corner
#import arviz as az
import numpy as np
import cpnest

from utils import *

from model.gmm import GaussianMixtureModel



    
def main():
    '''
    MAIN FUNCTION
    '''
    parser = OptionParser()
    parser.add_option("-g", "--n_gauss", default=None, help="Number of Gaussians to consider")

    parser.add_option("-c", "--n_cpu", default=os.cpu_count(), help="Number of CPUs")

    parser.add_option("-p", "--additional_priors", default=False, action="store_true", help="Uses additional priors")
    parser.add_option("-s", "--seed", default = 123, help="seed")
    parser.add_option("-l", "--nlive", default=1500, help="Number of Live Points")
    parser.add_option("-o", "--overwrite", default=False, action="store_true", help="Overwrites existing results")
    parser.add_option("-e", "--gaussian_errors", default=False, action="store_true", help="Considers Gaussian Errors on logT90")



    (options, args) = parser.parse_args()

    #output_dir = options.output
    n_gauss = int(options.n_gauss)
    n_cpu   = int(options.n_cpu)
    seed    = int(options.seed)
    nlive   = int(options.nlive)
    overwrite = options.overwrite
    additional_priors = options.additional_priors
    gaussian_errors   = options.gaussian_errors
    








    tstart = timeit.default_timer() #start time
    
    #define data directory and file
    work_dir = os.getcwd()
    
    data_file = os.path.join(work_dir, 'data/GRB_data.txt')
    
    #retrieve data
    data = np.genfromtxt(data_file)
    logT90, dlogT90, HR = data.T
    
    
    #plot histogram of data
    plot_data_histogram(logT90)
    
    #---- NESTED SAMPLING ----
    model=GaussianMixtureModel(data = {'y': logT90, 'dy':dlogT90}, num_gauss = n_gauss, 
                               additional_priors = additional_priors, gaussian_errors=gaussian_errors)
    if additional_priors:
        out_folder = 'results/'+ str(model.num_gauss)+'gauss_additional_priors'
    else:
         
         out_folder = 'results/'+str(model.num_gauss)+'gauss'

    if gaussian_errors:
        out_folder += '_gaussian_errors'
        

    chain_file = out_folder+'/chain_'+str(nlive)+str(seed)+'.txt'
    
    print('Starting Nested Sampling ...')
    print('----> number of mixture components = %d'%n_gauss)
    print('----> live points = %d'%nlive)
    print('----> additional priors = %s'%additional_priors)
    print('----> number of cpus = %d'%n_cpu)
    print('----> results will be saved at [ %s ]'%out_folder)
    
    work=cpnest.CPNest(model, verbose=2,
                    nthreads = n_cpu, 
                    nlive=nlive, maxmcmc=5000, nslice=0, nhamiltonian=0, seed = seed,
                    resume=True, periodic_checkpoint_interval=8*3600,output=out_folder, 
                    )
    if not os.path.exists(chain_file) or overwrite:
              
        work.run()
    else:
        print('Samples already produced. Skipping ...')
    print('Getting posterior samples...')
    work.get_posterior_samples(filename='posterior.txt')
        
    
    
    
    
    #print runtime
    tstop = timeit.default_timer()
    _ = print_runtime(tstart, tstop)
    

    '''
    On WINDOWS: cd to the results dir and type "watch tail -n10 cpnest.log" to see the logs during the sampling
    '''
    #print(best_estimates)
    return #best_estimates

     


'''==============================  MAIN ==========================================='''
if __name__=='__main__':
     main()
    