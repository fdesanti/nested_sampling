import numpy as np
import matplotlib.colors as mcolors
blue_color = mcolors.TABLEAU_COLORS['tab:blue']
import matplotlib.pyplot as plt



def print_runtime(tstart, tstop):
    runtime = tstop - tstart
    if (runtime >= 60) and (runtime < 3600) :
        runtime /= 60
        unit = "min"
    elif runtime >= 3600:
        runtime /= 3600
        unit = "hours"
    else:
        unit = "sec"

    print('Runtime took: %.1f %s'%(runtime, unit))  
    
    return runtime

def plot_data_histogram(data, data_label = 'logT90', nbins = 'fd', density = True):
    
    counts, bins = np.histogram(data, bins = nbins, density = density)
    #print('counts', counts)
    #print('bins', bins )

    bins = 0.5*( bins[0:-1] + bins[1:]) #bin centers
    
    bin_width = np.mean(np.diff(bins))
    counts /= bin_width

    plt.figure()
    plt.hist(bins, bins, weights=counts, histtype='stepfilled', alpha = 0.2)
    plt.hist(bins, bins, weights=counts, histtype='step', color = blue_color)
    plt.xlabel(data_label)
    plt.ylabel('Normalized density')
    plt.minorticks_on()
    #plt.xlim(bins[0], bins[-1])
    plt.savefig('results/'+data_label+'_data_histogram.png', dpi=300)
    #plt.show()
    plt.close()

    hist_data = {'x':bins, 'y':counts}

    return hist_data

def get_posterior_samples(chain_file):
        samples = np.genfromtxt(chain_file, skip_header = 1)
        samples = samples.T
        #get header
        with open(chain_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                header = line.strip().split('\t')[:-1]
                print(header, len(header))
                break
        
        posteriors_samples_dict = dict()
        for i, name in enumerate(header):
            
            posteriors_samples_dict[name] = samples[i]


        return header, posteriors_samples_dict

def convert_to_latex_label(header):
        var_labels = []
        for name in header:
            if not name.startswith('w'):
                var_name = "$\\"+name +"$"
            else:
                var_name = "$"+name+"$"
            var_labels.append(var_name)

        return var_labels

def get_best_estimates(posteriors):

        best_estimates = dict()

        for key in posteriors.keys():

            best_estimates[key] = np.quantile(posteriors[key], 0.5)
    

        return best_estimates

def determine_best_num_bins(posterior_samples_dict, mode = 'fd'):

    num_bins = []

    for key in posterior_samples_dict.keys():
         samples = posterior_samples_dict[key]
         nbins = len(np.histogram_bin_edges(samples, bins = mode)) - 1
         num_bins.append(nbins)

    return np.array(num_bins)


