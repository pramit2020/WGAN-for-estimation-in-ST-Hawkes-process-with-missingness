import pickle
import sys
import numpy as np
import random
import numpy.random as rand
from scipy.stats import poisson
import torch
import matplotlib
import matplotlib.pyplot as plt
import torch as torch
import torch.nn as nn
import pandas as pd
from torch import nn, optim, autograd
import time
torch.set_default_tensor_type(torch.DoubleTensor)
#torch.set_default_dtype(torch.DoubleTensor) 
import shapely 
import geopandas as gdp
import numpy as np
import math
import itertools
import numpy as np 
import pylab 
import scipy.stats as stats
import argparse
from scipy.stats import wasserstein_distance


#change the Bogota file path if needed
#load Bogota map and data
fp="diff-crime-reporting forked from nakpinar's Github repo/metadata/bogota.shp"
Bogota = gdp.read_file(fp)
# bogota vic, missing, population and maps
bogota_crime_stats=pd.read_csv("/home/pramitd/Bogota_folder/bogota_victimization.csv")
bogota_crime_stats[["LocNombre"]] = bogota_crime_stats[["District"]]
del bogota_crime_stats["District"]
#bogota_crime_stats
bogota_merged=pd.merge(Bogota, bogota_crime_stats, on ="LocNombre")

## scaling_factor for the population - this is not very important in our WGAN-based method, though
scaling_factor = 500
bogota_pop = np.array(bogota_merged[["Population"]])
bogota_vic_rate = np.array(bogota_merged[["Victimization"]])
bogota_missing_rate = 1- bogota_merged[["Percent_reported"]]
crime_numbers = (bogota_pop/scaling_factor)*bogota_vic_rate*12
bogota_merged["expected crime numbers"] = crime_numbers

#Bogota centres taken by Nil jana Akpinar
centers=torch.tensor([[6,20],[-6,20],[6,-20],[-6,-20],[6,-10],[6,10], [-6,-10],[-6,10],[-6,0],[6,0],[-6,-30],[-6,30],[6,30],[6,-30]], requires_grad = False)
#preparing map[ disyrict data from Bogota 
D = dict(zip(bogota_merged.LocNombre, bogota_merged.geometry))
polys = gdp.GeoSeries(D)
#new
def districts(data_stream):
    """
    Given a data stream containing (t_i, x_i, y_i) coordinates and a dictionary of polygons representing districts,
    assigns district membership to each point in the data stream based on spatial location.

    Args:
    - data_stream (torch.Tensor): Tensor of shape (3, N) containing (t_i, x_i, y_i) coordinates.
    - polys (dict): Dictionary of polygons representing districts.

    Returns:
    - district_list (list): List containing the district membership for each point in the data stream.
    """
    dummy_stream = torch.clone(data_stream).detach()  # Create a non-tensor numpy copy of the data stream
    spatial_comps = gdp.GeoDataFrame(geometry=gdp.points_from_xy(x=dummy_stream[1, :].numpy(),
                                                                 y=dummy_stream[2, :].numpy()))  # (x_i, y_i) pairs
    spatial_comps = spatial_comps.set_geometry('geometry')  # Set the geometry column explicitly
    M = spatial_comps.assign(**{key: spatial_comps.within(geom) for key, geom in polys.items()})
    M_array = np.array(M)
    district_list = []
    for i in range(data_stream.shape[1]):
        if np.sum(M_array[i, 1:]) == 0:
            district_list.append(-1)  # Point is outside
        else:
            index = np.where(M_array[i, 1:] == True)[0][0]  # Take the first district in case of multiple matches
            district_list.append(int(index))
    return district_list

#write a function to give how many points in each district from a data stream 
def district_wise_numbers(data_stream):
    d = districts(data_stream)
    num_list = np.zeros(19)
    for i in range(N):
        if d[i] !=-1:
            num_list[d[i]] = num_list[d[i]]+1     
    return num_list


cut_off = 50
burn_in = 75
sigma_background = torch.tensor(4.5, requires_grad = False)
#this is in exact form of Nil Jana akpinar's eqn upto constants
def bogota_ST_hawkes_parallel(N, mu, alpha, beta, sigma, K):
    #initiate blank lists
    t_events = [] #t_i's
    x_events = [] #x_i's
    y_events = [] #y_i's
    
    t1= time.time()
    

    #change these
    big_N_unif = K*N*(N+1)//2+ K*centers.shape[0]+K*N*centers.shape[0]+1000
    big_N_Gaussian = 2*(N*K*centers.shape[0]+K*centers.shape[0]+1000)
    #print("big_N_unif = ", big_N_unif)
    #print("big_N_Gaussian = ", big_N_Gaussian)
    seq_of_Gaussian_tensors = torch.randn(size=(big_N_Gaussian,))
    seq_of_uniform_tensors =torch.tensor(np.random.uniform(0,1,size=big_N_unif))
    
    t2 = time.time()
    
    print("time required to generate the uniforms/Gaussians is : ", round((t2-t1)/60,2), " minutes")

    uc=0 ## counter of uniforms used
    gc=0 ## counter of how many Gaussian's used for spatial component

    ## generate K many first events & need to tweak the background intenbsity function
    # we now would make a K by 14 tensor than a K sized vector
    partial_mu = mu/centers.shape[0]
    #t_first_matrix = -torch.log(1-seq_of_uniform_tensors[uc:uc+K*centers.shape[0]]).reshape((K, centers.shape[0]))/torch.mul(partial_mu, torch.pow(sigma,2))
    t_first_matrix = -torch.log(1-seq_of_uniform_tensors[uc:uc+K*centers.shape[0]]).reshape((K, centers.shape[0]))/partial_mu #first arrival time ok
    uc = uc+K*centers.shape[0] #uc+14K
    M = torch.min(t_first_matrix, axis=1)
    t_first = M[0] #this should be a K sized vector
    order_center_fires = np.array(M[1])
    
    #need to adjust this with which center is the event coming from 
    eta_x_first = sigma_background*seq_of_Gaussian_tensors[gc:gc+K]+centers[order_center_fires,0] #first x
    gc=gc+K  
    eta_y_first = sigma_background*seq_of_Gaussian_tensors[gc:gc+K]+centers[order_center_fires,1] #first y
    gc=gc+K 
    #concatenate 
    #first_occs=torch.stack((t_first, eta_x_first, eta_y_first), axis=1) ## (t_i,x_i,y_i) pairs for first events, i \in number of seqs.
    t_events.append(t_first)
    x_events.append(eta_x_first)
    y_events.append(eta_y_first)


    ##generate times first
    for p in range(N):
        if p>0 & p <= burn_in:
            t_empty_list = []
            x_empty_list =[]
            y_empty_list=[]
            
            u_vec = seq_of_uniform_tensors[uc:uc+K*p]
            uc=uc+K*p
            E=torch.stack(t_events[0:p],dim=1)
            F=torch.broadcast_to(t_events[p-1].reshape((K,1)), E.shape)
            hello = (1+beta/2*math.pi*torch.mul(alpha,torch.pow(sigma,2))*torch.exp(-beta*(E-F))*torch.log(1-u_vec).reshape((K,p))) ## if negative, put 0 , these are arrivals from triggering part 
            # this generates all possible next arrival times for K streams
           #change this, instead of t_base being a K sized vector, we need to get a K by 14 sized tensor
            t_base_matrix = -torch.log(1-seq_of_uniform_tensors[uc:uc+K*centers.shape[0]]).reshape((K,centers.shape[0]))/partial_mu #possible arrivals from background density from each center
            uc = uc+K*centers.shape[0] #uc=uc+14K
            M = torch.min(t_base_matrix, axis=1)
            order_center_fires = np.array(M[1])
            t_base = M[0] #this should be a K sized vector 
            vals_base = t_events[p-1]+t_base #this is a matrix
            
            for stream in range(K):
                hello_augmented = hello[stream,:][hello[stream,:]>=0]
                
                if len(hello_augmented)>0:

                    args = -torch.log(hello_augmented)/beta ## 0's would yield infinity here 
                    vals= t_events[p-1][stream]+torch.min(args) #this would give the pth arrival time in a particular stream 
                    position = torch.argmin(args)
                    if vals_base[stream]<vals:
                        position = -1 ## base
                    t_empty_list.append(torch.min(vals,vals_base[stream]))

                else:
                    position = -1  
                    t_empty_list.append(vals_base[stream])

                #now time to generate the pth spatial coords in a particular stream

                if position==-1:
                    x_empty_list.append(sigma_background*seq_of_Gaussian_tensors[gc]+centers[order_center_fires[stream],0]) #sigma_background triggering from base inetnsity
                    gc=gc+1
                    y_empty_list.append(sigma_background*seq_of_Gaussian_tensors[gc]+centers[order_center_fires[stream],1])
                    gc=gc+1
                else:
                    x_empty_list.append(torch.mul(sigma,seq_of_Gaussian_tensors[gc])+x_events[p-1][stream]) #sigma triggering from g 
                    gc=gc+1
                    y_empty_list.append(torch.mul(sigma,seq_of_Gaussian_tensors[gc])+y_events[p-1][stream])
                    gc=gc+1

            t_events.append(torch.stack(t_empty_list))  # works till here - generates t_n+1 if we know previous t_i's
            x_events.append(torch.stack(x_empty_list))
            y_events.append(torch.stack(y_empty_list))
        
        if p>0 & p> burn_in:
            
            # we shall ignore past events beyond cut_off steps to generate future triggers
            t_empty_list = []
            x_empty_list =[]
            y_empty_list=[]
            
            u_vec = seq_of_uniform_tensors[uc:uc+K*cut_off]
            uc=uc+K*cut_off
            E=torch.stack(t_events[p-cut_off:p],dim=1)
            F=torch.broadcast_to(t_events[p-1].reshape((K,1)), E.shape)
            hello = (1+beta/2*math.pi*torch.mul(alpha,torch.pow(sigma,2))*torch.exp(-beta*(E-F))*torch.log(1-u_vec).reshape((K,cut_off))) ## if negative, put 0 , triggering from g 
            # this generates all possible next arrival times for K streams
           #change this, instead of t_base being a K sized vector, we need to get a K by 14 sized tensor
            t_base_matrix = -torch.log(1-seq_of_uniform_tensors[uc:uc+K*centers.shape[0]]).reshape((K,centers.shape[0]))/partial_mu #possible arrivals from background density
            uc = uc+K*centers.shape[0] #uc=uc+14K
            M = torch.min(t_base_matrix, axis=1)
            order_center_fires = np.array(M[1])
            t_base = M[0] #this should be a K sized vector 
            vals_base = t_events[p-1]+t_base #this is a matrix
            
            for stream in range(K):
                hello_augmented = hello[stream,:][hello[stream,:]>=0]
                
                if len(hello_augmented)>0:

                    args = -torch.log(hello_augmented)/beta ## 0's would yield infinity here 
                    vals= t_events[p-1][stream]+torch.min(args) #this would give the pth arrival time in a particular stream 
                    position = torch.argmin(args)
                    if vals_base[stream]<vals:
                        position = -1 ## base
                    t_empty_list.append(torch.min(vals,vals_base[stream]))

                else:
                    position = -1  
                    t_empty_list.append(vals_base[stream])

                #now time to generate the pth spatial coords in a particular stream

                if position==-1:
                    x_empty_list.append(sigma_background*seq_of_Gaussian_tensors[gc]+centers[order_center_fires[stream],0]) #sigma_back triggering from 
                    gc=gc+1
                    y_empty_list.append(sigma_background*seq_of_Gaussian_tensors[gc]+centers[order_center_fires[stream],1])
                    gc=gc+1
                else:
                    x_empty_list.append(torch.mul(sigma,seq_of_Gaussian_tensors[gc])+x_events[p-1][stream]) #sigma trigger from g 
                    gc=gc+1
                    y_empty_list.append(torch.mul(sigma,seq_of_Gaussian_tensors[gc])+y_events[p-1][stream])
                    gc=gc+1

            t_events.append(torch.stack(t_empty_list))  # works till here - generates t_n+1 if we know previous t_i's
            x_events.append(torch.stack(x_empty_list))
            y_events.append(torch.stack(y_empty_list))
            
        
    T = torch.stack(t_events)
    gapped_T = torch.cat((T[0,:].reshape((1,K)), torch.diff(T, axis =0)), dim=0)
    X = torch.stack(x_events)
    Y = torch.stack(y_events)
    return torch.stack((gapped_T,X,Y), dim=0)


# Thinning function for a single stream
def one_stage_thinning_FAKE(data_stream):
    N = data_stream.shape[1]
    
    # Convert t_i - t_{i-1} to t_i
    t_cumsum = torch.cumsum(data_stream[0, :], dim=0)
    
    # Create a new data stream with cumulative times
    data_stream_cumsum = torch.stack((t_cumsum, data_stream[1, :], data_stream[2, :]))
    
    # Calculate district membership
    district_membership = districts(data_stream_cumsum)
    
    # Generate acceptance probabilities
    acceptance = np.zeros(N)
    unif_series = np.random.uniform(low=0, high=1, size=N)
    acceptance_prob = (1 - np.array(bogota_missing_rate)).reshape(19)

    # Apply the thinning process
    for i in range(N):
        if district_membership[i] != -1:
            if unif_series[i] <= acceptance_prob[district_membership[i]]:
                acceptance[i] = 1

    # Filter the accepted events
    accepted_data = data_stream_cumsum[:, acceptance.astype(bool)]
    
    # Compute the time differences between reported events
    if accepted_data.shape[1] > 0:
        t_differences = torch.cat((accepted_data[0, :1], torch.diff(accepted_data[0, :])))
    else:
        t_differences = torch.zeros(0)

    thinned_data = torch.stack((t_differences, accepted_data[1, :], accepted_data[2, :]), dim=0)

    # Pad with zeros to maintain the original size (3, N)
    padding_size = N - thinned_data.shape[1]
    padding = torch.zeros((3, padding_size))
    thinned_data = torch.cat((thinned_data, padding), dim=1)

    return thinned_data

# Thinning function for multiple streams
def one_stage_thinning_multiple_streams_FAKE(bogota_stream):
    thinned_list = []
    for i in range(bogota_stream.shape[2]):
        thinned_list.append(one_stage_thinning_FAKE(bogota_stream[:, :, i]))
    return torch.stack(thinned_list, dim=2)


# Generate synthetic crimes with thinning
def generate_FAKE_crimes_bogota(N, mu, alpha, beta, sigma, K):
    fake_crimes_all = bogota_ST_hawkes_parallel(N, mu, alpha, beta, sigma, K)
    return one_stage_thinning_multiple_streams_FAKE(fake_crimes_all)



def generate_synthetic_data(estimate, N, Ksynthetic):
    """
    Generate synthetic data streams based on the given estimate.
    
    Parameters:
    estimate (dict): Dictionary containing model parameters.
    N (int): Size of data stream.
    Ksynthetic (int): Number of synthetic replicates.
    
    Returns:
    list: List of synthetic data streams.
    """
    
    est_1= torch.tensor(estimate)
    mu,alpha,beta,sigma=est_1[0],est_1[1],est_1[2],est_1[3]
    synthetic_data = generate_FAKE_crimes_bogota(N,mu, alpha,beta,sigma,Ksynthetic)
    #synthetic_data = [generate_FAKE_crimes_bogota(N=N, mu=estimate['mu'], alpha=estimate['alpha'], beta=estimate['beta'], sigma=0.1, K=10) for _ in range(Ksynthetic)]
    return synthetic_data

#function to compute the chisquare statistic for goodness of fit
N = 250
Ksynthetic = 1000
comparison_results_filepath = "GoF_test_chisq_true_vs_synthetic_interarrival_times_Bogota_WGAN.txt"
num_bins = 50

#here, the training_data would have the replications of reported crime data
#the estimate theta_hat should be a tuple containing the values (mu_hat,alpha_hat,beta_hatmsigma_hat)

def compute_chisq_true_vs_synthetic_interarrival_times(training_data, estimate):
    t_start = time.time()
    # Load the real training data
    with open(training_data, 'rb') as file:
        training_data_bogota = pickle.load(file)
        
    # Extract interarrival times from the data
    interarrival_times = training_data_bogota[0, :, :]
    # Flatten the interarrival times tensor and filter out zero intervals
    flattened_interarrival_times = interarrival_times.flatten()
    training_non_zero_intervals = flattened_interarrival_times[flattened_interarrival_times > 0]

    # Generate synthetic data from estimates
    mu_hat, alpha_hat, beta_hat, sigma_hat = estimate
    #load_estimates(estimates_file_path)
    mu_hat_tensor = torch.tensor(mu_hat)
    alpha_hat_tensor = torch.tensor(alpha_hat)
    beta_hat_tensor = torch.tensor(beta_hat)
    sigma_hat_tensor = torch.tensor(sigma_hat)
    synthetic_data = generate_FAKE_crimes_bogota(N, mu_hat_tensor, alpha_hat_tensor, beta_hat_tensor, sigma_hat_tensor, Ksynthetic)

    # Extract interarrival times from the synthetic data
    synthetic_interarrival_times = synthetic_data[0, :, :].flatten()
    # Filter out zero intervals
    synthetic_interarrival_times = synthetic_interarrival_times[synthetic_interarrival_times > 0]
    
    t_max = max(np.max(training_non_zero_intervals.numpy()),np.max(synthetic_interarrival_times.numpy()))

    # Define the bin edges from 0 to 0.4 with 50 bins
    bin_edges = np.linspace(0, t_max, num_bins + 1)

    # Create histograms for the interarrival times
    hist_training, _ = np.histogram(training_non_zero_intervals.numpy(), bins=bin_edges, density=False)
    hist_synthetic, _ = np.histogram(synthetic_interarrival_times.numpy(), bins=bin_edges, density=False)

    # Add a small constant to avoid zero bins
    hist_training = hist_training.astype(float) + 1e-8
    hist_synthetic = hist_synthetic.astype(float) + 1e-8

    # Normalize histograms so that their sums match
    hist_training_normalized = hist_training / hist_training.sum()
    hist_synthetic_normalized = hist_synthetic / hist_synthetic.sum()

    # Calculate the Chi-Squared statistic
    chisq_stat = np.sum((hist_training_normalized - hist_synthetic_normalized) ** 2 / hist_training_normalized)

    # Plot histograms side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    # Histogram for training data
    axs[0].hist(training_non_zero_intervals.numpy(), bins=bin_edges, edgecolor='black', density=True)
    axs[0].set_title('Histogram of Non-Zero Interarrival Times in Training Data')
    axs[0].set_xlabel('Interarrival Time')
    axs[0].set_ylabel('Normalized Frequency')
    axs[0].grid(True)

    # Histogram for synthetic data
    axs[1].hist(synthetic_interarrival_times.numpy(), bins=bin_edges, edgecolor='black', density=True)
    axs[1].set_title('Histogram of Non-Zero Interarrival Times in Synthetic Data')
    axs[1].set_xlabel('Interarrival Time')
    axs[1].set_ylabel('Normalized Frequency')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Display the results with parameters
    result = f"Chi-Squared statistic between observed and synthetic interarrival times with parameters mu={mu_hat}, alpha={alpha_hat}, beta={beta_hat}, sigma={sigma_hat}: {round(chisq_stat, 4)}"
    print(result)
    
    # Save the result to a file
    with open(comparison_results_filepath, 'a') as f:
        f.write(result + '\n')
        
    t_end = time.time()
    
    print("Total time elapsed = :", round((t_end-t_start)/60, 2), " minutes")
    
    return chisq_stat





