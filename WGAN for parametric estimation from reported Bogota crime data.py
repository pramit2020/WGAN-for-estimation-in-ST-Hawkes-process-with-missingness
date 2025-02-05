#!/usr/bin/env python
# coding: utf-8


import argparse
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
import shapely 
import pickle
import geopandas as gdp
import numpy as np
import math
import itertools
import numpy as np 
import pylab 
import scipy.stats as stats
import sys

# Set up the argument parser
parser = argparse.ArgumentParser(description='Process the parameters.')
parser.add_argument('--a', type=float, required=True, help='Parameter a as a float')
parser.add_argument('--b', type=float, required=True, help='Parameter b as a float')
parser.add_argument('--c', type=float, required=True, help='Parameter c as a float')
parser.add_argument('--d', type=float, required=True, help='Parameter d as a float')

# Parse the arguments
args = parser.parse_args()

a = args.a
b = args.b
c = args.c
d = args.d

a_copy =a
b_copy = b
c_copy =c
d_copy = d

#this is the initial values of mu, alpha, beta, sigma in the optimization routine
print("Initialization point a,b,c,d = ",a,b,c,d)



#load Bogot.shp
fp = "~/Bogota_folder/bogota.shp"
Bogota = gdp.read_file(fp)
Bogota.head()


# In[23]:


pd.read_csv("~/Bogota_folder/bogota_victimization.csv")


# In[24]:


# bogota vic, missing, population and maps
bogota_crime_stats=pd.read_csv("~/Bogota_folder/bogota_victimization.csv")
bogota_crime_stats[["LocNombre"]] = bogota_crime_stats[["District"]]
del bogota_crime_stats["District"]
bogota_crime_stats


# In[25]:


bogota_merged=pd.merge(Bogota, bogota_crime_stats, on ="LocNombre")


# In[26]:


## scaling_factor 
scaling_factor = 500


# In[27]:


bogota_pop = np.array(bogota_merged[["Population"]])
bogota_vic_rate = np.array(bogota_merged[["Victimization"]])
bogota_missing_rate = 1- bogota_merged[["Percent_reported"]]
crime_numbers = (bogota_pop/scaling_factor)*bogota_vic_rate*12
bogota_merged["expected crime numbers"] = crime_numbers
fig, ax = plt.subplots(figsize=(50, 10))
bogota_merged.plot(column='expected crime numbers', ax=ax,legend=True)



#Bogota centres taken by Nil jana Akpinar
centers=torch.tensor([[6,20],[-6,20],[6,-20],[-6,-20],[6,-10],[6,10], [-6,-10],[-6,10],[-6,0],[6,0],[-6,-30],[-6,30],[6,30],[6,-30]], requires_grad = False)


#preparing map[ disyrict data from Bogota 
D = dict(zip(bogota_merged.LocNombre, bogota_merged.geometry))
polys = gdp.GeoSeries(D)

def districts(data_stream): #given (t_i,x_i,y_i) stream, it'd give the districts
     
    dummy_stream = torch.clone(data_stream).detach() #create a non-tensor numpy copy of the data stream
    spatial_comps =gdp.GeoDataFrame(gdp.points_from_xy(x=dummy_stream[1,:] , y=dummy_stream[2,:])) #(x_i,y_i) pairs
    spatial_comps.rename(columns={0: 'geometry'},inplace=True)
    M=spatial_comps.assign(**{key: spatial_comps.within(geom) for key, geom in polys.items()})
    M = np.array(M)
    N = M[:,1:] #this should contain 0,1's 
    
    district_list=[]
    for i in range(data_stream.shape[1]):
        if np.sum(N[i,:])==0:
            district_list.append(-1) #point is outside
        else:
            index = np.where(N[i,:]==1)[0] #take the first one in case of any anomoly 
            district_list.append(int(index))
    
    return np.array(district_list)

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




#write a function give how many points in ecah dfistriuct from a data stream 
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

    #change these
    big_N_unif = K*N*(N+1)//2+ K*centers.shape[0]+K*N*centers.shape[0]+1000
    big_N_Gaussian = 2*(N*K*centers.shape[0]+K*centers.shape[0]+1000)
    seq_of_Gaussian_tensors = torch.randn(size=(big_N_Gaussian,))
    seq_of_uniform_tensors =torch.tensor(np.random.uniform(0,1,size=big_N_unif))

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


# In[39]:


#newer version - supposedly faster. Does it work? 
def NEW_bogota_ST_hawkes_parallel(N,mu, alpha, beta, sigma, K):
    # Initiate blank lists
    t_events = []  # t_i's
    x_events = []  # x_i's
    y_events = []  # y_i's

    # Change these
    centers = torch.tensor([[6, 20], [-6, 20], [6, -20], [-6, -20], [6, -10], [6, 10], [-6, -10], [-6, 10],
                            [-6, 0], [6, 0], [-6, -30], [-6, 30], [6, 30], [6, -30]], requires_grad=False)
    sigma_background = torch.tensor(4.5, requires_grad=False)

    #change these
    big_N_unif = K*N*(N+1)//2+ K*centers.shape[0]+K*N*centers.shape[0]+1000
    big_N_Gaussian = 2*(N*K*centers.shape[0]+K*centers.shape[0]+1000)
    #if torch.is_tensor(big_N_unif):
    #    big_N_unif = int(big_N_unif.item())
    #if torch.is_tensor(big_N_Gaussian):
#        big_N_Gaussian = int(big_N_Gaussian.item())

    seq_of_Gaussian_tensors = torch.randn(size=(big_N_Gaussian,))
    seq_of_uniform_tensors =torch.tensor(np.random.uniform(0,1,size=big_N_unif))

    uc = 0  # Counter of uniforms used
    gc = 0  # Counter of how many Gaussian's used for spatial component

    # Generate K many first events & need to tweak the background intensity function
    partial_mu = mu / centers.shape[0]
    t_first_matrix = -torch.log(1 - seq_of_uniform_tensors[uc:uc + K * centers.shape[0]]).reshape(
        (K, centers.shape[0])) / partial_mu
    uc = uc + K * centers.shape[0]  # uc+14K
    M = torch.min(t_first_matrix, axis=1)
    t_first = M[0]  # This should be a K sized vector
    order_center_fires = np.array(M[1])

    # Need to adjust this with which center is the event coming from
    eta_x_first = sigma_background * seq_of_Gaussian_tensors[gc:gc + K] + centers[order_center_fires, 0]
    gc = gc + K
    eta_y_first = sigma_background * seq_of_Gaussian_tensors[gc:gc + K] + centers[order_center_fires, 1]
    gc = gc + K
    # Concatenate
    t_events.append(t_first)
    x_events.append(eta_x_first)
    y_events.append(eta_y_first)
    #print("first events generated for all streams!")

    ## Generate times first
    for p in range(1, N):
        if p > 0 and p <= burn_in:
            t_prev = t_events[p - 1]

            u_vec = seq_of_uniform_tensors[uc:uc + K * p]
            uc = uc + K * p
            E = torch.stack(t_events[0:p], dim=1)
            F = torch.broadcast_to(t_events[p - 1].reshape((K, 1)), E.shape)

            # Candidate arrival times from triggering
            hello = (1 + beta / (2 * math.pi) * alpha * sigma ** 2 * torch.exp(-beta * (E - F)) * torch.log(
                1 - u_vec).reshape((K, p)))

            # Candidate arrival times from background intensity
            t_base_matrix = -torch.log(1 - seq_of_uniform_tensors[uc:uc + K * centers.shape[0]]) / partial_mu
            uc = uc + K * centers.shape[0]
            t_base_matrix = t_base_matrix.reshape((K, centers.shape[0]))  # Reshape t_base_matrix
            M = torch.min(t_base_matrix, axis=1)
            order_center_fires = np.array(M[1])
            t_base = M[0]

            # Calculate vals_base
            vals_base = t_events[p - 1] + t_base  # This is a matrix

            # Reshape vals_base as a K by 1 tensor
            vals_base_reshaped = vals_base.unsqueeze(1)

            # Calculate the minimum row-wise
            t_empty_list = torch.min(torch.cat((hello, vals_base_reshaped), dim=1), dim=1)[0]

            # Append t_empty_list to t_events
            t_events.append(t_empty_list)

            x_empty_list = []
            y_empty_list = []
            for stream in range(K):
                x_empty_list.append(sigma * seq_of_Gaussian_tensors[gc] + x_events[p - 1][stream])
                gc = gc + 1
                y_empty_list.append(sigma * seq_of_Gaussian_tensors[gc] + y_events[p - 1][stream])
                gc = gc + 1
            x_events.append(torch.stack(x_empty_list))
            y_events.append(torch.stack(y_empty_list))

        if p > burn_in:
            #print("we are now in p> burn_in period")
            # We shall ignore past events beyond cut_off steps to generate future triggers
            t_prev = t_events[p - 1]
            mask_base = t_prev < cut_off
            u_vec = seq_of_uniform_tensors[uc:uc + K * cut_off]
            uc = uc + K * cut_off
            E = torch.stack(t_events[p - cut_off:p], dim=1)
            F = torch.broadcast_to(t_events[p - 1].reshape((K, 1)), E.shape)
            hello = (1 + beta / (2 * math.pi) * alpha * sigma ** 2 * torch.exp(-beta * (E - F)) * torch.log(
                1 - u_vec).reshape((K, cut_off)))

            t_base_matrix = -torch.log(1 - seq_of_uniform_tensors[uc:uc + K * centers.shape[0]]) / partial_mu
            uc = uc + K * centers.shape[0]
            t_base_matrix = t_base_matrix.reshape((K, centers.shape[0]))  # Reshape t_base_matrix
            M = torch.min(t_base_matrix, axis=1)
            order_center_fires = np.array(M[1])
            t_base = M[0]

            # Calculate vals_base
            vals_base = t_events[p - 1] + t_base  # This is a matrix

            # Reshape vals_base as a K by 1 tensor
            vals_base_reshaped = vals_base.unsqueeze(1)

            # Calculate the minimum row-wise
            t_empty_list = torch.min(torch.cat((hello, vals_base_reshaped), dim=1), dim=1)[0]

            # Append t_empty_list to t_events
            t_events.append(t_empty_list)

            x_empty_list = []
            y_empty_list = []
            for stream in range(K):
                if mask_base[stream]:
                    x_empty_list.append(sigma * seq_of_Gaussian_tensors[gc] + x_events[p - 1][stream])
                    gc = gc + 1
                    y_empty_list.append(sigma * seq_of_Gaussian_tensors[gc] + y_events[p - 1][stream])
                    gc = gc + 1
                else:
                    x_empty_list.append(sigma_background * seq_of_Gaussian_tensors[gc] + centers[order_center_fires[stream], 0])
                    gc = gc + 1
                    y_empty_list.append(sigma_background * seq_of_Gaussian_tensors[gc] + centers[order_center_fires[stream], 1])
                    gc = gc + 1
            x_events.append(torch.stack(x_empty_list))
            y_events.append(torch.stack(y_empty_list))
        #print("p = ",p, " done for all streams")

    T = torch.stack(t_events)
    gapped_T = torch.cat((T[0, :].reshape((1, K)), torch.diff(T, axis=0)), dim=0)
    X = torch.stack(x_events)
    Y = torch.stack(y_events)

    #return T, gapped_T, X, Y
    return torch.stack((gapped_T,X,Y), dim=0)


# In[40]:


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



# In[41]:


# Generate synthetic crimes with thinning
#use old hawkes function
def generate_FAKE_crimes_bogota(N, mu, alpha, beta, sigma, K):
    fake_crimes_all = bogota_ST_hawkes_parallel(N, mu, alpha, beta, sigma, K)
    return one_stage_thinning_multiple_streams_FAKE(fake_crimes_all)
#############
def generate_REAL_crimes_bogota(N, mu, alpha, beta, sigma, K):
    real_crimes_all = bogota_ST_hawkes_parallel(N, mu, alpha, beta, sigma, K)
    return one_stage_thinning_multiple_streams_FAKE(real_crimes_all)  


# In[23]:


##generate real data ~P_r
## real data ~ P_r = HP(mu,alpha,beta)
## generate these in batch rather than individually
## simulation 
# Set the random seed for PyTorch
torch.manual_seed(42)
# Set the random seed for NumPy
np.random.seed(42)
# Set the random seed for Python's built-in random module
random.seed(42)

real_mu = torch.tensor(a, requires_grad=False)
real_alpha =  torch.tensor(b, requires_grad=False)
real_beta = torch.tensor(c, requires_grad=False)
real_sigma = torch.tensor(d, requires_grad=False)

# stream length 
N=250
num_real_seqs=10000

#generate data
st = time.time()
#Extended_Real_data_bogota = generate_REAL_crimes_bogota(N, real_mu, real_alpha, real_beta, real_sigma, num_real_seqs)
Real_data_bogota = generate_REAL_crimes_bogota(N, real_mu, real_alpha, real_beta, real_sigma, num_real_seqs)
et = time.time()
print("OLD function: time required for generating ", str(num_real_seqs), " Bogota crime Spatiotemporal HP realisations of size", str(N), " is ", round((et-st)/60,3), " minutes")




# Assume that my_object is the object you want to save
file_name = 'real_bogota_data_size_250_with_params (a,b,c,d) = "+ str(a_copy)+" "+ str(b_copy)+ " "+ str(c_copy)+ " "+ str(d_copy)+".pkl'
with open(file_name, 'wb') as f:
    pickle.dump(Real_data_bogota, f) #save this



def batch_discriminator_RNN(batch_of_sequences):

  n = batch_of_sequences.shape[1]+1 
  l =  batch_of_sequences.shape[2] ##L kind of thing - how many streams
  input = batch_of_sequences.reshape((n-1,l,3))
 # input = torch.randn(50, 256, 1) #(L,N,h_in), L = sequence length, N = batch size, h_input = 10 
  
  h0 = torch.randn(3, l, k) ## (D*n_layers, N, h_out)
  global rnn
  output, hn = rnn(input, h0)
  vals=torch.sigmoid(torch.matmul(output,A)+b)

  return torch.sum(vals[:,:,0], dim=0)


# In[50]:


def batch_discriminator_LSTM(batch_of_sequences):

  n = batch_of_sequences.shape[1]+1
  l =  batch_of_sequences.shape[2]
  input = batch_of_sequences.reshape((n-1,l,3))
 # input = torch.randn(50, 256, 1) 
    #(L,N,h_in), L = sequence length, N = batch size, h_input = 10 
  

  global k 
  h0 = torch.randn(3, l, k) ## (D*n_layers, N, h_out)
  c0= torch.randn(3, l, k) 
  global lstm
  output, (hn,cn) = lstm(input, (h0,c0))
  vals=torch.sigmoid(torch.matmul(output,A)+b)
  
    #print(vals.shape)
  
  return torch.sum(vals[:,:,0], dim=0)
  #return torch.sum(vals, dim= 0).reshape((l,))





##new one with mask 
def batch_discriminator_LSTM(batch_of_sequences):
    n = batch_of_sequences.shape[1] + 1
    l = batch_of_sequences.shape[2]
    input = batch_of_sequences.reshape((n - 1, l, 3))

    # Create a mask for padded zeros
    mask = (input[:, :, 0] != 0).float()  # Assuming the first column represents the event occurrence

    global k
    h0 = torch.randn(3, l, k)  # (D*n_layers, N, h_out)
    c0 = torch.randn(3, l, k)
    global lstm
    output, (hn, cn) = lstm(input, (h0, c0))

    # Apply the mask to the output
    masked_output = output * mask.unsqueeze(-1)

    vals = torch.sigmoid(torch.matmul(masked_output, A) + b)

    return torch.sum(vals[:, :, 0], dim=0)



## here alpha, beta, mu would be passed as tensor arguments
## within the function, we shall convert them to numbers
##zeta is a generation of crimes from Bogota


def discriminator_loss_GP(zeta):
    global nu
    err = pow(10,-5)  
    real_size = (Real_data_bogota.shape)[2]
  ## eps is some sample from real data, i.e. a random chunk among the L chunks of real data
    sample = np.random.randint(low=0, high=real_size-1, size=L, dtype='int')
    eps = Real_data_bogota[:,:,sample] ## stream of L chunks from real missing data
   #compute loss
    a = batch_discriminator_LSTM(zeta)
    b = batch_discriminator_LSTM(eps)
  
    t = torch.rand(3,N,L,)
    mid = t * eps + (1 - t) * zeta ## doesn't have any gradients coming from real and fake data
  # set it to require grad info
    mid.requires_grad_()
    pred = batch_discriminator_LSTM(mid)
    grads = autograd.grad(outputs=pred, inputs=mid,
                      grad_outputs=torch.ones_like(pred),
                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    grads_padding = grads+pow(10,-10)    ## add a small constant to avoid NaN in gradient of torch.norm                
    gp = torch.pow(grads_padding.norm(2, dim=1) - 1, 2).mean()
    Wasserstein_dist =   torch.mean(b)- torch.mean(a)
    l = -torch.mean(b)+ torch.mean(a)+ nu*gp
    return (l,Wasserstein_dist)


# In[54]:


def generator_loss():
    zeta = generate_FAKE_crimes_bogota(N, mu, alpha, beta, sigma, L)
    #compute loss
    a = batch_discriminator_LSTM(zeta)
    l = - torch.mean(a)
    return l


# In[55]:


#set up training, optimizers 
from scipy.stats.distributions import betaprime
#initiate parameters
#torch.autograd.set_detect_anomaly(True)
N = 250
L = 128
num_epochs = 100
n_critic=5
nu = 0.3 


##initilaisation
#mu = torch.tensor(np.random.uniform(low=0.1, high=0.7, size=1)[0], requires_grad=True)
#alpha =  torch.tensor(np.random.uniform(low=0.2, high=0.7, size=1)[0], requires_grad=True)
#beta = torch.tensor(np.random.uniform(low=0.2, high=1, size=1)[0], requires_grad=True)

## let us initialise from true values
mu = torch.tensor(a, requires_grad=True)
alpha =  torch.tensor(b, requires_grad=True)
beta = torch.tensor(c, requires_grad=True)
sigma = torch.tensor(d,  requires_grad=True)

print("Training loop starts")
print("Initial mu is ", mu.item())
print("Initial alpha is ", alpha.item())
print("Initial beta is ", beta.item())



k=128
A = torch.randn(size=(k,1))
b = torch.randn(size=(1,))
lstm = torch.nn.LSTM(3, k, 3)

# Define the parameters
D_params = list(lstm.parameters()) + [A, b]
G_params = [mu, alpha, beta, sigma]

# Define different learning rates
lr_D = 5e-4
lr_G = {'mu': 1e-4, 'alpha': 1e-4, 'beta': 1e-4, 'sigma': 1e-4}

# Define the optimizers with different learning rates for G_params
optim_D = optim.Adam(D_params, lr=lr_D, betas=(0.5, 0.9))
optim_G = optim.Adam([
    {'params': mu, 'lr': lr_G['mu']},
    {'params': alpha, 'lr': lr_G['alpha']},
    {'params': beta, 'lr': lr_G['beta']},
    {'params': sigma, 'lr': lr_G['sigma']}
], betas=(0.5, 0.9))

## loss lists
Wass_distance_list=[]
D_loss_list = []
G_loss_list=[]
param_list = []



#training loop for WGAN
#torch.autograd.set_detect_anomaly(True)
st = time.time()
for _ in range(num_epochs):
    #zeta = ST_hawkes_parallel(N, mu, alpha, beta, sigma, L)
    zeta_prime = generate_FAKE_crimes_bogota(N, mu, alpha, beta, sigma, L)
    for i in range(n_critic):
        ## maximise l =  E(D(real))-E(D(G(z))) + penalty terms
        #loss_D = discriminator_loss()
        D_GP = discriminator_loss_GP(zeta_prime)
        loss_D = D_GP[0]
        Estimated_Wass = D_GP[1].item()
        # optimize LSTM weights A,b, etc.
        optim_D.zero_grad()
        #loss_D.backward()
        loss_D.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(D_params, max_norm=10.0, norm_type=2.0, error_if_nonfinite=False)
        #torch.nn.utils.clip_grad_norm_(D_params, max_norm=5.0, norm_type=2.0, error_if_nonfinite=False)
        optim_D.step()
        
        #torch.nn.utils.clip_grad_norm_(D_params, max_norm=10.0, norm_type=2.0, error_if_nonfinite=False)

    #put the loss/distances in list for plotting later  
    D_loss_list.append(loss_D.item())
    Wass_distance_list.append(Estimated_Wass)
    #### n_critic loop ends, time to update theta = (mu,alpha,beta)

    # train G now
    loss_G =generator_loss()
    G_loss_list.append(loss_G.item())
    #optimize
    optim_G.zero_grad()
    loss_G.backward()
    optim_G.step()




    for param in G_params:
        param_list.append(param.item())

    et = time.time()
    if _%10==0:
        print("\n")
        print("Epoch "+ str(_)+" is completed")
        print("Time spent till epoch", str(_), " is ", (et-st)/60, " minutes")
        print("Discriminator loss is ", str(loss_D.item()))
        print("Generator loss is ", str(loss_G.item()))
        print("Wasserstein distance is ", str(Estimated_Wass))
        print("mu is", round(mu.item(),4))
        print("alpha is", round(alpha.item(),4))
        print("beta is", round(beta.item(),4))
        print("sigma is", round(sigma.item(),4))
        #add code to flushout
        sys.stdout.flush()  # Explicitly flushing the output
        





# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 4))

# Plot Discriminator loss
ax.plot(np.arange(0, len(D_loss_list)), 
        np.array(D_loss_list),
        color='red', label='Discriminator loss')

# Plot Generator loss
ax.plot(np.arange(0, len(G_loss_list[:250])), 
        np.array(G_loss_list[:250]),
        color='blue', label='Generator loss')

# Set plot title and axes labels
ax.set(title='Discriminator and Generator Loss',
       xlabel='Iteration',
       ylabel='Loss')

# Add legend
ax.legend()

# Display the plot
plt.show()
# Save the plot with a filename that includes the parameters
plot_filename = f'loss_plot_bogota_a_{a_copy}_b_{b_copy}_c_{c_copy}_d_{d_copy}.png'
plt.savefig(plot_filename)

# Display the plot
plt.show()



# Save D_loss_list
with open(f'NEW_May_D_loss_list_bogota_a_{a_copy}_b_{b_copy}_c_{c_copy}_d_{d_copy}.pkl', 'wb') as f:
    pickle.dump(D_loss_list, f)

# Save G_loss_list
with open(f'NEW_May_G_loss_list_bogota_a_{a_copy}_b_{b_copy}_c_{c_copy}_d_{d_copy}.pkl', 'wb') as f:
    pickle.dump(G_loss_list, f)



#save Hawkes process parameter estimates across iterations 
alpha_list=[]
beta_list = []
mu_list =[]
sigma_list = []

for _ in range(len(param_list)):
    if _%4==0:
        mu_list.append(param_list[_])
    if _%4==1:
        alpha_list.append(param_list[_])
    if _%4 ==2:
        beta_list.append(param_list[_])
    if _%4==3:
        sigma_list.append(param_list[_])



# Save alpha_list
with open(f'alpha_list_bogota_a_{a_copy}_b_{b_copy}_c_{c_copy}_d_{d_copy}.pkl', 'wb') as f:
    pickle.dump(alpha_list, f)

# Save beta_list
with open(f'beta_list_bogota_a_{a_copy}_b_{b_copy}_c_{c_copy}_d_{d_copy}.pkl', 'wb') as f:
    pickle.dump(beta_list, f)

# Save mu_list
with open(f'mu_list_bogota_a_{a_copy}_b_{b_copy}_c_{c_copy}_d_{d_copy}.pkl', 'wb') as f:
    pickle.dump(mu_list, f)

# Save sigma_list
with open(f'sigma_list_bogota_a_{a_copy}_b_{b_copy}_c_{c_copy}_d_{d_copy}.pkl', 'wb') as f:
    pickle.dump(sigma_list, f)



#the final parameters 
G_params







