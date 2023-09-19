'''
This file contains all functions used in the slicer sampler method 
'''
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import metrics
from sklearn.cluster import KMeans
import pandas as pd
import collections
import numpy as np
import scipy.stats as stats
import math

# Initialisation 
def initialise_random(X, K):    
    '''
    Initalises variables required for the slice sampler algorithm
    --------------------------------------------------------------
    Input:
        X - numpy array(1,..,n): data matrix of y_i's 
        K - integer : maximum number of clusters 
        m_0 - numpy array(1,...,c): number of observations assigned to each cluster
        d_0 - numpy arrary(1,...n): cluster assignments for each observation

    Output:
        K - integer : maximum number of clusters 
        m_0 - numpy array(1,...,c): number of observations assigned to each cluster
        d_0 - numpy array(1,...n): cluster assignments for each observation
        cluster_counters - numpy array length 1: number of non-empty clusters
    '''    
    # Randomly assign each data point to ag cluster
    cluster_assignments = np.zeros(len(X), dtype=int)
    for i in range(0, len(X)):
        cluster_assignments[i] = np.random.randint(0, K) 
    
    d_0 = cluster_assignments

    # Count # of points assigned to each cluster
    m_0 = collections.Counter(d_0)
    for i in range(0,K):
        if i not in m_0.keys(): # even if cluster has zero entries make sure it's still included
            m_0[i] = 0
    
    # Initialise vector to store all count updates
    cluster_counts = np.array(len(np.unique(d_0))).reshape(1,)
    mu_vec = np.zeros((K, 2)) #initialise mean vector

    return(K, d_0, m_0, cluster_counts,mu_vec)

####################################################################################################################################################################
def outlier_region(X, far=1):
    ''' 
    Defines a cluster center for unhabited clusters squared (or different) distance far from the observations
    --------------------------------------------------------------
    Input:
        X - numpy arrary(1,..,n): data matrix of y_i's 
        far - optional, power of the distance to take (>1)

    Output:
        outlier_center - numpy array size 2 : a cluster center for unhabited clusters
    '''
    outlier_center0 = (abs(max(X[:,0])-min(X[:,0]))+1)**far
    outlier_center1 = (abs(max(X[:,1])-min(X[:,1]))+1)**far

    return np.array([outlier_center0,outlier_center1])

####################################################################################################################################################################

# Sample means - we need to store the means for use in step 4
def sample_means(X, K, m_0,d_0,var_y=1,outlier_center=[1,1]):
    '''
    Samples posterior means given the prior distribution and data 
    - we need to store the means for use in step 4
    --------------------------------------------------------------
    Input:
        X - numpy arrary(1,..,n): data matrix of y_i's 
        K - integer : number of clusters 
        m_0 - numpy array(1,...,c): number of observations assigned to each cluster
        d_0 - numpy arrary(1,...n): cluster assignments for each observation

    Output:
        mu_vec - numpy array (1,...,K) :  sample from posterior distribution of cluster centers mu_j
    '''
    # sigma = 1
    mu_vec = np.zeros((K, 2)) #initialise mean vector

    for k in range(0, K):
        m = m_0[k] #number of elements in cluster
        y_1 = X[np.where(d_0 == k)][:,0] #d_0 cluster assignment from previous iteration x axis 
        y_2 = X[np.where(d_0 == k)][:,1] #d_0 cluster assignment from previous iteration y axis 
        if len(y_1)==0:
            mu_vec[k, 0] = outlier_center[0]#np.random.normal(0, 1, 1)#100 #update mu x axis 
            mu_vec[k, 1] = outlier_center[1]#np.random.normal(0, 1, 1)#100 #update mu y axis


            # mu_vec[k, 0] = mu_j_1 #update mu x axis 
            # mu_vec[k, 1] = mu_j_2 #update mu y axis  
            pass
        else:
            mu_mean = (m/(m+var_y)) * np.mean(y_1)
            mu_var = 1 / (1 + m/var_y)
            mu_j_1 = np.random.normal(mu_mean, np.sqrt(mu_var), 1) #take sample from posterior x-axis
            mu_mean = (m/(m+var_y)) * np.mean(y_2) 
            mu_var = 1 / (1 + m/var_y)
            mu_j_2 = np.random.normal(mu_mean, np.sqrt(mu_var), 1) #take sample from posterior y-axis

            mu_vec[k, 0] = mu_j_1 #update mu x axis 
            mu_vec[k, 1] = mu_j_2 #update mu y axis 
    return(mu_vec)

####################################################################################################################################################################




# Sample the v_j 
def sample_v(K, m_0):
    '''
    Samples the probability v_j from a beta distribution with parameters alpha_j, beta_j
    is defined as a part of stick - breaking framework
    --------------------------------------------------------------
    Input:       
        K - integer : number of clusters 
        m_0 - numpy array(1,...,c): number of observations assigned to each cluster

    Output:
        w - numpy arrary(1,...,K): weight associated with cluster 
        v_vec - numpy array (1,...,K) :  sample from pbeta distribution with parameters alpha_j, beta_j
    '''
    v_vec = np.zeros(K) #initialise v vector
    
    for k in range(0, K): #for each cluster 
        a_j = 1 + m_0[k] #compute # of elelments in cluster 
        higher_order_k = sum([value for key, value in m_0.items() if key > k]) # compute # of elements in higher order clusters (remaining breaking sticks)
        b_j = 1 + higher_order_k 
        v_j = np.random.beta(a_j, b_j, 1) #sample from distribution
        v_vec[k] = v_j #set value 
    
    # Create w
    w = np.zeros(K) #initialise w vector
    w[0] = v_vec[0] #first entry of w is equal to first entry of v
# Compute re-weighting $w_j = v_j\prod_{j = 1}^{j-1}(1-v_j)$
    for k in range(1, K):        
        w[k] = v_vec[k] * np.prod(1 - v_vec[:k]) #normalise remaining element of w
    
    return(w, v_vec) #return w and v vectors

####################################################################################################################################################################



# sample u
def sample_u(X, d_0, w):
    '''
    BREAK A STICK - samples from a uniform variable from [0,w_i] i.e. the weight associated with the cluster containing y_i for all i to obtain u_i
    --------------------------------------------------------------
    Input:
        X - numpy arrary(1,..,n): data matrix of y_i's
        d_0 - numpy array(1,...,n) : cluster assignment for each y_i
        w - numpy array(1,...,K): weight associated with cluster 

    Output:
        u_vec - numpy array (1,...,n) :  sample from marginal distribution of u 
    '''
    u_vec = np.zeros(len(X)) #initialise u vector
    
    for i in range(0, len(X)):
        d_i = d_0[i] # find cluster associated with observation i 
        w_i = w[d_i] # find weight associated with this cluster
        u_i = np.random.uniform(0, w_i, 1) # sample uniformly from [0, w_i]
        u_vec[i] = u_i
    return(u_vec)

####################################################################################################################################################################



# Update probabilities
def compute_probs(X, K, u_vec, w, mu_vec,var_y = 1):
    '''
    Computes the probability that observation y_i belongs to cluster k with mean mu_k
    --------------------------------------------------------------
    Input:
        X - numpy arrary(1,..,n): data matrix of y_i's
        K - integer : number of clusters 
        u_vec - numpy array (1,...,n) :  sample from marginal distribution of u 
        w - numpy array(1,...,K): weight associated with cluster
        mu_vec - numpy array (1,...,K) :  sample from posterior distribution of cluster centers mu_j

    Output:
        prob_vec - numpy array (1,...,n)x(1,...,K) :  computed probabilities 
    '''
    prob_vec = np.zeros((len(X), K)) #initialise probability matrix
    tracker = 0
    for i in range(0, len(X)):
        for k in range(0, K):
            if w[k] > u_vec[i]:
                p_k = stats.multivariate_normal.pdf(X[i,:], mean=mu_vec[k], cov= var_y * np.identity(2)) 
                prob_vec[i, k] = p_k # will need to normalise probabilities
                tracker = k
            else:
                prob_vec[i, k] = 0

        if sum(prob_vec[i,:])==0:
            prob_vec[i,k]= prob_vec[i,tracker]+10**(-6)

    return(prob_vec)

####################################################################################################################################################################


# Normalise probabilities
def normalise_probs(X, prob_vec, K):
    '''
   NORMALIZES the probability that observation y_i belongs to cluster k with mean mu_k computed in the previous step
    --------------------------------------------------------------
    Input:
        X - numpy arrary(1,..,n): data matrix of y_i's
        prob_vec - numpy array(1,...,K): priors set 0's, updated iteratively
        K - integer : number of clusters 

    Output:
        prob_vec - numpy array (1,...,n)x(1,...,K) :  computed probabilities, NORMALISED
    '''    
    for i in range(0, len(X)):
        row_sum = sum(prob_vec[i,:])
        if row_sum==0:
            print(prob_vec[i,:])
            raise 'SUM is 0'
        for k in range(0, K):
            prob_vec[i,k] = (prob_vec[i,k] / row_sum)
    return(prob_vec)

####################################################################################################################################################################



def reassign_clusters(X,prob_vec,d_0):
    '''
    Reassigns y_i's as a sample from multinomial distribution with parameter prob_vec
    --------------------------------------------------------------
    Input:
        X - numpy arrary(1,..,n): data matrix of y_i's
        prob_vec - numpy array(1,...,K): priors set 0's, updated iteratively
        d_0 - numpy array(1,...,n) : cluster assignment for each y_i
    Output:
        d_0 - numpy array (1,...,n) :  updated cluster assignment for each y_i
    '''
    new_d = []
    for i in range(0, len(X)):
        new_d.append(np.random.multinomial(1, prob_vec[i], size=None))
    d_0 = np.nonzero(np.asarray(new_d))[1]
    return d_0

####################################################################################################################################################################



def update_cluster_counts(d_0, K):
    '''
    Updates record of the number of data points assigned to each cluster 
    --------------------------------------------------------------
     Input:
         d_0 - numpy array(1,...,n): cluster assignment for each y_i
         K - integer: number of clusters
     Output:
         m_0 - numpy array(1,...,c): number of observations assigned to each cluster 
    '''     
    m_0 = collections.Counter(d_0)
    for i in range(0,K):
        if i not in m_0.keys(): # even if cluster has zero entries make sure it's still included
            m_0[i] = 0
    return(m_0)

####################################################################################################################################################################




def slicey_time(X, K, max_iter = 100,var_y = 1,verbose=True):
    
    ### Initialisation steps
    K, d_0, m_0, cluster_counts, mu_vec = initialise_random(X, K)
    outlier_center = outlier_region(X)
    # Initialise vector to store all count updates
    cluster_counts = np.array(len(np.unique(d_0))).reshape(1,) 
    historical_labels = []
    all_cluster_centers = []
    ### Sampling 
    t = 0
    while t < max_iter:
        if verbose:
            print('iteration ' + str(t))

        # Step 1) update means
        mu_vec =  sample_means(X, K, m_0,d_0,var_y = var_y,outlier_center=outlier_center)
        # print(mu_vec)
        # Step 2) update v's an w's
        w, v_vec = sample_v(K, m_0)
         
        # Step 3) update u's
        u_vec = sample_u(X, d_0, w)

        # Step 4)
        # compute likelihoods
        prob_vec = compute_probs(X , K, u_vec, w, mu_vec,var_y = var_y)

        # normalise probabilites
        prob_vec = normalise_probs(X, prob_vec, K)

        # Step 5)
        # reassign clusters 
        d_0 = reassign_clusters(X,prob_vec,d_0)  


        # count number of active clusters
        n_active_cluster = np.array(len(np.unique(d_0))).reshape(1,) 
        if verbose:
            print('# clusters: ' + str(n_active_cluster))

        # update cluster counts
        m_0 = update_cluster_counts(d_0, K)

        # store update
        cluster_counts = np.concatenate((cluster_counts, n_active_cluster))
        historical_labels.append(d_0)
        
        all_cluster_centers.append(mu_vec)
        # increment counter
        t +=1   
        
    return(m_0, cluster_counts,d_0, historical_labels, mu_vec,all_cluster_centers)
