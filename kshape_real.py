import numpy as np
from scipy.spatial.distance import euclidean
#from fastdtw import fastdtw
from statistics import median
from scipy.stats import mode
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib.pyplot import MultipleLocator
from scipy import signal
from scipy.stats import rankdata
import random
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import mean_squared_error
#from sklearn.cluster import kmeans_plusplus
from sklearn.cluster import KMeans
from statistics import median
from scipy.stats import mode
from collections import defaultdict
import seaborn as sns
from sklearn.metrics.cluster import adjusted_rand_score
#import dcor
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import ccf
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel
from dtaidistance import dtw, clustering
from dtaidistance.subsequence.dtw import subsequence_alignment
from dtaidistance import dtw_visualisation as dtwvis
from tslearn.metrics import dtw_subsequence_path
import random
from sklearn.metrics.pairwise import euclidean_distances
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
import datetime
from matplotlib.dates import YearLocator, DateFormatter
from scipy.stats import norm,kurtosis,skew
import pickle
import plotly.graph_objects as go
import time
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from numba import jit
import random
from scipy.cluster.hierarchy import fcluster


# The number of stocks is n
#n = 120 #90 #120
# The number of lags is m
m = 5
# The length of time series is T
#T = 500
# The value of sub time series shift is s
s = 1
# The length of sub time series is q
q = 10#91#11
l = 21#101#21
# The number of sub time series in each time series is h
h = int((l-q)/s) + 1
# The sd of noise
#sigma = 1
# The threshold of votes count
#vote_threshold = int(0.2*h)
w = 1
#cate_list = ["lag","lead"]
cate_list = ["lag"]
#spn_list = [7]
spn_list = [7,5,3,1]
# forward f
#f_list = [7]
f_list = [7,5,3,1]
# ratio list
#ratio_list = [0.15,0.2,0.25,0.3]
#ratio_list = [0.15,0.25]
ratio_list = [0.2]
vote_threshold = 6
#k_list = [5]
k_list = [5,10]





data_name = "stock_data_600"
plot_folder = "stock_plot"
result_folder = "stock_result"

#data_name = "etf_data"
#plot_folder = "etf_plot"
#result_folder = "etf_result"


#data_name = "pinnacle_data"
#plot_folder = "pinnacle_plot"
#result_folder = "pinnacle_result"



def gen_model(n, k, T, m, sigma):

    factor = np.random.normal(0, 1, size=(k, T))
#     factor = np.random.normal(0, 1, size=(1, T))
#     for k_index in range(1,k):
#         factor = np.append(factor, np.random.normal(k_index**3, k_index+1, size=(1, T)), axis=0)
    
    error = np.random.normal(0, sigma, size=(n,T)) 
    
    lag = np.zeros((n,k))
    beta = np.zeros((n,k))
    lag_index = []
    for k_index in range(0,k):
        for n_index in range(0,n,k):
            lag_index.append(n_index+k_index*n+k_index)
            
    i = 0 
    for item in lag_index:
        lag.ravel()[item] = int(i /(n/((m+1)*k)))  #np.random.randint(0, m+1)
        beta.ravel()[item] = 1 #random.uniform(0.8,1.2) #1
        i += 1
        if i == (n/k):
            i = 0
            
    signal_dict = {}
    for k_index in range(0,k):
        ts_list = []
        for n_index in range(0,int(n/k)):
            ts_list.append(n_index+k_index*int(n/k)+1)
        signal_dict[k_index] = ts_list
    return beta, factor, error, lag, signal_dict

def gen_returns(beta, factor, lag, error, error_add = True):
    k,T = factor.shape
    n,k = beta.shape
    returns_i = []
    for n_index in range(n):
        returns_it = []
        for t_index in range(T):
            returns_it_value = 0
            for k_index in range(k):
                returns_it_value += beta[n_index][k_index]*factor[k_index][min(max(t_index-int(lag[n_index][k_index]),0),T-1)]
            if error_add:
                returns_it_value += error[n_index][t_index]
            returns_it.append(returns_it_value)
        returns_i.append(returns_it)
    return np.array(returns_i)

def gen_ts_universe(returns,q,s):
    n,_ = returns.shape
    returns = returns.tolist()
    ts_universe = []
    for n_index in range(n):
        ts = returns[n_index]
        sub_ts = [ts[i:i+q] for i in range(0,len(ts)-q+s,s)]
        ts_universe = ts_universe + sub_ts
    return np.array(ts_universe)

def gen_asset_sub_ts_name(df,h,s,des=1):
    asset_sub_ts_name_list = []
    for asset_name in list(df):
        for h_index in range(0,h):
            if des == 0:
                asset_sub_ts_name = asset_name + " & start at "+str(h_index*s)
            if des == 1:
                asset_sub_ts_name = (asset_name, h_index*s)
            asset_sub_ts_name_list.append(asset_sub_ts_name)
    return asset_sub_ts_name_list

#def gen_distance_corr(ts_universe_df, asset_sub_ts_name):
#    col_list = list(ts_universe_df)
#    col_list_len = len(col_list)
#    diag_one_mat = np.diag(np.ones(col_list_len))
#    distance_corr_df = pd.DataFrame(diag_one_mat, index=asset_sub_ts_name, columns=asset_sub_ts_name)
#    for i in range(0,col_list_len):
#        for j in range(i+1,col_list_len):
#                distance_corr_df.iloc[i,j] = distance_corr_df.iloc[j,i] = dcor.distance_correlation(np.array(ts_universe_df.iloc[:,i]),np.array(ts_universe_df.iloc[:,j]))
#    return distance_corr_df

# def gen_heatmap(ts_universe, asset_sub_ts_name, method):
#     ts_universe_df = pd.DataFrame(ts_universe.transpose(),columns=asset_sub_ts_name)
#     if method == "pearson":
#         corr = ts_universe_df.corr(method='pearson')
#     elif method == "distance":
#         corr = gen_distance_corr(ts_universe_df,asset_sub_ts_name)
#     ax = sns.heatmap(
#         corr, 
#         vmin=-1, vmax=1, center=0,
#         cmap= plt.cm.bwr,
#     )
#     ax.set_xticklabels(
#         ax.get_xticklabels(),
#         rotation=60,
#         horizontalalignment='right'
#     )
    
    
#     fig = plt.gcf()
#     fig.set_size_inches(14,12)
#     plt.savefig('plots/'+method+'_heatmap.', dpi=1000)
#     plt.show()

def gen_corr_mat(ts_universe, method):
    if method == "pearson":
        corr = np.corrcoef(ts_universe)
    elif method == "distance":
        corr = distance_corr(ts_universe)
    return corr
    
def gen_eigen(corr_matrix):
    D = np.diag(abs(corr_matrix).sum(axis=1))
    L = D - corr_matrix
    vals, vecs = np.linalg.eig(L)
    vals = vals.real
    vecs = vecs.real
    
    vecs = vecs[:,np.argsort(vals)]
    vals = vals[np.argsort(vals)]
    return vecs, vals

def gen_cluster_dic(label,asset_sub_ts_name):
    cluster_dic={}
    for l_index in set(label):
        sub_ts_index = [i for i,x in enumerate(label) if x==l_index]
        sub_ts_value_list = [asset_sub_ts_name[j] for j in sub_ts_index]
        sub_dict = {}
        for value in sub_ts_value_list:
            sub_dict.setdefault(value[0], []).append(value[1])
        cluster_dic[l_index] = sub_dict
    return cluster_dic

def gen_cluster_delta_dic(cluster_dic):
    cluster_delta_dic = {}
    for key in cluster_dic.keys():
        vec = np.array([np.append(np.array(()), np.array(i)) for i in cluster_dic[key].values()],dtype=object)
        l = len(vec)
        delta_dic = {}
        keys = list(cluster_dic[key].keys())
        for i in range(0,l):
            for j in range(i+1,l):
                delta_dic[(keys[i],keys[j])] = [delta1_value - delta2_value for delta2_value in vec[j] for delta1_value in vec[i]]
        cluster_delta_dic[key] = delta_dic
    return cluster_delta_dic

def gen_vote_dic(cluster_delta_dic):
    vote_dic = defaultdict(list)
    for d in tuple(cluster_delta_dic.values()):
        for key, value in d.items():
            vote_dic[key].extend(value)
    return vote_dic

def gen_vote_count_result(vote_dic,n,threshold):
    vote_count_dic = {}
    vote_count_mat = np.zeros((n,n))
    for key, value in vote_dic.items():
        if len(value) >= threshold:
            vote_count_dic[key] = len(value)
#             (i,j) = key
#             vote_count_mat[i-1,j-1] = vote_count_mat[j-1,i-1] = len(value)
    return vote_count_dic#, vote_count_mat

# def gen_vote_count_matrix_plot(vote_count_mat, threshold):
#     n = len(vote_count_mat)
#     fig, ax = plt.subplots()
#     ax.matshow(vote_count_mat, cmap=plt.cm.tab20c)
#     for i in range(n):
#         for j in range(n):
#             ax.text(i, j, vote_count_mat[j,i], va='center', ha='center')
#     label = ["asset " + str(index) for index in list(range(0,n+1))]
#     ax.set_xticklabels(label)
#     ax.set_yticklabels(label)
#     plt.savefig('plots/vote_count_matrix_threshold_'+str(threshold)+'.pdf', dpi=1000)
#     plt.show()

def gen_vote_result(vote_dic,method,threshold,verbose = False):
    vote_result_dic = {}
    if method == "mode":
        if verbose == True:
            vote_result_dic = {key: mode(value) for key, value in vote_dic.items() if len(value) >= threshold}
        else:
            vote_result_dic = {key: mode(value)[0][0] for key, value in vote_dic.items() if len(value) >= threshold}
    if method == "median":
        vote_result_dic = {key: median(value) for key, value in vote_dic.items() if len(value) >= threshold}
    return vote_result_dic

def ccf_auc(returns,m):
    n, _ = returns.shape
    lead_lag_mat = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            I_ij = np.sum(np.abs(ccf(returns[i],returns[j])[1:m+1]))
            I_ji = np.sum(np.abs(ccf(returns[j],returns[i])[1:m+1]))
            S_ij = np.sign(I_ij-I_ji)*max(I_ij,I_ji)/(I_ij+I_ji)
            lead_lag_mat[i,j] = S_ij
            lead_lag_mat[j,i] = -S_ij
    lead_lag_mat[np.isnan(lead_lag_mat)] = 0
    return lead_lag_mat

def map_asset_to_number(asset_df):
    asset_name = list(asset_df.index)
    asset_dict = {}
    i = 0
    for asset in asset_name:
        asset_dict[asset] = i
        i +=1
    return asset_dict

def lead_lag_result(lead_lag_dict,asset_to_number_dict,n):
    lead_lag_mat = np.zeros((n,n))
    for key, value in lead_lag_dict.items():
        (i,j) = key
        lead_lag_mat[asset_to_number_dict[i],asset_to_number_dict[j]] = int(value)
        lead_lag_mat[asset_to_number_dict[j],asset_to_number_dict[i]] = -int(value)
    return lead_lag_mat

def ranking(lead_lag_matrix):
    row_sum = lead_lag_matrix.sum(axis=1)
    rank = [sorted(row_sum).index(x) for x in row_sum]
    return rank

def ranking_index(lead_lag_matrix, ratio):
    row_sum = lead_lag_matrix.sum(axis=1)
    index_rank = sorted(range(len(row_sum)), key=row_sum.__getitem__, reverse=True)
    lagger_list = index_rank[:int(len(index_rank)*ratio)]
    leadder_list = index_rank[int(len(index_rank)*(1-ratio)):]
    return lagger_list,leadder_list



##############################################
def gen_dtw_lag(path,method):
    diff_list = []
    for [map_x, map_y] in path:
        diff = map_x - map_y
        diff_list.append(diff)
    if method == "mode":
        lag = mode(diff_list)[0][0]
    if method == "median":
        lag = median(diff_list)
    return lag
    
def gen_dtw_matrix(cluster_idx, returns):
    n, _ = returns.shape
    #print(n)
    lead_lag_mat_mod = np.zeros((n,n))
    lead_lag_mat_med = np.zeros((n,n))

    for key, value in cluster_idx.items():
        for i in range(len(value)-1):
            for j in range(i+1,len(value)):

                path_matirx = dtw.warping_paths_fast(returns[list(value)[i]],returns[list(value)[j]],window = m+1)[1]
                best_path = dtw.best_path(path_matirx)
                dtw_lag_ij_mod = gen_dtw_lag(best_path,"mode")
                dtw_lag_ij_med = gen_dtw_lag(best_path,"median")
                lead_lag_mat_mod[list(value)[i]][list(value)[j]] = dtw_lag_ij_mod
                lead_lag_mat_med[list(value)[i]][list(value)[j]] = dtw_lag_ij_med

    lead_lag_mat_mod = lead_lag_mat_mod-lead_lag_mat_mod.T
    lead_lag_mat_med = lead_lag_mat_med-lead_lag_mat_med.T
    return lead_lag_mat_mod,lead_lag_mat_med


def gen_dtw_label(cluster_idx, n):
    label =  [0] * n
    for key, value in cluster_idx.items():
        for index in value:
            label[index] = key
    return label
    

    
def gen_dtw_true_label(n,k):
    true_label = []
    for i in range(k):
        for _ in range(int(n/k)):
            true_label.append(i)
    return true_label
    
###################################
def gen_kshape_ccf(s,t):
    cross_corr = np.correlate(s, t, mode='full')
    #positive_corr_indices = np.where(cross_corr > 0)[0]
    #max_corr_index = positive_corr_indices[np.argmax(cross_corr[positive_corr_indices])]
    #lag_at_max_corr = max_corr_index - len(s) + 1
    lag_at_max_corr = np.argmax(cross_corr) - len(s) + 1
    return lag_at_max_corr




def gen_kshape_matrix(kshape_label, kshape_center, scale_returns):
    (n,l,_) = scale_returns.shape
    scale_returns = scale_returns.flatten().reshape(n,l)
    lead_lag_mat = np.zeros((n,n))
    
    for i,label in enumerate(kshape_label):
        for j in range(i+1,len(kshape_label)):
            if kshape_label[j]==label:
                ref = kshape_center[label].flatten()
                lag_ir = gen_kshape_ccf(scale_returns[i],ref)
                lag_jr = gen_kshape_ccf(scale_returns[j],ref)
                lag_ij = lag_ir-lag_jr

                if lag_ij > m or lag_ij < -m:
                    lag_ij = 0
                lead_lag_mat[i,j] = lag_ij
                lead_lag_mat[j,i] = -lag_ij
    return lead_lag_mat
    
#def gen_kshape_matrix(kshape_label, kshape_center, scale_returns):
#    (n,l,_) = scale_returns.shape
#    scale_returns = scale_returns.flatten().reshape(n,l)
#    lead_lag_mat = np.zeros((n,n))
#    
#    for i,label in enumerate(kshape_label):
#        for j in range(i+1,len(kshape_label)):
#            if kshape_label[j]==label:
#                I_ij = np.sum(np.abs(ccf(scale_returns[i],scale_returns[j])[1:m+1]))
#                I_ji = np.sum(np.abs(ccf(scale_returns[j],scale_returns[i])[1:m+1]))
#                S_ij = np.sign(I_ij-I_ji)*max(I_ij,I_ji)/(I_ij+I_ji)
#                lead_lag_mat[i,j] = S_ij
#                lead_lag_mat[j,i] = -S_ij
#    return lead_lag_mat


    
def gen_daily_pnl(returns,lead_lag_matrix,spn,f,ratio,cate,i):
    lagger_index_list,leadder_index_list = ranking_index(lead_lag_matrix, ratio)
    returns_df = pd.DataFrame(returns)
    ewm_cols = returns_df.iloc[leadder_index_list, l+i*w-spn:l+i*w].T
    lead_signal = np.sign(np.mean(ewm_cols.ewm(span=spn, adjust=False).mean().iloc[-1]))
    ##ewm_cols = asset_df.iloc[leadder_index_list, i*w:l+i*w].T
    ##ewm_cols = ewm_cols.ewm(span=spn, adjust=False).mean()
    ##lead_signal = np.sign(np.mean(np.mean(ewm_cols.T.iloc[:, -spn:])))
    lagger_col = returns_df.iloc[lagger_index_list, l+i*w-1+f]
    leadder_col = returns_df.iloc[leadder_index_list, l+i*w-1+f]
    if cate == "lead":
        PnL = (np.mean(leadder_col))*lead_signal
    if cate == "lag":
        PnL = (np.mean(lagger_col))*lead_signal
    return PnL


    
def robust_lead_lag_matrix(asset_sub_ts_name,asset_to_number_dict,sub_returns,cluster,method):

    n, _ = sub_returns.shape
    ts_universe = gen_ts_universe(sub_returns,q,s)
    #asset_sub_ts_name = gen_asset_sub_ts_name(returns_df.T, h,s)
    corr_matrix = gen_corr_mat(ts_universe,"pearson")
    
    A = ts_universe
    S = rbf_kernel(A)
    nbrs = NearestNeighbors(n_neighbors=3).fit(A)
    sparse_matrix = nbrs.kneighbors_graph(A).toarray()
    SS = S*sparse_matrix    
    
    if cluster == "K-means++":
        kmean = KMeans(n_clusters=h,random_state=42).fit(A)
        label = kmean.labels_.tolist()

    if cluster == "Spectral":
    
        try:
            spectral = SpectralClustering(n_clusters=h, assign_labels='discretize',random_state=42).fit(SS) #corr_matrix #SS
        except:
            spectral = SpectralClustering(n_clusters=h, assign_labels='discretize',random_state=42).fit(S)
        label = spectral.labels_

    cluster_dic = gen_cluster_dic(label, asset_sub_ts_name)
    cluster_delta_dic = gen_cluster_delta_dic(cluster_dic)
    vote_dic = gen_vote_dic(cluster_delta_dic)
    
    
    vote_result = gen_vote_result(vote_dic, method, vote_threshold)
    lead_lag_matrix = lead_lag_result(vote_result, asset_to_number_dict, n)
    return lead_lag_matrix
    
def gen_hierarchy_cluster_idx(linkage, k):

    clusters = fcluster(linkage, t=k, criterion='maxclust')

    # Create a dictionary to store the cluster numbers and their corresponding data points
    cluster_dict = {}
    for idx, cluster_num in enumerate(clusters):
        if cluster_num not in cluster_dict:
            cluster_dict[cluster_num] = set()
        cluster_dict[cluster_num].add(idx)

    return cluster_dict
    
def gen_robust_lead_lag_parallel(returns,asset_sub_ts_name,asset_to_number_dict,i):


    #print(i)
    date = returns.columns[l+i*w]
    returns = np.array(returns)
    sub_returns = returns[:, 0 + i * w: l + i * w]
    

    #CCF
    ccf_matrix = ccf_auc(sub_returns,m)
    #Robust
    kmeans_mode_matrix = robust_lead_lag_matrix(asset_sub_ts_name,asset_to_number_dict,sub_returns,"K-means++","mode")
    kmeans_median_matrix = robust_lead_lag_matrix(asset_sub_ts_name,asset_to_number_dict,sub_returns,"K-means++","median")
    spectral_mode_matrix = robust_lead_lag_matrix(asset_sub_ts_name,asset_to_number_dict,sub_returns,"Spectral","mode")
    spectral_median_matrix = robust_lead_lag_matrix(asset_sub_ts_name,asset_to_number_dict,sub_returns,"Spectral","median")

    np.save(result_folder + "/CCF_"+ str(date) + ".npy", ccf_matrix)
    np.save(result_folder + "/KM_Mod_"+ str(date) + ".npy", kmeans_mode_matrix)
    np.save(result_folder + "/KM_Med_"+ str(date) + ".npy", kmeans_median_matrix)
    np.save(result_folder + "/SP_Mod_"+ str(date) + ".npy", spectral_mode_matrix)
    np.save(result_folder + "/SP_Med_"+ str(date) + ".npy", spectral_median_matrix)
    
def gen_ccf_lead_lag_parallel(returns,asset_sub_ts_name,asset_to_number_dict,i):


    #print(i)
    date = returns.columns[l+i*w]
    returns = np.array(returns)
    sub_returns = returns[:, 0 + i * w: l + i * w]
    

    #CCF
    ccf_matrix = ccf_auc(sub_returns,m)

    np.save(result_folder + "/CCF_"+ str(date) + ".npy", ccf_matrix)
    


def gen_dtw_lead_lag_parallel(returns,asset_sub_ts_name,asset_to_number_dict,k_list,i):


    #print(i)
    date = returns.columns[l+i*w]
    returns = np.array(returns)
    sub_returns = returns[:, 0 + i * w: l + i * w]

    #DTW
    for k in k_list:
        random.seed(42)
        kmedoids = clustering.KMedoids(dtw.distance_matrix_fast, {"window": m+1, "only_triu":True}, k)
        cluster_idx = kmedoids.fit(sub_returns)
        #kmedoids_label = gen_dtw_label(cluster_idx, n)
        dtw_mod_matrix,dtw_med_matrix = gen_dtw_matrix(cluster_idx, sub_returns)
        
        np.save(result_folder + "/DTW_Mod_k_"+str(k)+"_"+ str(date) + ".npy", dtw_mod_matrix)
        np.save(result_folder + "/DTW_Med_k_"+str(k)+"_"+ str(date) + ".npy", dtw_med_matrix)
        
        #DBA
        dba = clustering.KMeans(k=k, max_it=10, max_dba_it=10, dists_options={"window": m+1})
        dba_cluster_idx, _ = dba.fit(sub_returns, use_c=False, use_parallel=False)
        dba_mod_matrix,dba_med_matrix = gen_dtw_matrix(dba_cluster_idx, sub_returns)
        
        np.save(result_folder + "/DBA_Mod_k_"+str(k)+"_"+ str(date) + ".npy", dba_mod_matrix)
        np.save(result_folder + "/DBA_Med_k_"+str(k)+"_"+ str(date) + ".npy", dba_med_matrix)
        
        #hierarchy
        hierarchy = clustering.LinkageTree(dtw.distance_matrix_fast, {"window": m+1})
        linkage = hierarchy.fit(sub_returns)
        hierarchy_cluster_idx = gen_hierarchy_cluster_idx(linkage, k)
        hierarchy_mod_matrix,hierarchy_med_matrix = gen_dtw_matrix(hierarchy_cluster_idx, sub_returns)
        
        np.save(result_folder + "/Hierarchy_Mod_k_"+str(k)+"_"+ str(date) + ".npy", hierarchy_mod_matrix)
        np.save(result_folder + "/Hierarchy_Med_k_"+str(k)+"_"+ str(date) + ".npy", hierarchy_med_matrix)
        

def gen_kshape_lead_lag_parallel(returns,asset_sub_ts_name,asset_to_number_dict,k_list,i):


    print(i)
    date = returns.columns[l+i*w]
    returns = np.array(returns)
    sub_returns = returns[:, 0 + i * w: l + i * w]

    #kshape
    for k in k_list:
        random.seed(42)
        scale_sub_returns = TimeSeriesScalerMeanVariance(mu=0, std=1).fit_transform(sub_returns)
        kshape = KShape(n_clusters=k, n_init=10, random_state=42).fit(scale_sub_returns)
        kshape_label = list(kshape.labels_)
        kshape_center = kshape.cluster_centers_
        kshape_matrix = gen_kshape_matrix(kshape_label, kshape_center, scale_sub_returns)
        np.save(result_folder + "/KShape_k_"+str(k)+"_"+ str(date) + ".npy", kshape_matrix)





def gen_dtw_PnL_parallel(returns,asset_sub_ts_name,asset_to_number_dict,spn,f,ratio,cate,k,i):

    date = returns.columns[l+i*w]
    
    #DTW
    dtw_mod_matrix = np.load(result_folder + "/DTW_Mod_k_"+str(k)+"_"+ str(date) + ".npy")
    dtw_med_matrix = np.load(result_folder + "/DTW_Med_k_"+str(k)+"_"+ str(date) + ".npy")

    dtw_mod_pnl = gen_daily_pnl(returns,dtw_mod_matrix,spn,f,ratio,cate,i)
    dtw_med_pnl = gen_daily_pnl(returns,dtw_med_matrix,spn,f,ratio,cate,i)
    
    
    #DBA
    dba_mod_matrix = np.load(result_folder + "/DBA_Mod_k_"+str(k)+"_"+ str(date) + ".npy")
    dba_med_matrix = np.load(result_folder + "/DBA_Med_k_"+str(k)+"_"+ str(date) + ".npy")
    
    dba_mod_pnl = gen_daily_pnl(returns,dba_mod_matrix,spn,f,ratio,cate,i)
    dba_med_pnl = gen_daily_pnl(returns,dba_med_matrix,spn,f,ratio,cate,i)
    
    
    #hierarchy
    hierarchy_mod_matrix = np.load(result_folder + "/Hierarchy_Mod_k_"+str(k)+"_"+ str(date) + ".npy")
    hierarchy_med_matrix = np.load(result_folder + "/Hierarchy_Med_k_"+str(k)+"_"+ str(date) + ".npy")
    
    hierarchy_mod_pnl = gen_daily_pnl(returns,hierarchy_mod_matrix,spn,f,ratio,cate,i)
    hierarchy_med_pnl = gen_daily_pnl(returns,hierarchy_med_matrix,spn,f,ratio,cate,i)

    return (dtw_mod_pnl,dtw_med_pnl,dba_mod_pnl,dba_med_pnl,hierarchy_mod_pnl,hierarchy_med_pnl)



def gen_kshape_PnL_parallel(returns,asset_sub_ts_name,asset_to_number_dict,spn,f,ratio,cate,k,i):

    date = returns.columns[l+i*w]
    
        
    kshape_matrix = np.load(result_folder + "/KShape_k_"+str(k)+"_"+ str(date) + ".npy")
    kshape_pnl = gen_daily_pnl(returns,kshape_matrix,spn,f,ratio,cate,i)

    return kshape_pnl


def gen_robust_PnL_parallel(returns,asset_sub_ts_name,asset_to_number_dict,spn,f,ratio,cate,i):

    date = returns.columns[l+i*w]
    
    
    kmeans_mode_matrix = np.load(result_folder + "/KM_Mod_"+ str(date) + ".npy")
    kmeans_median_matrix = np.load(result_folder + "/KM_Med_"+ str(date) + ".npy")
    spectral_mode_matrix = np.load(result_folder + "/SP_Mod_"+ str(date) + ".npy")
    spectral_median_matrix = np.load(result_folder + "/SP_Med_"+ str(date) + ".npy")
    
    #PnL

    kmeans_mode_pnl = gen_daily_pnl(returns,kmeans_mode_matrix,spn,f,ratio,cate,i)
    kmeans_median_pnl = gen_daily_pnl(returns,kmeans_median_matrix,spn,f,ratio,cate,i)
    spectral_mode_pnl = gen_daily_pnl(returns,spectral_mode_matrix,spn,f,ratio,cate,i)
    spectral_median_pnl = gen_daily_pnl(returns,spectral_median_matrix,spn,f,ratio,cate,i)

    return (date,kmeans_mode_pnl,kmeans_median_pnl,spectral_mode_pnl,spectral_median_pnl)
    
def gen_ccf_PnL_parallel(returns,asset_sub_ts_name,asset_to_number_dict,spn,f,ratio,cate,i):

    date = returns.columns[l+i*w]
    
    
    ccf_matrix = np.load(result_folder + "/CCF_"+ str(date) + ".npy")
    
    
    #PnL

    ccf_pnl = gen_daily_pnl(returns,ccf_matrix,spn,f,ratio,cate,i)


    return (date,ccf_pnl)
    
def gen_metric(PnL,spn,f,ratio,cate,K,estimation):
     
    #Vol_adjusted
    PnL = np.array(PnL)
    
    
    target_vol = 0.15
    vol_scaling = target_vol / (np.std(PnL) * np.sqrt(252))
    PnL = vol_scaling * PnL
    
    
    
    T = len(PnL)
    #metric
    #E(mean)
    mean_return = np.mean(PnL) * 252
    #Vol
    vol_annual = np.std(PnL) * np.sqrt(252)
    #Sharpe
    SR2 = round(mean_return/vol_annual,2)
    SR3 = round(mean_return/vol_annual,3)
    #PPT
    PPT = round(np.mean(PnL) * 10000,3)
    #P_value
    SR_nonannual = np.mean(PnL)/np.std(PnL)
    DSR = norm.cdf(SR_nonannual / np.sqrt((1-skew(PnL)*SR_nonannual+(kurtosis(PnL)-1)*np.square(SR_nonannual)/4)/(T-1)))
    p_value = round(min(DSR,1-DSR)*2,3)
    #Sortino
    neg_returns = PnL[PnL < 0]
    neg_std_dev = neg_returns.std() * np.sqrt(252)
    sortino_ratio = round(mean_return / neg_std_dev,3)
    #MDD    
    comp_ret = (pd.Series(PnL)+1).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    drawdown = (comp_ret/peak)-1
    max_drawdown = round(drawdown.min(),3)
    #Calmar
    calmar_ratio = round(mean_return / np.abs(max_drawdown),3)
    #Hit
    hit_rate = round(len(PnL[PnL > 0]) / T,3)
    #Avg profit / Avg loss
    avg_profit = PnL[PnL > 0].mean()
    avg_loss = PnL[PnL < 0].mean()
    avg_profit_loss = round(abs(avg_profit / avg_loss),3)
    #Downside Deviation
    neg_std_dev = round(neg_std_dev,3)
    
    mean_return = round(mean_return,3)
    cum_PnL = 100*(pd.Series(PnL).cumsum())
    
    if cate == "lead":
        results = pd.Series({
            'Estimation': estimation, 
            'Method': r"$D_\alpha$", 
            'P': spn, 
            "Delta":f, 
            'Alpha':1-ratio,
            'Beta':ratio,
            'K': str(K),
            'E[Return]': mean_return,
            'Vol': vol_annual,
            'Downside Deviation': neg_std_dev,
            'Max. Drawdown': max_drawdown,
            'Sortino': sortino_ratio,
            'Calmar': calmar_ratio,
            'Hit Rate': hit_rate,
            'Avg. Profit / Avg. Loss': avg_profit_loss,
            'PPT': PPT,
            'Sharpe': SR3,
            'P-value':p_value
        })
        
    if cate == "lag":
        results = pd.Series({
            'Estimation': estimation, 
            'Method': r"$G_\beta$", 
            'P': spn, 
            "Delta":f, 
            'Alpha':1-ratio,
            'Beta':ratio,
            'K': str(K),
            'E[Return]': mean_return,
            'Vol': vol_annual,
            'Downside Deviation': neg_std_dev,
            'Max. Drawdown': max_drawdown,
            'Sortino': sortino_ratio,
            'Calmar': calmar_ratio,
            'Hit Rate': hit_rate,
            'Avg. Profit / Avg. Loss': avg_profit_loss,
            'PPT': PPT,
            'Sharpe': SR3,
            'P-value':p_value
        })
    
    return [SR2, cum_PnL, results]
    
    
    
    
#def gen_cum_PnL_plot(date_list,cum_pnl_dict,spn,f,ratio,cate,k,plot_folder):

 #   SMALL_SIZE = 8
  #  MEDIUM_SIZE = 16
   # BIGGER_SIZE = 30

    #plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    #plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    #plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    #plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    #plt.rc('legend', fontsize=10)    # legend fontsize
    #plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    

    #if data_name == "pinnacle_data":
     #   dates = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in date_list]
    #else:
     #   dates = [datetime.datetime.strptime(date, '%Y%m%d').date() for date in date_list]
    ##dates = list(range(1, 1+len(cum_pnl_dict["dtw_mod1"][1])))
    
    #fig = plt.figure()
    
    #plt.plot(dates,cum_pnl_dict["dtw_mod"][1],"-b",linestyle="solid", label= "DTW_Mod (SR = "+str(cum_pnl_dict["dtw_mod"][0])+")")
    #plt.plot(dates,cum_pnl_dict["dtw_med"][1],"-b",linestyle="dotted", label= "DTW_Med (SR = "+str(cum_pnl_dict["dtw_med"][0])+")")
    #plt.plot(dates,cum_pnl_dict["ccf"][1],"-r", linestyle="solid",label= "CCF (SR = "+str(cum_pnl_dict["ccf"][0])+")")
    #plt.plot(dates,cum_pnl_dict["kmeans_mode"][1],"-g",linestyle="solid", label= "KM_Mod (SR = "+str(cum_pnl_dict["kmeans_mode"][0])+")")
    #plt.plot(dates,cum_pnl_dict["kmeans_median"][1],"-g",linestyle="dotted", label= "KM_Med (SR = "+str(cum_pnl_dict["kmeans_median"][0])+")")
    #plt.plot(dates,cum_pnl_dict["spectral_mode"][1],"-g", linestyle="dashed",label= "SP_Mod (SR = "+str(cum_pnl_dict["spectral_mode"][0])+")")
    #plt.plot(dates,cum_pnl_dict["spectral_median"][1],"-g", linestyle="dashdot",label= "SP_Med (SR = "+str(cum_pnl_dict["spectral_median"][0])+")")  
    
    #plt.xlabel("Time (Year)")
    #plt.ylabel("Cumulative Returns (%)")
    #plt.legend(loc=4)
    #plt.gca().xaxis.set_major_locator(YearLocator(2))
    #plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))
    #plt.xticks(rotation=30)


    #plt.savefig(plot_folder+'/pnl_span_'+str(spn)+'_f_'+str(f)+'_ratio_'+str(ratio)+'_'+cate+'_k_'+str(k)+'.pdf', dpi=1000,bbox_inches="tight")
    ##plt.close()
    
    

    
def gen_cum_PnL_plot(cum_pnl_dict,spn,f,ratio,cate,k,plot_folder):
    
    
    if data_name == "pinnacle_data":
        dates = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in cum_pnl_dict["date"]]
    else:
        dates = [datetime.datetime.strptime(date, '%Y%m%d').date() for date in cum_pnl_dict["date"]]


    
    # Line chart 1
    trace1 = go.Scatter(x=dates, y=cum_pnl_dict["DTW_KMed_Mod"][1], mode='lines', name="DTW_KMed_Mod (SR = "+str(cum_pnl_dict["DTW_KMed_Mod"][0])+")",line=dict(color='blue', dash='solid'))
    
    # Line chart 2
    trace2 = go.Scatter(x=dates, y=cum_pnl_dict["DTW_KMed_Med"][1], mode='lines', name="DTW_KMed_Med (SR = "+str(cum_pnl_dict["DTW_KMed_Med"][0])+")",line=dict(color='blue', dash='dash'))
    
    # Line chart 3
    trace3 = go.Scatter(x=dates, y=cum_pnl_dict["DTW_BA_KM_Mod"][1], mode='lines', name="DTW_BA_KM_Mod (SR = "+str(cum_pnl_dict["DTW_BA_KM_Mod"][0])+")",line=dict(color='yellow', dash='solid'))
    
    # Line chart 4
    trace4 = go.Scatter(x=dates, y=cum_pnl_dict["DTW_BA_KM_Med"][1], mode='lines', name="DTW_BA_KM_Med (SR = "+str(cum_pnl_dict["DTW_BA_KM_Med"][0])+")",line=dict(color='yellow', dash='dash'))
    
    # Line chart 5
    trace5 = go.Scatter(x=dates, y=cum_pnl_dict["DTW_KH_Mod"][1], mode='lines', name="DTW_KH_Mod (SR = "+str(cum_pnl_dict["DTW_KH_Mod"][0])+")",line=dict(color='purple', dash='solid'))
    
    # Line chart 6
    trace6 = go.Scatter(x=dates, y=cum_pnl_dict["DTW_KH_Med"][1], mode='lines', name="DTW_KH_Med (SR = "+str(cum_pnl_dict["DTW_KH_Med"][0])+")",line=dict(color='purple', dash='dash'))
    
    # Line chart 7
    #trace7 = go.Scatter(x=dates, y=cum_pnl_dict["KM_Mod"][1], mode='lines', name="KM_Mod (SR = "+str(cum_pnl_dict["KM_Mod"][0])+")", line=dict(color='green', dash='solid'))
    
    # Line chart 8
    #trace8 = go.Scatter(x=dates, y=cum_pnl_dict["KM_Med"][1], mode='lines', name="KM_Med (SR = "+str(cum_pnl_dict["KM_Med"][0])+")",line=dict(color='green', dash='dash'))
    
    # Line chart 9
    #trace9 = go.Scatter(x=dates, y=cum_pnl_dict["SP_Mod"][1], mode='lines', name="SP_Mod (SR = "+str(cum_pnl_dict["SP_Mod"][0])+")",line=dict(color='green', dash='dot'))
    
    # Line chart 10
    #trace10 = go.Scatter(x=dates, y=cum_pnl_dict["SP_Med"][1], mode='lines', name="SP_Med (SR = "+str(cum_pnl_dict["SP_Med"][0])+")",line=dict(color='green', dash='dashdot'))
    
    # Line chart 11
    trace11 = go.Scatter(x=dates, y=cum_pnl_dict["CCF"][1], mode='lines', name="CCF (SR = "+str(cum_pnl_dict["CCF"][0])+")",line=dict(color='red', dash='solid') )
    
    # Line chart 12
    trace12 = go.Scatter(x=dates, y=cum_pnl_dict["KS"][1], mode='lines', name="KS (SR = "+str(cum_pnl_dict["KS"][0])+")",line=dict(color='pink', dash='solid') )
    
    
    # Create a figure
    #fig = go.Figure(data=[trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11,trace12])
    fig = go.Figure(data=[trace1, trace2, trace3, trace4, trace5, trace6, trace11,trace12])
    
    


    
    fig.update_layout(
        xaxis_title="Time (Year)",
        yaxis_title="Cumulative Returns (%)",
        xaxis_title_font=dict(size=32),
        yaxis_title_font=dict(size=32),
        xaxis_tickangle=-30,
        legend=dict(
        x=1,#1
        xanchor='right',#right
        y=0,#0
        yanchor='bottom',#bottom
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(0, 0, 0, 0.5)',
        borderwidth=1,
        font=dict(
            size= 18
        )
        ),
        margin=dict(t=0, l=0, r=0, b=0, pad=0)
    )
    
    
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=dates[::252*2],  # Ticks every 2 years
            ticktext=[x.strftime('%Y') for x in dates[::252*2]]
        )
    )
    

    
    # Update the font size of x-axis tick labels
    fig.update_xaxes(tickfont=dict(size=28))
    
    # Update the font size of y-axis tick labels
    fig.update_yaxes(tickfont=dict(size=28))
    
    # Save the chart as an PDF file
    #fig.show()
    fig.write_image(plot_folder+'/pnl_span_'+str(spn)+'_f_'+str(f)+'_ratio_'+str(ratio)+'_'+cate+'_k_'+str(k)+'.pdf', format='pdf')
    time.sleep(2)
    fig.write_image(plot_folder+'/pnl_span_'+str(spn)+'_f_'+str(f)+'_ratio_'+str(ratio)+'_'+cate+'_k_'+str(k)+'.pdf', format='pdf') 
    
    
    
    
    
def main():
    print("Start!")
    random.seed(42)
    column_names = ['Estimation', 'Method', "P","Delta",'Alpha','Beta','K','E[Return]','Vol','Downside Deviation','Max. Drawdown','Sortino','Calmar','Hit Rate','Avg. Profit / Avg. Loss', "PPT","Sharpe","P-value"]
    
    kshape_metric = pd.DataFrame(columns=column_names, dtype=object)
    dtw_mod_metric = pd.DataFrame(columns=column_names, dtype=object)
    dtw_med_metric = pd.DataFrame(columns=column_names, dtype=object)
    dba_mod_metric = pd.DataFrame(columns=column_names, dtype=object)
    dba_med_metric = pd.DataFrame(columns=column_names, dtype=object)
    hierarchy_mod_metric = pd.DataFrame(columns=column_names, dtype=object)
    hierarchy_med_metric = pd.DataFrame(columns=column_names, dtype=object)
    
    ccf_metric = pd.DataFrame(columns=column_names, dtype=object)
    #kmeans_mode_metric = pd.DataFrame(columns=column_names, dtype=object)
    #kmeans_median_metric = pd.DataFrame(columns=column_names, dtype=object)
    #spectral_mode_metric = pd.DataFrame(columns=column_names, dtype=object)
    #spectral_median_metric = pd.DataFrame(columns=column_names, dtype=object)

                              
    data = pd.read_csv(data_name+".csv", index_col=0)
    n,T = data.shape

         
    asset_sub_ts_name = gen_asset_sub_ts_name(data.T, h,s)
    asset_to_number_dict = map_asset_to_number(data)

    #Parallel(n_jobs=-1, verbose=0)(delayed(gen_robust_lead_lag_parallel)(data,asset_sub_ts_name,asset_to_number_dict,i) for i in range((T - 10 + 1 - l) // w))
    #Parallel(n_jobs=-1, verbose=0)(delayed(gen_ccf_lead_lag_parallel)(data,asset_sub_ts_name,asset_to_number_dict,i) for i in range((T - 10 + 1 - l) // w))
    
    #for k in k_list:
        #Parallel(n_jobs=-1, verbose=0)(delayed(gen_dtw_lead_lag_parallel)(data,asset_sub_ts_name,asset_to_number_dict,k_list,i) for i in range((T - 10 + 1 - l) // w))
        #Parallel(n_jobs=-1, verbose=0)(delayed(gen_kshape_lead_lag_parallel)(data,asset_sub_ts_name,asset_to_number_dict,k_list,i) for i in range((T - 10 + 1 - l) // w))

    
    for ratio in ratio_list:
          for spn in spn_list:
              for f in f_list:             
                  for cate in cate_list:

                      #robust_result = Parallel(n_jobs=-1)(delayed(gen_robust_PnL_parallel)(data,asset_sub_ts_name,asset_to_number_dict,spn,f,ratio,cate,i) for i in range((T - 10 + 1 - l) // w))
                      #date,kmeans_mode_pnl,kmeans_median_pnl,spectral_mode_pnl,spectral_median_pnl = zip(*robust_result)
                      
                      ccf_result = Parallel(n_jobs=-1)(delayed(gen_ccf_PnL_parallel)(data,asset_sub_ts_name,asset_to_number_dict,spn,f,ratio,cate,i) for i in range((T - 10 + 1 - l) // w))
                      date,ccf_pnl = zip(*ccf_result)
                      
                      for k in k_list:
                          dtw_result = Parallel(n_jobs=-1)(delayed(gen_dtw_PnL_parallel)(data,asset_sub_ts_name,asset_to_number_dict,spn,f,ratio,cate,k,i) for i in range((T - 10 + 1 - l) // w))
                          dtw_mod_pnl,dtw_med_pnl,dba_mod_pnl,dba_med_pnl,hierarchy_mod_pnl,hierarchy_med_pnl = zip(*dtw_result)
                          kshape_PnL = Parallel(n_jobs=-1)(delayed(gen_kshape_PnL_parallel)(data,asset_sub_ts_name,asset_to_number_dict,spn,f,ratio,cate,k,i) for i in range((T - 10 + 1 - l) // w))
                          
                          cum_pnl_dict = {}
                          cum_pnl_dict["KS"] = gen_metric(kshape_PnL,spn,f,ratio,cate, k,"KS")
                          
                          cum_pnl_dict["DTW_KMed_Mod"] = gen_metric(dtw_mod_pnl,spn,f,ratio,cate,k, "DTW_KMed_Mod")
                          cum_pnl_dict["DTW_KMed_Med"] = gen_metric(dtw_med_pnl,spn,f,ratio,cate, k,"DTW_KMed_Med")
                          cum_pnl_dict["DTW_BA_KM_Mod"] = gen_metric(dba_mod_pnl,spn,f,ratio,cate,k, "DTW_BA_KM_Mod")
                          cum_pnl_dict["DTW_BA_KM_Med"] = gen_metric(dba_med_pnl,spn,f,ratio,cate, k,"DTW_BA_KM_Med")
                          cum_pnl_dict["DTW_KH_Mod"] = gen_metric(hierarchy_mod_pnl,spn,f,ratio,cate,k, "DTW_KH_Mod")
                          cum_pnl_dict["DTW_KH_Med"] = gen_metric(hierarchy_med_pnl,spn,f,ratio,cate, k,"DTW_KH_Med")
     
                          cum_pnl_dict["date"] = date
                          cum_pnl_dict["CCF"] = gen_metric(ccf_pnl,spn,f,ratio,cate, "NaN","CCF")
                          #cum_pnl_dict["KM_Mod"] = gen_metric(kmeans_mode_pnl,spn,f,ratio,cate, h,"KM_Mod")
                          #cum_pnl_dict["KM_Med"] = gen_metric(kmeans_median_pnl,spn,f,ratio,cate,h, "KM_Med")
                          #cum_pnl_dict["SP_Mod"] = gen_metric(spectral_mode_pnl,spn,f,ratio,cate, h,"SP_Mod")
                          #cum_pnl_dict["SP_Med"] = gen_metric(spectral_median_pnl,spn,f,ratio,cate, h,"SP_Med")

                      
                          with open(result_folder+'/dict_span_'+str(spn)+'_f_'+str(f)+'_ratio_'+str(ratio)+'_'+cate+'_k_'+str(k)+'.pkl', "wb") as file:
                              pickle.dump(cum_pnl_dict, file)
                              
                          
                          with open(result_folder+'/dict_span_'+str(spn)+'_f_'+str(f)+'_ratio_'+str(ratio)+'_'+cate+'_k_'+str(k)+'.pkl', "rb") as file:
                              cum_pnl_dict = pickle.load(file)
                          
                          
                          gen_cum_PnL_plot(cum_pnl_dict,spn,f,ratio,cate,k,plot_folder)
                          
                          dtw_mod_metric = dtw_mod_metric.append(cum_pnl_dict["DTW_KMed_Mod"][2], ignore_index=True)
                          dtw_med_metric = dtw_med_metric.append(cum_pnl_dict["DTW_KMed_Med"][2], ignore_index=True)
                          
                          dba_mod_metric = dba_mod_metric.append(cum_pnl_dict["DTW_BA_KM_Mod"][2], ignore_index=True)
                          dba_med_metric = dba_med_metric.append(cum_pnl_dict["DTW_BA_KM_Med"][2], ignore_index=True)
                          hierarchy_mod_metric = hierarchy_mod_metric.append(cum_pnl_dict["DTW_KH_Mod"][2], ignore_index=True)
                          hierarchy_med_metric = hierarchy_med_metric.append(cum_pnl_dict["DTW_KH_Med"][2], ignore_index=True)
                          
                          kshape_metric = kshape_metric.append(cum_pnl_dict["KS"][2], ignore_index=True)
                          
                      ccf_metric = ccf_metric.append(cum_pnl_dict["CCF"][2], ignore_index=True)
                      #kmeans_mode_metric = kmeans_mode_metric.append(cum_pnl_dict["KM_Mod"][2], ignore_index=True)
                      #kmeans_median_metric = kmeans_median_metric.append(cum_pnl_dict["KM_Med"][2], ignore_index=True)
                      #spectral_mode_metric = spectral_mode_metric.append(cum_pnl_dict["SP_Mod"][2], ignore_index=True)
                      #spectral_median_metric = spectral_median_metric.append(cum_pnl_dict["SP_Med"][2], ignore_index=True)

            
    #metric_table = pd.concat([kshape_metric,dtw_mod_metric, dtw_med_metric, dba_mod_metric,dba_med_metric,hierarchy_mod_metric,hierarchy_med_metric,ccf_metric, kmeans_mode_metric, kmeans_median_metric,spectral_mode_metric,spectral_median_metric], ignore_index=True)
    metric_table = pd.concat([kshape_metric,dtw_mod_metric, dtw_med_metric, dba_mod_metric,dba_med_metric,hierarchy_mod_metric,hierarchy_med_metric,ccf_metric], ignore_index=True)
    
    metric_table.to_csv(plot_folder+'/kshape_metric_result.csv',index=False)        
    print("Camellia!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl+C pressed. Stopping...")