################################################################################
# Module: methods.py
# Description: Compute all the metrics and derive distributions
# Ding Luo @ TU Delft, The Netherlands
################################################################################


from __future__ import division
import math
import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats

def compute_benchmark_metric(G_L,delta = 0.2):
    """
    Compute the benchmark travel impedance metric for all the stops
    
    Parameters
    ----------
    G_L : networkx graph object
        unweighted L-space graph for a PTN
    delta: fraction   
        A parameter determining the minimum percentage of the number of nodes 
        that should be connected to the rest of the network. If below this 
        minimum, a node is not considered a usable one in the following analysis.     

    Returns
    -------
    df : dataframe
    
    """ 
    result = list(nx.shortest_path_length(G_L))
    temp_dict = {}
    num_nodes = G_L.number_of_nodes()

    for x in result:
        try:
            if len(x[1]) > num_nodes * delta:
                total = sum(x[1].values())
                temp_dict[x[0]] = round(total/(len(x[1])-1),1)
            else:
                temp_dict[x[0]] = math.nan
        except:
            temp_dict[x[0]] = math.nan   
    
    x_list = list(nx.get_node_attributes(G_L,'x').values())
    y_list = list(nx.get_node_attributes(G_L,'y').values())   
    df = pd.DataFrame({'node_id':list(temp_dict.keys()),\
                       'x':x_list,\
                       'y':y_list,\
                       'hops':list(temp_dict.values())})                  
    return df

def compute_GTCbased_metric(G,transfer_penalty=300,delta = 0.2):
    '''
    Compute the average travel impedance associated with each stop in the public
    transport network. The travel impedance is based on the generalized travel
    cost (GTC) which includes initial and transfer waiting time, in-vehicle 
    times and time-equivalent transfer penalty time.
    
    Paramters:
    -------
    G: networkx graph object
        A weighted space-of-service graph (P-space)
    transfer_penalty: int 
        A constant indicating the time-equivalent transfer penalty cost. 
        The unit is second in this program
    delta: fraction   
        A parameter determining the minimum percentage of the number of nodes 
        that should be connected to the rest of the network. If below this 
        minimum, a node is not considered a usable one in the following analysis.         

    Returns
    -------
    df: dataframe
        
    '''  

    # shortest path 
    sp = nx.shortest_path(G,weight = 'total_travel_time')
    # create a dictionary for stop travel impedance values
    # The travel impedance is also decomposed
    # GTC: total generalized travel cost
    # IVT: in-vehicle travel time
    # NONIVT: the remaining part related to transfer and waiting times
    ti = {}
    for key in sp.keys():
        ti[key] = {}
        ti[key]['GTC'] = {}
        ti[key]['IVT'] = {}
        ti[key]['NONIVT'] = {}
    for source in sp.keys():
        for target in sp[source].keys():
            cur_sp = sp[source][target]
            ti[source]['GTC'][target] = 0
            ti[source]['IVT'][target] = 0
            ti[source]['NONIVT'][target] = 0
            if not len(cur_sp) == 1:
                # if not the node itself
                for k in range(len(cur_sp)-1):
                    i = cur_sp[k]
                    j = cur_sp[k+1]
                    ti[source]['IVT'][target] += G[i][j]['ivt']
                    ti[source]['NONIVT'][target] += G[i][j]['wt']
                ti[source]['NONIVT'][target] += (len(cur_sp)-2) * transfer_penalty
                ti[source]['GTC'][target] = ti[source]['IVT'][target] + ti[source]['NONIVT'][target]
                  
    GTC_list = []
    IVT_list = []
    NONIVT_list = []
    num_nodes = G.number_of_nodes()

    for x in ti.keys():
        try:
            if len(ti[x]['GTC']) > num_nodes * delta:
                tot_GTC = sum(ti[x]['GTC'].values())/60 # unit: minutes
                tot_IVT = sum(ti[x]['IVT'].values())/60 # unit: minutes
                tot_NONIVT = sum(ti[x]['NONIVT'].values())/60 # unit: minutes
                avg_CTC = round(tot_GTC/(len(ti[x]['GTC'])-1),1)
                avg_IVT = round(tot_IVT/(len(ti[x]['GTC'])-1),1)
                avg_NONIVT = round(tot_NONIVT/(len(ti[x]['GTC'])-1),1)
                GTC_list.append(avg_CTC)
                IVT_list.append(avg_IVT)
                NONIVT_list.append(avg_NONIVT)
            else:
                GTC_list.append(math.nan)
                IVT_list.append(math.nan)
                NONIVT_list.append(math.nan)
        except ZeroDivisionError:
            GTC_list.append(math.nan)
            IVT_list.append(math.nan)
            NONIVT_list.append(math.nan)
    
    x_list = list(nx.get_node_attributes(G,'x').values())
    y_list = list(nx.get_node_attributes(G,'y').values())  
    df = pd.DataFrame({'node_id':list(sp.keys()),'x':x_list,'y':y_list,\
                       'gtc':GTC_list,'ivt':IVT_list,'nonivt':NONIVT_list})               
    return df   

def compute_gap_between_metrics(df,x_clm_name,y_clm_name,new_clm_name = 'gap'):
    x = df[x_clm_name]
    y = df[y_clm_name]
    mask = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask],y[mask])
    residuals = y - (slope * x + intercept)
    df[new_clm_name] = residuals
    return df,r_value 

def derive_pdf_ccdf(data):
    '''
    derive the Prbability Density Function (PDF) and 
    Complementary Cumulative Distribution Function (CCDF)
    '''
    unique_counts = np.unique(data,return_counts = True)
    prob =[]
    cum_prob = []
    for x in range(len(unique_counts[1])):
        cur_prob = unique_counts[1][x] / sum(unique_counts[1])
        cur_sum_prob = sum(unique_counts[1][x:-1])/sum(unique_counts[1])
        prob.append(cur_prob)
        cum_prob.append(cur_sum_prob)
    
    data_dict = {}
    data_dict['variable'] = unique_counts[0]
    data_dict['frequency']  = unique_counts[1]
    data_dict['pdf']  = prob
    data_dict['ccdf'] = cum_prob
    return data_dict

