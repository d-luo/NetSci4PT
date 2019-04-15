# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:17:11 2019

@author: dingluo
"""
from __future__ import division
import math
import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns;
from network_loading import *
from sklearn.preprocessing import scale
from scipy import stats

def compute_benchmark_metric(G_L,min_connected_nodes_perc):
    """
    Compute the benchmark travel impedance metric for all the stops
    
    Parameters
    ----------
    G_L : networkx graph object
        unweighted L-space graph for a PTN

    min_connected_nodes_perc: float
        a parameter to determine the minimum percentage of the number of nodes 
        that should be connected to the rest of the network. When below this
        minimum, a node is not considered a usable one in the following analysis. 
    Returns
    -------
    result_dict : dict
        a dictionary contains the metric value for each stop
    
    """
    result = list(nx.shortest_path_length(G_L))
    farness_dict = {}
    max_shortest_path_length = 0
#    mean_shortest_path_length = round(nx.average_shortest_path_length(G),1)
    num_nodes = G_L.number_of_nodes()
    for x in result:
        max_shortest_path_length = max(max_shortest_path_length, max(x[1].values()))
        try:
            if len(x[1]) > num_nodes * min_connected_nodes_perc:
                total = sum(x[1].values())
                farness_dict[x[0]] = round(total/(len(x[1])-1),1)
            else:
                farness_dict[x[0]] = math.nan
        except:
            farness_dict[x[0]] = math.nan   
    mean_shortest_path_length = round(np.nanmean(list(farness_dict.values())),1)    
    
    x_list = list(nx.get_node_attributes(G_L,'x').values())
    y_list = list(nx.get_node_attributes(G_L,'y').values())   
    df = pd.DataFrame({'node_id':list(farness_dict.keys()),\
                       'x':x_list,\
                       'y':y_list,\
                       'values':list(farness_dict.values())})        
    result_dict = {}
    result_dict['max_shortest_path_length'] = max_shortest_path_length
    result_dict['mean_shortest_path_length'] = mean_shortest_path_length
    result_dict['df'] = df           
    return result_dict

def compute_GTCbased_metric(G,transfer_penalty):
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
        A constant indicating the transfer penalty time of which unit is seconds   
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
#            print(source,target)
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
            # The minimum percentage of the number of nodes that should be connected 
            # to the rest of the network. If below this minimum, a node is not considered 
            # a usable one in the following analysis. 
            min_connected_nodes_perc = 0.2
            if len(ti[x]['GTC']) > num_nodes * min_connected_nodes_perc:
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

