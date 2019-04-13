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

def compute_metric_benchmark(G_L,min_connected_nodes_perc):
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

def compute_metric_GTC(G,transfer_penalty_cost,min_connected_nodes_perc):
    '''
    G: Networkx DiGraph
    transfer_penalty_cost: the amount of time that should be reduced from the
    first leg of the trip since there should be no transfer penalty for it.
    min_connected_nodes_perc: The minimum percentage of the number of nodes that
    should be connected to the rest of the network. If below this minimum, a node
    is not considered a usable one in the following analysis.    
    '''
    result = list(nx.shortest_path_length(G,weight = 'total_travel_time'))
    # subtract the initial transfer penalty cost from each value
    for x in result:
        for key in x[1]:
            if not x[1][key] == 0:
                x[1][key] = x[1][key] - transfer_penalty_cost
   
    farness_dict = {}
    max_shortest_path_length = 0
    num_nodes = G.number_of_nodes()
    for x in result:
        max_shortest_path_length = max(max_shortest_path_length, max(x[1].values())-transfer_penalty_cost)
        try:
            if len(x[1]) > num_nodes * min_connected_nodes_perc:
                total = sum(x[1].values())/60
                farness_dict[x[0]] = round(total/(len(x[1])-1),1)
            else:
                farness_dict[x[0]] = math.nan
        except ZeroDivisionError:
            farness_dict[x[0]] = math.nan
    
    x_list = list(nx.get_node_attributes(G,'x').values())
    y_list = list(nx.get_node_attributes(G,'y').values())  

    df = pd.DataFrame({'node_id':list(farness_dict.keys()),\
                       'x':x_list,\
                       'y':y_list,\
                       'values':list(farness_dict.values())})               
    max_shortest_path_length = round(max_shortest_path_length/60,1)     
    mean_shortest_path_length = round(np.nanmean(list(farness_dict.values())),1)
    
    result_dict = {}
    result_dict['max_shortest_path_length'] = max_shortest_path_length
    result_dict['mean_shortest_path_length'] = mean_shortest_path_length
    result_dict['df'] = df           
    return result_dict

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

