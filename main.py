# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 14:05:05 2018

@author: dingluo
"""

from __future__ import division
import math
import pandas as pd
import numpy as np
import networkx as nx

from network_loading import *
from sklearn.preprocessing import scale
from scipy import stats
from mpl_toolkits.axes_grid1 import AxesGrid
from method import *
from plot import *
from utils import *


    

def add_linreg_residuals(df,x_clm,y_clm,diff_clm):
    x = df[x_clm]
    y = df[y_clm]
    mask = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask],y[mask])
    residuals = y - (slope * x + intercept)
    df[diff_clm] = residuals
    return df,r_value

def gen_results_accessiblility_viz(graph_dict,city_list,city_names):
    '''
    This function is responsible for generating three types of visualization
      (1) Hop-based accessibility for all the stops in the network;
      (2) GTC-based accessibility for all the stops in the network;
      (3) Comparison between the two different kinds
    '''
    # figure parameters
    fig_para = {}
    fig_para['ax1'] = [0.001,0.01,0.68,0.8]
    fig_para['ax2'] = [0.75,0.04,0.02,0.43]
    fig_para['ax3'] = [0.67,0.6,0.24,0.28]
    fig_para['node_size'] = 10 
    # switch for whether the distribution will be included in the final diagram      
    with_dist = True
    # switch for whether the plot will be saved as a picture finally
    save_pic = True
    
    for x in range(len(city_list)):
        cur_city = city_list[x]
        G_L = graph_dict[cur_city]['L']
        G_P = graph_dict[cur_city]['P']
        # hop-based computing solely using the unweighted L-space network       
        result_dict_unweighted_L = compute_metric_benchmark(G_L,min_connected_nodes_perc)
        # GTC-based computing solely using the weighted P-space network    
        result_dict_weighted_P = compute_GTCbased_accessibility(G_P,transfer_penalty_cost,min_connected_nodes_perc)   
        # combining two results for a final dataframe
        final_df = pd.DataFrame({'node_id':result_dict_unweighted_L['df']['node_id'],\
                                 'x':result_dict_unweighted_L['df']['x'],\
                                 'y':result_dict_unweighted_L['df']['y'],\
                                 'num_hops':result_dict_unweighted_L['df']['values'],\
                                 'travel_time':result_dict_weighted_P['df']['values']})
        # hop-based 
        plot_networkwide_accessibility(G_L,result_dict_unweighted_L['df'],\
                          'Travel Cost [# hops]','# hops',with_dist,fig_para,city_names[cur_city],save_pic) 
        # GTC-based                  
        plot_networkwide_accessibility(G_L,result_dict_weighted_P['df'],\
                          'Travel Cost [min]','min',with_dist,fig_para,city_names[cur_city],save_pic)
        # Comparison    
        final_df,r_value = add_linreg_residuals(final_df,'num_hops','travel_time','residual_of_hops')
        plot_comparison_map(G_L,final_df,r_value,fig_para,city_names[cur_city],'num_hops','travel_time','residual_of_hops',save_pic) 

    
def gen_results_accessiblity_ccdf(graph_dict,city_list,city_names):
    marker_list = ['s','o','v','p','*','<','+','p','d']
    color_list = ['tab:purple','tab:blue','tab:green','r','c','m','tab:orange','k']
    #---- hop-based accessibility
    f1 = plt.figure()
    for x in range(len(city_list)):
        cur_city = city_list[x]
        cur_cityname = city_names[cur_city]
        G_L = graph_dict[cur_city]['L']
        G_P = graph_dict[cur_city]['P']          
        # computation       
        result_by_hops = compute_hopbased_accessibility(G_L,min_connected_nodes_perc)
        dist_dict = derive_pdf_ccdf(result_by_hops['df']['values'])
        line, = plt.plot(dist_dict['variable'],dist_dict['ccdf'],'--',markersize=1,
                 marker = marker_list[x],color = color_list[x],label = cur_cityname)
        plt.legend(loc = 'upper right') 
        plt.xlabel('# Hops')
        plt.ylabel('Probability')
    plt.savefig('ccdf_hop_based.png', format='png', dpi=300)    
    
    #---- GTC-based accessibility
    f2 = plt.figure()
    for x in range(len(city_list)):
        cur_city = city_list[x]
        cur_cityname = city_names[cur_city]
        G_L = graph_dict[cur_city]['L']
        G_P = graph_dict[cur_city]['P']          
        # computation       
        result_by_hops = compute_GTCbased_accessibility(G_P,transfer_penalty_cost,min_connected_nodes_perc)  
        dist_dict = derive_pdf_ccdf(result_by_hops['df']['values'])
        line, = plt.plot(dist_dict['variable'],dist_dict['ccdf'],'--',markersize=1,
                 marker = marker_list[x],color = color_list[x],label = cur_cityname)
        plt.legend(loc = 'upper right') 
        plt.xlabel('Generalized Travel Cost [min]')
        plt.ylabel('Probability')
    plt.savefig('ccdf_GTC_based.png', format='png', dpi=300)  


if __name__ == '__main__': 
    

    # Initial parameters
    min_connected_nodes_perc = 0.2
    transfer_penalty_cost = 300 # 300 seconds = 5 min
#    city_names = {'amsterdam':'Amsterdam','milan':'Milan','denhaag':'The Hague',\
#                  'melbourne':'Melbourne','vienna':'Vienna','zurich':'Zurich',\
#                  'toronto':'Toronto','budapest':'Budapest'}
    city_names = {'amsterdam':'Amsterdam'}

    city_list = list(city_names.keys())
#    city_list = ['zurich','vienna','denhaag']
    space_list = ['L','P']
    graph_dict = load_graphs(city_list,space_list)
    
    gen_results_accessiblility_viz(graph_dict,city_list,city_names)
#    
#    df_dict ={}
#    for x in range(len(city_list)):
#        cur_city = city_list[x]
#        cur_cityname = city_names[cur_city]
#        G_L = graph_dict[cur_city]['L']
#        G_P = graph_dict[cur_city]['P']
#        # hop-based computing solely using the unweighted L-space network       
#        result_dict_unweighted_L = compute_hopbased_accessibility(G_L,min_connected_nodes_perc)
#        # GTC-based computing solely using the weighted P-space network    
#        result_dict_weighted_P = compute_GTCbased_accessibility(G_P,transfer_penalty_cost,min_connected_nodes_perc)   
#        # combining two results for a final dataframe
#        
#        df_dict[cur_city] = pd.DataFrame({'node_id':result_dict_unweighted_L['df']['node_id'],\
#                                 'x':result_dict_unweighted_L['df']['x'],\
#                                 'y':result_dict_unweighted_L['df']['y'],\
#                                 'num_hops':result_dict_unweighted_L['df']['values'],\
#                                 'travel_time':result_dict_weighted_P['df']['values']})
#        df_dict[cur_city]['city'] = cur_cityname
#        
#    final_df = pd.concat(list(df_dict.values()))
#    
#    # Set up the matplotlib figure
#    f, ax = plt.subplots(figsize=(11, 5))
#    ax.set_axisbelow(True)
#    ax.grid(color='k', alpha=0.5, linestyle='--',linewidth=1)
#    ax = sns.violinplot(x = 'city',y='travel_time',data = final_df,
#                        order=["Melbourne", "Milan",'Budapest','Vienna',
#                               'Toronto','The Hague','Amsterdam','Zurich'])
#    ax.set(xlabel='', ylabel='Generalized Travel Cost [min]')
#    plt.savefig('violinplot.png', format='png', dpi=300)     
    
#    gen_results_accessiblility_viz(graph_dict,city_list,city_names)
 
   
