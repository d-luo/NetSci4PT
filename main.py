# -*- coding: utf-8 -*-
"""

@author: Ding Luo
"""


import pandas as pd
from network_loading import *
from methods import *
from plots import *

def compute_all_networks():
    transfer_penalty = 300
    space_list = ['L','P']
    # specify all the networks
#    cities = {'amsterdam':'Amsterdam','milan':'Milan',\
#              'denhaag':'The Hague', 'melbourne':'Melbourne',\
#              'vienna':'Vienna','zurich':'Zurich',\
#              'toronto':'Toronto','budapest':'Budapest'}
    
    cities = {'amsterdam':'Amsterdam'}
    
    # load all the networks
    city_keys = list(cities.keys())
    graphs = load_graphs(city_keys,space_list,transfer_penalty)
    df_dict = {} # a dictionary containing all the results
    for city_key in city_keys:
        G_L = graphs[city_key]['L']
        G_P = graphs[city_key]['P']
        df_benchmark = compute_benchmark_metric(G_L)
        df_GTC = compute_GTCbased_metric(G_P,transfer_penalty)  
        
        df_dict[city_key] = {}
        df_dict[city_key]['L'] = G_L # This is for the sake of subsequent viz
        df_dict[city_key]['df'] = pd.DataFrame({'node_id':df_GTC['node_id'],\
                                 'x':df_GTC['x'],'y':df_GTC['y'],\
                                 'hops':df_benchmark['hops'],\
                                 'gtc':df_GTC['gtc'],\
                                 'gtc_ivt':df_GTC['ivt'],\
                                 'gtc_nonivt':df_GTC['nonivt']})
        df_dict[city_key]['df'], df_dict[city_key]['r'] = \
             compute_gap_between_metrics(df_dict[city_key]['df'],'hops','gtc')
        df_dict[city_key]['cityname'] = cities[city_key]
    return df_dict
        
def plot_all_travel_impedance_maps(df_dict):
    for key in df_dict.keys():
        G_L = df_dict[key]['L']
        # plot the benchmark map
        plot_travel_impedance_map(G_L,df_dict[key]['df'],'hops',\
                                  '# Hops','# Hops',df_dict[key]['cityname'])      
        # plot the GTC-based map                          
        plot_travel_impedance_map(G_L,df_dict[key]['df'],'gtc',\
                                  'Minutes','Minutes',df_dict[key]['cityname'])          
        # plot the comparison map                   
        plot_travel_impedance_comparison_map(G_L,df_dict[key]['df'],df_dict[key]['r'],\
                                             df_dict[key]['cityname'],'hops','gtc','gap')                             

def plot_ccdf():
    pass

def plot_all_violin_graphs():
    pass


    
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
    # specify all the inputs first
    cities = {'amsterdam':'Amsterdam','milan':'Milan',\
              'denhaag':'The Hague', 'melbourne':'Melbourne',\
              'vienna':'Vienna','zurich':'Zurich',\
              'toronto':'Toronto','budapest':'Budapest'}
    
    df_dict = compute_all_networks()
    
    plot_all_travel_impedance_maps(df_dict)
#    plot_violin_graph(df_dict,'Cost of waiting and transfers')

#    # Initial parameters
#    min_connected_nodes_perc = 0.2
#    transfer_penalty_time = 300 # 300 seconds = 5 min
#    # figure parameters
#    fig_para = {}
#    fig_para['ax1'] = [0.015,0.01,0.68,0.8]
#    fig_para['ax2'] = [0.75,0.04,0.02,0.43]
#    fig_para['ax3'] = [0.67,0.6,0.24,0.28]
#    fig_para['node_size'] = 10 
##    city_names = {'amsterdam':'Amsterdam','milan':'Milan','denhaag':'The Hague',\
##                  'melbourne':'Melbourne','vienna':'Vienna','zurich':'Zurich',\
##                  'toronto':'Toronto','budapest':'Budapest'}
#    
#    city_names = {'amsterdam':'Amsterdam'}
#
#    city_list = list(city_names.keys())
#    space_list = ['L','P']
#    graph_dict = load_graphs(city_list,space_list,transfer_penalty_time)
#    G_L = graph_dict[city_list[0]]['L']
#    G_P = graph_dict[city_list[0]]['P']
#    
#    df = compute_GTCbased_metric(G_P,transfer_penalty_time)
#    plot_travel_impedance_map(G_L,df,'gtc','min','min','True',fig_para,city_names['amsterdam'],\
#                              'True')
#    gen_results_accessiblility_viz(graph_dict,city_list,city_names)
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
  
    
#    gen_results_accessiblility_viz(graph_dict,city_list,city_names)
 
   
