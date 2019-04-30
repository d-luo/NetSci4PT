################################################################################
# Module: main.py
# Description: pipeline of the following paper
#    Integrating Network Science and Public Transport Accessibiliy Analysis 
#    for Comparative Assessment
# Ding Luo @ TU Delft, The Netherlands
################################################################################


import pandas as pd
from network_loading import *
from methods import *
from plots import *

def compute_all_networks():
    transfer_penalty = 300
    space_list = ['L','P']
    # specify all the networks
    cities = {'amsterdam':'Amsterdam','milan':'Milan',\
              'denhaag':'The Hague', 'melbourne':'Melbourne',\
              'vienna':'Vienna','zurich':'Zurich',\
              'toronto':'Toronto','budapest':'Budapest'}
#    cities = {'amsterdam':'Amsterdam','denhaag':'The Hague'}
 
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
        df_dict[city_key]['df']['cityname'] = cities[city_key]
    return df_dict
        
def plot_all_travel_impedance_maps(df_dict):
    for key in df_dict.keys():
        G_L = df_dict[key]['L']
        # plot the benchmark map
        plot_travel_impedance_map(G_L,df_dict[key]['df'],'hops',\
                                  '# Hops','# Hops','Benchmark metric')      
        # plot the GTC-based map                          
        plot_travel_impedance_map(G_L,df_dict[key]['df'],'gtc',\
                                  'Minutes','Minutes','GTC-based metric')          
        # plot the comparison map                   
        plot_travel_impedance_comparison_map(G_L,df_dict[key]['df'],df_dict[key]['r'],
                                             'hops','gtc','gap','Comparison')                             

def plot_all_violin_graphs(df_dict):
    plot_violin_graph(df_dict,'gtc','Minutes','Generalized travel cost')
    plot_violin_graph(df_dict,'gtc_ivt','Minutes','In-vehicle travel times')
    plot_violin_graph(df_dict,'gtc_nonivt','Minutes','Waiting and transfer times (with penalty)')

   
    
if __name__ == '__main__': 
    # The pipeline is as follows:
    df_dict = compute_all_networks()
    plot_all_travel_impedance_maps(df_dict)
    plot_all_violin_graphs(df_dict)
  
 
   
