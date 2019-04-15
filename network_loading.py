# -*- coding: utf-8 -*-
###############################################################################
# 
# Author: Ding Luo @ TU Delft, The Netherlands
###############################################################################

import pandas as pd
import networkx as nx

def load_Lspace_graph(folder,city):
    """
    Load the L-space graph stored in csv files into a networkx graph 
    
    Parameters
    ----------
    folder : string
        where to load the csv files for networks
    city : string
        name of the city
    Returns
    -------
    G : directed graph as a networkx object with two types of weights:
        TravelTime and ServiceFrequency
    
    """
    links = pd.read_csv(folder + '/lg_ms_links_' + city +'.csv')
    nodes = pd.read_csv(folder + '/lg_ms_nodes_' + city +'.csv') 
    # repalce NaN values
#    links = links.dropna()
    # ini the graph
    G = nx.DiGraph()
    # add nodes
    for i in range(len(nodes)):
        G.add_node(nodes.iloc[i]['id'],\
                   node_id = nodes.iloc[i]['id'],\
                   name = nodes.iloc[i]['name'],\
                   x = nodes.iloc[i]['x'],\
                   y = nodes.iloc[i]['y'],\
                   coords = [nodes.iloc[i]['x'], nodes.iloc[i]['y']])
    # add links    
    for i in range(len(links)):
        G.add_edge(links.iloc[i]['EndNodes_1'],\
                   links.iloc[i]['EndNodes_2'],\
                   travel_time = links.iloc[i]['TravelTime'],\
                   service_frequency = links.iloc[i]['ServiceFrequency'])
    print('The L-space graph of ' + city + ' has been loaded...')
    return G
 
def load_Pspace_graph(folder_path,city,transfer_penalty_time):
    """
    Load the P-space graph stored in csv files into a networkx graph 
    
    Parameters
    ----------
    folder : string
        where to save the shapefile, if none, then default folder
    city : string
        name of the city
    Returns
    -------
    G : directed graph as a networkx object with four types of weights:
        total in-vehicle travel time, ServiceFrequency, wating time and total
        travel time
    
    """
    links = pd.read_csv(folder_path + '/pg_ms_links_' + city +'.csv')
    links = links.dropna()
    nodes = pd.read_csv(folder_path + '/pg_ms_nodes_' + city +'.csv') 
    # ini the graph
    G = nx.DiGraph()
    # add nodes
    for i in range(len(nodes)):
        G.add_node(nodes.iloc[i]['id'],\
                   node_id = nodes.iloc[i]['id'],\
                   name = nodes.iloc[i]['name'],\
                   x = nodes.iloc[i]['x'],\
                   y = nodes.iloc[i]['y'],\
                   coords = [nodes.iloc[i]['x'], nodes.iloc[i]['y']])
    # add links    
    # ivt = in-vehicle travel times
    # sf = service frequency
    # wt = waiting times              
    for i in range(len(links)):
        G.add_edge(links.iloc[i]['EndNodes_1'],\
                   links.iloc[i]['EndNodes_2'],\
                   ivt = links.iloc[i]['TravelTime'],\
                   sf = links.iloc[i]['ServiceFrequency'],\
                   wt = links.iloc[i]['WaitingTime'] * 60,\
                   total_travel_time = links.iloc[i]['TravelTime'] + \
                                       links.iloc[i]['WaitingTime'] * 60 + \
                                       transfer_penalty_time)
    print('The P-space graph of ' + city + ' has been loaded...')
    return G   

def load_graphs(city_list,space_list,transfer_penalty_time):
    folder_path = r'D:/dingluo/SURFdrive/research/4_codes/tramaccess/data/tram_graphs'
    graph_dict = {}
    
    for city in city_list:
        graph_dict[city] = {}
        if 'L' in space_list:
            graph_dict[city]['L'] = load_Lspace_graph(folder_path,city)
        if 'P' in space_list:    
            graph_dict[city]['P'] = load_Pspace_graph(folder_path,city,transfer_penalty_time)
        
    return graph_dict    

if __name__ == '__main__':
    city_list = ['amsterdam']
    space_list = ['L','P']
    graph_dict = load_graphs(city_list,space_list)