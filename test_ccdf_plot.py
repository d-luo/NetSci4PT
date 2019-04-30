# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:56:11 2019

@author: dingluo
"""

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