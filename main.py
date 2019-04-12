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
import matplotlib.pyplot as plt
import seaborn as sns;
from network_loading import *
from accessibility_computing import *
from sklearn.preprocessing import scale
from scipy import stats
from mpl_toolkits.axes_grid1 import AxesGrid
from MidPointNorm import *

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


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
    
    
def plot_networkwide_accessibility(G_L,df,cb_label,dist_x_label,with_dist,fig_para,city_name,save_pic):
    '''
    some parameters
    '''
    cmap_name = 'inferno_r'
    opacity_value = 0.7
    
    fig = plt.figure(figsize=(5,4))

    # select NonNaN rows
    idx_nonnan = df['values'].notnull()
    lat_nonnan, lon_nonnan = df['y'].loc[idx_nonnan], df['x'].loc[idx_nonnan]
    nonnan_values = df['values'].loc[idx_nonnan]
    # select NaN rows
    idx_nan = df['values'].isnull()
    lat_nan, lon_nan = df['y'].loc[idx_nan], df['x'].loc[idx_nan]   
    # draw the underlying links first
    pos = nx.get_node_attributes(G_L,'coords')
    
    # ax1 is the major plot
    ax1 = fig.add_axes(fig_para['ax1'])
    ax1.axis('off')
    ax1.set_title(city_name,loc = 'center')
    nx.draw_networkx_edges(G_L,pos,edge_color = '#b7c9e2',width=1,arrows=False,\
                           alpha = opacity_value,ax = ax1)                  
    # draw the NaN points 
    ax1.scatter(lon_nan, lat_nan, label=None, color = 'k', marker = 'x', s= fig_para['node_size'])                          
    # Scatter the nonnan points, using size and color but no label
    sc = ax1.scatter(lon_nonnan, lat_nonnan, label=None,
                c=nonnan_values, cmap=cmap_name,
                s= fig_para['node_size'], linewidth=0, alpha=opacity_value)
    # ax2 is the colorbar plot
    ax2 = fig.add_axes(fig_para['ax2'])
    ax2.tick_params(labelsize = 9)
    cb = plt.colorbar(sc,cax = ax2)
    cb.set_label(cb_label,fontsize=10)
    if with_dist:
        # this is an inset axes over the main axes
        # ax3 is the distribution plot
        sns.set(font_scale=2)
        sns.set(style="white", palette="muted", color_codes=True)
        ax3 = fig.add_axes(fig_para['ax3'])
        sns.distplot(nonnan_values,kde=False, color="k",norm_hist = False,ax = ax3)
        ax3.tick_params(axis='both', labelsize=8,pad = 0.1)
        ax3.set_xlabel(dist_x_label,fontsize=9)
        ax3.set_ylabel('Count',fontsize=9)
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position("right")
        ax3.tick_params(axis='both', labelsize=8,pad = 0.1)
        ax3.grid(color='k', alpha=0.5, linestyle='--',linewidth=1)
        
    if save_pic:
        file_name = city_name + '_' + cb_label + '.png'
        plt.savefig(file_name, format='png', dpi=300)
    
def plot_accessibility_cmp(G_L,df,r_value,fig_para,city_name,x_clm,y_clm,diff_clm):
    orig_cmap = 'coolwarm'
    opacity_value = 0.7
    norm = MidPointNorm(midpoint=0)
    
    fig = plt.figure(figsize=(5,4))
    ax1 = fig.add_axes(fig_para['ax1'])
    ax1.axis('off')

    ax1.set_title(city_name,loc = 'center')

    pos = nx.get_node_attributes(G_L,'coords')
    nx.draw_networkx_edges(G_L,pos,edge_color = '#b7c9e2',width=1,arrows=False,\
                           alpha = opacity_value,ax = ax1)  
    idx_nan = df[diff_clm].isnull()
    idx_nonnan = ~idx_nan
    # draw the NaN points 
    ax1.scatter(df['x'].loc[idx_nan], df['y'].loc[idx_nan], \
                label=None, color = 'k', marker = 'x', s= fig_para['node_size']-1)    
                         
    sc =ax1.scatter(df['x'].loc[idx_nonnan], df['y'].loc[idx_nonnan], label=None,
                    c=df[diff_clm].loc[idx_nonnan], norm = norm, cmap=orig_cmap,
                    vmin = -10, vmax = 10,
                    s= fig_para['node_size'], linewidth=0, alpha=opacity_value)
    # ax2 is the colorbar plot
    ax2 = fig.add_axes(fig_para['ax2'])
    ax2.tick_params(labelsize = 9)
    cb = plt.colorbar(sc,cax = ax2,extend = 'both')
    cb.set_label('Gap [min]',fontsize=10)   
    ax3 = fig.add_axes(fig_para['ax3'])
    plot_scatter_comparison(df,r_value,ax3,x_clm,y_clm,diff_clm)
    
    if save_pic:
        file_name = city_name + '_cmp.png'
        plt.savefig(file_name, format='png', dpi=300)

def plot_scatter_comparison(df,r_value,cur_ax,x_clm,y_clm,diff_clm):
    cur_cmap = 'coolwarm'
    norm = MidPointNorm(midpoint=0)
    idx_nonnan = ~df[diff_clm].isnull()
#    rsquare = lambda a, b: stats.pearsonr(a, b)[0] ** 2

    sns.regplot(x=x_clm, y=y_clm,data = df,dropna = True,color="b",\
                scatter = True,scatter_kws = {'s':0.5},line_kws = {'color':'k','linewidth':1},ax = cur_ax)
    cur_ax.scatter(df[x_clm].loc[idx_nonnan],df[y_clm].loc[idx_nonnan],\
               norm = norm, c= df[diff_clm].loc[idx_nonnan], cmap = cur_cmap,s = 2)
    cur_ax.set_xlabel('# Hops',fontsize=10)
    cur_ax.set_ylabel('GTC [min]',fontsize=10)
    cur_ax.tick_params(axis='both', labelsize=8,pad = 0.1)
    cur_ax.yaxis.set_label_position("right")
    cur_ax.yaxis.tick_right()
    cur_ax.grid(color='k', alpha= 0.3, linestyle='--',linewidth=0.5)
    # add the correlation coefficient in the scatter plot
    r_value = round(r_value,2)
    cur_str = f"r = {r_value}"
    x = 0.6 * max(df[x_clm].loc[idx_nonnan]) 
    y = 1.2 * min(df[y_clm].loc[idx_nonnan])
    cur_ax.text(x,y,cur_str,fontsize=10,style='italic')        


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
        result_dict_unweighted_L = compute_hopbased_accessibility(G_L,min_connected_nodes_perc)
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
        plot_accessibility_cmp(G_L,final_df,r_value,fig_para,city_names[cur_city],'num_hops','travel_time','residual_of_hops') 

    
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
    city_names = {'amsterdam':'Amsterdam','milan':'Milan','denhaag':'The Hague',\
                  'melbourne':'Melbourne','vienna':'Vienna','zurich':'Zurich',\
                  'toronto':'Toronto','budapest':'Budapest'}

    city_list = list(city_names.keys())
#    city_list = ['zurich','vienna','denhaag']
    space_list = ['L','P']
    graph_dict = load_graphs(city_list,space_list)
    
    df_dict ={}
    for x in range(len(city_list)):
        cur_city = city_list[x]
        cur_cityname = city_names[cur_city]
        G_L = graph_dict[cur_city]['L']
        G_P = graph_dict[cur_city]['P']
        # hop-based computing solely using the unweighted L-space network       
        result_dict_unweighted_L = compute_hopbased_accessibility(G_L,min_connected_nodes_perc)
        # GTC-based computing solely using the weighted P-space network    
        result_dict_weighted_P = compute_GTCbased_accessibility(G_P,transfer_penalty_cost,min_connected_nodes_perc)   
        # combining two results for a final dataframe
        
        df_dict[cur_city] = pd.DataFrame({'node_id':result_dict_unweighted_L['df']['node_id'],\
                                 'x':result_dict_unweighted_L['df']['x'],\
                                 'y':result_dict_unweighted_L['df']['y'],\
                                 'num_hops':result_dict_unweighted_L['df']['values'],\
                                 'travel_time':result_dict_weighted_P['df']['values']})
        df_dict[cur_city]['city'] = cur_cityname
        
    final_df = pd.concat(list(df_dict.values()))
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 5))
    ax.set_axisbelow(True)
    ax.grid(color='k', alpha=0.5, linestyle='--',linewidth=1)
    ax = sns.violinplot(x = 'city',y='travel_time',data = final_df,
                        order=["Melbourne", "Milan",'Budapest','Vienna',
                               'Toronto','The Hague','Amsterdam','Zurich'])
    ax.set(xlabel='', ylabel='Generalized Travel Cost [min]')
    plt.savefig('violinplot.png', format='png', dpi=300)     
    
#    gen_results_accessiblility_viz(graph_dict,city_list,city_names)
 
   
