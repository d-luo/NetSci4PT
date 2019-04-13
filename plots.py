# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:54:04 2019

@author: dingluo
"""

from numpy import ma
from matplotlib import cbook
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns;



def plot_violin_graph():
    pass

def plot_travel_impedance_map(G_L,df,cb_label,dist_x_label,with_dist,fig_para,city_name,save_pic):
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



def plot_travel_impedance_comparison_map(G_L,df,r_value,fig_para,city_name,x_clm,y_clm,diff_clm,save_pic):
    """
    Plot the map of the comparison between the benchmark and GTC-based metrics
    
    Parameters
    ----------
    G_L : networkx 

    city : string
        name of the city
    Returns
    -------
    G : directed graph as a networkx object with two types of weights:
        TravelTime and ServiceFrequency
    
    """
    
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
 
def add_linreg_residuals(df,x_clm,y_clm,diff_clm):
    x = df[x_clm]
    y = df[y_clm]
    mask = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask],y[mask])
    residuals = y - (slope * x + intercept)
    df[diff_clm] = residuals
    return df,r_value       
        
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


def plot_network_properties():
    '''
    This function makes the following figure: 
    Figure 3: Illustration of the basic properties of the studied tram networks.
    '''
    df = pd.read_csv('tram_networks.csv')
    
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_axes([0.15,0.2,0.6,0.7])
    #cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
    ax = sns.scatterplot(x="nStops", y="nLinks", alpha=0.8,hue = '# Routes',size = '# Routes',
                         sizes=(20, 100),legend = 'full',
                         data=df)
    ax.set(xlabel='# Stops', ylabel='# Links')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)       

class MidPointNorm(Normalize):    
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")       
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint            
            resdat[resdat>0] /= abs(vmax - midpoint)            
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)                

        if is_scalar:
            result = result[0]            
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)  
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0: 
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint