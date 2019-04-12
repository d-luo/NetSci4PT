# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 10:04:59 2019

@author: dingluo
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set(style="whitegrid")


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






df = pd.read_csv('tram_networks.csv')

fig = plt.figure(figsize=(6,3))
ax = fig.add_axes([0.15,0.2,0.6,0.7])
#cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
ax = sns.scatterplot(x="nStops", y="nLinks", alpha=0.8,hue = '# Routes',size = '# Routes',
                     sizes=(20, 100),legend = 'full',
                     data=df)
ax.set(xlabel='# Stops', ylabel='# Links')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


#for line in range(0,df.shape[0]):
#     ax.text(df.nStops[line]+0.2, df.nLinks[line], df.City[line], 
#             horizontalalignment='left', size='small', color='black', weight='semibold')

plt.savefig('network_description.png', format='png', dpi=300)