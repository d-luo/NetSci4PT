# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 10:04:59 2019

@author: dingluo
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set(style="whitegrid")

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