# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 16:43:30 2025

@author: engs2608
"""

from LabelAnalysis import LabelAnalysis
import numpy as np

## To check the info, type LabelAnalysis? in the console for help


## Load the data:
## Change the file name to your .csv file saved in the same folder as this code
psd = LabelAnalysis("example_avizo_output.csv")


## Set the number of bins to group the equivalent diameter into 
## (The number of bars on the histogram)
n_bins = 11

## Set whether you want to plot the percentage of the volume or the total
## volume (in um3) on they axis
plot_percentage = False




psd.make_distribution(n_bins=11)
f, ax = psd.distribution.plot_dist_bars(percentage=plot_percentage)

if plot_percentage == True:
    ax.set_ylabel("Percentage of particle \nvolume at eq diam (%)")
else:
    ax.set_ylabel("Volume at eq diam ($\mu$m$^{3}$)")
    
ax.set_xlabel("Eq Diam ($\mu$m)")
