import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import re

class ParticleSizeDist(object):
    def __init__(self, fname):
        '''
        Load and analyse particle size distribution data from ImageJ using XLib plugin
        'Features for printing' specified as: itvardsbg (with 'g' at end for bary centres)
        
        Inputs
        ----------
        fname (str):
            .txt file containing outputs from XLib plugin
            
        Attributes
        ----------
        df (pandas dataframe):
            Dataframe of values from txt file
            
        Methods
        ----------
        make_volume_dist(nbins=20, min_radius=None, max_radius=None, use_bins=None):
            makes a binned distribution of the volume as a function of radius
        
        '''
        header = True
        data = {}
        
        self._version = "13/09/2024"

        with open(fname) as f:
            for line in f.readlines():
                if header == False:
                    line_data = [value for value in line.split("  ") if len(value)>0]
                    bary_center = line_data[-1]
                    bary_center_values = re.findall("\d+.\d+", bary_center)
                    line_data = line_data[:-1]+bary_center_values
                    for nvalue, value in enumerate(line_data):
                        [*data.values()][nvalue].append(float(value))

                if " ID " in line:
                    header_line = [name.strip("\n").strip(" ") for name in line.split("  ") if len(name)>0]
                    header = False
                    header_line = header_line[:-1]
                    [header_line.append(name) for name in ["BaryX", "BaryY", "BaryZ"]]
                    data.update([(name, []) for name in header_line])
                    
        data.update([("sphericity", 
                      np.pi**(1/3)*((6*np.array(data["volume"]))**(2/3))/np.array(data["surface"]))])
        self.data = data
        self._headers = header_line
        self.df = pd.DataFrame(data)
        
    def make_volume_dist(self, nbins=20, min_radius=None, 
                         max_radius=None, use_bins=None, 
                         min_sphericity=0, max_sphericity=None):
        if min_radius == None:
            min_radius = np.min(self.df["radius"])
        if max_radius == None:
            max_radius = np.max(self.df["radius"])
        
        if type(use_bins) == type(None):
            bins = np.linspace(min_radius, max_radius, nbins+1)
        else:
            bins = use_bins
            nbins = bins.shape[0]-1
        volume_dist = np.full((nbins, 3), np.nan)
        
        if type(max_sphericity) == type(None):
            max_sphericity = 1e6
            
        for n in range(nbins):
            volume_dist[n, 0] = bins[n]
            volume_dist[n, 1] = bins[n+1]
            volume_dist[n, 2] = np.sum(self.df.loc[(self.df["radius"]>=bins[n]) & 
                                                   (self.df["radius"]<bins[n+1]) &
                                                   (self.df["sphericity"]>min_sphericity) &
                                                   (self.df["sphericity"]<max_sphericity)]["volume"])
            
        return pd.DataFrame({"min radius": volume_dist[:, 0],
                "max_radius": volume_dist[:, 1],
                "total volume": volume_dist[:, 2]})
