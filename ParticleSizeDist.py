class ParticleSizeDist(object):
    def __init__(self, fname, scale=1):
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
        
        from copy import deepcopy
        self._data = deepcopy(data)
        
        for keys, values in data.items():
            if keys in ["volume"]:
                data[keys] = np.array(values)*scale**3
                
            if keys in ["surface"]:
                data[keys] = np.array(values)*scale**2
                
            if keys in ["radius", "diameter", 'BaryX', 'BaryY', 'BaryZ']:
                data[keys] = np.array(values)*scale
        
        self._headers = header_line
        self.df = pd.DataFrame(data)
        
    def make_distribution(self, distribution_variable="radius", sort_by_variable="volume", 
                          use_bins=None, n_bins=20, percentile_min=0, percentile_max=100,
                          dist_min=None, dist_max=None):
        """
        Makes a histogram-style distribution of the selected variable.
        Bins the data into subgroups of selected variable and finds total volume of particles in each bin. 

        Inputs
        ----------
        distribution_variable (str, default="EqDiameter"):
            the variable to find the distribution of (x axis on histogram)

        sort_by_variable (str, default="Volume3d"):
            the variable over which to sum the groups of data (volume by default)

        use_bins (defualt=None):
            bin sizes from a different LabelAnalysis object, for matching x axes)

        n_bins (int, default=20):
            number of bins into which to sort the data (if use_bins==None)

        percentile_min (float, default=0):
            exclude the min x% of particles using the 'sort by' variable (e.g. by volume)

        percentile_max (float, default=100):
            exclude the max x% of particles using the 'sort by' variable (e.g. by volume)

        dist_min (float, default=None):
            minimum measurable size to include using the 'distribution' variable (e.g. equivalent diameter)

        dist_max (float, default=None):
            maximum measurable size to include using the 'distribution' variable (e.g. equivalent diameter)

        mm_to_um (bool, default=True):
            returns e.g. diameters, volumes in microns rather than mm

        Returns
        ----------
        self.distribution:

            Attributes
            ----------
            - x_bins: use for labelling - values of the x axis groups (e.g. in um)
            - y_dist_abs: absolute values of the summed volumes (e.g. in um^3)
            - y_dist: the percentage of the total volume in each bin (%)
            - x_tick_pos: the x positions to use when plotting a histogram

            Methods
            ----------
            plot_dist_bars(dist_self, xtickformatter="{:.1f}", percentage=True):
                plots the distribution of the selected variable

            Example
            ----------
            The x, y data for plotting without the automatic function can be accessed using:

            ax.bar(self.distribution.x_tick_pos, self.distribution.y_dist)
            ax.set_xticks(self.distribution.x_tick_pos)
            ax.set_xticklabels(self.distribution.x_bins)

        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        if all([type(limit)!=type(None) for limit in [dist_min, dist_max]]):
            self.df = self.df.loc[(self.df[distribution_variable]>dist_min) & (self.df[distribution_variable]<dist_max)]
        x_meas = np.array(self.df[sort_by_variable])
        y_meas = np.array(self.df[distribution_variable])

        sort_idx = np.argsort(x_meas)
        sorted_x_all = x_meas[sort_idx]
        sorted_y_all = y_meas[sort_idx]

        lo_idx = int(sort_idx.shape[0]*percentile_min/100)
        hi_idx = int(sort_idx.shape[0]*percentile_max/100)

        sorted_x = sorted_x_all[lo_idx: hi_idx]
        sorted_y = sorted_y_all[lo_idx: hi_idx]

        if type(use_bins) ==  type(None):
            y_min = np.min(sorted_y)
            y_max = np.max(sorted_y)

            bins = np.linspace(y_min, y_max, n_bins+1)

        else:
            bins = use_bins
            n_bins = len(use_bins)-1

        self.bins = bins

        binned_idx = [np.argmin(abs(sorted_y-val)) for val in bins]
        x_binned = [sorted_x[binned_idx[b]: binned_idx[b+1]] for b in range(n_bins)]
        x_bin_sums = np.array([np.sum(x) for x in x_binned])

        class _Dist(object):

            def __init__(dist_self):
                import pandas as pd
                import numpy as np
                import matplotlib.pyplot as plt
                dist_self.x_bins = bins*scale
                dist_self.y_dist_abs = np.array(x_bin_sums*scale**3)
                dist_self.y_dist = dist_self.y_dist_abs/np.sum(dist_self.y_dist_abs)
                dist_self.x_tick_pos = np.arange(n_bins)

            def plot_dist_bars(dist_self, xtickformatter="{:.1f}", percentage=True):
                import pandas as pd
                import numpy as np
                import matplotlib.pyplot as plt
                """
                Automatically plots the distribution as a bar chart (with axes formatted like histogram)

                Inputs
                ----------
                xtickformatter (str, default={:.1f}):
                    modifies the decimal places shown on the x axis tick labels

                percentage (bool, default=True):
                    if true, shows the bar heights as a percentage of the total
                    if false, shows absolute units (e.g. um^3)

                Returns
                ----------
                f, ax (can be used to modify axis, labels, etc)
                """


                f, ax = plt.subplots()
                if percentage == True:
                    ax.bar(self.distribution.x_tick_pos, self.distribution.y_dist)
                else:
                    ax.bar(self.distribution.x_tick_pos, self.distribution.y_dist_abs)

                x_bins = self.distribution.x_bins
                bin_spacing = x_bins[1]-x_bins[0]
                bin_ticks = x_bins-bin_spacing
                bin_tick_labels = []
                for val in bin_ticks:
                    if np.sign(val) == 1:
                        bin_tick_labels.append(xtickformatter.format(val))
                    else:
                        bin_tick_labels.append(xtickformatter.format(0))

                ax.set_xticks(list(self.distribution.x_tick_pos-0.5)+[self.distribution.x_tick_pos[-1]+0.5])

                ax.set_xticklabels(bin_tick_labels)
                dist_self._bin_tick_labels = bin_tick_labels
                return f, ax

        self.distribution = _Dist()
