class LabelAnalysis(object):  
    def __init__(self, fname):
        """
        Load and analyse label analysis data output from Avizo. Uses 'Label analysis' output including headers: 
        BaryCenterX, BaryCenterY, BaryCenterZ, Volume3d, EqDiameter (with or without units and trailing spaces)
        
        Inputs
        ----------
        fname (str):
            .csv file containing outputs from Avizo
            
        Attributes
        ----------
        extent (Extent subclass):
            for each of (x, y, z): min, max, extent
            
        estimated_areal_loading (float):
            estimated mass of sulfur in g/m2 for the analysed volume
            
        Methods
        ----------
        estimate_thickness(n_grids=10, return_grid=True):
            estimates the electrode thickness, returning either a grid of pre-defined resolution, or a single median value
            
        make_distribution(distribution_variable="EqDiameter", sort_by_variable="Volume3d", use_bins=None,
                          n_bins=20, percentile_min=0, percentile_max=100, dist_min=None, dist_max=None, mm_to_um=True):
            makes a binned distribution of the selected variable, according to the share of the volume falling into each range                          
                          
        estimate_neighbourhood(self, fast_grid=True, grid_size=4):
            finds the distance from each particle to the nearest larger particle
        
        """        
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        self._version = "20.11.2024"
        self._log = ["20.11.2024: fixed error on self.estimate_neighbourhood(fast_grid=False)"]
        
        self.df = pd.read_csv(fname, header=1)
        self.df.columns = [name.split(" ")[0] for name in self.df.columns]
        
        class Extent(object):
            def __init__(extent_self):
                extent_self.x_extent, extent_self.y_extent, extent_self.z_extent = [max(self.df["BaryCenter{}".format(coord)])-min(self.df["BaryCenter{}".format(coord)])
                                        for coord in ["X", "Y", "Z"]]
                extent_self.x_min, extent_self.y_min, extent_self.z_min = [min(self.df["BaryCenter{}".format(coord)]) for coord in ["X", "Y", "Z"]]
                extent_self.x_max, extent_self.y_max, extent_self.z_max = [max(self.df["BaryCenter{}".format(coord)]) for coord in ["X", "Y", "Z"]]
        self.extent = Extent()
        
        coords = ["x", "y", "z"]
        thickness_direction = coords[np.argmin([vars(self.extent)["{}_extent".format(coord)] for coord in coords])]
        lateral_dimensions = [name for name in coords if name != thickness_direction]

        ## find area
        area_mm2 = np.product([vars(self.extent)["{}_extent".format(coord)] for coord in lateral_dimensions])
        
        area = area_mm2*(1e-3)**2

        total_volume = np.sum(self.df["Volume3d"])*(0.1)**3
        density_sulfur = 2.01
        mass_sulfur_g = total_volume*density_sulfur
        self.estimated_areal_loading = mass_sulfur_g/area
        
        self._thickness_direction = thickness_direction
        self._lateral_dimensions = lateral_dimensions

    def estimate_thickness(self, n_grids=10, return_grid=True):
        """
        Estimates the thickness of the electrode based on the centres of the particles in the dataset. 
        NB only accounts for particles in dataset (e.g. excludes surrounding composite/ matrix)
        
        Inputs
        ----------
        n_grids (int, default=10):
            defines the number of grid squres in to divide volume into in lateral dimension (see 'Returns')
        
        return_grid (bool, default=True):
            if True, returns a n_grids x n_grids array of local thickness estimates
            if False, returns a single median thickness value
            
        
        Returns
        ----------
        for return_grid == True:
            returns n_grids x n_grids array with maximum-minimum thickness extent within each grid square
        for return grid == False:
            returns single median thickness value
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        lateral_dimensions = self._lateral_dimensions
        thickness_direction = self._thickness_direction
        
        grids = dict([(coord, np.linspace(vars(self.extent)["{}_min".format(coord)],
                                          vars(self.extent)["{}_max".format(coord)],
                                          n_grids+1))
                      for coord in lateral_dimensions])

        subvol = np.zeros((n_grids, n_grids), dtype=object)
        for hi in range(n_grids):
            for wi in range(n_grids):
                subvol[hi, wi] = self.df.loc[(self.df["BaryCenter{}".format(str.upper(lateral_dimensions[0]))]>grids[lateral_dimensions[0]][hi]) & 
                                             (self.df["BaryCenter{}".format(str.upper(lateral_dimensions[0]))]<=grids[lateral_dimensions[0]][hi+1]) &
                                             (self.df["BaryCenter{}".format(str.upper(lateral_dimensions[1]))]>grids[lateral_dimensions[1]][wi]) & 
                                             (self.df["BaryCenter{}".format(str.upper(lateral_dimensions[1]))]<=grids[lateral_dimensions[1]][wi+1])]


        thickness_estimate = np.array([[np.max(subvol[hi, wi]["BaryCenter{}".format(str.upper(thickness_direction))])-
                                        np.min(subvol[hi, wi]["BaryCenter{}".format(str.upper(thickness_direction))])
                                        for hi in range(n_grids)]
                                        for wi in range(n_grids)])
        if return_grid == True:
            return thickness_estimate
        else:
            return np.nanmedian(thickness_estimate)

    def make_distribution(self, distribution_variable="EqDiameter", sort_by_variable="Volume3d", 
                          use_bins=None, n_bins=20, percentile_min=0, percentile_max=100,
                          dist_min=None, dist_max=None, mm_to_um=True):
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
            
        if mm_to_um == True:
            scale = 1000
        else:
            scale = 1

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
        
    def estimate_neighbourhood(self, fast_grid=True, n_grids=4):
        """
        Estimates the distance from each particle to the nearest particle with larger volume
        
        Inputs:
        ----------
        fast_grid (bool, default=True):
            subdivides the dataframe using grid_size (see below)
            improves speed because only a local subset of the dataframe needs to be sorted and analysed to find nearby particles
            
        grid_size (int, default=4):
            the number of grid squares to subdivide the volume into (default gives 4x4 grid)
            
        Returns:
        ----------
        equivalent_diameters, minimum_distances:
            x, y data to plot diameters against nearest neighbours
        
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm import notebook
        
        
        def find_min_distance(df):
            min_distance = []

            for idx in range(len(df)):
                high_volume = df.loc[df["Volume3d"]>df["Volume3d"].iloc[idx]]
                if len(high_volume) > 0:
                    min_distance_data = high_volume.iloc[np.argmin(np.sqrt(np.sum([(high_volume[xyz]-df[xyz].iloc[idx])**2 for xyz in ["BaryCenterX", "BaryCenterY", "BaryCenterZ"]], axis=0)))]
                    min_distance.append(np.sqrt(np.sum([(min_distance_data[xyz]-df[xyz].iloc[idx])**2 for xyz in ["BaryCenterX", "BaryCenterY", "BaryCenterZ"]])))
                else:
                    min_distance.append(np.nan)
            return min_distance
        
        
        if fast_grid == False:
            return self.df["EqDiameter"], find_min_distance(self.df)

        else:
            lateral_dimensions = self._lateral_dimensions
            thickness_direction = self._thickness_direction

            h = lateral_dimensions[0]
            w = lateral_dimensions[1]

            grid_dict = {}
            for coord in self._lateral_dimensions:
                grid_dict.update([(coord, np.linspace(vars(self.extent)["{}_min".format(coord)],
                                  vars(self.extent)["{}_max".format(coord)],
                                  n_grids+1))])

            grid_idx = np.arange(n_grids**2).reshape(n_grids, n_grids)
            subvols = np.zeros((n_grids, n_grids), dtype=object)
            nearest_neighbours = np.zeros((n_grids, n_grids), dtype=object)
            eq_diams = np.zeros((n_grids, n_grids), dtype=object)

            for hiwi in notebook.tqdm(range(n_grids**2)):    
                hi, wi = np.argwhere(grid_idx==hiwi)[0]
                subvols[hi, wi] = self.df.loc[(self.df["BaryCenter{}".format(str.upper(self._lateral_dimensions[0]))]>=grid_dict[self._lateral_dimensions[0]][hi]) &
                                    (self.df["BaryCenter{}".format(str.upper(self._lateral_dimensions[0]))]<grid_dict[self._lateral_dimensions[0]][hi+1]) & 
                                    (self.df["BaryCenter{}".format(str.upper(self._lateral_dimensions[1]))]>=grid_dict[self._lateral_dimensions[1]][wi]) &
                                    (self.df["BaryCenter{}".format(str.upper(self._lateral_dimensions[1]))]<grid_dict[self._lateral_dimensions[1]][wi+1])]   
                nearest_neighbour_dist = np.full(len(subvols[hi, wi]), np.nan, dtype=float)

                for idx in range(len(subvols[hi, wi])):
                    larger_particles = subvols[hi, wi].loc[subvols[hi, wi]["Volume3d"]>subvols[hi, wi].iloc[idx]["Volume3d"]]
                    if len(larger_particles) > 0:
                        xi, yi, zi = subvols[hi, wi].iloc[idx][["BaryCenter{}".format(coord) for coord in ["X", "Y", "Z"]]]
                        larger_pos = np.array(larger_particles[["BaryCenter{}".format(coord) for coord in ["X", "Y", "Z"]]])
                        nearest_neighbour_dist[idx] = np.nanmin([np.sqrt((x**2+y**2+z**2)) for x, y, z in larger_pos-np.array([xi, yi, zi])])
                nearest_neighbours[hi, wi] = nearest_neighbour_dist
                eq_diams[hi, wi] = np.array(subvols[hi, wi]["EqDiameter"])
                
        return eq_diams, nearest_neighbours

    def borderkill(self):
        """
        Function to remove partial particles from volume and leave only particles fully within volume (approximates)
        Deletes particles whose centroid is within one equivalent diameter of a boundary.
        Can be undone using 'undo_borderkill' to restore original data to self.df
        """
        import pandas as pd
        from copy import deepcopy
        setattr(self, "_df_original", self.df)
        setattr(self, "df", self.df.loc[pd.concat([abs(self.df["BaryCenter{}".format(str.capitalize(coord))]
                                     -vars(self.extent)["{}_{}".format(coord, dimension)])>self.df["EqDiameter"] 
                                 for coord in ["x", "y", "z"] for dimension in ["min", "max"]], axis=1).all(1)])
        
    def undo_borderkill(self):
        if "_df_original" in vars(self):
            setattr(self, "df", self._df_original)
            print("Recovered original data")
        else:
            print("No border kill applied, no changes made")
