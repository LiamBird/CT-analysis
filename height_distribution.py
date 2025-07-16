def height_distribution(sample, heightdim="y", dimh1="x", dimh2="z", 
                        dimh1_min=None, dimh2_min=None,
                        grid_steps=25, smooth_range=5, y_bins = 20, min_height_only=False, return_sphericity=False):
    import pandas as pd
    import warnings
    import numpy as np
    warnings.filterwarnings(action='ignore', category=RuntimeWarning)

    H1 = str.capitalize(dimh1)
    H2 = str.capitalize(dimh2)
    HEIGHT = str.capitalize(heightdim)

    if dimh1_min == None:
        h1_grid = np.linspace(vars(sample.extent)[dimh1+"_min"], vars(sample.extent)[dimh1+"_max"], grid_steps)
    else:
        h1_grid = np.linspace(dimh1_min, vars(sample.extent)[dimh1+"_max"], grid_steps)

    if dimh2_min == None:
        h2_grid = np.linspace(vars(sample.extent)[dimh2+"_min"], vars(sample.extent)[dimh2+"_max"], grid_steps)
    else:
        h2_grid = np.linspace(dimh2_min, vars(sample.extent)[dimh2+"_max"], grid_steps)

    df_positions = np.array([[sample.df.loc[(sample.df["BaryCenter"+H1]>= h1_grid[nxi])
                              & (sample.df["BaryCenter"+H1]< h1_grid[nxi+1])
                                & (sample.df["BaryCenter"+H2] >= h2_grid[nzi])
                                & (sample.df["BaryCenter"+H2] < h2_grid[nzi+1])]
                                 for nxi, xi in enumerate(h1_grid[:-1])]
                                for nzi, zi in enumerate(h2_grid[:-1])], dtype=object)

    min_height = np.array([[np.min(df_positions[nxi, nzi]["BaryCenter"+HEIGHT]) for 
                            nxi, xi in enumerate(h1_grid[:-1])] for nzi, zi in enumerate(h2_grid[:-1])])

    stack_min_height = np.full((min_height.shape[0]+smooth_range, min_height.shape[1]+smooth_range, 
                            smooth_range**2), np.nan)
    offsets = np.arange(smooth_range**2).reshape(smooth_range, smooth_range)
    # stack_min_height
    for n in range(smooth_range**2):
        start_idx = np.argwhere(offsets == n).flatten()
        stack_min_height[start_idx[0]: start_idx[0]+min_height.shape[0],
                         start_idx[1]: start_idx[1]+min_height.shape[1],
                         n]  = min_height
    min_height_smooth = (np.nansum(stack_min_height, axis=-1)[:-smooth_range, :-smooth_range]/\
                         np.count_nonzero(np.isfinite(stack_min_height), axis=-1)[:-smooth_range, :-smooth_range])
    
    if min_height_only == True:
        return min_height_smooth
    
    else:
        height_adjust = np.array([[(df_positions[nxi, nzi]["BaryCenterY"]-min_height_smooth[nxi, nzi]) for 
                               nxi, xi in enumerate(h1_grid[:-1])]
                              for nzi, zi in enumerate(h2_grid[:-1])],
                            dtype=object)

        volume_grid = np.array([[df_positions[nxi, nzi]["Volume3d"] for 
                                 nxi, xi in enumerate(h1_grid[:-1])]
                                for nzi, zi in enumerate(h2_grid[:-1])],
                               dtype=object)
        
        sphericity_grid = np.array([[df_positions[nxi, nzi]["Sphericity"] for 
                                 nxi, xi in enumerate(h1_grid[:-1])]
                                for nzi, zi in enumerate(h2_grid[:-1])],
                               dtype=object)

        heights_combined = np.hstack(np.array([[height_adjust[nxi, nzi].to_numpy() for nxi, xi in enumerate(h1_grid[:-1])]
                                          for nzi, zi in enumerate(h2_grid[:-1])], dtype=object).flatten())

        heights_combined = heights_combined[~np.isnan(heights_combined)]


        extent_adjust = np.linspace(np.min(heights_combined), np.max(heights_combined), y_bins)


        height_volume_combo = pd.DataFrame({"height": pd.concat(list(height_adjust.flatten())),
                                             "volume": pd.concat(list(volume_grid.flatten())),
                                            "sphericity": pd.concat(list(sphericity_grid.flatten()))})

        volume_dist = [np.sum(height_volume_combo.loc[(height_volume_combo["height"] > extent_adjust[n]) & \
                             (height_volume_combo["height"] < extent_adjust[n+1])]["volume"])
                         for n in range(y_bins-1)]
        
        sphericity_dist = [height_volume_combo.loc[(height_volume_combo["height"] > extent_adjust[n]) & \
                             (height_volume_combo["height"] < extent_adjust[n+1])]["volume"]
                         for n in range(y_bins-1)]
        
        if return_sphericity == False:
            return extent_adjust, volume_dist
        else:
            return extent_adjust, volume_dist, sphericity_dist