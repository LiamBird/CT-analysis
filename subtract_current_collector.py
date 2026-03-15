import open3d as o3d
import numpy as np
import pandas as pd
from tqdm import notebook

def subtract_current_collector_ply(self, filename, thickness_steps=20,
                                   lateral_steps=50, cc_thickness_steps=20):
 

    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points)
    df = pd.DataFrame(points, columns=["x", "y", "z"])
    
    unique_x, unique_counts = np.unique(df["x"], return_counts=True)
    x_grid = unique_x[unique_counts>5]

    unique_y, unique_counts = np.unique(df["y"], return_counts=True)
    y_grid = unique_y[unique_counts>5]

    unique_z, unique_counts = np.unique(df["z"], return_counts=True)
    z_grid = unique_z[unique_counts>5]

    coord_ranges = dict([(coord, df[coord].max()-df[coord].min())
                         for coord in ["x", "y", "z"]])
    lateral_axes = [[*coord_ranges.keys()][idx] for idx in np.argsort([*coord_ranges.values()])[1:]]
    height_axis = [*coord_ranges.keys()][np.argsort([*coord_ranges.values()])[0]]

    axis_grids = {}
    for coord in lateral_axes:
        if lateral_steps == None:
            unique_values, unique_counts = np.unique(df[coord], return_counts=True)
            axis_grids.update([(coord, unique_values[unique_counts>5])])

        else:
            lateral_min, lateral_max = [df[coord].min(), df[coord].max()]
            axis_grids.update([(coord, np.linspace(lateral_min, lateral_max, lateral_steps))])

    if cc_thickness_steps == None:
        unique_values, unique_counts = np.unique(df[height_axis], return_counts=True)
        axis_grids.update([(height_axis, unique_values[unique_counts>5])])
    else:
        thickness_min, thickness_max = [df[height_axis].min(), df[height_axis].max()]
        axis_grids.update([(height_axis, np.linspace(thickness_min, thickness_max, cc_thickness_steps))])
        
    u_name, v_name = lateral_axes

    points_grid = np.full((axis_grids[u_name].shape[0]-1, axis_grids[v_name].shape[0]-1), np.nan, dtype=float)

    for nui, ui in enumerate(axis_grids[u_name][:-1]):
        for nvi, vi in enumerate(axis_grids[v_name][:-1]):
            points_grid[nui, nvi] = np.nanmedian(df.loc[(df[u_name]>=axis_grids[u_name][nui]) & (df[u_name]<axis_grids[u_name][nui+1])
                                                      & (df[v_name]>=axis_grids[v_name][nvi]) & (df[v_name]<axis_grids[v_name][nvi+1])].drop_duplicates(subset=lateral_axes)[height_axis])
        
    column_u = "BaryCenter"+str.capitalize(u_name)
    column_v = "BaryCenter"+str.capitalize(v_name)
    column_h = "BaryCenter"+str.capitalize(height_axis)
    
    if type(thickness_steps) == int or type(thickness_steps) == float:
        sample_thickness_grid = np.linspace(self.df[column_h].min(),
                                    self.df[column_h].max(),
                                    thickness_steps)
    else:
        sample_thickness_grid = thickness_steps
    
    particles_grid = np.full((axis_grids[u_name].shape[0]-1, 
                          axis_grids[v_name].shape[0]-1,
                          sample_thickness_grid.shape[0]-1), np.nan)



    for nui, ui in notebook.tqdm(enumerate(axis_grids[u_name][:-1])):
        for nvi, vi in enumerate(axis_grids[v_name][:-1]):
            selected_particles = self.df.loc[(self.df[column_u]>=axis_grids[u_name][nui]) &
                                                   (self.df[column_u]< axis_grids[u_name][nui+1]) &
                                                   (self.df[column_v]>=axis_grids[v_name][nvi]) &
                                                   (self.df[column_v]< axis_grids[v_name][nvi+1])]

            for nhi, hi in enumerate(sample_thickness_grid[:-1]):
                particles_grid[nui, nvi, nhi] = np.sum(selected_particles.loc[(selected_particles[column_h]>=sample_thickness_grid[nhi]) & 
                                                                              (selected_particles[column_h]< sample_thickness_grid[nhi+1])]["Volume3d"])

    volume_at_height = np.sum(np.sum(particles_grid, axis=1), axis=0)
    return sample_thickness_grid, volume_at_height