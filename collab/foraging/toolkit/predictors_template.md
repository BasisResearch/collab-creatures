# local_window function

FUNCTION local_window(...):
    
    Inputs: 
        foragers : (x,y) positions of foragers grouped by forager index (list of DataFrames)
        grid_size : number of grid points (int)
        sampling_rate : fraction of grid points to keep (float [0,1])
        window_size : perception radius over which predictor scores are to be calculated (int)
        random_sample : True (random sample of grid points) or False (evenly spaced sample. useful for debugging!)
        fixed_sample :
            True => grid points are sampled once and fixed in time
            False => grid points are sampled at every time step (##PP_comment : do they need to be the same across foragers? Whether to keep the grid points consistent across foragers would depend on how exactly inference is implemented. TBD)
        random_seed: for reproducibility (int)

    Returns: 
        local_windowsDF:
            DataFrame containing gridpoints to compute predictor scores over, for each forager at each time step.
            DataFrame will have 4 columns : "x", "y", "time", "forager"
            Let n_points be the number of grid points within a radius of window_size. Length of this DataFrame roughly would be (n_points * sampling_rate * num_foragers * num_frames), barring any edge cases.

        local_windows : 
            List (length: num_foragers) of DataFrames (length : n_points * sampling_rate * num_frames, columns: "x","y","time","forager") grouped by forager index


    Psuedocode implementation:
        #set random seed

        #initialize a common grid
        grid = get_grid(grid_size, sampling_rate, random_sample) #a function that first generates a DataFrame of grid points and then subsamples from it either randomly or evenly depending on value of random_sample

        local_windows = []
        for f in range(num_foragers): 
            local_windows_f = []
            for t in range(num_frames):
                #choose grid
                if fixed_sample:
                    g = grid.copy()
                
                else:
                    g = get_grid(...) ##PP_comment : if we want to keep grid_points fixed across foragers, will have to reorder some of these code chunks

                #calculate distance of all grid_points to the current position of forager f
                ...
                #select grid_points with distance < window_size
                ...
                #append DataFrame of selected grid_points to local_windows_f
                ...

            #concatenate local_windows_f and append to local_windows
        
        local_windowsDF = concatentate local_windows
        return local_windows, local_windowsDF



            

                









# General template for calculating a particular predictor
FUNCTION derive_predictor_X (foragersDF, grid, params)



