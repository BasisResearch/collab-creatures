# Design of local_window function

def local_window(...):
    
    Inputs: 
        foragers_object : data object (result of simulation or from experiments)
        ##PP comment : I think it is cleaner & more robust to pass the data object (the attributes of which can be accessed in the function, eg: foragers_object.foragerDF, foragers_object.grid_size) unless there are any specific objections?
        sampling_rate : fraction of grid points to keep (float [0,1])
        window_size : radius over which predictor scores are to be calculated (int)
        random_sample : True (random sample of grid points) or False (evenly spaced sample. useful for debugging!)
        fixed_sample :
            True => grid points are sampled once and fixed in time
            False => grid points are sampled at every time step (##PP_comment : do they need to be the same across foragers? Whether to keep the grid points consistent across foragers would depend on how exactly inference is implemented. TBD)
        random_seed: for reproducibility (int)
        ##PP_comment: construct_visibility() takes additional arguments start,end,time_shift. what is the use case for these arguments, and is it important to include them in this function?

    Returns: 
        local_windowsDF:
            DataFrame containing gridpoints to compute predictor scores over, for each forager at each time step.
            DataFrame will have 4 columns : "x", "y", "time", "forager"
            Let n_points be the number of grid points within a radius of window_size. Length of this DataFrame roughly would be (n_points * sampling_rate * num_foragers * num_frames), barring any edge cases.

        local_windows : 
            List grouped by forager index (length: num_foragers). Each element of the list is a list of num_frames DataFrames (each DataFrame has length : n_points * sampling_rate, columns: "x","y","time","forager") 

        ##PPcomment: saw that all predictor functions return data in both formats. Should we choose one or keep both?

    Psuedocode implementation:
        #set random seed
        ...

        #grab relevant parameters, e.g
        foragers = foragers_object.foragers
        grid_size = foragers_object.grid_size
        ...

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

                #calculate distance of points in g to the current position of forager f
                ...
                #select grid points with distance < window_size
                ...
                #append DataFrame of selected grid points to local_windows_f
                ...

            #append local_windows_f to local_windows
        
        # compute local_windowsDF by concatenating local_windows
        return local_windows, local_windowsDF


# Template for calculating a general predictor
def generate_predictor_X (...):
    Specifications:
        - Function needs to return predictors for all frames in the data. For edge cases where certain quanities don't exist (i.e. t=0 or t=-1), function must return nan values for the predictors
        - Function should be able to handle nan values in data (arising due to tracking errors in experiments) and return nan values for the corresponding frames   

    Inputs:
        foragers_object : data object (from simulation or experiments)
        local_windows: list of DataFrames (grouped by forager index and frames), each containing grid points to calculate predictors over
        interaction_length : 
            radius of influence if predictor depends on the state of other foragers. Defaults to window_size, but it is useful to keep it separate for clarity and special cases (int)
        params: other parameters specific to the predictor

    Returns:
        predictor_X_DF : 
            DataFrame (of the same shape as local_windowsDF) containing a column with predictor values for each grid point
        predictor_X : 
            List of DataFrames (grouped by forager index and frames) containing predictor values for each grid points (same structure as local_windows)


    Psuedocode implementation: 
        #compute any secondary attributes of the data object if necessary (eg. velocity)
        add_quantity_X_to_object(...)

        #initialize output variable
        predictor_X = local_windows.copy()

        for f in range(num_foragers):
            for t in range(num_frames):
                #add a column for predictor value
                predictor_X[f][t]["predictor_X"] = 0 

                #if predictor depends on the state of other foragers, identify which foragers can influence
                    #find forager_to_forager_distances at time t 
                    ...
                    #find index of foragers with distance < interation_length
                    ...
                    ##PP_comment : we could use existing function filter_by_visibility() here but that combines visibility w/ rewards so might be better to separate these and define new functions for this purpose

                #grab relevant state variables from selected foragers (e.g., distance, velocity etc)
                ...

                #additively combine predictor values corresponding to each selected forager
                for f_i in selected_foragers:
                    predictor_X[f][t]["predictor_X"] += predictor_X_calculator(...)
                    ##PP_comments:
                        - predictor_X_calculator(...) takes in grid point locations (predictor_X[f][t]["x", "y"]), forager f variables, relevant forager f_i variables (+ other params) and calculates the value of the predictor at every grid point based on the chosen functional form of the predictor
                        - this function must be able to handle nan values of the primary and derived quantites (x, y, v, etc)

                #normalize predictor values if needed
                ...

        #generate predictor_X_DF by concatenating predictor_X
        ...

        return predictor_X, predictor_X_DF

        

                    
                    





