# General plan for generate_predictor functions:

    - First define underscored versions of functions (_generate_predictor_X(..)) which take in specific attributes of foragers_object, and parameters to compute predictors
    - Then write wrapper functions generate_predictor_X(..) that only take the foragers_object, access relevant quantities from it, and then call _generate_predictor_X(..) 
    - Both of these functions will take an additional argument "predictor_ID" which decides what name to save the predictor using (useful for the case when we want to simulataneously test the same predictor with different parameters, eg: "velocity_10", "velocity_20", etc )

PROBLEMS WITH THIS PLAN:

    - _generate_local_windows(...) should still be given a foragers_object, so it can access grid_size, etc. from the object as we need to enforce that the grid_size is the same as that used to create the foragers_object otherwise results don't make sense (don't have grid_size be an exposed parameter). It also needs both the DF and foragers object for computations (maybe it's okay that we have to pass both?)
    - for predictors that require computation of secondary quantities like velocity, we cannot add these quantities to the object unless it is passed to the function (but maybe we don't need to? or alternately, hidden function can return both predictor and derived quantity. exposed function adds derived quantity to foragers_object)
    - We cannot have a single "filter_by_interaction_distance" function that takes in a general constraint that works for both cases like foragers on reward, or of specific age, because such functions would need additional attributes of the foragers_object (such as rewardsDF) that are not originally passed to _generate_predictor_X(..). So one option is to have different "filter_by" functions for different cases, or only have generate_predictor_X(...).

# General specifications for nan handling 

Two reasons why nans may arise: 

- Real experimental data can have missing values for animal positions either due to tracking errors (but animal still in frame) or due to animals leaving the frame
- Derived quantities (such as velocity, how_far_score) that depend on past/future positions may not be defined for edge cases, leading to nans for predictors that depend on them. 

Code design plan to work with nans:

- object_from_data(...) raises a warning if there are nan values in data 
- generate_local_windows(...): 
    - Default behavior: set local_windows[f][t]=None for all timepoints t that forager f's location is missing & raise warning 
    - Optional behavior: set local_windows[:][t]=None, i.e insert a None element for *all* foragers when *any* forager is missing
- Handling of missing data while calculating predictor values follows the lead of generate_local_windows(...):
    - add_quantity_X_to_data(...) adds nan values both for frames when positional data is missing and when derived quantities are not defined
    - If local_windows[f][t] is None (indicates that predictor values should not be calculated):
        - generate_predictor_X(...) returns an empty element in predictor_X for corresponding frame (i.e. predictor[f][t] = None )
    - If local_windows[f][t] is not empty:
        - identify other foragers that influence predictor values - here, only consider foragers whose positional data exists (even if their derived quantities are nans)
        - predictor_X_calculator(...) returns nans if relevant derived quantities of focal or influential foragers are not defined
- generate_DF_from_predictor(...) throws away all empty elements in predictor_X (but keeps DFs with nans)
- generate_combined_predictorDF(...) : generates combined DF and (optional) deletes all rows where *any* predictor values are nan for inference 

# Design of local_windows (& related) functions

def _generate_local_windows(...):
    
    Inputs: 
        foragers : list of positional DFs for each forager
        foragersDF : flattened positional DF for all foragers
        grid_size : size of grid used to rescale forager positional data (int)
        sampling_fraction : fraction of grid points to keep (float [0,1]) 
        window_size : radius over which predictor scores are to be calculated (int)
        random_seed: for reproducibility (int)
        skip_incomplete_frames : False (only skip frames of the missing forager) or True (skip frames for all foragers) 
        grid_constraint: Optional function to model inaccessible points in grid (for eg, tank boundaries). Function returns True for accessible points

    Returns: 
        local_windows : 
            List grouped by forager index (length: num_foragers). Each element of the list is a list of num_frames DataFrames (each DataFrame has length : (number of points within window_size) * sampling_fraction (barring edge cases!), columns: "x","y","time","forager") containing grid points to compute predictor scores over  

    Psuedocode implementation:

        #initialize a common grid
        grid = get_grid(grid_size, sampling_fraction, random_seed, grid_constraint) #a function that first generates a DataFrame of valid grid points and then subsamples from it randomly 
    
        #identify time_points where any forager is missing
        missing_time_points_all = []
        if skip_incomplete_frames:
            missing_time_points_all=foragersDF["time"][foragersDF["x"].isna()].unique().to_list()

        local_windows = []
        for f in range(num_foragers): 
            #initialize local_windows_f to None
            local_windows_f = [None for _ in range(num_frames)]

            #find time points where forager's positional data is missing 
            missing_time_points_f=foragers[f]["time"][foragers[f]["x"].isna()].to_list()

            #find frames for which local windows need to be computed
            compute_frames = (set(range(num_frames)) - set(missing_time_points_f))-set(missing_time_points_all)
            
            for t in compute_frames:
                    #copy grid
                    g = grid.copy()

                    #calculate distance of points in g to the current position of forager f
                    ...

                    #select grid points with distance < window_size
                    ...

                    #add forager and time info to the DF
                    ...

                    #update the corresponding element of local_windows_f to DF with computed grid points
                    local_windows_f[t] = computed_grid_points

            #add local_windows_f to local_windows
            local_windows.append(local_windows_f)
        
        return local_windows

##
def generate_local_windows(foragers_object):

    #grab parameters specific to local_windows
    params = foragers_object.predictor_params["local_windows"] #this returns a dictionary

    #call hidden function with keyword arguments
    local_windows =  _generate_local_windows(foragers=foragers_object.foragers, foragersDF=foragers_object.foragersDF, grid_size=foragers_object.grid_size, **params)

    return local_windows


##
def get_grid(...):

    Inputs:
        grid_size: size of grid (int)
        sampling_fraction: fraction of grid points to keep (float [0,1]) 
        random_seed: for reproducibility
        grid_constraint: Optional function to model inaccessible points in grid (for eg, tank boundaries). Function returns True for accessible points

    Returns:
        grid : DataFrame with 2 columns "x", "y" of selected grid points where predictors can be calculated 

    Psuedocode implementation: 
    #generate grid of all points
    grid = list(product(range(1, grid_size+ 1), repeat=2)) 
    grid =pd.DataFrame(grid, columns=["x", "y"])

    #only keep accessible points
    if grid_constraint is not None: 
        grid = grid[grid_constraint(grid["x"],grid["y])]
    
    #example of a constraint function - for a circular tank centered at the grid center
    #    def constraint(x,y,grid_size):
    #        return (x-grid_size/2 - 0.5)**2 + (y-grid_size/2-0.5)**2 < grid_size**2
    
    #subsample the grid
    grid = grid.sample(frac=sampling_fraction,random_state=random_seed)

    return grid.sort_index()

# Template for _generate_predictor_X (& related) functions: 

def _generate_predictor_X (...):

    Inputs:
        foragers : list of positional DFs of each forager 
        foragersDF : flattened DF of positions for each forager
        local_windows: list of DataFrames (grouped by forager index and frames), each containing grid points to calculate predictors over
        predictor_ID : name to be used for the predictor
        interaction_length : 
            radius of influence if predictor depends on the state of other foragers. Defaults to window_size, but it is useful to keep it separate for clarity and special cases (int)
        params: other parameters specific to the predictor

    Returns:
        predictor_X : 
            List of DataFrames (grouped by forager index and frames) containing predictor values for all grid points (same structure as local_windows)
        quantity_X:
            Derived quantities (datatype TBD)

    Psuedocode implementation: 
        #compute any secondary attributes of the data object if necessary (eg. velocity)
        if "quantity_X" not in foragers[0].columns:
            quantity_X = compute_quantity_X(...)

        #initialize output variable
        predictor_X = local_windows.copy()

        for f in range(num_foragers):
            for t in range(num_frames):
                #calculate predictor scores if grid is not empty
                if predictor_X[f][t] is not None:
                    #add a column for predictor value
                    predictor_X[f][t][predictor_ID] = 0 

                    #if predictor depends on the state of other foragers, identify which foragers can influence
                    interaction_partners = filter_by_interaction_distance(foragersDF, f, t, interaction_length)
                      
                    #additively combine predictor values corresponding to each selected forager
                    for f_i in interaction_partners:
                        predictor_X[f][t][predictor_ID] += predictor_X_pairwise_calculator(...)
                        ##PP_comments:
                            - predictor_X_calculator(...) takes in grid point locations (predictor_X[f][t]["x", "y"]), forager f variables, relevant forager f_i variables (+ other params) and calculates the value of the predictor at every grid point based on the chosen functional form of the predictor
                            - treat divide-by-zeros on a case-to-case basis 
                            - this function must return nan values when derived quantities (eg. v) are not defined

                    #normalize predictor values if needed. (z score it over grid points) 
                    ...

        return quantity_X, predictor_X

##
def generate_predictor_X(foragers_object, predictor_ID):

    #grab relevant parameters
    params = foragers_object.predictor_params[predictor_ID] #this returns a dictionary

    quantity_X, predictor_X = _generate_predictor_X(foragers=foragers_object.foragers, foragersDF=foragers_object.foragersDF, local_windows=foragers_object.local_windows, predictor_ID=predictor_ID, **params)

    #add quantity_X to foragers_object
    ...

    return predictor_X

##
def filter_by_interaction_distance(foragersDF, f, t, interaction_length):

    positions = foragersDF[foragersDF["time"]==t].copy()
    distances = (positions["x"] - positions["x"][f])**2 + (positions["y"] - positions["y"][f])**2
    distances[f] = np.nan
    forager_ind = positions.loc[distances<interaction_length**2, "forager"].tolist()

    return forager_ind

# Design of generate_all_predictors function

def generate_all_predictors(...):

##RU_comment : this should also scale all the predictors (across foragers and time), and add that as a column in combined_predictorsDF 
##RU_comment : this a dict of dicts ("function_kwargs") as arguments instead of individual params
##RU_comment : instead of passing predictors, we infer which predictors to compute from function_kwargs. this can have multiple runs of the same predictor with different parameters (convention "predictorX_*" to name the different versions )

    Specifications:
        - The function calculates all predictors as specified in "predictors" by calling individual generate_predictor_X() functions
        - The outputs of every generate_predictor_X() call is added as an attribute to the foragers_object 
        - ##PP_comment: What should this function return? forager_object is modified in place so it is not necessary to return it. Can return a combined predictorsDF? Both?

    Inputs:
        foragers_object : data object (from simulation or experiments)
        ##EM_comment : modifying in place might break if code is run in parallel?
        predictors : list of strings, e.g ["visibility", "proximity", "rewards"] 
        # arguments for local_window
            sampling_fraction
            window_size 
            random_sample  
            random_seed
            drop_all_missing_frames
            constraint
        # arguments for each predictor type, e.g.:
            proximity_preferred_distance
            proximity_decay
            ...
            ##PP_comment: as the number of predictors increase, it will be hard to keep track of all the parameters, so can establish a convention that names of parameters specific to a particular predictor start with a predictor identifier 
            ##PP_comment: I need to understand what exactly time_shift is doing and where to implement it [potentially just need to implement it in local_windows] **RESOLVED** decided to not implement time_shift 
            dropna : True (filter out rows with nans in combined_predictorDF) or False (keep nans)
    Returns:
        foragers_object : modified foragers_object which contains all computed predictors as attributes
        combined_predictorDF : a combined DataFrame containing all computed predictor values for each forager and time step at all selected grid points, with nans filtered out 

    Psuedocode implementation: 
        #save local_windows parameter values as attributes of the foragers_object, e.g. 
        foragers_object.window_size = window_size
        ...

        #generate local_windows
        local_windows = generate_local_windows(...)

        #add output to foragers_object
        foragers_object.local_windows = local_windows

        list_predictors = []

        ##EM_comment : pick function given the regular expression, so users don't need to modify derive predictors
        #repeated code chunks to compute each predictor if selected
        if "predictor_X" in predictors:
            #save specified parameter values as attributes of the foragers_object
            ...

            #calculate predictor value
            predictor_X = generate_predictor_X(foragers_object, local_windows, ...)

            #add outputs to foragers_object
            foragers_object.predictor_X = predictor_X
            ## RU_comment : save predictors as a dictionary!
                foragers_object.predictors["velocity_10"] = predictor_X
            #add to all_predictors_list
            list_predictors.append(predictor_X)

        #generate combined_predictorDF
        combined_predictorDF = generate_combined_predictorDF(list_predictors,dropna)

        ##RU_comment : also add scaled columns for predictors 

        #save to object
        foragers_object.combined_predictorDF = combined_predictorDF

        return foragers_object, combined_predictorDF

# Design of DataFrame merging functions

def generate_DF_from_predictor(predictor_X):

    return pd.concat([pd.concat(p, axis=0) for p in predictor_X], axis=0) #this automatically ignores None elements! 
        
def generate_combined_predictorDF(list_predictors,dropna):

    list_predictorDFs = [generate_DF_from_predictor(p) for p in list_predictors]
    combined_predictorDF = list_predictorDFs[0]

    for i in range(1,len(list_predictorDFs)):
        combined_predictorDF = combined_predictorDF.merge(list_predictorDFs[i],how='inner')

    if dropna:
        combined_predictorDF.dropna()

    return combined_predictorDF

            

                    
                    






