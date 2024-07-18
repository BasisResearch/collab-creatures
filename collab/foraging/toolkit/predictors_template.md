##PP_comment : should we continue to name everything wrt foraging? (eg, foragerDF, forager_object, etc.)

# General specifications for nan handling 

Two reasons why nans may arise: 

- Real experimental data can have missing values for animal positions either due to tracking errors (but animal still in frame) or due to animals leaving the frame
- Derived quantities (such as velocity, how_far_score) that depend on past/future positions may not be defined for edge cases, leading to nans for predictors that depend on them 

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

# Design of generate_local_window function

def generate_local_window(...):
    
    Inputs: 
        foragers_object : data object (result of simulation or from experiments)
        ##PP_comment : I think it is cleaner & more robust to pass the data object (the attributes of which can be accessed in the function, eg: foragers_object.foragerDF, foragers_object.grid_size) unless there are any specific objections?
        ##PP_comment : Can use functools.singledispatch to allow function to take both kinds of inputs -- might be an overkill though
        sampling_fraction : fraction of grid points to keep (float [0,1]) ##EM_comment : update var name. **RESOLVED**
        window_size : radius over which predictor scores are to be calculated (int)
        random_sample : True (random sample of grid points) or False (evenly spaced sample. useful for debugging!)
        ##EM_comment : keep sample fixed in time always **RESOLVED** by removing fixed_sample argument
        random_seed: for reproducibility (int)
        ##PP_comment: construct_visibility() takes additional arguments start,end,time_shift. what is the use case for these arguments, and is it important to include them in this function? **RESOLVED** see below
        ##RU/EM_comment : have a separate function to first crop data object and pass to generate_all_predictors. if it gets very annoying w backward compatibility -- revisit.
        drop_all_missing_frames : False (only drop frames of missing forager) True (drop frames for all foragers) 
        constraint: Optional function to model inaccessible points in grid (for eg, tank boundaries). Function returns True for accessible points

    Returns: 
        local_windows : 
            List grouped by forager index (length: num_foragers). Each element of the list is a list of num_frames DataFrames (each DataFrame has length : (number of points within window_size) * sampling_fraction (barring edge cases!), columns: "x","y","time","forager") containing grid points to compute predictor scores over  

        ##PPcomment: saw that all predictor functions return data in list and flattened DataFrame formats. Should we choose one or keep both?
        ##RU/EM_comment : only output local_windows. make sure no existing functions need the DF. **RESOLVED** by only returning lists from all functions

    Psuedocode implementation:
        #grab relevant parameters, e.g
        foragers = foragers_object.foragers
        grid_size = foragers_object.grid_size
        ...

        #initialize a common grid
        ##RU/EM_comment : pass a constraint function f(x,y) to model inaccessible points in the grid. find eligible points BEFORE subsample **RESOLVED** see implementation below
        grid = get_grid(grid_size, sampling_fraction, random_sample, random_seed,constraint) #a function that first generates a DataFrame of grid points and then subsamples from it either randomly or evenly depending on value of random_sample

        ##RU/EM_comment : nan handling!! Empty local_window for missing frames. Raise warning to preprocess? (also at point of object creation) **RESOLVED** w/ two types of behavior depending on drop_all_missing_frames
    
        #identify time_points where any forager is missing
        nan_time_points_all = []
        if drop_all_missing_frames:
            nan_time_points_all=foragersDF["time"][foragersDF["x"].isna()].unique().to_list()

        local_windows = []
        for f in range(num_foragers): 
            #initialize local_windows_f to None
            local_windows_f = [None for _ in range(num_frames)]

            #find time points where forager's positional data is missing 
            nan_time_points_f=foragers[f]["time"][foragers[f]["x"].isna()].to_list()

            #find frames for which local windows need to be computed
            compute_frames = (set(range(num_frames)) - set(nan_time_points_f))-set(nan_time_points_all)
            
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

# Design of get_grid function 
def get_grid(...):

    Inputs:
        grid_size: size of grid (int)
        sampling_fraction: fraction of grid points to keep (float [0,1]) 
        random_sample: True (sample grid points randomly) or False (sample evenly)
        random_seed: for reproducibility
        constraint: Optional function to model inaccessible points in grid (for eg, tank boundaries). Function returns True for accessible points

    Returns:
        grid : DataFrame with 2 columns "x", "y" of selected grid points where predictors can be calculated 

    Psuedocode implementation: 
    #generate grid of all points
    grid = list(product(range(1, grid_size+ 1), repeat=2)) 
    grid =pd.DataFrame(grid, columns=["x", "y"])

    #only keep accessible points
    if constraint is not None: 
        grid = grid[constraint(grid["x"],grid["y])]
    
    #example of a constraint function - for a circular tank centered at the grid center
    #    def constraint(x,y,grid_size):
    #        return (x-grid_size/2 - 0.5)**2 + (y-grid_size/2-0.5)**2 < grid_size**2
    
    #subsample the grid
    if random_sample:
        grid = grid.sample(frac=sampling_fraction,random_state=random_seed) ##PP_comment: there are ways to get random samples that cover the space evenly (for eg: stratified sampling, Poisson disk, etc) do we want to implement that?
    else:
        drop = np.floor(np.sqrt(1/(sampling_fraction))) ##PP_comment: the need to convert to int severely limits the range of the true sampling_fraction. Probably doesn't matter since we are only using this for debugging
        ind = grid["x"]%drop==0 & grid["y"]%drop==0
        grid = grid[ind]
    return grid

# Template for calculating a general predictor
##PP_comment : do we want to enforce this template using abstract classes?

def generate_predictor_X (...):

    Specifications:
        - When a forager is missing, i.e. local_windows[f][t]=[], ensure predictor_X[f][t]=[]
        - Elements of predictor_X[f][t] are nans when derived quanties are not defined   
        ##RU/EM_comment : if local_window is empty -- return empty element! **RESOLVED** handling of missing data follows local_windows

    Inputs:
        foragers_object : data object (from simulation or experiments)
        local_windows: list of DataFrames (grouped by forager index and frames), each containing grid points to calculate predictors over
        interaction_length : 
            radius of influence if predictor depends on the state of other foragers. Defaults to window_size, but it is useful to keep it separate for clarity and special cases (int)
        params: other parameters specific to the predictor
        ##PP_comment : in generate_all_predictors I am saving specified predictor values to the foragers_object before calling generate_predictor_X, so potentially these parameters can just be accessed from foragers_object, and don't need to be passed separately to the function 

    Returns:
        predictor_X : 
            List of DataFrames (grouped by forager index and frames) containing predictor values for all grid points (same structure as local_windows)


    Psuedocode implementation: 
        #compute any secondary attributes of the data object if necessary (eg. velocity)
        add_quantity_X_to_object(...)

        #initialize output variable
        predictor_X = local_windows.copy()

        for f in range(num_foragers):
            for t in range(num_frames):
                #calculate predictor scores if grid is not empty
                if predictor_X[f][t] is not None:
                    #add a column for predictor value
                    predictor_X[f][t]["predictor_X"] = 0 

                    #if predictor depends on the state of other foragers, identify which foragers can influence
                        #find forager_to_forager_distances at time t 
                        ...
                        #find index of foragers with distance < interation_length (ignore foragers with missing positional data)
                        ...
                        ##PP_comment : we could use existing function filter_by_visibility() here but that combines visibility w/ rewards so might be better to separate these and define new functions for this purpose

                    #additively combine predictor values corresponding to each selected forager
                    for f_i in selected_foragers:
                        predictor_X[f][t]["predictor_X"] += predictor_X_calculator(...)
                        ##PP_comments:
                            - predictor_X_calculator(...) takes in grid point locations (predictor_X[f][t]["x", "y"]), forager f variables, relevant forager f_i variables (+ other params) and calculates the value of the predictor at every grid point based on the chosen functional form of the predictor
                            - this function must return nan values when derived quantities (eg. v) are not defined

                    #normalize predictor values if needed
                    ...

        return predictor_X

# Design of generate_all_predictors function

def generate_all_predictors(...):

    Specifications:
        - The function calculates all predictors as specified in "predictors" by calling individual generate_predictor_X() functions
        - The outputs of every generate_predictor_X() call is added as an attribute to the foragers_object 
        - ##PP_comment: What should this function return? forager_object is modified in place so it is not necessary to return it. Can return a combined predictorsDF? Both?

    Inputs:
        foragers_object : data object (from simulation or experiments)
        predictors : list of strings, e.g ["visibility", "proximity", "rewards"] 
        # arguments for local_window
            sampling_fraction
            window_size 
            random_sample  
            random_seed
            drop_all_missing_frames
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

        #repeated code chunks to compute each predictor if selected
        if "predictor_X" in predictors:
            #save specified parameter values as attributes of the foragers_object
            ...

            #calculate predictor value
            predictor_X = generate_predictor_X(foragers_object, local_windows, ...)

            #add outputs to foragers_object
            foragers_object.predictor_X = predictor_X

            #add to all_predictors_list
            list_predictors.append(predictor_X)

        #generate combined_predictorDF
        combined_predictorDF = generate_combined_predictorDF(list_predictors,dropna)

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

            

                    
                    






