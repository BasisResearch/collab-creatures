# Rewrite of object_from_data function as a class instantiation 

class dataObject:
    def __init__(self, foragersDF, grid_size=None, rewardsDF=None, frames=None):
        if frames is None:
            frames = foragersDF["time"].nunique()

        if grid_size is None:
            grid_size = int(max(max(foragersDF["x"]), max(foragersDF["y"])))

        self.grid_size = grid_size
        self.num_frames = frames
        self.foragersDF = foragersDF
        if self.foragersDF["forager"].min() == 0:
            self.foragersDF["forager"] = self.foragersDF["forager"] + 1

        self.foragers = [group for _, group in foragersDF.groupby("forager")]

        if rewardsDF is not None:
            self.rewardsDF = rewardsDF
            self.rewards = [group for _, group in rewardsDF.groupby("time")]

        self.num_foragers = len(sim.foragers)

    def calculate_step_size_max(self):
        step_maxes = []

        for b in range(len(self.foragers)):
            df = self.foragers[b]
            step_maxes.append(
                max(
                    max(
                        [
                            abs(df["x"].iloc[t + 1] - df["x"].iloc[t])
                            for t in range(len(df) - 1)
                        ]
                    ),
                    max(
                        [
                            abs(df["y"].iloc[t + 1] - df["y"].iloc[t])
                            for t in range(len(df) - 1)
                        ]
                    ),
                )
            )
        self.step_size_max = max(step_maxes)

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
    params = foragers_object.local_windows_kwargs

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
        interaction_constraint: 
            (Optional) function that takes a list of foragers within interaction length and applies additional constraints on it 
        interaction_constraint_params :
            Dictionary of parameters for constraint function 
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
                    interaction_partners = filter_by_interaction_distance(foragersDF, f, t, interaction_length,interaction_constraint, interaction_constraint_params)
                      
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
    params = foragers_object.predictor_kwargs[predictor_ID] #this returns a dictionary

    quantity_X, predictor_X = _generate_predictor_X(foragers=foragers_object.foragers, foragersDF=foragers_object.foragersDF, local_windows=foragers_object.local_windows, predictor_ID=predictor_ID, **params)

    #add quantity_X to foragers_object
    ...

    return predictor_X

##
def filter_by_interaction_distance(foragersDF, f, t, interaction_length, interaction_constraint=None, interaction_constraint_params: dict = None):
    positions = foragersDF[foragersDF["time"]==t].copy()
    distances = (positions["x"] - positions["x"][f])**2 + (positions["y"] - positions["y"][f])**2
    distances[f] = np.nan
    forager_ind = positions.loc[distances<interaction_length**2, "forager"].tolist()
    if interaction_constraint is not None:
        foragers_ind = interaction_constraint(forager_ind, f, t, foragersDF, **interaction_constraint_parameters)
    return foragers_ind

# Design of generate_all_predictors function

def generate_all_predictors(...):

    Inputs:
        foragers_object : data object (from simulation or experiments)
        local_windows_kwargs : dictionary of keyword arguments for local_windows 
            local_windows_kwargs = {"sampling_fraction":5, "window_size":30, ...}
        predictor_kwargs : nested dictionary of keyword arguments for all the predictors to be computed. we can have multiple versions of the same predictor (with different parameters) by using underscores in the naming. i.e.  ["visibility", "proximity_10", "proximity_20_2"]. 
        !!! The part of the string before the first underscore indicates the predictor type !!!
            predictor_kwargs = {
                "proximity_10" : {"optimal_dist":10, "decay":1, ...},
                "proximity_20" : {"optimal_dist":20, "decay":2, ...},
                "proximity_w_constraint" : {...,"interaction_constraint" : constraint_function, "interaction_constraint_params": {"rewardsDF" : foragers_object.rewardsDF,...}}
            }
        dropna : True (filter out rows with nans in combined_predictorDF) or False (keep nans)
        add_scaled_scores :  True (include scaled values of predictors) or False 

    Returns:
        foragers_object : modified foragers_object which contains all computed predictors in a dictionary
        combined_predictorDF : a combined DataFrame containing all computed predictor values for each forager and time step at all selected grid points

    Psuedocode implementation: 
        #save chosen parameters to object
        foragers_object.local_windows_kwargs = local_windows_kwargs
        foragers_object.predictor_kwargs = predictor_kwargs
        
        #generate local_windows and add to object
        local_windows = generate_local_windows(foragers_object)
        foragers_object.local_windows = local_windows

        computed_predictors = {}

        for predictor_ID in predictor_kwargs.keys():
            predictor_type = predictor_ID.split('_')[0]
            function_name = f"generate_{predictor_type}_predictor"

            #How to fetch the correct generating function depends on how functions are organised into files / imported. Eg, if we save all generate_predictor_X() functions in a single file generate_predictors.py and import that as "gen_pred" into the file with derive predictors
            generate_predictor_function = getattr(gen_pred,function_name)

            computed_predictors[predictor_ID] = generate_predictor_function(foragers_object, predictor_ID)
        
        #save computed_predictors to object
        foragers_object.predictors = computed_predictors

        #generate combined_predictorDF
        combined_predictorDF = generate_combined_predictorDF(computed_predictors,dropna,add_scaled_scores)

        #save to object
        foragers_object.combined_predictorDF = combined_predictorDF

        return foragers_object, combined_predictorDF

# Design of DataFrame merging functions

def generate_DF_from_predictor(predictor_X):

    return pd.concat([pd.concat(p, axis=0) for p in predictor_X], axis=0) #this automatically ignores None elements! 
        
def generate_combined_predictorDF(dict_predictors,dropna,add_scaled_scores):

    list_predictorDFs = [generate_DF_from_predictor(p) for p in dict_predictors.values()]
    combined_predictorDF = list_predictorDFs[0]

    for i in range(1,len(list_predictorDFs)):
        combined_predictorDF = combined_predictorDF.merge(list_predictorDFs[i],how='inner')

    if dropna:
        combined_predictorDF.dropna()

    #scale predictor columns
    if add_scaled_scores:
        for predictor_ID in dict_predictors.keys():
            column_min = np.nanmin(combined_predictorDF[predictor_ID])
            column_max = np.nanmax(combined_predictorDF[predictor_ID])
            combined_predictorDF[f"{predictor_ID}_scaled"] = (combined_predictorDF[predictor_ID] - column_min)/(column_max - column_min)

    return combined_predictorDF

            

                    
                    






