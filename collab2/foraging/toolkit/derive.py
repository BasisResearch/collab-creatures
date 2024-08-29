import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from collab2.foraging import toolkit as ftk
from collab2.foraging.toolkit.local_windows import generate_local_windows
from collab2.foraging.toolkit.utils import dataObject

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
derivation_logger = logging.getLogger(__name__)


def generate_DF_from_predictor(df_list: List[List[pd.DataFrame]]) -> pd.DataFrame:
    """
    A utils function that concatenates a nested list of DataFrames into a single, flattened DataFrame.
    List elements that are `None` are automatically discarded.

    :param df_list: nested list of DataFrames, e.g., computed predictor DataFrames that are grouped
        by forager_index and time
    :return: flattened DataFrame
    """
    return pd.concat(
        [pd.concat(df, axis=0) for df in df_list], axis=0
    )  # this automatically ignores None elements!


def generate_combined_predictorDF(
    dict_predictors: Dict[str, List[List[pd.DataFrame]]],
    dropna: Optional[bool] = True,
    add_scaled_scores: Optional[bool] = False,
) -> pd.DataFrame:
    """
    A utils function that takes a dictionary of computed predictors (as nested lists of DataFrames),
    and returns a single, flattened DataFrame, containing each predictor as a column.

    :param dict_predictors: dictionary of computed predictors
    :param dropna: set to `True` to drop NaN elements from final DataFrame
    :param add_scaled_scores: set to `True` to compute scaled predictor scores
        and add them as additional columns in final DataFrame
    :return: final, flattened DataFrame containing all computed predictors as columns
    """
    list_predictorDFs = [
        generate_DF_from_predictor(p) for p in dict_predictors.values()
    ]
    combined_predictorDF = list_predictorDFs[0]

    for i in range(1, len(list_predictorDFs)):
        combined_predictorDF = combined_predictorDF.merge(
            list_predictorDFs[i], how="inner"
        )

    if dropna:
        combined_predictorDF.dropna()

    # scale predictor columns
    if add_scaled_scores:
        for predictor_ID in dict_predictors.keys():
            column_min = combined_predictorDF[predictor_ID].min()
            column_max = combined_predictorDF[predictor_ID].max()
            combined_predictorDF[f"{predictor_ID}_scaled"] = (
                combined_predictorDF[predictor_ID] - column_min
            ) / (column_max - column_min)

    return combined_predictorDF


def derive_predictors(
    foragers_object: dataObject,
    local_windows_kwargs: Dict[str, Any],
    predictor_kwargs: Dict[str, Dict[str, Any]],
    dropna: Optional[bool] = True,
    add_scaled_scores: Optional[bool] = False,
) -> pd.DataFrame:
    """
    A function that calculates a chosen set of predictors for data by inferring their names from keys in `predictor_kwargs`,
    and dynamically calling the corresponding functions.
    :param foragers_object: instance of dataObject class containing the trajectory data of foragers
    :param local_window_kwargs: dictionary of keyword arguments for `generate_local_windows` function.
    :param predictor_kwargs: nested dictionary of keyword arguments for predictors to be computed.
                        Keys of predictor_kwargs set the name of the predictor to be computed. The predictor name can have underscores,
                        however, the substring before the first underscore must correspond to the name of a predictor type in Collab.
                        Thus, we can have multiple versions of the same predictor type (with different parameters) by naming. them as follows
                        predictor_kwargs = {
                            "proximity_10" : {"optimal_dist":10, "decay":1, ...},
                            "proximity_20" : {"optimal_dist":20, "decay":2, ...},
                            "proximity_w_constraint" : {...,"interaction_constraint" : constraint_function, "interaction_constraint_params": {...}}
                        }

    :param dropna: set to `True` to drop NaN elements from final DataFrame
    :param add_scaled_scores: set to `True` to compute scaled predictor scores
        and add them as additional columns in final DataFrame
    :return: final, flattened DataFrame containing all computed predictors as columns
    """

    # save chosen parameters to object
    foragers_object.local_windows_kwargs = local_windows_kwargs
    foragers_object.predictor_kwargs = predictor_kwargs

    # generate local_windows and add to object
    local_windows = generate_local_windows(foragers_object)
    foragers_object.local_windows = local_windows

    computed_predictors = {}
    for predictor_name in predictor_kwargs.keys():
        predictor_type = predictor_name.split("_")[0]
        function_name = f"generate_{predictor_type}_predictor"
        generate_predictor_function = getattr(ftk, function_name)
        computed_predictors[predictor_name] = generate_predictor_function(
            foragers_object, predictor_name
        )
        derivation_logger.info(f"{predictor_name} completed")

    # save predictors to object
    foragers_object.predictors = computed_predictors

    # generate combined_predictorDF
    combined_predictorDF = generate_combined_predictorDF(
        computed_predictors, dropna, add_scaled_scores
    )

    # save to object
    foragers_object.combined_predictorDF = combined_predictorDF

    return combined_predictorDF
