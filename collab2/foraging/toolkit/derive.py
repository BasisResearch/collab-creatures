import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from collab2.foraging import toolkit as ftk
from collab2.foraging.toolkit.local_windows import generate_local_windows
from collab2.foraging.toolkit.utils import dataObject

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
derivation_logger = logging.getLogger(__name__)


def _generate_DF_from_nestedList(df_list: List[List[pd.DataFrame]]) -> pd.DataFrame:
    """
    A helper function that concatenates a nested list of DataFrames into a single, flattened DataFrame.
    List elements that are `None` are automatically discarded.

    :param df_list: nested list of DataFrames, e.g., computed predictor DataFrames that are grouped
        by forager_index and time
    :return: flattened DataFrame
    """
    return pd.concat(
        [pd.concat(df, axis=0) for df in df_list], axis=0
    )  # this automatically ignores None elements!


def _generate_combined_DF(
    predictors_and_scores: Dict[str, List[List[pd.DataFrame]]],
    dropna: Optional[bool] = True,
    add_scaled_values: Optional[bool] = False,
) -> pd.DataFrame:
    """
    A helper function that takes a dictionary of computed predictors/scores (as nested lists of DataFrames),
    and returns a single, flattened DataFrame, containing each predictor/score as a column.

    :param predictors_and_scoress: dictionary of computed predictors/scores
    :param dropna: set to `True` to drop NaN elements from final DataFrame
    :param add_scaled_values: set to `True` to scale the predictor/score columns and
        add the values as additional columns in final DataFrame
    :return: final, flattened DataFrame containing all computed predictors as columns
    """
    list_DFs = [_generate_DF_from_nestedList(p) for p in predictors_and_scores.values()]
    combinedDF = list_DFs[0]

    for i in range(1, len(list_DFs)):
        combinedDF = combinedDF.merge(list_DFs[i], how="inner")

    if dropna:
        combinedDF.dropna(inplace=True)

    # scale predictor columns
    if add_scaled_values:
        for key in predictors_and_scores.keys():
            column_min = combinedDF[key].min()
            column_max = combinedDF[key].max()
            combinedDF[f"{key}_scaled"] = (combinedDF[key] - column_min) / (
                column_max - column_min
            )

    return combinedDF


def derive_predictors_and_scores(
    foragers_object: dataObject,
    local_windows_kwargs: Dict[str, Any],
    predictor_kwargs: Dict[str, Dict[str, Any]],
    score_kwargs: Dict[str, Dict[str, Any]],
    dropna: Optional[bool] = True,
    add_scaled_values: Optional[bool] = False,
) -> pd.DataFrame:
    """
    A function that calculates a chosen set of predictors and scores for data by inferring their names from
    keys in `predictor_kwargs` & `score_kwargs`, and dynamically calling the corresponding functions.
    :param foragers_object: instance of dataObject class containing the trajectory data of foragers
    :param local_window_kwargs: dictionary of keyword arguments for `generate_local_windows` function.
    :param predictor_kwargs: nested dictionary of keyword arguments for predictors to be computed.
            Keys of predictor_kwargs set the name of the predictor to be computed.
            The predictor name can have underscores, however, the substring before the first underscore must correspond
            to the name of a predictor type in Collab. Thus, we can have multiple versions of the same predictor type
            (with different parameters) by naming them as follows
            predictor_kwargs = {
                "proximity_10" : {"optimal_dist":10, "decay":1, ...},
                "proximity_20" : {"optimal_dist":20, "decay":2, ...},
                "proximity_w_constraint" : {...,"interaction_constraint" : constraint_function,
                                        "interaction_constraint_params": {...}}
            }
    :param score_kwargs: nested dictionary of keyword arguments for scores to be computed.
            The substring before the first underscore in dictionary keys must correspond to the name of
            a score type in Collab, same as in `predictor_kwargs`
    :param dropna: set to `True` to drop NaN elements from final DataFrame
    :param add_scaled_values: set to `True` to compute scaled predictor scores
        and add them as additional columns in final DataFrame
    :return: final, flattened DataFrame containing all computed predictors as columns
    """

    # save chosen parameters to object
    foragers_object.local_windows_kwargs = local_windows_kwargs
    foragers_object.predictor_kwargs = predictor_kwargs
    foragers_object.score_kwargs = score_kwargs

    # generate local_windows and add to object
    local_windows = generate_local_windows(foragers_object)
    foragers_object.local_windows = local_windows

    derived_quantities = {}

    # calculate predictors
    for predictor_name in predictor_kwargs.keys():
        predictor_type = predictor_name.split("_")[0]
        function_name = f"generate_{predictor_type}_predictor"
        generate_function = getattr(ftk, function_name)
        derived_quantities[predictor_name] = generate_function(
            foragers_object, predictor_name
        )
        derivation_logger.info(f"{predictor_name} completed")

    # calculate scores
    for score_name in score_kwargs.keys():
        score_type = score_name.split("_")[0]
        function_name = f"generate_{score_type}_score"
        generate_function = getattr(ftk, function_name)
        derived_quantities[score_name] = generate_function(foragers_object, score_name)
        derivation_logger.info(f"{score_name} completed")

    # save to object
    foragers_object.derived_quantities = derived_quantities

    # generate combined DF
    derivedDF = _generate_combined_DF(derived_quantities, dropna, add_scaled_values)

    # save to object
    foragers_object.derivedDF = derivedDF

    return derivedDF
