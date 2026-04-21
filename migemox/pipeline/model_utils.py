"""
Model Utilities

This file contains generic utilities for handling models.
"""

from cobra import Model
from cobra_structural import Model as StructuralModel
import time

def reset_solver(model: Model, solver: str, temp_solver: str | None=None):
    """
    Reset a model's solver object. This function works by setting the model's solver to a different
    solver temporarily, then switching it back to the original solver, forcing COBRApy to create
    a new solver object without any state.

    The default temp_solver is 'glpk', but if the desired solver is 'glpk', the solver is temporarily 
    switched to 'glpk_exact' instead. If you as the user would like to control which temporary solver
    is used, you can do so by setting `temp_solver` to the name of the solver you want to temporarily
    switch to.

    This function modifies the model in place. It does not return a new model.

    :param model: The model object for which to reset the solver. Must not be a `StructuralModel`, as these
    do not contain solvers.
    :param solver: The name of the current solver in the model, which will be reset.
    :param temp_solver: The name of the temporary solver to switch to to force a solver reset. If set
    to `None`, defaults to using `'glpk'` as the temp solver unless `solver` is set to `'glpk``, in which
    case it uses `'glpk_exact'`.
    """
    # Resolve default temp_solver before the ValueError check
    if temp_solver is None:
        if solver == "glpk":
            temp_solver = "glpk_exact"
        else:
            temp_solver = "glpk"

    # Validate that temp_solver and solver are not the same
    if temp_solver == solver:
        raise ValueError(
            f"temp_solver '{temp_solver}' cannot be the same as solver '{solver}'. "
            f"The temporary solver must be a different solver type in order to force "
            f"COBRApy to create a new solver object on the switch back."
        )

    # Record solver id before reset
    id_before = id(model.solver)

    # Switch to temp solver, then back to the original solver
    # This forces COBRApy to create a brand new solver object
    model.solver = temp_solver
    model.solver = solver

    # Record solver id after reset
    id_after = id(model.solver)

    # Confirm that a new solver object was actually created
    if id_before == id_after:
        # try again, with some sleep time
        id_before2 = id(model.solver)
        model.solver = temp_solver
        time.sleep(0.3)
        model.solver = solver
        id_after2 = id(model.solver)

        if id_before2 == id_after2:
            raise RuntimeError(
                f"Solver reset failed: the solver object id did not change after switching "
                f"from '{solver}' -> '{temp_solver}' -> '{solver}'. "
                f"id_before={id_before}, id_after={id_after}."
            )

def find_biomass_candidates(
        model: Model | StructuralModel,
        microbe_name: str | None = None,
    ):
    """
    Find candidate reactions that could be the biomass reaction.

    By default, or if `microbe_name = None` this function searches the model for reactions whose ID
    starts with 'bio' (case sensitive). This mirrors the behavior of MMT.

    However, if the reactions in a microbe are tagged with the microbe's name, then this will break
    that functionality, as the biomass reaction ID will start with the microbe name instead of `bio`.
    To work around this issue, if the reaction IDs are tagged with the microbe name, you must pass in
    the microbe name in the `microbe_name` parameter. Please note that this should just be the microbe
    name itself (e.g. E_coli) **without** a trailing underscore (i.e. not E_coli_). This `microbe_name`
    (plus a trailing underscore) will be removed from any reaction ID in which it is present prior to 
    checking if the reaction id starts with `'bio'`. Please note that this check and removal is 
    **case sensitive**.

    **Another note:** Previously, MiGEMox searched for biomass reactions as reactions containing the word 
    `'biomass'` in them. However, this was insufficient as some AGORA2 models have biomass reactions named 
    `bio1`. It is also insufficient to just search for reactions whose IDs contain the string `'bio'` in them 
    because there are reactions such as `'pbiosynthesis'` which are not the biomass reaction and would therefore 
    get wrongly flagged. For that reason, we must search specifically for srings that start with `'bio'`. This is
    the behavior MMT uses.
    
    :param model: The model or structural model to search to find biomass reaction candidates.
    :type model: Model | StructuralModel
    :param microbe_name: The name of the microbe being checked, if reactions are tagged with microbe name. E.g. `E_coli`
    :type microbe_name: str | None
    :return: A list of reaction objects corresponding to potential biomass reactions.
    """

    biomass_candidate_rxns = []
    for rxn in model.reactions:

        # copy rxn id so don't override original
        rxn_id = rxn.id

        # remove microbe_name from rxn_id if present
        if microbe_name is not None:
            rxn_id = rxn_id.replace(microbe_name + '_', '')

        # check if it is a biomass reaction (case sensitive)
        if rxn_id.startswith('bio'):
            biomass_candidate_rxns.append(rxn)

    return biomass_candidate_rxns
