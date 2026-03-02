"""
Model Utilities

This file contains generic utilities for handling models.
"""

from cobra import Model
from cobra_structural import Model as StructuralModel

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
