"""
I/O Utilities for MiGEMox Pipeline

This module provides functions for loading input data (e.g., abundance files),
preparing model data for saving to disk, and general pipeline helpers such as
memory usage tracking. It centralizes file operations and utility functions
to improve code organization and reusability.
"""

from datetime import timezone
import pandas as pd
import psutil
import os
import re
from cobra_structural import Model as StructuralModel
from cobra.io import load_matlab_model
from cobra.util import create_stoichiometric_matrix
from cobra.io import read_sbml_model, write_sbml_model
from cobra_structural.io import read_sbml_model as read_structural_sbml_model
from cobra_structural.io import write_sbml_model as write_structural_sbml_model
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz
import pickle
import sys
from datetime import datetime

def log_with_timestamp(message: str):
    """
    Logs a message with the current UTC timestamp.

    Args:
        message: The message to log.
    """
    current_time = datetime.now(tz=timezone.utc)
    print(f"[{current_time}] {message}")

def pickle_structural_model(model: StructuralModel, file_path: str):
    """
    Pickle a cobra StructuralModel to a file.

    Args:
        model: The cobra StructuralModel to pickle.
        file_path: The path to the file where the model should be pickled.
    """
    log_with_timestamp(f"Pickling StructuralModel to {file_path}...")
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    log_with_timestamp(f"Pickling complete.")

def load_structural_model_pickle(file_path: str) -> StructuralModel:
    """
    Load a pickled cobra StructuralModel from a file.

    Args:
        file_path: The path to the file from which to load the model.
    Returns:
        The loaded cobra StructuralModel.
    """
    log_with_timestamp(f"Loading StructuralModel from pickle {file_path}...")
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    log_with_timestamp(f"Loading complete.")
    return model

def save_model_and_constraints(model, C, d, dsense, ctrs, model_name, out_dir, save_format="sbml"):
    """
    Save a cobra model and its associated coupling constraints to disk.

    This function works on both cobra.Model and cobra.StructuralModel objects.

    You can specify the save format for the model to be SBML or pickle. Pickle only works on StructuralModel objects.
    
    :param model: A cobra Model or StructuralModel object
    :param C: Coupling matrix
    :param d: Right-hand side of coupling constraints
    :param dsense: Sense of coupling constraints
    :param ctrs: Names of coupling constraints
    :param model_name: Name of the model
    :param out_dir: Directory to save the model and constraints
    :param save_format: "sbml" or "pickle"
    """

    base = os.path.join(out_dir, model_name)

    # save GEM
    if save_format == "pickle":
        if not isinstance(model, StructuralModel):
            raise ValueError("Pickle format only supported for StructuralModel objects.")
        # 1) GEM → pickle
        pickle_path = base + ".pkl"
        pickle_structural_model(model, pickle_path)

    elif save_format == "sbml":
        # 1) GEM → SBML
        sbml_path = base + ".sbml"
        if isinstance(model, StructuralModel):
            log_with_timestamp(f"Writing StructuralModel to SBML at {sbml_path}...")
            write_structural_sbml_model(model, sbml_path)
            log_with_timestamp(f"Writing complete.")
        else:
            log_with_timestamp(f"Writing Model to SBML at {sbml_path}...")
            write_sbml_model(model, sbml_path)
            log_with_timestamp(f"Writing complete.")

    # Save constraints
    log_with_timestamp(f"Saving coupling constraints to {base}_C.npz and {base}_constraints.npz...")
    C = csr_matrix(C)
    save_npz(base + "_C.npz", C)

    np.savez(
        base + "_constraints.npz",
        d=d,
        dsense=dsense,
        ctrs=ctrs,
    )
    log_with_timestamp(f"Saving constraints complete.")

def load_model_and_constraints(model_name, model_dir, model_type="standard",  save_format="sbml"):
    """
    Load a COBRA model and its associated coupling constraints from disk.

    This function works on both cobra.Model and cobra.StructuralModel objects. The type of model to load is specified by the `model_type` parameter.
    
    :param model_name: The name of the model to load
    :param model_dir: The directory where the model and constraints are stored
    :param model_type: "standard" for cobra.Model, "structural" for cobra.StructuralModel
    :param save_format: "sbml" or "pickle"
    :return: A tuple of (model, C, d, dsense, ctrs)
    """
    base = os.path.join(model_dir, model_name)

    if save_format == "pickle":
        if model_type != "structural":
            raise ValueError("Pickle format only supported for StructuralModel objects.")
        # Load GEM from pickle
        model = load_structural_model_pickle(base + ".pkl")


    elif save_format == "sbml":
        if model_type == "standard":
            # Load GEM from SBML
            log_with_timestamp(f"Loading Model from SBML at {base + '.sbml'}...")
            model = read_sbml_model(base + ".sbml")
            log_with_timestamp(f"Loading complete.")
        elif model_type == "structural":
            # Load StructuralModel from SBML
            log_with_timestamp(f"Loading StructuralModel from SBML at {base + '.sbml'}...")
            model = read_structural_sbml_model(base + ".sbml")
            log_with_timestamp(f"Loading complete.")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # Load constraints
    log_with_timestamp(f"Loading coupling constraints from {base + '_C.npz'} and {base + '_constraints.npz'}...")
    C = load_npz(base + "_C.npz")
    data = np.load(base + "_constraints.npz", allow_pickle=True)
    log_with_timestamp(f"Loading constraints complete.")
    d = data["d"]
    dsense = data["dsense"]
    ctrs = data["ctrs"]

    return model, C, d, dsense, ctrs

def total_size(o, seen=None):
    """
    Check total size of a Python dict
    
    Args:
        o: the dict to check size of
        seen: set of seen object ids to avoid double counting (always leave as default when calling this function)
    """

    if seen is None:
        seen = set()
    oid = id(o)
    if oid in seen:
        return 0
    seen.add(oid)
    size = sys.getsizeof(o)
    if isinstance(o, dict):
        size += sum(total_size(v, seen) for v in o.values())
    elif isinstance(o, (list, tuple, set)):
        size += sum(total_size(i, seen) for i in o)
    return size

def ensure_parent_dir(file_path: str):
    """
    Ensure a parent directory to a file path exists.

    This function creates the directory if it does not exist.

    Args:
        file_path: the path to the file you want to ensure exists
    """
    parent_dir = os.path.dirname(file_path)
    if parent_dir:  # This will be empty string if file is in current directory
        os.makedirs(parent_dir, exist_ok=True)

def print_memory_usage(stage=""):
    """
    Prints the current memory usage (in MB) of the running Python process.

    Args:
        stage (str): Description of the pipeline stage.
    """
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    print(f"[MEMORY] {stage}: {mem_mb:.2f} MB")

def get_individual_size_name(abun_file_path: str, mod_path: str) -> tuple:
    """
    Reads abundance data from a CSV file and extracts sample names, organisms,
    and extracellular metabolites from associated model files or a dict of already
    loaded models.

    Args:
        abun_file_path: Path to the abundance CSV file.
        mod_path: Path to the directory containing organism model files (.mat).

    Returns:
        A tuple containing:
            - clean_samp_names: List of cleaned sample names (valid Python identifiers).
            - organisms: List of organism names from the abundance file.
            - ex_mets: List of sorted unique extracellular metabolites found in the models.

    Raises:
        ValueError: If there's an error reading the abundance file or loading a model.
        FileNotFoundError: If a model file is not found.
    """
    # Step 1: Read abundance CSV
    df = pd.read_csv(abun_file_path, index_col=0)
    samp_names, organisms = list(df.columns), list(df.index)

    # Step 2: Clean sample names to be valid Python identifiers
    clean_samp_names = []
    for name in samp_names:
        if not name.isidentifier():
            name = re.sub(r'\W', '_', name)
            if not name[0].isalpha():
                name = 'sample_' + name
        clean_samp_names.append(name)

    # Step 3: Load models and extract [e] metabolites
    ex_mets = set()
    for organism in organisms:
        model_file = os.path.join(mod_path, organism + '.mat')
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        try:
            model = load_matlab_model(model_file)
        except Exception as e:
            raise ValueError(f"Error loading model {organism}: {e}")

        # Extract extracellular metabolites (assuming suffix '[e]')
        mets = [met.id for met in model.metabolites if met.id.endswith('[e]')]
        ex_mets.update(mets)

    return clean_samp_names, organisms, list(sorted(ex_mets))

def make_community_gem_dict(model, C=None, d=None, dsense=None, ctrs=None):
    """
    Creates a dictionary representation of a cobra model, including additional
    parameters (C, d, dsense, ctrs) for saving in a .mat file format compatible
    with mgPipe/Microbiome Modeling Toolbox.

    Args:
        model (cobra.Model): The cobra model object.
        C (scipy.sparse.csr_matrix, optional): Coupling matrix. Defaults to None.
        d (numpy.ndarray, optional): Right-hand side of coupling constraints. Defaults to None.
        dsense (numpy.ndarray, optional): Sense of coupling constraints ('E', 'L', 'G'). Defaults to None.
        ctrs (list, optional): Names of coupling constraints. Defaults to None.

    Returns:
        dict: A dictionary containing model components and additional parameters
              suitable for `scipy.io.savemat`.
    """
    num_rxns = len(model.reactions)
    num_mets = len(model.metabolites)

    # Objective vector
    c = np.zeros((num_rxns, 1))
    obj_rxn_id = str(model.objective.expression).split('*')[1].split('-')[0].strip()
    for i, rxn in enumerate(model.reactions):
        if rxn.id == obj_rxn_id:
            c[i, 0] = 1.0
            break

    # S matrix
    S = create_stoichiometric_matrix(model)

    # Bounds
    lb = np.array([rxn.lower_bound for rxn in model.reactions], dtype=np.float64).reshape(-1, 1)
    ub = np.array([rxn.upper_bound for rxn in model.reactions], dtype=np.float64).reshape(-1, 1)

    # Met and rxn info
    rxns = np.array([[rxn.id] for rxn in model.reactions], dtype=object)
    rxnNames = np.array([[rxn.name] for rxn in model.reactions], dtype=object)
    metNames = np.array([[met.name] for met in model.metabolites], dtype=object)
    mets = np.array([[met.id] for met in model.metabolites], dtype=object)

    # Sense
    csense = np.array(['E'] * num_mets, dtype='U1')

    # Default coupling matrices
    if C is None: C = csr_matrix((0, num_rxns))
    if d is None: d = np.zeros((0, 1))
    if dsense is None: dsense = np.array([], dtype='<U1')
    if ctrs is None: ctrs = np.array([], dtype=object).reshape(-1, 1)

    C = csr_matrix((0, num_rxns)) if C is None else C
    d = np.zeros((0, 1)) if d is None else d
    dsense = np.array([], dtype='<U1') if dsense is None else dsense
    ctrs = np.array([], dtype=object).reshape(-1, 1) if ctrs is None else ctrs.reshape(-1, 1)

    # Model name
    model_name = np.array([model.name], dtype=object)
    osenseStr = np.array(['max'], dtype='U3')

    return {
        'rxns': rxns,
        'rxnNames': rxnNames,
        'mets': mets,
        'metNames': metNames,
        'S': S,
        'b': np.zeros((num_mets, 1)),
        'c': c,
        'lb': lb,
        'ub': ub,
        'metChEBIID': np.array([
            m.annotation['chebi'][0].replace('CHEBI:', '') if 'chebi' in m.annotation and isinstance(m.annotation['chebi'], list)
            else m.annotation['chebi'].replace('CHEBI:', '') if 'chebi' in m.annotation and isinstance(m.annotation['chebi'], str)
            else ''
            for m in model.metabolites
        ], dtype=object).reshape(-1, 1),
        'metCharges': np.array([m.charge if getattr(m, 'charge', None) is not None else np.nan for m in model.metabolites]).reshape(-1, 1),
        'metFormulas': np.array([m.formula if getattr(m, 'formula', None) is not None else np.nan for m in model.metabolites]).reshape(-1, 1),
        'rules': np.array([getattr(r, 'gene_reaction_rule', '') for r in model.reactions], dtype=object).reshape(-1, 1),
        'subSystems': np.array([getattr(r, 'subsystem', '') for r in model.reactions], dtype=object).reshape(-1, 1),
        'osenseStr': osenseStr,
        'csense': csense,
        'C': C,
        'd': d,
        'dsense': dsense,
        'ctrs': ctrs,
        'name': model_name
    }

def collect_flux_profiles(samp_names: List[str], exchanges: List[str],
                          net_production: Dict[str, Dict[str, float]],
                          net_uptake: Dict[str, Dict[str, float]],
                          res_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts net production and net uptake dictionaries, which contain results
    for each sample, into structured Pandas DataFrames.

    Args:
        res_path: path to results directory where flux.csv files should be stored
        samp_names: List of sample identifiers.
        exchanges: List of exchange reaction IDs (metabolites) analyzed.
        net_production: Dictionary where keys are sample names and values are
                        dictionaries of metabolite-to-flux for net production.
        net_uptake: Dictionary where keys are sample names and values are
                    dictionaries of metabolite-to-flux for net uptake.

    Returns:
        Tuple of:
            - net_secretion_df (pd.DataFrame): DataFrame of net production (secretion)
                                               fluxes with metabolites as index and samples as columns.
            - net_uptake_df (pd.DataFrame): DataFrame of net uptake fluxes with
                                            metabolites as index and samples as columns.
    """
    # Ensure exchanges are sorted for consistent DataFrame indexing
    exchanges_sorted = sorted(exchanges)

    res_path = Path.cwd() / 'Results' if not res_path else Path(res_path)
    logger.info(f"Results will be saved to: {res_path}")

    # Prepare data for net production DataFrame
    prod_data, uptk_data = {}, {}
    for samp in samp_names:
        prod_data[samp] = [net_production.get(samp, {}).get(r, 0.0) for r in exchanges_sorted]
        uptk_data[samp] = [net_uptake.get(samp, {}).get(r, 0.0) for r in exchanges_sorted]
    
    net_secretion_df = pd.DataFrame(prod_data, index=exchanges_sorted)
    net_uptake_df = pd.DataFrame(uptk_data, index=exchanges_sorted)

    net_secretion_df.to_csv(res_path / 'inputDiet_net_secretion_fluxes.csv', index=True, index_label='Net secretion')
    net_uptake_df.to_csv(res_path / 'inputDiet_net_uptake_fluxes.csv', index=True, index_label='Net uptake')


    return net_secretion_df, net_uptake_df

def extract_positive_net_prod_constraints(csv_path: str, threshold: float = 0) -> Dict[str, Dict[str, float]]:
    """
    Reads a net secretion/production CSV and returns a dict of
    {met_id: {model_name: flux_value}} for positive fluxes.
    """
    df = pd.read_csv(csv_path, index_col=0)
    # keep only rows with at least one non-zero flux
    mask = (df.iloc[:, 0:].abs() > threshold).any(axis=1)
    filtered = df[mask]
    filtered.index = filtered.index.to_series().apply(lambda x: x.split("[")[0].replace("EX_", ""))
    return filtered.dropna(how="all", axis=1).to_dict(orient="index")
