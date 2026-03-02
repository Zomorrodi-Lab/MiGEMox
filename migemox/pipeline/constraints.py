"""
Constraints Management for MiGEMox Pipeline

This module centralizes the formulation, pruning, and application of
metabolic constraints, especially the biomass coupling constraints,
to cobra models. It ensures that complex constraints are consistently
defined and integrated into the optimization problems.
"""

import cobra
from cobra_structural import Model as StructuralModel
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import csr_matrix, vstack, hstack, lil_matrix
from optlang import Constraint, Variable, Objective, Model as OptModel
from tqdm import tqdm
from migemox.pipeline.io_utils import print_memory_usage, log_with_timestamp
from migemox.pipeline.model_utils import find_biomass_candidates
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Union

# Default Coupling Factor (Used in https://doi.org/10.4161/gmic.22370)
COUPLING_FACTOR = 400



def build_global_coupling_constraints(model: StructuralModel, microbe_list: list[str], coupling_factor: float=400):
    """
    Build sparse coupling constraint matrix for microbe biomass relationships.
    
    Biological Constraint: v_reaction ≤ coupling_factor × v_biomass
    
    Rationale: Individual reaction rates cannot exceed microbe biomass production
    by more than the coupling factor. Prevents unrealistic flux distributions
    where microbe have high metabolic activity but low biomass.
    
    Args:
        model: Community model with all microbe reactions
        microbe_list: List of microbe identifiers  
        coupling_factor: Biomass coupling strength (default from config)
        
    Returns:
        Coupling matrix (C), bounds (d), constraint sense (dsense), names (ctrs)
    """
    print(f"{datetime.now(tz=timezone.utc)}: Building global coupling constraints...")
    rxn_id_to_index = {r.id: i for i, r in enumerate(model.reactions)}

    all_constraints = []
    all_d = []
    all_dsense = []
    all_ctrs = []

    for microbe in microbe_list:

        print(f"{datetime.now(tz=timezone.utc)}: Microbe: {microbe}")
        # Find microbe reactions and biomass reaction
        microbe_rxns = [r for r in model.reactions if r.id.startswith(microbe + '_')]
        # biomass_rxns = [r for r in microbe_rxns if 'biomass' in r.id.lower()] # an old way to find biomass; inconsistent with MMT
        biomass_rxns = find_biomass_candidates(model, microbe)

        if not biomass_rxns:
            raise RuntimeError(
                f"Could not find biomass reaction for model {microbe}. \
                    Please ensure the biomass reaction ID starts with 'bio' for this model."
                )

        biomass_rxn = biomass_rxns[0]
        biomass_idx = rxn_id_to_index[biomass_rxns[0].id]
        
        const_count = 0
        for rxn in microbe_rxns:
            if rxn.id == biomass_rxn.id:
                continue  # Don't couple biomass to itself

            rxn_idx = rxn_id_to_index[rxn.id]

            # Create sparse constraint: v_rxn - 400*v_biomass <= 0
            row_indices = [rxn_idx, biomass_idx]
            row_data = [1.0, -coupling_factor]
            constraint_row = csr_matrix((row_data, ([0, 0], row_indices)), shape=(1, len(model.reactions)))
            const_count += 1

            all_constraints.append(constraint_row)
            all_d.append(0.0)
            all_dsense.append('L')  # <= constraint
            all_ctrs.append(f"slack_{rxn.id}")

            # Also add reverse constraint: v_rxn + 400*v_biomass >= 0 (for reversible reactions)
            if rxn.lower_bound < 0:
                row_indices_rev = [rxn_idx, biomass_idx]
                row_data_rev = [1.0, coupling_factor]
                constraint_row_rev = csr_matrix((row_data_rev, ([0, 0], row_indices_rev)), shape=(1, len(model.reactions)))
                const_count += 1

                all_constraints.append(constraint_row_rev)
                all_d.append(0.0)
                all_dsense.append('G')
                all_ctrs.append(f"slack_{rxn.id}_R")
                
        print(microbe)
        print(f"{const_count}x{len(microbe_rxns)}")

    if all_constraints:
        C = vstack(all_constraints)
        d = np.array(all_d).reshape(-1, 1)
        dsense = np.array(all_dsense, dtype='<U1')
        ctrs = np.array(all_ctrs, dtype=object)
    else:
        C = csr_matrix((0, len(model.reactions)))
        d = np.zeros((0, 1))
        dsense = np.array([], dtype='<U1')
        ctrs = np.array([], dtype=object)
    print(f"{datetime.now(tz=timezone.utc)}: Completed building global coupling constraints.")
    print_memory_usage()

    return C, d, dsense, ctrs

def prune_coupling_constraints_by_microbe_fast(
    global_rxn_ids: List[str],
    global_C: csr_matrix,
    global_d: np.ndarray,
    global_dsense: np.ndarray,
    global_ctrs,
    present_microbe: List[str],
    sample_model
) -> Tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """
    Faster version: avoid per-column slicing and hstack, but keep behavior of
    inserting zero columns for reactions not present in the global model.
    """

    print("Dimensions of matrices loaded in prune_coupling_constraints_by_microbe:")
    print(f"global_C: {global_C.shape}")
    print(f"global_d: {global_d.shape}")
    print(f"global_dsense: {global_dsense.shape}")
    # if global_ctrs is a list, this will fail; if it's np.array it’s fine.
    # You can guard it if needed:
    try:
        print(f"global_ctrs: {global_ctrs.shape}")
    except AttributeError:
        print(f"global_ctrs length: {len(global_ctrs)}")

    present_microbe_set = set(present_microbe)
    slack_prefix = "slack_"
    keep_rows: List[int] = []

    # ---- 1) Determine which constraint rows to keep ----
    for i, ctr_name in enumerate(global_ctrs):
        if ctr_name.startswith(slack_prefix):
            microbe_part = ctr_name[len(slack_prefix):]
            for microbe in present_microbe_set:
                if microbe_part.startswith(microbe):
                    keep_rows.append(i)
                    break

    log_with_timestamp(f"Length of keep_rows: {len(keep_rows)}")

    if not keep_rows:
        # No constraints relevant for this sample
        pruned_C = csr_matrix((0, len(sample_model.reactions)))
        pruned_d = np.zeros((0, 1))
        pruned_dsense = np.array([], dtype='<U1')
        pruned_ctrs = np.array([], dtype=object)
        return pruned_C, pruned_d, pruned_dsense, pruned_ctrs

    # Convert to array for fancy indexing
    keep_rows = np.asarray(keep_rows, dtype=int)

    # ---- 2) Build mapping from global reaction IDs to column indices ----
    sample_rxn_ids = [r.id for r in sample_model.reactions]
    log_with_timestamp(f"length of sample_rxn_ids: {len(sample_rxn_ids)}")

    global_rxn_idx_map = {rid: i for i, rid in enumerate(global_rxn_ids)}
    log_with_timestamp(f"length of global_rxn_idx_map: {len(global_rxn_idx_map)}")

    # ---- 3) Figure out which sample reactions exist in the global model ----
    present_pairs = []   # (sample_col_index, global_col_index)
    missing = []         # (sample_col_index, reaction_id)

    for j, rid in enumerate(sample_rxn_ids):
        gidx = global_rxn_idx_map.get(rid)
        if gidx is None:
            missing.append((j, rid))
        else:
            present_pairs.append((j, gidx))

    if missing:
        # Optional: log a few missing reactions for debugging
        example_missing = [rid for _, rid in missing[:10]]
        log_with_timestamp(
            f"{len(missing)} reactions from sample model not found in global_rxn_ids, "
            f"e.g. {example_missing}"
        )

    n_rows = len(keep_rows)
    n_sample = len(sample_rxn_ids)

    if not present_pairs:
        # None of the sample reactions are in the global model -> all-zero matrix
        pruned_C = csr_matrix((n_rows, n_sample))
    else:
        # ---- 4) Slice once on rows and existing columns ----
        sample_positions, global_positions = zip(*present_pairs)
        sample_positions = np.asarray(sample_positions, dtype=int)
        global_positions = np.asarray(global_positions, dtype=int)

        # First slice rows
        subC_rows = global_C[keep_rows, :]           # shape: (n_rows, n_global)

        # Then slice the existing columns as a block
        subC_present = subC_rows[:, global_positions]  # shape: (n_rows, n_present)

        # ---- 5) Remap columns to full sample size, zero where missing ----
        # subC_present currently has columns 0..n_present-1.
        # Each of these corresponds to some column index in the sample model order.
        # Build a mapping: local_col_index -> sample_col_index
        n_present = subC_present.shape[1]
        col_map = np.empty(n_present, dtype=int)
        col_map[np.arange(n_present)] = sample_positions

        # Create a new CSR matrix with remapped column indices and full width
        subC_present = subC_present.tocsr()
        new_indices = col_map[subC_present.indices]   # vectorized remap

        pruned_C = csr_matrix(
            (subC_present.data, new_indices, subC_present.indptr),
            shape=(n_rows, n_sample)
        )

    # ---- 6) Slice vectors for the kept rows ----
    pruned_d = global_d[keep_rows, :]
    pruned_dsense = global_dsense[keep_rows]
    pruned_ctrs = global_ctrs[keep_rows]

    return pruned_C, pruned_d, pruned_dsense, pruned_ctrs

def prune_coupling_constraints_by_microbe(
    global_rxn_ids: list[str],
    global_C: csr_matrix,
    global_d: np.ndarray,
    global_dsense: np.ndarray,
    global_ctrs: list,
    present_microbe: list[str],
    sample_model: StructuralModel
) -> tuple:
    """
    Prunes the global coupling constraints (C, d, dsense, ctrs) to match the
    microbe present in the sample-specific model.

    Args:
        global_rxn_ids (list[str]): The list of all the reaction IDs in the global model.
        global_C (csr_matrix): The global coupling matrix.
        global_d (np.ndarray): The global 'd' vector.
        global_dsense (np.ndarray): The global 'dsense' vector.
        global_ctrs (list): The global constraint names.
        present_microbe (list[str]): List of microbe present in the sample.
        sample_model (cobra_structural.Model): The sample-specific model after pruning microbe.

    Returns:
        tuple: A tuple containing the pruned:
            - C (csr_matrix): Sample-specific coupling matrix.
            - d (np.ndarray): Sample-specific 'd' vector.
            - dsense (np.ndarray): Sample-specific 'dsense' vector.
            - ctrs (list): Sample-specific constraint names.
    """

    # print matrices for debugging
    print(f"Dimensions of matrices loaded in prune_coupling_constraints_by_microbe:")
    print(f"global_C: {global_C.shape}")
    print(f"global_d: {global_d.shape}")
    print(f"global_dsense: {global_dsense.shape}")
    print(f"global_ctrs: {global_ctrs.shape}")

    present_microbe_set = set(present_microbe)
    slack_prefix = "slack_"
    keep_rows = []

    for i, ctr_name in enumerate(tqdm(global_ctrs)):
        # Extract microbe name from constraint name (e.g., "slack_Bacteroides_sp_2_1_33B_IEX_12ppd_S[u]tr")
        if ctr_name.startswith(slack_prefix):
            microbe_part = ctr_name[len(slack_prefix):]
            for microbe in present_microbe_set:
                if microbe_part.startswith(microbe):
                    keep_rows.append(i)
                    break
    
    log_with_timestamp(f"Length of keep_rows: {len(keep_rows)}")

    if keep_rows:
          # Remap columns to match sample-specific model
        sample_rxn_ids = [r.id for r in sample_model.reactions]

        log_with_timestamp(f"length of sample_rxn_ids: {len(sample_rxn_ids)}")

        global_rxn_idx_map = {rid: i for i, rid in enumerate(global_rxn_ids)}

        log_with_timestamp(f"length of global_rxn_idx_map: {len(global_rxn_idx_map)}")

        cols = []
        # for rid in sample_rxn_ids:
        for i in range(len(sample_rxn_ids)):
            print(f"Processing reaction {i} of {len(sample_rxn_ids)}")
            rid = sample_rxn_ids[i]
            idx = global_rxn_idx_map.get(rid)
            if idx is not None:
                cols.append(global_C[keep_rows, idx])
            else:
                cols.append(csr_matrix((len(keep_rows), 1)))
        pruned_C = hstack(cols)
        
        pruned_d = global_d[keep_rows, :]
        pruned_dsense = global_dsense[keep_rows]
        pruned_ctrs = global_ctrs[keep_rows]
    else:
        pruned_C = csr_matrix((0, len(sample_model.reactions)))
        pruned_d = np.zeros((0, 1))
        pruned_dsense = np.array([], dtype='<U1')
        pruned_ctrs = np.array([], dtype=object)
    
    return pruned_C, pruned_d, pruned_dsense, pruned_ctrs

def apply_couple_constraints(model: cobra.Model, model_data: dict) -> cobra.Model:
    """
    Applies biomass coupling constraints to a cobra model using optlang.

    This function reads the pre-calculated coupling matrix (C), right-hand side (d),
    and sense vector (dsense) from model_data, and adds them as linear
    constraints to the provided cobra model's solver interface.

    Args:
        model (cobra.Model): The cobra model to which constraints will be added.
        model_data: Dictionary containing the coupling data (C, d, dsense, ctrs)

    Returns:
        cobra.Model: The model with added coupling constraints.
    """

    # S = csr_matrix(modelscipy['S'])     # stoichiometric matrix
    # C = csr_matrix(modelscipy.get('C', np.empty((0, S.shape[1]))))  # fallback to empty
    C = csr_matrix(model_data['C'])  # NOTE: used to fallback to matrix of zeros of dimension of stoich matrix if empty, but now does not
    # d = modelscipy['d'].reshape(-1, 1).astype(float)
    d = model_data['d'].reshape(-1, 1).astype(float)

    C_csr = C.tocsr()
    # dsense = modelscipy.get('dsense')
    dsense = model_data['dsense']
    
    forward_vars = np.array([rxn.forward_variable for rxn in model.reactions])
    reverse_vars = np.array([rxn.reverse_variable for rxn in model.reactions])
    constraints_to_add = []

    for i in tqdm(range(C.shape[0])):
        row = C_csr.getrow(i)
        if row.nnz == 0:  # Skip empty rows
            continue
        
        expr = sum(row.data[k] * (forward_vars[row.indices[k]] - reverse_vars[row.indices[k]]) for k in range(row.nnz))
        if dsense[i] == 'E':
            constraints_to_add.append(Constraint(expr, lb=d[i,0], ub=d[i,0]))
        elif dsense[i] == 'L':
            constraints_to_add.append(Constraint(expr, ub=d[i,0]))
        elif dsense[i] == 'G':
            constraints_to_add.append(Constraint(expr, lb=d[i,0]))

    model.add_cons_vars(constraints_to_add)

    return model

# Running FVA by building an optlang model from scratch
# Good for deeply understanding how coupling constraints work
def build_constraint_matrix(model_path):
    """
    Build the constraint matrix A and RHS vector from a scipy model dictionary of a .mat file.
    
    Args:
        model_path (str): Path to the .mat file containing the model.

    Returns:
        A (scipy.sparse.csr_matrix), rhs (numpy.ndarray), csense (numpy.ndarray)
    """

    # Load the model from the .mat file using scipy
    data = loadmat(model_path, simplify_cells=True)
    modelscipy = data["model"]

    S = csr_matrix(modelscipy['S'])     # stoichiometric matrix
    C = csr_matrix(modelscipy.get('C', np.empty((0, S.shape[1]))))  # fallback to empty
    d = modelscipy['d'].reshape(-1, 1).astype(float)
    b = modelscipy['b'].reshape(-1, 1).astype(float)
    csense = modelscipy.get('csense')   # character vector like ['E', 'E', ..., 'L', ...]

    # Final constraint matrix A and RHS
    A = vstack([S, C])                           # shape: (m + k, n)
    rhs = np.vstack([b, d])                      # shape: (m + k, 1)

    # Constraint sense vector
    csense = np.concatenate([modelscipy['csense'].flatten(), modelscipy['dsense'].flatten()])

    lb = modelscipy['lb'].flatten()  # Lower bounds, shape (n,)
    ub = modelscipy['ub'].flatten()  # Upper bounds, shape (n,)
    c = modelscipy['c'].flatten()    # Objective coefficients, shape (n,)
    
    return A, rhs, csense, lb, ub, c

def build_optlang_model(A: csr_matrix, rhs: np.ndarray, csense: np.ndarray,
                       lb: np.ndarray, ub: np.ndarray, c: np.ndarray) -> tuple:
    """
    Build optlang optimization model from constraint matrix components.
    
    This creates an optimization model compatible with multiple solvers
    for performing flux variability analysis on large constraint systems.
    
    Performance Optimization: Batch constraint creation instead of individual loops.
    
    Args:
        A: Sparse constraint matrix A
        rhs: Right-hand side vector
        csense: Constraint sense vector ('E', 'L', 'G')
        lb/ub: Variable lower/upper bounds
        c: Objective coefficient vector
        
    Returns:
        Tuple of (optimization_model, variables, objective_expression)
    """
    n_vars = A.shape[1]
    vars = [Variable(f'v_{i}', lb=lb[i], ub=ub[i]) for i in range(n_vars)]
    opt_model = OptModel()
    opt_model.add(vars)
    A_csr = A.tocsr()  # Ensure CSR format for efficient row access
    
    for batch_start in tqdm(range(0, A.shape[0], 1000), desc="Building constraints"):
        batch_end = min(batch_start + 1000, A.shape[0])
        batch_constraints = []
        
        for i in range(batch_start, batch_end):
            row = A_csr.getrow(i)
            if row.nnz == 0:  # Skip empty rows
                continue
                
            expr = sum(row.data[k] * vars[row.indices[k]] for k in range(row.nnz))
            sense = csense[i]
            
            if sense == 'E':
                constr = Constraint(expr, lb=rhs[i, 0], ub=rhs[i, 0])
            elif sense == 'L':
                constr = Constraint(expr, ub=rhs[i, 0])
            elif sense == 'G':
                constr = Constraint(expr, lb=rhs[i, 0])
            
            batch_constraints.append(constr)
        
        opt_model.add(batch_constraints)
    obj_expr = sum(c[j] * vars[j] for j in range(n_vars))
    return opt_model, vars, obj_expr

def run_sequential_fva(opt_model, vars: list, obj_expr, 
                      rxn_ids: list, opt_percentage: float = 99.99) -> tuple:
    """
    Perform sequential flux variability analysis with optimality constraints.
    
    Biological Context: FVA determines the range of possible flux values for each
    reaction while maintaining near-optimal growth. This reveals metabolic
    flexibility and identifies essential vs. dispensable pathways.

    Args:
        opt_model (optlang.Model): The optimization model.
        vars (list): List of optlang variables.
        obj_expr (optlang.Expression): Objective expression.
        rxn_ids (list): List of reaction IDs to analyze.
        opt_percentage (float): Percentage of optimal objective to constrain FVA.

    Returns:
        min_fluxes (dict), max_fluxes (dict): Minimum and maximum fluxes for each reaction.
    """
    # Get optimal objective value
    opt_model.objective = Objective(obj_expr, direction='max')
    print(f'Model Status after optimization: {opt_model.optimize()}')
    optimal_obj = opt_model.objective.value

    min_flux = []
    max_flux = []
    for j in tqdm(rxn_ids, desc="FVA (sequential)"):
        # Minimize
        opt_model.objective = Objective(vars[j], direction='min')
        if opt_percentage < 100:
            opt_constr = Constraint(obj_expr, lb=opt_percentage/100 * optimal_obj)
            opt_model.add(opt_constr)
        opt_model.optimize()
        min_flux.append(vars[j].primal if opt_model.status == 'optimal' else None)
        if opt_percentage < 100:
            opt_model.remove(opt_constr)
        # Maximize
        opt_model.objective = Objective(vars[j], direction='max')
        if opt_percentage < 100:
            opt_constr = Constraint(obj_expr, lb=opt_percentage/100 * optimal_obj)
            opt_model.add(opt_constr)
        opt_model.optimize()
        max_flux.append(vars[j].primal if opt_model.status == 'optimal' else None)
        if opt_percentage < 100:
            opt_model.remove(opt_constr)
    
    # Using a pd Dataframe with rxn_id as index and min and max as columns, return two dicts
    df = pd.DataFrame({
        'rxn_id': [vars[j].name for j in rxn_ids],
        'min_flux': min_flux,
        'max_flux': max_flux
    }).set_index('rxn_id')

    return df['min_flux'].to_dict(), df['max_flux'].to_dict()

def couple_rxn_list_to_rxn(
    model: cobra.Model,
    rxn_list: Optional[List[str]] = None,
    rxn_c: Union[str, cobra.Reaction, List[str]] = None,
    c: Union[float, int, List[float], np.ndarray] = 1000.0,
    u: Union[float, int, List[float], np.ndarray] = 0.01,
) -> cobra.Model:
    """
    Python/CobraPy implementation of the MATLAB function coupleRxnList2Rxn.

    Adds coupling constraints vi ~ v_rxnC for each reaction in rxn_list,
    encoded as linear constraints and applied via `apply_couple_constraints`.

    For irreversible reactions in rxn_list:
        vi - c * v_rxnC <= u

    For reversible reactions in rxn_list (reverse direction):
        vi + c * v_rxnC >= u

    Args:
        model:      cobra.Model
        rxn_list:   list of reaction IDs to be coupled. If None/empty, all reactions.
        rxn_c:      reaction to couple to (ID, Reaction, or list with single ID).
        c:          scalar or vector of coupling factors, length == len(rxn_list)
                    (default 1000)
        u:          scalar or vector of thresholds, length == len(rxn_list)
                    (default 0.01)

    Returns:
        cobra.Model with added coupling constraints (via optlang).
    """
    if rxn_list is None or len(rxn_list) == 0:
        rxn_list = [rxn.id for rxn in model.reactions]

    # Normalize rxn_c to a single reaction id string
    if rxn_c is None:
        raise ValueError("rxn_c (the coupled reaction) must be provided.")

    if isinstance(rxn_c, cobra.Reaction):
        rxn_c_id = rxn_c.id
    elif isinstance(rxn_c, str):
        rxn_c_id = rxn_c
    elif isinstance(rxn_c, (list, tuple, np.ndarray)):
        if len(rxn_c) != 1:
            raise ValueError("rxn_c should be a single reaction (or a list of length 1).")
        rxn_c_id = rxn_c[0] if not isinstance(rxn_c[0], cobra.Reaction) else rxn_c[0].id
    else:
        raise TypeError("rxn_c must be a reaction ID, cobra.Reaction, or list of one of those.")

    # Sanity checks
    model_rxn_ids = [rxn.id for rxn in model.reactions]
    rxn_index = {rid: i for i, rid in enumerate(model_rxn_ids)}

    missing = [rid for rid in rxn_list if rid not in rxn_index]
    if missing:
        raise ValueError(f"The following reactions in rxn_list are not in the model: {missing}")

    if rxn_c_id not in rxn_index:
        # Mirror the original MATLAB behavior: warn but proceed
        import warnings
        warnings.warn(f"Coupled reaction '{rxn_c_id}' is not in the model.", UserWarning)

    # Original nRxnList is defined *before* possibly adding rxnC
    n_rxn_list = len(rxn_list)

    # Handle c and u as vectors of length n_rxn_list (to mimic MATLAB behavior)
    def _make_vec(param, name: str):
        if np.isscalar(param):
            return np.full(n_rxn_list, float(param))
        arr = np.asarray(param, dtype=float).ravel()
        if arr.size != n_rxn_list:
            raise ValueError(
                f"Parameter '{name}' must be scalar or of length len(rxn_list)={n_rxn_list}, "
                f"got length {arr.size}."
            )
        return arr

    c_vec = _make_vec(c, "c")  # shape (n,)
    u_vec = _make_vec(u, "u")  # shape (n,)

    # Reversibility: revs = model.lb(rxn_list) < 0   (True if reversible)
    lbs = np.array([model.reactions[rxn_index[rid]].lower_bound for rid in rxn_list])
    revs = lbs < 0  # boolean array length n_rxn_list

    # Constraint IDs: slack_<rxn>, slack_<rxn>_R (forward and reverse form)
    ctrs_forward = [f"slack_{rid}" for rid in rxn_list]
    ctrs_reverse = [f"slack_{rid}_R" for rid in rxn_list]
    ctrs_mat = np.vstack([ctrs_forward, ctrs_reverse])  # shape (2, n)
    # MATLAB column-major linearization: [col1; col2; ...]
    ctrs_vec = ctrs_mat.flatten(order="F")  # length 2n

    # plusminus = [ones(1,n); -ones(1,n)]   (2 x n)
    plusminus_mat = np.vstack([np.ones(n_rxn_list), -np.ones(n_rxn_list)])  # (2, n)
    plusminus_vec = plusminus_mat.flatten(order="F")  # (2n,) -> [1,-1,1,-1,...]

    # toRemove = [false(1,n); ~revs'] then (:)
    #   -> keep all forward rows; drop reverse rows for irreversible reactions
    to_remove_mat = np.vstack([np.zeros(n_rxn_list, dtype=bool), ~revs])
    to_remove_vec = to_remove_mat.flatten(order="F")  # (2n,)

    # coefs: sparse(2*n, n + numel(setdiff(rxnC,rxnList)))
    # Here rxn_c_id is a single reaction; if not in rxn_list, we add one extra column.
    extra_cols = 0 if rxn_c_id in rxn_list else 1
    n_cols = n_rxn_list + extra_cols  # constrained rxns + optional extra rxn_c column

    coefs = lil_matrix((2 * n_rxn_list, n_cols), dtype=float)

    # rxnInd = [1:n; 1:n]; rxnInd(:) (MATLAB 1-based). For zero-based:
    # rxnInd_mat = [0..n-1; 0..n-1]
    rxn_ind_mat = np.vstack([np.arange(n_rxn_list), np.arange(n_rxn_list)])  # (2, n)
    rxn_ind_vec = rxn_ind_mat.flatten(order="F")  # (2n,)

    # constInd = 1:2*n (MATLAB 1-based); here zero-based row indices 0..2n-1
    row_indices = np.arange(2 * n_rxn_list)

    # Set diagonal entries for vi (each constraint row has its own vi with coeff 1)
    for r, c_idx in zip(row_indices, rxn_ind_vec):
        coefs[r, c_idx] = 1.0

    # Determine the column index for rxn_c_id in this small coefs matrix
    if rxn_c_id in rxn_list:
        rxn_cid_col = rxn_list.index(rxn_c_id)  # among the first n_rxn_list columns
    else:
        rxn_cid_col = n_cols - 1  # extra column at the end

    # cs = - plusminus * c  (MATLAB implicit expansion)
    # plusminus_mat: (2, n); c_vec: (n,) -> broadcast to (2, n)
    cs_mat = -plusminus_mat * c_vec  # (2, n)
    cs_vec = cs_mat.flatten(order="F")  # (2n,)

    # Put coupling coefficients in column rxn_cid_col
    for r, val in zip(row_indices, cs_vec):
        coefs[r, rxn_cid_col] += val  # += in case rxn_c is also in rxn_list

    # dsenses = [repmat('L',1,n); repmat('G',1,n)]; dsenses(:)
    dsenses_mat = np.vstack([np.full(n_rxn_list, "L"), np.full(n_rxn_list, "G")])
    dsenses_vec = dsenses_mat.flatten(order="F")  # (2n,)

    # ds = plusminus * u
    ds_mat = plusminus_mat * u_vec  # (2, n)
    ds_vec = ds_mat.flatten(order="F")  # (2n,)

    # Remove rows corresponding to non-reversible reactions (reverse constraint removed)
    keep_mask = ~to_remove_vec
    coefs_kept = coefs[keep_mask, :]          # (n_kept, n_cols)
    ds_kept = ds_vec[keep_mask]               # (n_kept,)
    dsenses_kept = dsenses_vec[keep_mask]     # (n_kept,)
    ctrs_kept = np.array(ctrs_vec, dtype=object)[keep_mask]  # (n_kept,)

    # Map the small coefficient matrix to full model matrix C of size (n_kept, n_model_rxns)
    n_model_rxns = len(model.reactions)
    # Column mapping from our n_cols to model reaction indices:
    #   first n_rxn_list columns -> rxn_list
    #   optional extra column    -> rxn_c_id
    col_rxn_ids = list(rxn_list) + ([] if extra_cols == 0 else [rxn_c_id])
    mapping_rows = np.arange(n_cols)
    mapping_cols = np.array([rxn_index[rid] for rid in col_rxn_ids], dtype=int)

    # P has 1 at (j, model_index_of_col_j)
    P = csr_matrix(
        (np.ones(n_cols), (mapping_rows, mapping_cols)),
        shape=(n_cols, n_model_rxns),
    )

    C_small_csr = coefs_kept.tocsr()
    C_full = C_small_csr.dot(P)  # shape (n_kept, n_model_rxns)

    # Prepare model_data dict for apply_couple_constraints
    model_data = {
        "C": C_full,
        "d": ds_kept.astype(float),
        "dsense": np.array(dsenses_kept, dtype=str),
        "ctrs": ctrs_kept,  # not used by apply_couple_constraints, but kept for completeness
    }

    # Apply constraints to the model (in-place) and return it
    model_coupled = apply_couple_constraints(model, model_data)

    return model_coupled
