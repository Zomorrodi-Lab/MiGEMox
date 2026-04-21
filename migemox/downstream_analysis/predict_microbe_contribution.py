import os
import pandas as pd
import numpy as np
import docplex
import cplex
from pathlib import Path
from cobra.io import load_matlab_model, write_sbml_model, read_sbml_model
from cobra.flux_analysis.variability import flux_variability_analysis
from typing import Optional, List, Tuple, Dict, Literal
import logging
from migemox.pipeline.constraints import apply_couple_constraints
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from migemox.pipeline.io_utils import load_model_and_constraints, log_with_timestamp
from migemox.pipeline.model_utils import reset_solver
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _append_fecalmax_failure_row(
    diet_mod_dir: str,
    model_name: str,
    sample_id: str,
    met_id: str,
    ex_rxn_id: str,
    stage: str,
    error_message: str,
    failing_iex: Optional[str] = None,
) -> None:
    """
    Append a single row describing a fecal_max failure to a CSV file in the
    Diet/Debug directory for this run.

    Columns:
        model_name, sample_id, met_id, ex_rxn_id, failing_iex, stage, error
    """
    debug_dir = Path(diet_mod_dir) / "Debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    csv_path = debug_dir / "fecalmax_failures.csv"

    row = {
        "model_name": model_name,
        "sample_id": sample_id,
        "met_id": met_id,
        "ex_rxn_id": ex_rxn_id,
        "failing_iex": failing_iex if failing_iex is not None else "",
        "stage": stage,
        "error": error_message.replace("\n", " "),
    }

    df = pd.DataFrame([row])
    # Append if file exists, else create with header
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, mode="w", header=True, index=False)

def _save_debug_model_with_constraints(
    model: object,
    model_name: str,
    diet_mod_dir: str,
    sample_id: str,
    met_id: str,
    model_data: Dict,
    tag: str = "fecalmax_failure",
) -> None:
    """
    Save the current model (after apply_couple_constraints) to SBML, and also
    save the coupling constraints (C, d, dsense, ctrs) to a .npz file so the
    exact state can be reconstructed in a notebook.
    """
    try:
        debug_dir = Path(diet_mod_dir) / "Debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{model_name}_{tag}_{sample_id}_{met_id}"
        sbml_path = debug_dir / f"{base_name}.sbml"
        npz_path = debug_dir / f"{base_name}_constraints.npz"

        # Save SBML
        write_sbml_model(model, sbml_path)

        # Save constraints
        C = model_data.get('C')
        d = model_data.get('d')
        dsense = model_data.get('dsense')
        ctrs = model_data.get('ctrs')
        np.savez_compressed(
            npz_path,
            C_data=C.data if C is not None else np.array([]),
            C_indices=C.indices if C is not None else np.array([], dtype=int),
            C_indptr=C.indptr if C is not None else np.array([0], dtype=int),
            C_shape=C.shape if C is not None else (0, 0),
            d=d,
            dsense=dsense,
            ctrs=ctrs,
        )

        logger.warning(f"Saved debug model to {sbml_path}")
        logger.warning(f"Saved debug constraints to {npz_path}")
    except Exception as e:
        logger.error(f"Failed to save debug model for {model_name}, {met_id}, {sample_id}: {e}")
    
def _get_sample_id_from_model_name(model_name: str) -> str:
    """
    Extract the original sample ID from a diet-adapted model filename.

    Expected pattern: 'microbiota_model_diet_<sample_name>'.
    Raises if the pattern does not match.
    """
    prefix = "microbiota_model_diet_"
    if not model_name.startswith(prefix):
        raise ValueError(
            f"Model name '{model_name}' does not follow expected pattern "
            f"'{prefix}<sample_name>'."
        )
    return model_name[len(prefix):]
    
def _get_exchange_reactions(model: object, mets_list: Optional[List[str]] = None, mets_as_iex=False) -> List[str]:
    """
    Extract relevant exchange reactions from model
    
    if mets_as_iex is set to True, assumes mets_list metabolites have been formatted like 
    this prior to passing into this function.

    ```python
    [f"IEX_{m}[u]tr" for m in mets_list]
    ```

    Otherwise, assumes metabolites just have metabolite ID
    """
    all_iex = [rxn.id for rxn in model.reactions if 'IEX_' in rxn.id]
    if not mets_list:
        # All reactions containing 'IEX_'
        return all_iex
    else:
        # Only reactions matching provided metabolites
        if mets_as_iex:
            return [rxn for rxn in all_iex if any(m in rxn for m in mets_list)]
        else:
            return [rxn for rxn in all_iex if any(f"IEX_{m}[u]tr" in rxn for m in mets_list)]
        

def _perform_fva(model: object, rxns_in_model: List[str], solver: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Perform flux variability analysis with fallbacks"""
    try:
        fva_result = flux_variability_analysis(
            model, reaction_list=rxns_in_model,
            fraction_of_optimum=0.9999, processes=4)
        
        min_fluxes, max_fluxes = fva_result['minimum'].to_dict(), fva_result['maximum'].to_dict()
        return min_fluxes, max_fluxes
        
    except Exception as e:
        logger.warning(f"FVA failed, falling back to individual FBA: {str(e)}")
        min_fluxes, max_fluxes = {}, {}
        
        for rxn_id in rxns_in_model:
            try:
                model.objective = rxn_id
                sol_min = model.optimize(objective_sense='minimize')
                min_fluxes[rxn_id] = sol_min.objective_value if sol_min.status == 'optimal' else 0
                
                sol_max = model.optimize(objective_sense='maximize')
                max_fluxes[rxn_id] = sol_max.objective_value if sol_max.status == 'optimal' else 0
            except Exception as rxn_e:
                logger.error(f"Failed to optimize reaction {rxn_id}: {str(rxn_e)}")
                min_fluxes[rxn_id] = 0
                max_fluxes[rxn_id] = 0
        
        return min_fluxes, max_fluxes

def _fva_min_max_for_reactions(
    model: object,
    rxn_ids: List[str],
    infeasible: Literal['raise', 'warn'] = 'raise'
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Run flux variability analysis on the given reactions *without* imposing
    a biomass fraction-of-optimum constraint (i.e., use current model bounds
    as-is). Returns dicts of min and max fluxes. Raises/warns on failure.
    """
    try:
        with model:
            fva_result = flux_variability_analysis(
                model,
                reaction_list=rxn_ids,
                fraction_of_optimum=None,  # do NOT constrain to optimum objective
                processes=1,               # single-process for numerical stability
            )
        min_fluxes = fva_result['minimum'].to_dict()
        max_fluxes = fva_result['maximum'].to_dict()
        return min_fluxes, max_fluxes
    except Exception as e:
        if infeasible == 'raise':
            raise RuntimeError(
                f"FVA infeasible or failed for reactions {rxn_ids} in model {model.name}: {str(e)}"
            ) from e
        elif infeasible == 'warn':
            logger.warning(
                f"FVA failed for reactions {rxn_ids} in model {model.name}: {str(e)}. "
                f"Setting min/max to 0."
            )
            min_fluxes = {rid: 0.0 for rid in rxn_ids}
            max_fluxes = {rid: 0.0 for rid in rxn_ids}
            return min_fluxes, max_fluxes
        else:
            raise ValueError(f"Invalid setting for `infeasible`: {infeasible}")

def _all_max_then_min_for_reactions(
    model: object,
    rxn_ids: List[str],
    solver: str,
    infeasible: Literal['raise', 'warn'] = 'raise',
    max_retries: int = 5,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    For each reaction in rxn_ids, first compute its maximal flux, then in a
    second pass compute its minimal flux, using the current model bounds.

    For each individual LP (max or min), if the solver status is not 'optimal',
    this function will:
        - retry up to max_retries times, each time calling reset_solver(model, solver),
        - and only then raise (when infeasible='raise') or warn+set 0 if infeasible='warn'.

    This is intended to mitigate sporadic numerical infeasibilities due to
    solver state without changing the underlying LP.
    """
    max_fluxes: Dict[str, float] = {}
    min_fluxes: Dict[str, float] = {}

    # First pass: all maximizations
    for rxn_id in rxn_ids:
        log_with_timestamp(f"[all_max] Reaction: {rxn_id}")
        success = False
        last_status = None

        for attempt in range(1, max_retries + 1):
            with model:
                rxn = model.reactions.get_by_id(rxn_id)
                model.objective = rxn
                log_with_timestamp(f"  maximizing... (attempt {attempt}/{max_retries})")
                sol_max = model.optimize(objective_sense="maximize")
                last_status = sol_max.status

            if sol_max.status == "optimal":
                max_fluxes[rxn_id] = sol_max.objective_value
                success = True
                break
            else:
                log_with_timestamp(
                    f"  WARNING: max solve attempt {attempt} for {rxn_id} "
                    f"returned status '{sol_max.status}'. Resetting solver and retrying."
                )
                # Reset solver state before next attempt
                reset_solver(model, solver)

        if not success:
            if infeasible == "raise":
                raise RuntimeError(
                    f"Maximization infeasible or non-optimal for reaction {rxn_id} "
                    f"after {max_retries} attempts: last status {last_status}"
                )
            elif infeasible == "warn":
                max_fluxes[rxn_id] = 0.0
                print(
                    f"WARNING: max infeasible/non-optimal for {rxn_id} "
                    f"after {max_retries} attempts (last status {last_status}). "
                    f"Setting max flux to 0."
                )
            else:
                raise ValueError(f"Invalid setting for `infeasible`: {infeasible}")

    # Second pass: all minimizations
    for rxn_id in rxn_ids:
        log_with_timestamp(f"[all_min] Reaction: {rxn_id}")
        success = False
        last_status = None

        for attempt in range(1, max_retries + 1):
            with model:
                rxn = model.reactions.get_by_id(rxn_id)
                model.objective = rxn
                log_with_timestamp(f"  minimizing... (attempt {attempt}/{max_retries})")
                sol_min = model.optimize(objective_sense="minimize")
                last_status = sol_min.status

            if sol_min.status == "optimal":
                min_fluxes[rxn_id] = sol_min.objective_value
                success = True
                break
            else:
                log_with_timestamp(
                    f"  WARNING: min solve attempt {attempt} for {rxn_id} "
                    f"returned status '{sol_min.status}'. Resetting solver and retrying."
                )
                reset_solver(model, solver)

        if not success:
            if infeasible == "raise":
                raise RuntimeError(
                    f"Minimization infeasible or non-optimal for reaction {rxn_id} "
                    f"after {max_retries} attempts: last status {last_status}"
                )
            elif infeasible == "warn":
                min_fluxes[rxn_id] = 0.0
                print(
                    f"WARNING: min infeasible/non-optimal for {rxn_id} "
                    f"after {max_retries} attempts (last status {last_status}). "
                    f"Setting min flux to 0."
                )
            else:
                raise ValueError(f"Invalid setting for `infeasible`: {infeasible}")

    return min_fluxes, max_fluxes

def _min_max_flux_per_reaction(
    model: object,
    rxn_ids: List[str],
    infeasible: Literal['raise', 'warn'] = 'raise'
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute min/max flux for each reaction in rxn_ids without imposing
    a fraction-of-optimum constraint on the existing model objective.
    Uses context managers to ensure model state is restored after each solve.
    """
    min_fluxes, max_fluxes = {}, {}

    for rxn_id in rxn_ids:
        log_with_timestamp(f"Reaction: {rxn_id}")

        # Minimize
        with model:
            rxn = model.reactions.get_by_id(rxn_id)
            model.objective = rxn

            log_with_timestamp("minimizing...")
            sol_min = model.optimize(objective_sense='minimize')
            if sol_min.status != 'optimal':
                if infeasible == 'raise':
                    raise RuntimeError(
                        f"Minimization infeasible or non-optimal for reaction {rxn_id}: status {sol_min.status}"
                    )
                elif infeasible == 'warn':
                    # NOTE: may want to modify the warn functionality to be consistent with MMT behavior.
                    # However, as of April 2026, the warn functionality is not being used by MiGEMox.
                    min_fluxes[rxn_id] = 0.0
                    print(f"WARNING: solver status was {sol_min.status} for rxn_id {rxn_id} in model {model.name}")
                else:
                    raise ValueError(f"Invalid setting for `infeasible`: {infeasible}")
            else:
                min_fluxes[rxn_id] = sol_min.objective_value

        # Maximize
        with model:
            rxn = model.reactions.get_by_id(rxn_id)
            model.objective = rxn

            log_with_timestamp("maximizing...")
            sol_max = model.optimize(objective_sense='maximize')
            if sol_max.status != 'optimal':
                if infeasible == 'raise':
                    raise RuntimeError(
                        f"Maximization infeasible or non-optimal for reaction {rxn_id}: status {sol_max.status}"
                    )
                elif infeasible == 'warn':
                    max_fluxes[rxn_id] = 0.0
                    print(f"WARNING: solver status was {sol_max.status} for rxn_id {rxn_id} in model {model.name}")
                else:
                    raise ValueError(f"Invalid setting for `infeasible`: {infeasible}")
            else:
                max_fluxes[rxn_id] = sol_max.objective_value

    log_with_timestamp("success!")
    return min_fluxes, max_fluxes

def _process_batch_parallel(
    current_batch: List[Path],
    diet_mod_dir: str,
    mets_list: Optional[List[str]], 
    net_production_dict: Optional[Dict[str, Dict[str, float]]],
    solver: str,
    workers: int,
    method: str,
    raw_fva_df: Optional[pd.DataFrame],
    fraction: float = 0.98,
) -> Dict:
    """Process batch of models in parallel"""
    batch_results = {}
    
    # --- SEQUENTIAL VERSION FOR DEBUGGING ---
    # for model_file in tqdm(current_batch, desc="Processing batches (sequential)"):
    #     result = _process_single_model(
    #         model_file,
    #         diet_mod_dir,
    #         mets_list,
    #         net_production_dict,
    #         solver,
    #         method,
    #         raw_fva_df if method in {"fecal_max", "net_exchange"} else None,
    #         fraction: float = 0.98,
    #     )
    #     if result is not None:
    #         batch_results[result['model_name']] = {
    #             'min_fluxes': result['min_fluxes'],
    #             'max_fluxes': result['max_fluxes'],
    #             'rxns': result['rxns']
    #         }

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _process_single_model,
                model_file,
                diet_mod_dir,
                mets_list,
                net_production_dict,
                solver,
                method,
                raw_fva_df if method in {"fecal_max", "net_exchange"} else None,
                fraction,
            )
            for model_file in current_batch
        ]
    
        for future in tqdm(as_completed(futures), total=len(futures), 
                        desc=f'Processing batches'):
            result = future.result()
            if result is not None:
                batch_results[result['model_name']] = {
                    'min_fluxes': result['min_fluxes'],
                    'max_fluxes': result['max_fluxes'],
                    'rxns': result['rxns']
                }
    
    return batch_results

def _process_single_model(
    model_file: Path,
    diet_mod_dir: str,
    mets_list: Optional[List[str]], 
    net_production_dict: Optional[Dict[str, Dict[str, float]]],
    solver: str,
    method: str,
    raw_fva_df: Optional[pd.DataFrame],
    fraction: float = 0.98,
) -> Optional[Dict]:
    """
    Process a single model file
    
    Returns:
        Dict with model results or None if failed
    """
    model_name = model_file.stem
    try:
        model, C, d, dsense, ctrs = load_model_and_constraints(
            model_name, diet_mod_dir, model_type="standard", save_format="sbml")
        
        model_data = {
            'C': C,
            'd': d,
            'dsense': dsense,
            'ctrs': ctrs
        }
        model.solver = solver
        model = apply_couple_constraints(model, model_data)
        if model is None: return None
            
        min_fluxes, max_fluxes, rxns = {}, {}, []
        
        if method == "biomass":
            log_with_timestamp("Using method 'biomass'")
            if net_production_dict:
                # Only process mets with a constraint for this model
                sample_id = _get_sample_id_from_model_name(model_name)

                for met, model_fluxes in tqdm(net_production_dict.items()):
                    ex_rxn_id = f"EX_{met}[fe]"
                    iex_rxn_ids = _get_exchange_reactions(model, [met])
                    # Set EX lower bound temporarily
                    if ex_rxn_id in model.reactions and iex_rxn_ids:
                        ex_rxn = model.reactions.get_by_id(ex_rxn_id)
                        orig_lb = ex_rxn.lower_bound

                        # OLD: ex_rxn.lower_bound = model_fluxes[model_name.split('_')[-1]]
                        ex_rxn.lower_bound = model_fluxes[sample_id]

                         # Perform FVA on only IEX rxns associated with current metabolite
                        minf, maxf = _perform_fva(model, iex_rxn_ids, solver)
                        min_fluxes.update(minf)
                        max_fluxes.update(maxf)
                        rxns.extend(iex_rxn_ids)
                        ex_rxn.lower_bound = orig_lb
            else:
                rxns_in_model = _get_exchange_reactions(model, mets_list, mets_as_iex=True)
                if not rxns_in_model:
                    logger.warning(f"No exchange reactions found in model {model_name}")
                    return None
                min_fluxes, max_fluxes = _perform_fva(model, rxns_in_model, solver)
                rxns = rxns_in_model

        elif method == "fecal_max":
            log_with_timestamp("Using method 'fecal_max'")
            if mets_list is None:
                raise ValueError("mets_list must be provided when method='fecal_max'.")
            if raw_fva_df is None:
                raise ValueError("raw_fva_df must be provided when method='fecal_max'.")

            # Tolerances for comparing raw vs local fecal_max
            fecal_max_atol = 1e-6
            fecal_max_rtol = 1e-3  # 0.1% relative

            # Number of decimal places for rounding based on atol (e.g. 1e-6 -> 6 decimals)
            if fecal_max_atol < 1.0:
                decimal_places = int(round(-math.log10(fecal_max_atol)))
            else:
                decimal_places = 0

            # Basic structural check on raw_fva_df
            if (not isinstance(raw_fva_df.index, pd.MultiIndex) or
                    raw_fva_df.index.names != ['Sample', 'Reaction']):
                raise ValueError(
                    "raw_fva_df must have a MultiIndex with levels ['Sample', 'Reaction'] "
                    "as built in run_community_fva."
                )
            if 'max_flux_fecal' not in raw_fva_df.columns:
                raise ValueError("raw_fva_df must contain a 'max_flux_fecal' column.")

            # Sample ID convention: microbiota_model_diet_<sample_name>
            sample_id = _get_sample_id_from_model_name(model_name)
            if sample_id not in raw_fva_df.index.get_level_values('Sample'):
                raise KeyError(
                    f"Sample ID '{sample_id}' not found in raw_fva_df index."
                )

            log_with_timestamp("Initial checks passed")

            for iex_pattern in mets_list:
                log_with_timestamp(f"Processing IEX pattern {iex_pattern}")
                # iex_pattern is like "IEX_glu_D[u]tr"
                if not iex_pattern.startswith("IEX_") or not iex_pattern.endswith("[u]tr"):
                    raise ValueError(f"Unexpected IEX metabolite format: {iex_pattern}")
                met_id = iex_pattern[len("IEX_"):-len("[u]tr")]

                ex_rxn_id = f"EX_{met_id}[fe]"
                if ex_rxn_id not in model.reactions:
                    msg = (f"Fecal exchange reaction {ex_rxn_id} not found in model {model_name}. "
                           f"Cannot apply 'fecal_max' for metabolite {met_id}.")
                    log_with_timestamp("ERROR: " + msg)
                    _append_fecalmax_failure_row(
                        diet_mod_dir,
                        model_name,
                        sample_id,
                        met_id,
                        ex_rxn_id,
                        stage="missing_ex_rxn",
                        error_message=msg,
                    )
                    raise RuntimeError(msg)

                ex_rxn = model.reactions.get_by_id(ex_rxn_id)
                orig_lb, orig_ub = ex_rxn.lower_bound, ex_rxn.upper_bound

                # Collect all IEX reactions for this metabolite
                pattern = f"_IEX_{met_id}[u]tr"
                iex_rxn_ids = [rxn.id for rxn in model.reactions if pattern in rxn.id]
                if not iex_rxn_ids:
                    # No microbe IEX for this metabolite in this model
                    log_with_timestamp(f"WARNING: No IEX reactions found for metabolite {met_id}. Skipping.")
                    continue

                # --- 1) Get upstream fecal max from raw_fva_df and round it ---
                try:
                    row = raw_fva_df.loc[(sample_id, ex_rxn_id)]
                except KeyError:
                    msg = (f"FVA results for sample '{sample_id}', reaction '{ex_rxn_id}' "
                           f"not found in raw_fva_df.")
                    log_with_timestamp("ERROR: " + msg)
                    _append_fecalmax_failure_row(
                        diet_mod_dir,
                        model_name,
                        sample_id,
                        met_id,
                        ex_rxn_id,
                        stage="missing_raw_fva",
                        error_message=msg,
                    )
                    raise RuntimeError(msg)

                fecal_max_raw = float(row['max_flux_fecal'])
                fecal_max_rounded = round(fecal_max_raw, decimal_places)

                if fecal_max_rounded <= 1e-10:
                    msg = (f"Maximal fecal secretion (rounded) for {ex_rxn_id} in sample {sample_id} "
                           f"is non-positive ({fecal_max_rounded}); cannot apply 'fecal_max' method.")
                    log_with_timestamp("ERROR: " + msg)
                    _append_fecalmax_failure_row(
                        diet_mod_dir,
                        model_name,
                        sample_id,
                        met_id,
                        ex_rxn_id,
                        stage="raw_fecal_max_nonpositive", 
                        error_message=msg
                    )
                    raise RuntimeError(msg)

                # --- 2) First attempt: use 0.98 * rounded raw fecal_max ---
                log_with_timestamp(f"Using fraction of fecal_max: {fraction}")
                new_lb = max(orig_lb, fraction * fecal_max_rounded)
                if new_lb > orig_ub + 1e-10:
                    msg = (f"Inconsistent bounds for {ex_rxn_id} after applying {fraction}*fecal_max_rounded "
                           f"in model {model_name} (new_lb={new_lb}, orig_ub={orig_ub}).")
                    log_with_timestamp("ERROR: " + msg)
                    _append_fecalmax_failure_row(
                        diet_mod_dir,
                        model_name,
                        sample_id,
                        met_id,
                        ex_rxn_id,
                        stage="lb_rounded_inconsistent", 
                        error_message=msg
                    )
                    raise RuntimeError(msg)

                ex_rxn.lower_bound = new_lb

                # Global feasibility check
                with model:
                    sol_feas = model.optimize()
                    if sol_feas.status != 'optimal':
                        _save_debug_model_with_constraints(
                            model,
                            model_name=model_name,
                            diet_mod_dir = diet_mod_dir,
                            sample_id = sample_id,
                            met_id = met_id,
                            model_data = model_data,
                            tag = "first_feasibility_check_failure"
                        )
                if sol_feas.status != 'optimal':
                    msg = (f"Model {model_name} infeasible after applying fecal_max constraint "
                           f"on {ex_rxn_id} (lb={new_lb}). Status: {sol_feas.status}")
                    log_with_timestamp("ERROR: " + msg)
                    _append_fecalmax_failure_row(
                        diet_mod_dir,
                        model_name,
                        sample_id,
                        met_id,
                        ex_rxn_id,
                        stage="global_feasibility_rounded", 
                        error_message=msg
                    )
                    ex_rxn.lower_bound = orig_lb
                    ex_rxn.upper_bound = orig_ub
                    raise RuntimeError(msg)

                log_with_timestamp('feasibility check passed (rounded raw fecal_max)')

                # Helper to actually run min/max on IEX reactions
                def _run_iex_minmax() -> Tuple[Dict[str, float], Dict[str, float]]:
                    log_with_timestamp("running IEX min/max (manual)")
                    minf, maxf = _min_max_flux_per_reaction(model, iex_rxn_ids, infeasible='raise')
                    log_with_timestamp("IEX min/max complete (manual)")
                    return minf, maxf
                
                # Helper: run all max then all min for IEX reactions
                def _run_all_max_then_min() -> Tuple[Dict[str, float], Dict[str, float]]:
                    log_with_timestamp("running IEX all-max-then-min (manual)")
                    minf, maxf = _all_max_then_min_for_reactions(
                        model, iex_rxn_ids, solver=solver, infeasible="raise"
                    )
                    log_with_timestamp("IEX all-max-then-min complete (manual)")
                    return minf, maxf

                try:
                    # Attempt 1: current solver state, rounded fecal_max_rounded LB
                    try:
                        minf, maxf = _run_all_max_then_min()
                        for rid in iex_rxn_ids:
                            min_fluxes[rid] = minf[rid]
                            max_fluxes[rid] = maxf[rid]
                        rxns.extend(iex_rxn_ids)
                        continue  # success, move to next metabolite

                    except RuntimeError as e1:
                        msg1 = str(e1)
                        if "infeasible" not in msg1 and "non-optimal" not in msg1:
                            # Unexpected error: re-raise
                            raise

                        log_with_timestamp(
                            f"WARNING: IEX all-max-then-min infeasible for {met_id} "
                            f"(sample {sample_id}) with rounded fecal_max. "
                            f"Resetting solver and retrying."
                        )

                    # Attempt 2: reset solver state, same LB, run all-max-then-min again
                    try:
                        # Reset solver to clear any internal state
                        reset_solver(model, solver)
                        minf, maxf = _run_all_max_then_min()
                        for rid in iex_rxn_ids:
                            min_fluxes[rid] = minf[rid]
                            max_fluxes[rid] = maxf[rid]
                        rxns.extend(iex_rxn_ids)
                        continue  # success

                    except RuntimeError as e2:
                        msg2 = str(e2)
                        if "infeasible" not in msg2 and "non-optimal" not in msg2:
                            raise

                        log_with_timestamp(
                            f"WARNING: IEX all-max-then-min still infeasible for {met_id} "
                            f"(sample {sample_id}) after solver reset. "
                            f"Recomputing fecal_max locally and trying once more."
                        )

                    # Attempt 3: reset solver, recompute local fecal_max via FVA, set new LB,
                    # global feasibility check, then run all-max-then-min again
                    failing_iex = None
                    if "reaction " in msg2:
                        failing_iex = msg2.split("reaction ", 1)[-1].split(":", 1)[0]

                    try:
                        # reset solver again
                        reset_solver(model, solver)
                        # Recompute local fecal_max with biomass fraction 0.99
                        with model:
                            fva_local = flux_variability_analysis(
                                model,
                                reaction_list=[ex_rxn_id],
                                fraction_of_optimum=0.99,
                                processes=1,
                            )
                        fecal_max_local = float(fva_local.loc[ex_rxn_id, "maximum"])

                        if fecal_max_local <= 1e-10:
                            raise RuntimeError(
                                f"Local maximal fecal secretion for {ex_rxn_id} in sample {sample_id} "
                                f"is non-positive ({fecal_max_local}); cannot apply 'fecal_max' fallback."
                            )

                        diff = abs(fecal_max_local - fecal_max_raw)
                        tol = fecal_max_atol + fecal_max_rtol * max(1.0, abs(fecal_max_raw))
                        log_with_timestamp(f"fecal_max_local: {fecal_max_local}")
                        log_with_timestamp(f"fecal_max_raw:   {fecal_max_raw}")
                        log_with_timestamp(f"diff:           {diff}, tol: {tol}")

                        if diff > tol:
                            raise RuntimeError(
                                f"Inconsistent fecal_max for {ex_rxn_id} in sample {sample_id}: "
                                f"raw_fva_df={fecal_max_raw}, local={fecal_max_local}, "
                                f"diff={diff} > tol={tol}."
                            )

                        # Use local fecal_max for LB (same fraction as above)
                        new_lb_local = max(orig_lb, fraction * fecal_max_local)
                        if new_lb_local > orig_ub + 1e-10:
                            raise RuntimeError(
                                f"Inconsistent bounds for {ex_rxn_id} after applying {fraction}*fecal_max_local "
                                f"in model {model_name} (new_lb={new_lb_local}, orig_ub={orig_ub})."
                            )
                        ex_rxn.lower_bound = new_lb_local

                        # Global feasibility check under local fecal_max
                        with model:
                            sol_feas2 = model.optimize()
                        if sol_feas2.status != "optimal":
                            raise RuntimeError(
                                f"Model {model_name} infeasible even after local fecal_max fallback "
                                f"on {ex_rxn_id} (lb={new_lb_local}). Status: {sol_feas2.status}"
                            )

                        log_with_timestamp("feasibility check passed (local fecal_max fallback)")

                        # Reset solver and run all-max-then-min a final time
                        reset_solver(model, solver)
                        minf, maxf = _run_all_max_then_min()
                        for rid in iex_rxn_ids:
                            min_fluxes[rid] = minf[rid]
                            max_fluxes[rid] = maxf[rid]
                        rxns.extend(iex_rxn_ids)
                        continue  # success

                    except Exception as e3:
                        msg3 = (
                            f"Final all-max-then-min attempt failed for {met_id} (sample {sample_id}) "
                            f"even after local fecal_max recomputation: {e3}"
                        )
                        log_with_timestamp("ERROR: " + msg3)
                        _append_fecalmax_failure_row(
                            diet_mod_dir,
                            model_name,
                            sample_id,
                            met_id,
                            ex_rxn_id,
                            stage="final_all_max_min_failure",
                            error_message=msg3,
                            failing_iex=failing_iex,
                        )

                        # Save full debug model at the point of final failure
                        _save_debug_model_with_constraints(
                            model=model,
                            model_name=model_name,
                            diet_mod_dir=diet_mod_dir,
                            sample_id=sample_id,
                            met_id=met_id,
                            model_data=model_data,
                            tag="final_all_max_min_failure",
                        )

                        # Hard error: propagate up so the run fails noisily
                        raise

                finally:
                    # Restore original fecal bounds before moving on
                    ex_rxn.lower_bound = orig_lb
                    ex_rxn.upper_bound = orig_ub

        elif method == "net_exchange":
            log_with_timestamp("Using method 'net_exchange'")
            if mets_list is None:
                raise ValueError("mets_list must be provided when method='net_exchange'.")
            if raw_fva_df is None:
                raise ValueError("raw_fva_df must be provided when method='net_exchange'.")

            # Basic structural check on raw_fva_df, similar to fecal_max
            if not isinstance(raw_fva_df.index, pd.MultiIndex) or raw_fva_df.index.names != ['Sample', 'Reaction']:
                raise ValueError(
                    "raw_fva_df must have a MultiIndex with levels ['Sample', 'Reaction'] as built in run_community_fva."
                )

            required_cols = ['min_flux_diet', 'max_flux_diet', 'min_flux_fecal', 'max_flux_fecal']
            for col in required_cols:
                if col not in raw_fva_df.columns:
                    raise ValueError(
                        f"raw_fva_df must contain a '{col}' column for method='net_exchange'."
                    )

            sample_id = _get_sample_id_from_model_name(model_name)
            if sample_id not in raw_fva_df.index.get_level_values('Sample'):
                raise KeyError(
                    f"Sample ID '{sample_id}' not found in raw_fva_df index."
                )

            tol = 1e-10

            for iex_pattern in mets_list:
                if not iex_pattern.startswith("IEX_") or not iex_pattern.endswith("[u]tr"):
                    raise ValueError(f"Unexpected IEX metabolite format: {iex_pattern}")
                met_id = iex_pattern[len("IEX_"):-len("[u]tr")]

                fecal_ex_rxn_id = f"EX_{met_id}[fe]"
                diet_ex_rxn_id = f"Diet_EX_{met_id}[d]"

                if fecal_ex_rxn_id not in model.reactions:
                    raise RuntimeError(f"Fecal exchange reaction {fecal_ex_rxn_id} not found in model {model_name}.")
                if diet_ex_rxn_id not in model.reactions:
                    raise RuntimeError(f"Diet exchange reaction {diet_ex_rxn_id} not found in model {model_name}.")

                # Look up FVA bounds for diet and fecal exchanges
                try:
                    row_diet = raw_fva_df.loc[(sample_id, fecal_ex_rxn_id)]
                except KeyError:
                    raise RuntimeError(
                        f"Diet FVA results for sample '{sample_id}', reaction '{fecal_ex_rxn_id}' "
                        f"not found in raw_fva_df."
                    )
                try:
                    row_fecal = raw_fva_df.loc[(sample_id, fecal_ex_rxn_id)]
                except KeyError:
                    raise RuntimeError(
                        f"Fecal FVA results for sample '{sample_id}', reaction '{fecal_ex_rxn_id}' "
                        f"not found in raw_fva_df."
                    )

                min_diet = float(row_diet['min_flux_diet'])
                max_fecal = float(row_fecal['max_flux_fecal'])

                # Signed net-exchange indicator
                S = min_diet + max_fecal

                fecal_rxn = model.reactions.get_by_id(fecal_ex_rxn_id)
                diet_rxn = model.reactions.get_by_id(diet_ex_rxn_id)

                # Build linear constraint depending on sign of S
                interface = model.solver.interface
                expr = diet_rxn.flux_expression + fecal_rxn.flux_expression
                constr_name = f"net_exchange_{met_id}_{sample_id}"

                # Decide which constraint to impose
                if S > tol:
                    # Secretion-capable: v_diet + v_fecal >= 0.98 * S
                    # NOTE: we use 0.98 because 0.99 can cause random infeasibilities
                    # due to numerical tolerance issues
                    net_constr = interface.Constraint(expr, lb=0.98 * S, name=constr_name)
                elif S < -tol:
                    # Uptake-capable: v_diet + v_fecal <= 0.98 * S  (S is negative)
                    net_constr = interface.Constraint(expr, ub=0.98 * S, name=constr_name)
                else:
                    # |S| ~ 0: no meaningful net exchange; skip constraint + FVA for this met
                    # (or you could still do unconstrained FVA if you prefer)
                    continue

                model.add_cons_vars([net_constr])

                try:
                    pattern = f"_IEX_{met_id}[u]tr"
                    iex_rxn_ids = [rxn.id for rxn in model.reactions if pattern in rxn.id]
                    if not iex_rxn_ids:
                        raise RuntimeError(
                            f"No IEX reactions found for metabolite {met_id} in model {model_name}."
                        )

                    minf, maxf = _min_max_flux_per_reaction(model, iex_rxn_ids, infeasible='raise')
                    min_fluxes.update(minf)
                    max_fluxes.update(maxf)
                    rxns.extend(iex_rxn_ids)
                finally:
                    model.remove_cons_vars([net_constr])
        
        return {
            'model_name': model_name,
            'min_fluxes': min_fluxes,
            'max_fluxes': max_fluxes,
            'rxns': rxns
        }
        
    except Exception as e:
        logger.error(f"Failed to process model {model_name}: {str(e)}")
        if method in {"net_exchange", "fecal_max"}:
            # Propagate error so the whole run fails visibly
            raise
        return None
    
def _round_df_with_format(df: pd.DataFrame, fmt: str) -> pd.DataFrame:
    """
    Round all float values in a DataFrame using a Python format spec, e.g. ':.2f' or ':.3g'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing float values (NaNs allowed).
    fmt : str
        A format specifier such as ':.2f', ':.3g', '.2f', '.3g', etc.

    Returns
    -------
    pd.DataFrame
        DataFrame with values rounded according to the given format.
    """
    # Allow either ':.2f' or '.2f'
    if fmt.startswith(':'):
        fmt = fmt[1:]  # strip leading ':'

    def _round_value(x):
        # Leave NaNs as-is
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return x
        # Format then cast back to float
        return float(format(x, fmt))

    # Apply elementwise
    rounded = df.applymap(_round_value)

    return rounded

def _calculate_flux_spans(min_df: pd.DataFrame, max_df: pd.DataFrame, precision: str=None) -> pd.DataFrame:
    """Calculate flux spans with proper handling of positive/negative fluxes"""

    # If the user specifies a precision to use, truncate the numbers to use that precision
    if precision is not None:
        min_df = _round_df_with_format(min_df, precision)
        max_df = _round_df_with_format(max_df, precision)

    min_vals = min_df.values
    max_vals = max_df.values
    
    # Create result array with same shape
    spans = np.zeros_like(min_vals, dtype=float)
    
    mask1 = (max_vals > 1e-10) & (min_vals > 1e-10)
    mask2 = (max_vals > 1e-10) & (min_vals < -1e-10)
    mask3 = (max_vals < -1e-10) & (min_vals < -1e-10)
    mask4 = (max_vals > 1e-10) & (np.abs(min_vals) < 1e-10)
    mask5 = (min_vals < -1e-10) & (np.abs(max_vals) < 1e-10)
    
    spans[mask1] = max_vals[mask1] - min_vals[mask1]
    spans[mask2] = max_vals[mask2] + np.abs(min_vals[mask2])
    spans[mask3] = np.abs(min_vals[mask3]) - np.abs(max_vals[mask3])
    spans[mask4] = max_vals[mask4]
    spans[mask5] = np.abs(min_vals[mask5])
    
    return pd.DataFrame(spans, index=min_df.index, columns=min_df.columns)

def _clean_and_filter_dataframes(min_df: pd.DataFrame, max_df: pd.DataFrame, 
                                flux_spans_df: pd.DataFrame, tolerance: float = 1e-7) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combined cleaning and filtering for better performance"""
    dataframes = [min_df, max_df, flux_spans_df]
    for df in dataframes:
        # Clean reaction names
        df.index = df.index.str.replace('_IEX', '', regex=False)\
                   .str.replace('[u]tr', '', regex=False)\
                   .str.replace('pan', '', regex=False)
        
        # Clean model names
        df.columns = df.columns.str.replace('microbiota_model_samp_', '', regex=False)\
                     .str.replace('microbiota_model_diet_', '', regex=False)
    
    # Filter zero rows once for all dataframes
    min_df_filtered = min_df[min_df.abs().sum(axis=1) >= tolerance]
    max_df_filtered = max_df[max_df.abs().sum(axis=1) >= tolerance]
    flux_spans_filtered = flux_spans_df[flux_spans_df.abs().sum(axis=1) >= tolerance]
    
    return min_df_filtered, max_df_filtered, flux_spans_filtered


def predict_microbe_contributions(
    diet_mod_dir: str,
    res_path: Optional[str] = None, 
    mets_list: Optional[List[str]] = None,
    net_production_dict: Optional[Dict[str, Dict[str, float]]] = None,
    solver: str = 'cplex',
    workers: int = 1,
    method: str = "biomass",
    raw_fva_df: Optional[pd.DataFrame] = None,
    precision: str = None,
    fraction: float = 0.98,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Predicts the minimal and maximal fluxes through internal exchange
    reactions in microbes in a list of microbiome community models for a list
    of metabolites. This allows for the prediction of the individual
    contribution of each microbe to total metabolite uptake and secretion by
    the community.

    Args:
        diet_mod_dir: Directory containing diet-constrained models
        res_path: Where to store the results of strain-level contributions
        mets_list: List of VMH IDs for metabolites to analyze (Ex. 'ac', '2obut')
                   (default: all exchanged metabolites)
        net_production_dict: Dictionary mapping metabolite IDs to their net production rates
                             When supplied, LB of the corresponding exchange reaction will be 
                             temporarily set to the net production rate for each metabolite
        solver: Solver to use for solving FVA
        workers: Number of processes to use for parallelization
        method: "biomass" (default), "fecal_max", or "net_exchange". TODO: add more description for biomass and fecal_max
        * 'net_exchange': Use signed FVA bounds on diet + fecal exchanges to
          constrain total net exchange (v_diet + v_fecal) per metabolite:
          if min_diet + max_fecal > 0, treat as secretion-capable and impose
          a lower bound; if < 0, treat as uptake-capable and impose an upper
          bound. Then run FVA on IEX reactions.
        raw_fa_df: only required/used for method="fecal_max". Ignored otherwise. The DataFrame of raw
        FVA results (must contain the fecal max)
        precision: A format specifier such as ':.2f', ':.3g', '.2f', '.3g', etc. Used when calculating
        the flux spans to round the min and max fluxes to a certain number of decimal points or sig figs.
        fraction: Only used when computing fecal_max; the fraction of the fecal max secretion flux to 
        constrain the fecal exchange reaction to when maximizing and minimizing IEX reactions.

    Returns:
        minFluxes:  Minimal fluxes through analyzed exchange reactions,
                    corresponding to secretion fluxes for each microbe
        maxFluxes:  Maximal fluxes through analyzed exchange reactions,
                    corresponding to uptake fluxes for each microbe
        fluxSpans:  Range between min and max fluxes for analyzed
                    exchange reactions
    '''    

    if method not in {"biomass", "fecal_max", "net_exchange"}:
        raise ValueError(f"Unknown method '{method}'. "
                         "Expected 'biomass', 'fecal_max', or 'net_exchange'.")
    if method in {"fecal_max", "net_exchange"} and mets_list is None:
        raise ValueError(f"mets_list must be provided when method='{method}'.")
    if method == "fecal_max" and raw_fva_df is None:
        raise ValueError("raw_fva_df must be provided when method='fecal_max'.")
    if method == "net_exchange" and raw_fva_df is None:
        raise ValueError("raw_fva_df must be provided when method='net_exchange'.")

    res_path = Path.cwd() / 'Contributions' if not res_path else Path(res_path)
    os.makedirs(res_path, exist_ok=True)

    # Format met_list to match exchange reaction IDs if provided
    mets_list = [f"IEX_{m}[u]tr" for m in mets_list] if mets_list else None

    logger.info(f"Processing models from: {diet_mod_dir}")
    logger.info(f"Results will be saved to: {res_path}")
    logger.info(f"Analyzing {len(mets_list) if mets_list else 'all'} metabolites")

    # Gather all model files
    model_files = []
    for ext in ['.mat', '.sbml', '.xml']:
        model_files.extend([Path(f) for f in glob(f"*{ext}", root_dir=diet_mod_dir)])
    
    if not model_files:
        raise FileNotFoundError(f'No model files found in {diet_mod_dir}')
    
    logger.info(f"Found {len(model_files)} model files")

    # Check for partial results and resume if needed
    min_flux_file = res_path / 'minFluxes.csv'
    max_flux_file = res_path / 'maxFluxes.csv'
    
    if min_flux_file.exists() and max_flux_file.exists():
        logger.info("Found partial results, resuming from where left off")
        min_fluxes_df = pd.read_csv(min_flux_file, index_col=0)
        max_fluxes_df = pd.read_csv(max_flux_file, index_col=0)
        processed_models = set(min_fluxes_df.columns)
        remaining_models = [f for f in model_files if f.stem not in processed_models]
        logger.info(f"Resuming: {len(remaining_models)} models remaining")
    else:
        logger.info("Starting fresh analysis")
        min_fluxes_df = pd.DataFrame()
        max_fluxes_df = pd.DataFrame()
        remaining_models = model_files

    # Determine batch size based on number of models
    batch_size = 100 if len(remaining_models) > 200 else 25
    
    # Process models in batches
    for batch_start in range(0, len(remaining_models), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining_models))
        current_batch = remaining_models[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}: "
                   f"models {batch_start + 1}-{batch_end} of {len(remaining_models)}")
        
        # Process batch in parallel
        batch_results = _process_batch_parallel(
            current_batch, diet_mod_dir, mets_list,
            net_production_dict, solver, workers,
            method, raw_fva_df, fraction
        )

        for model_name, results in batch_results.items():
            if min_fluxes_df.empty:
                reactions = list(results['rxns'])
                min_fluxes_df = pd.DataFrame(index=reactions, columns=[model_name], dtype=float)
                max_fluxes_df = pd.DataFrame(index=reactions, columns=[model_name], dtype=float)
            else:
                # Add new column for this model
                min_fluxes_df[model_name] = 0.0
                max_fluxes_df[model_name] = 0.0
                
                # Add any new reactions as rows
                new_reactions = set(results['rxns']) - set(min_fluxes_df.index)
                for rxn in new_reactions:
                    min_fluxes_df.loc[rxn] = 0.0
                    max_fluxes_df.loc[rxn] = 0.0
            
            # Populate values for this model
            for rxn_id, min_val in results['min_fluxes'].items():
                min_fluxes_df.loc[rxn_id, model_name] = min_val
            for rxn_id, max_val in results['max_fluxes'].items():
                max_fluxes_df.loc[rxn_id, model_name] = max_val
            
            # Save intermediate results
            min_fluxes_df.to_csv(res_path / 'minFluxes.csv')
            max_fluxes_df.to_csv(res_path / 'maxFluxes.csv')
            
            logger.info(f"Saved intermediate results after batch {batch_start//batch_size + 1}")
    
    flux_spans_df = _calculate_flux_spans(min_fluxes_df, max_fluxes_df, precision)

    min_fluxes_df, max_fluxes_df, flux_spans_df = _clean_and_filter_dataframes(min_fluxes_df, max_fluxes_df, flux_spans_df)
    
    # Step 9: Save final results
    logger.info("Saving final results...")
    min_fluxes_df.to_csv(res_path / 'Microbe_Secretion.csv')
    max_fluxes_df.to_csv(res_path / 'Microbe_Uptake.csv')
    flux_spans_df.to_csv(res_path / 'Microbe_Flux_Spans.csv')

    # Remove temporary files from batch processing
    os.remove(min_flux_file)
    os.remove(max_flux_file)

    logger.info(f"Analysis complete! Results saved to {res_path}")
    
    return min_fluxes_df, max_fluxes_df, flux_spans_df
