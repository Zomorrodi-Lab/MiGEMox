import os
import pandas as pd
import numpy as np
import docplex
import cplex
from pathlib import Path
from cobra.io import load_matlab_model
from cobra.flux_analysis.variability import flux_variability_analysis
from typing import Optional, List, Tuple, Dict
import logging
from migemox.pipeline.constraints import apply_couple_constraints
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from migemox.pipeline.io_utils import load_model_and_constraints

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _load_model_safely(model_path: str) -> Optional[object]:
    """Handle model loading with fallbacks"""
    try:
        model = load_matlab_model(model_path)
        logger.debug(f"Successfully loaded model: {Path(model_path).stem}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {Path(model_path).stem}: {str(e)}")
        return None
    
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

def _process_batch_parallel(current_batch: List[Path], diet_mod_dir: str, mets_list: Optional[List[str]], 
                           net_production_dict: Optional[Dict[str, Dict[str, float]]], solver: str, workers: int) -> Dict:
    """Process batch of models in parallel"""
    batch_results = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_process_single_model, model_file, diet_mod_dir, mets_list, net_production_dict, solver)
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

def _process_single_model(model_file: Path, diet_mod_dir: str, mets_list: Optional[List[str]], 
                         net_production_dict: Optional[Dict[str, Dict[str, float]]], solver: str) -> Optional[Dict]:
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
        
        if net_production_dict:
            # Only process mets with a constraint for this model
            for met, model_fluxes in tqdm(net_production_dict.items()):
                ex_rxn_id = f"EX_{met}[fe]"
                iex_rxn_ids = _get_exchange_reactions(model, [met])
                # Set EX lower bound temporarily
                if ex_rxn_id in model.reactions and iex_rxn_ids:
                    ex_rxn = model.reactions.get_by_id(ex_rxn_id)
                    orig_lb = ex_rxn.lower_bound
                    ex_rxn.lower_bound = model_fluxes[model_name.split('_')[-1]]
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
        
        return {
            'model_name': model_name,
            'min_fluxes': min_fluxes,
            'max_fluxes': max_fluxes,
            'rxns': rxns
        }
        
    except Exception as e:
        logger.error(f"Failed to process model {model_name}: {str(e)}")
        return None

def _calculate_flux_spans(min_df: pd.DataFrame, max_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate flux spans with proper handling of positive/negative fluxes"""
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


def predict_microbe_contributions(diet_mod_dir: str, res_path: Optional[str] = None, 
                                  mets_list: Optional[List[str]] = None, net_production_dict: Optional[Dict[str, Dict[str, float]]] = None,
                                  solver: str = 'cplex',
                                  workers: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    Returns:
        minFluxes:  Minimal fluxes through analyzed exchange reactions,
                    corresponding to secretion fluxes for each microbe
        maxFluxes:  Maximal fluxes through analyzed exchange reactions,
                    corresponding to uptake fluxes for each microbe
        fluxSpans:  Range between min and max fluxes for analyzed
                    exchange reactions
    '''    

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
        batch_results = _process_batch_parallel(current_batch, diet_mod_dir, mets_list, net_production_dict, solver, workers)

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
    
    flux_spans_df = _calculate_flux_spans(min_fluxes_df, max_fluxes_df)

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
