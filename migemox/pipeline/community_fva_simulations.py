"""
Simulation Module for MiGEMox Pipeline

This module encapsulates the core logic for running metabolic simulations
on community models. It handles the application of dietary constraints,
setting of physiological bounds, optimization of models, and the
execution of Flux Variability Analysis (FVA) for individual samples.
"""

import cobra
from cobra.flux_analysis import flux_variability_analysis
import pandas as pd
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from migemox.pipeline.io_utils import load_model_and_constraints, save_model_and_constraints, log_with_timestamp
from migemox.pipeline.diet_adapter import adapt_vmh_diet_to_agora
from migemox.pipeline.constraints import apply_couple_constraints

# simulate_microbiota_models: define human-derived metabolites present in the gut: primary bile acids, amines, mucins, host glycans
HUMAN_METS = {
    'gchola': -10, 'tdchola': -10, 'tchola': -10, 'dgchol': -10,
    '34dhphe': -10, '5htrp': -10, 'Lkynr': -10, 'f1a': -1,
    'gncore1': -1, 'gncore2': -1, 'dsT_antigen': -1, 'sTn_antigen': -1,
    'core8': -1, 'core7': -1, 'core5': -1, 'core4': -1,
    'ha': -1, 'cspg_a': -1, 'cspg_b': -1, 'cspg_c': -1,
    'cspg_d': -1, 'cspg_e': -1, 'hspg': -1
}

def run_single_fva(sample_name: str, ex_mets: list, model_dir: str, diet_constraints: pd.DataFrame, 
                   res_path: str, biomass_bounds: tuple, solver: str, humanMets: dict) -> tuple:
    """
    Process individual sample: apply diet, optimize, and run flux variability analysis.
    
    Biological Workflow:
    1. Load sample-specific community model
    2. Apply dietary constraints to limit nutrient availability  
    3. Set community biomass bounds for realistic growth
    4. Optimize and save diet-adapted model
    5. Calculate net metabolite production and uptake fluxes
    
    Args:
        sample_name: Sample identifier
        ex_mets: List of metabolites for exchange analysis
        model_dir: Directory containing community models
        diet_constraints: DataFrame with diet reaction constraints
        results_path: Directory to save diet-adapted models
        biomass_bounds: Community biomass growth bounds
        solver: Optimization solver (cplex, gurobi, etc.)
        humanMets (dict): Human metabolites dict
        
    Returns:
        Tuple of (sample_name, net_production_dict, net_uptake_dict)
    """
    diet_model_dir = os.path.join(res_path, 'Diet')
    os.makedirs(diet_model_dir, exist_ok=True)
    diet_model_name = f"microbiota_model_diet_{sample_name}"

    try:
        if os.path.exists(os.path.join(diet_model_dir, f"{diet_model_name}.sbml")):
            log_with_timestamp(f"Diet-adapted model for {sample_name} already exists. Skipping steps 1-4, Running FVA directly.")
            model, C, d, dsense, ctrs = load_model_and_constraints(
                diet_model_name, diet_model_dir, model_type="standard", save_format="sbml")
            model_data = {
                'C': C,
                'd': d,
                'dsense': dsense,
                'ctrs': ctrs
            }
        else:
            # Step 1: Load and configure sample model
            model, C, d, dsense, ctrs = load_model_and_constraints(
                f"microbiota_model_samp_{sample_name}", model_dir, model_type="standard", save_format="sbml")
            model.solver = solver
            model.name = sample_name
            model_data = {
                'C': C,
                'd': d,
                'dsense': dsense,
                'ctrs': ctrs
            }
            log_with_timestamp(f"Processing {sample_name}: got model")
            
            # Step 2: Apply dietary constraints
            model = _apply_dietary_constraints(model, sample_name, diet_constraints)
            
            # Step 3: Set physiological bounds
            model = _configure_physiological_bounds(model, biomass_bounds, humanMets, diet_constraints) 
            
            # Step 4: Optimize and save diet-adapted model
            diet_model_name = _optimize_and_save_model(model, model_data, sample_name, res_path)

        # Step 5: Perform flux variability analysis
        net_production_samp, net_uptake_samp, min_net_fecal_excretion, raw_fva_results = _perform_fva(
            model, ex_mets, sample_name, model_data)
        return sample_name, net_production_samp, net_uptake_samp, min_net_fecal_excretion, raw_fva_results
        
    except Exception as e:
        log_with_timestamp(f"Error processing sample {sample_name}: {str(e)}")
        raise e
    
def _apply_dietary_constraints(model: cobra.Model, sample_name: str, diet_constraints: pd.DataFrame) -> cobra.Model:
    """Apply diet and host-derived metabolite constraints to model."""
    diet_rxns = [r.id for r in model.reactions if '[d]' in r.id and r.id.startswith('EX_')]
    for rxn_id in diet_rxns:
        new_id = rxn_id.replace('EX_', 'Diet_EX_')
        if new_id not in model.reactions:
            model.reactions.get_by_id(rxn_id).id = new_id

    # First: Set ALL Diet_EX_ reactions to lower bound 0 (like useDiet.m does)
    for rxn in model.reactions:
        if rxn.id.startswith('Diet_EX_'):
            rxn.lower_bound = 0

    # Apply diet
    for _, row in diet_constraints.iterrows():
        rxn = row['rxn_id']
        if rxn in model.reactions:
            model.reactions.get_by_id(rxn).lower_bound = float(row['lower_bound'])
            if pd.notnull(row['upper_bound']):
                model.reactions.get_by_id(rxn).upper_bound = float(row['upper_bound'])

    log_with_timestamp(f"Processing {sample_name}: diet applied")
    return model

def _configure_physiological_bounds(model: cobra.Model, biomass_bounds: tuple, humanMets: dict, diet_constraints: pd.DataFrame) -> cobra.Model:
    """Set physiologically realistic bounds on transport and biomass reactions."""
    # Constrain community biomass growth
    if 'communityBiomass' in model.reactions:
        model.reactions.communityBiomass.bounds = biomass_bounds

    for rxn in model.reactions:
        if rxn.id.startswith('UFEt_') or rxn.id.startswith('DUt_') or rxn.id.startswith('EX_'):
            rxn.upper_bound = 1e6

    # Change the bound of the humanMets if not included in the diet BUT it is in the existing model's reactions
    for met_id, bound in humanMets.items():
        rxn_id = f'Diet_EX_{met_id}[d]'
        if rxn_id not in diet_constraints['rxn_id'].values and rxn_id in model.reactions:
            model.reactions.get_by_id(rxn_id).bounds = bound, 10000.

    # close demand and limit sink reactions
    for rxn in model.reactions:
        if '_DM_' in rxn.id: rxn.lower_bound = 0
        elif '_sink_' in rxn.id: rxn.lower_bound = -1
    return model

def _optimize_and_save_model(model: cobra.Model, model_data: dict, sample_name: str, results_path: str) -> None:
    """Optimize model and save diet-adapted version."""
    # Set objective to community biomass export & Optimize to ensure feasibility
    model.objective = 'EX_microbeBiomass[fe]'
    solution = model.optimize()
    log_with_timestamp(f"Processing {sample_name}: model optimized") if solution.status == 'optimal' else log_with_timestamp(f"  ⚠️ Warning: Model optimization status: {solution.status}")
    
    # Save diet-adapted model
    diet_model_dir = os.path.join(results_path, 'Diet')
    os.makedirs(diet_model_dir, exist_ok=True)

    model_name = f"microbiota_model_diet_{sample_name}"
    save_model_and_constraints(
        model, 
        model_data['C'],
        model_data['d'],
        model_data['dsense'],
        model_data['ctrs'],
        model_name=model_name,
        out_dir=diet_model_dir,
        save_format="sbml"
    )

    log_with_timestamp(f"  Saved diet-adapted model")
    return model_name

def _perform_fva(model: cobra.Model, exchanges: list, sample_name: str, model_data: dict) -> tuple:
    """Perform flux variability analysis and calculate net metabolite fluxes."""
    log_with_timestamp(f"  Starting FVA for {sample_name}")
    
    # Get reaction indices for FVA
    # fecal_rxn_ids = [model.reactions.index(rxn) for rxn in model.exchanges]

    # diet_rxn_ids = [rxn.id.replace('EX_', 'Diet_EX_').replace('[fe]', '[d]') for rxn in model.exchanges]
    # diet_rxn_ids = [model.reactions.index(model.reactions.get_by_id(rid)) for rid in diet_rxn_ids if rid in model.reactions]
    
    # A, rhs, csense, lb, ub, c = build_constraint_matrix(diet_model_path)
    # opt_model, vars, obj_expr = build_optlang_model(A, rhs, csense, lb, ub, c)
    # min_flux_fecal, max_flux_fecal = run_sequential_fva(opt_model, vars, obj_expr, fecal_rxn_ids, opt_percentage=99.99)
    # min_flux_diet, max_flux_diet = run_sequential_fva(opt_model, vars, obj_expr, diet_rxn_ids, opt_percentage=99.99)

    model = apply_couple_constraints(model, model_data)
    fecal_rxn_list = [rxn for rxn in model.reactions if rxn.id.startswith('EX_')]

    fecal_result = flux_variability_analysis(model, 
                                            # reaction_list=[rxn for rxn in model.exchanges],
                                            reaction_list=fecal_rxn_list,
                                            fraction_of_optimum=0.9999, processes=4)
    diet_rxn_list = [rxn for rxn in model.reactions if 'Diet_EX_' in rxn.id]

    diet_result = flux_variability_analysis(model, 
                                            reaction_list=diet_rxn_list,
                                            fraction_of_optimum=0.9999, processes=4)
    
    min_flux_fecal, max_flux_fecal = fecal_result['minimum'].to_dict(), fecal_result['maximum'].to_dict()
    min_flux_diet, max_flux_diet = diet_result['minimum'].to_dict(), diet_result['maximum'].to_dict()

    # Calculate net production and uptake
    net_production_samp, net_uptake_samp, min_net_fecal_excretion, raw_fva_results = {}, {}, {}, {}

    # exchanges derived from exMets (all exchanged metabolites across all individual models) -> intersect it with rxns in this particular model
    # fecal_rxns = [r.id for r in model.exchanges]
    fecal_rxns = [r.id for r in model.reactions if r.id.startswith('EX_')]
    exchanges = set(fecal_rxns).intersection(set(exchanges))

    # cut off very small values below solver sensitivity
    tol = 1e-07
    for rxn in exchanges:
        fecal_var = rxn
        diet_var = rxn.replace('EX_', 'Diet_EX_').replace('[fe]', '[d]')

        # if abs(max_flux_fecal.get(fecal_var, 0)) < tol: max_flux_fecal.get(fecal_var, 0) == 0

        # Safely clamp tiny max_flux_fecal values to 0 (only if key exists)
        if fecal_var in max_flux_fecal and abs(max_flux_fecal[fecal_var]) < tol:
            max_flux_fecal[fecal_var] = 0.0

        prod = abs(min_flux_diet.get(diet_var, 0) + max_flux_fecal.get(fecal_var, 0))
        uptk = abs(max_flux_diet.get(diet_var, 0) + min_flux_fecal.get(fecal_var, 0))
        min_net_fe_ex = min_flux_fecal.get(fecal_var, 0) + min_flux_diet.get(diet_var, 0)
    
        net_production_samp[rxn] = prod
        net_uptake_samp[rxn] = uptk
        min_net_fecal_excretion[rxn] = min_net_fe_ex
        raw_fva_results[rxn] = {
            'min_flux_diet': min_flux_diet.get(diet_var, 0),
            'max_flux_diet': max_flux_diet.get(diet_var, 0),
            'min_flux_fecal': min_flux_fecal.get(fecal_var, 0),
            'max_flux_fecal': max_flux_fecal.get(fecal_var, 0)
        }
    
    log_with_timestamp(f"  Completed FVA analysis for {sample_name}")
    return net_production_samp, net_uptake_samp, min_net_fecal_excretion, raw_fva_results

def run_community_fva(
    sample_names: list, ex_mets: list, model_dir: str, diet_file: str, res_path: str,
    biomass_bounds: tuple=(0.4, 1.0), solver: str = 'cplex', workers: int = 1) -> tuple:
    """
    Apply dietary constraints and perform flux variability analysis on community models.
    
    This is the main pipeline function that processes all samples in parallel,
    applying dietary constraints and calculating metabolite production/uptake profiles
    through flux variability analysis.
    
    Args:
        sample_names: List of sample identifiers
        ex_mets: List of metabolites for exchange analysis
        model_dir: Directory containing sample community models
        diet_file: Path to VMH diet constraint file
        res_path: Directory for saving results
        biomass_bounds: Community biomass bounds
        solver: Optimization solver name
        workers: Number of parallel workers
        
    Returns:
        Tuple of (exchange_reactions, net_production_dict, net_uptake_dict, min_net_fecal_excretion, raw_fva_results)
    """
    os.makedirs(res_path, exist_ok=True)
    exchanges = [f"EX_{m.replace('[e]', '[fe]')}" for m in ex_mets if m != 'biomass[e]']

    net_production = {}
    net_uptake = {}
    min_net_fecal_excretion = {}
    raw_fva_results = {}

    # Adapt diet
    diet_constraints = adapt_vmh_diet_to_agora(diet_file, setup_type='Microbiota')

    log_with_timestamp("Got constraints, starting parallel processing")

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all sample processing jobs
        futures = [
            executor.submit(
                run_single_fva, 
                samp, exchanges, model_dir, diet_constraints, res_path,
                biomass_bounds, solver, HUMAN_METS
            ) 
            for samp in sample_names
        ]
        
        # Collect results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing samples'):
            try:
                sample_name, net_production_samp, net_uptake_samp, min_net_fe_ex, raw_fva = future.result()
                net_production[sample_name] = net_production_samp
                net_uptake[sample_name] = net_uptake_samp
                min_net_fecal_excretion[sample_name] = min_net_fe_ex
                raw_fva_results[sample_name] = raw_fva
                
            except Exception as e:
                log_with_timestamp(f"Sample {sample_name} failed with error: {e}")
                # Continue processing other samples
                continue

    log_with_timestamp("All samples processed successfully")
    return exchanges, net_production, net_uptake, min_net_fecal_excretion, raw_fva_results
