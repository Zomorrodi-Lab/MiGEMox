"""
MiGEMox Pipeline Main Entry Point

This script orchestrates the entire MiGEMox workflow, from model construction
and diet-constrained simulation to downstream analysis of strain contributions.
It serves as the high-level control script, calling functions from
various specialized modules.
"""

import pandas as pd
import os
import shutil
import argparse
from pathlib import Path
# Import functions from our new modules
from migemox.pipeline.community_gem_builder import community_gem_builder
from migemox.pipeline.community_fva_simulations import run_community_fva
from migemox.pipeline.io_utils import collect_flux_profiles, extract_positive_net_prod_constraints, log_with_timestamp
from migemox.downstream_analysis.predict_microbe_contribution import predict_microbe_contributions
from datetime import datetime, timezone
from migemox.pipeline.io_utils import print_memory_usage
import json

def run_migemox_pipeline(abun_filepath: str, mod_filepath: str, diet_filepath: str,
                         res_filepath: str = 'Results', workers: int = 1, solver: str = 'cplex',
                         biomass_bounds: tuple = (0.4, 1.0), contr_filepath: str = 'Contributions',
                         analyze_contributions: bool = False, fresh_start: bool = False,
                         use_net_production_dict: bool = False):
    """
    Main function to run the MiGEMox pipeline.

    Args:
        abun_filepath: Path to the abundance CSV file (MUST be normalized before running MiGEMox).
        mod_filepath: Path to the directory containing organism model files (.mat).
        res_filepath: Base directory for saving output models and results.
        contr_filepath: Directory for saving strain contribution analysis results.
                          If None, default is 'Contributions'
        diet_filepath: Path to the VMH diet file.
        workers: Number of parallel workers to use for sample processing.
        solver: Optimization solver to use (e.g., 'cplex', 'gurobi').
        biomass_bounds: Tuple (lower, upper) for community biomass reaction.
        analyze_contributions: Boolean, whether to run strain contribution analysis.
    """
    log_with_timestamp(f"--- MiGEMox Pipeline Started at {datetime.now(tz=timezone.utc)} ---")
    log_with_timestamp(f"Current memory usage: {print_memory_usage()}")
    if fresh_start and os.path.exists(res_filepath):
        shutil.rmtree(res_filepath)
        log_with_timestamp("Output directory cleared for fresh start.")

    clean_samp_names, organisms, ex_mets, active_ex_mets, global_rxn_ids = community_gem_builder(
        abun_filepath=abun_filepath,
        mod_filepath=mod_filepath,
        out_dir=f'{res_filepath}/Personalized_Models',
        workers=workers
    )

    log_with_timestamp("Writing ex_mets to file:")
    ex_mets_path = os.path.join(res_filepath, 'ex_mets.json')
    with open(ex_mets_path, 'w') as f:
        json.dump(ex_mets, f)
    log_with_timestamp(f"Written to: {ex_mets_path}")

    log_with_timestamp("Writing active_ex_mets to file:")
    active_ex_mets_path = os.path.join(res_filepath, 'active_ex_mets.json')
    with open(active_ex_mets_path, 'w') as f:
        json.dump(active_ex_mets, f)
    log_with_timestamp(f"Written to: {active_ex_mets_path}")
    
    # 2. Adapt Diet
    log_with_timestamp(f"--- Stage 1 Finished at {datetime.now(tz=timezone.utc)} ---")
    log_with_timestamp("\n--- Stage 2: Adapting Diet and Running Simulations ---")

    # 3. Simulate Microbiota Models
    exchanges, net_production, net_uptake, min_net_fecal_excretion, raw_fva_results = run_community_fva(
        sample_names=clean_samp_names,
        ex_mets=active_ex_mets,
        model_dir=f'{res_filepath}/Personalized_Models',
        diet_file=diet_filepath,
        res_path=res_filepath,
        biomass_bounds=biomass_bounds,
        solver=solver,
        workers=2
    )

    # 4. Collect Flux Profiles and Save
    log_with_timestamp(f"--- Stage 2 Finished at {datetime.now(tz=timezone.utc)} ---")
    log_with_timestamp("\n--- Collecting and Saving Simulation Results ---")
    net_secretion_df, net_uptake_df = collect_flux_profiles(
        samp_names=clean_samp_names,
        exchanges=exchanges,
        net_production=net_production,
        net_uptake=net_uptake,
        res_path=res_filepath
    )
    pd.DataFrame(min_net_fecal_excretion).to_csv(Path(res_filepath) / 'inputDiet_min_net_fecal_excretion.csv')
    raw_fva_df = pd.concat({k: pd.DataFrame(v).T for k, v in raw_fva_results.items()}, axis=0)
    raw_fva_df.index.names = ['Sample', 'Reaction']
    raw_fva_df.to_csv(Path(res_filepath) / 'inputDiet_raw_fva_results.csv')
    
    log_with_timestamp(f"Net secretion and uptake results saved to {res_filepath}.")

    # 5. Run Strain Contribution Analysis (Optional)
    if analyze_contributions:
        log_with_timestamp("\n--- Downstream Analysis: Predicting Strain Contributions ---")
        log_with_timestamp(f"--- Started at {datetime.now(tz=timezone.utc)} ---")

        # VMH_ID from met names like ac[fe] -> ac
        # mets = [x.split('[')[0] for x in ex_mets]
        diet_mod_dir_for_contributions = os.path.join(res_filepath, 'Diet')
        pos_net_prod = extract_positive_net_prod_constraints(Path(res_filepath) / 'inputDiet_net_secretion_fluxes.csv')
        mets = sorted(list(pos_net_prod.keys()))

        kwargs = dict(
            diet_mod_dir=diet_mod_dir_for_contributions,
            res_path=contr_filepath,
            mets_list=mets,
            solver=solver,
            workers=workers
        )
        if use_net_production_dict: kwargs['net_production_dict'] = pos_net_prod
        min_fluxes_df, max_fluxes_df, flux_spans_df = predict_microbe_contributions(**kwargs)
        log_with_timestamp(f"Strain contribution analysis completed at {datetime.now(tz=timezone.utc)} and results saved.")

    log_with_timestamp("\n--- MiGEMox Pipeline Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MiGEMox (Microbiome Genome-Scale Modeling) pipeline.")
    parser.add_argument("-a", "--abun_filepath", type=str, required=True,
                        help="Path to the abundance CSV file (e.g., test_data_input/normCoverageReduced.csv)")
    parser.add_argument("-m", "--mod_filepath", type=str, required=True,
                        help="Path to the directory containing organism model files (.mat) (e.g., test_data_input/AGORA103)")
    parser.add_argument("-r", "--res_filepath", type=str, default="Results",
                        help="Base directory for saving all output models and results (e.g., Results)")
    parser.add_argument("-c", "--contr_filepath", type=str, default="Contributions",
                        help="Directory for saving strain contribution analysis results (e.g., Contributions)")
    parser.add_argument("-d", "--diet_filepath", type=str, required=True,
                        help="Path to the VMH diet file (e.g., test_data_input/AverageEU_diet_fluxes.txt)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers to use for sample processing (default: 1)")
    parser.add_argument("--solver", type=str, default="cplex",
                        help="Optimization solver to use (e.g., 'cplex', 'gurobi', 'glpk') (default: cplex)")
    parser.add_argument("--biomass_lb", type=float, default=0.4,
                        help="Lower bound for community biomass growth (default: 0.4)")
    parser.add_argument("--biomass_ub", type=float, default=1.0,
                        help="Upper bound for community biomass growth (default: 1.0)")
    parser.add_argument("--analyze_contributions", action="store_true",
                        help="Set this flag to run strain contribution analysis.")
    parser.add_argument("--use_net_production_dict", action="store_true",
                        help="If set, pass net_production_dict to predict_microbe_contributions.")
    parser.add_argument("--fresh_start", action="store_true",
                        help="Set this flag to clear previous results and start fresh.")

    args = parser.parse_args()

    # Combine biomass bounds
    biomass_bounds_tuple = (args.biomass_lb, args.biomass_ub)

    run_migemox_pipeline(
        abun_filepath=args.abun_filepath,
        mod_filepath=args.mod_filepath,
        res_filepath=args.res_filepath,
        contr_filepath=args.contr_filepath,
        diet_filepath=args.diet_filepath,
        workers=args.workers,
        solver=args.solver,
        biomass_bounds=biomass_bounds_tuple,
        analyze_contributions=args.analyze_contributions,
        use_net_production_dict=args.use_net_production_dict,
        fresh_start=args.fresh_start,
    )
