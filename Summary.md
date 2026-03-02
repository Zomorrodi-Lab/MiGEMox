# MiGEMox: A Python Toolbox for Microbiome Genome-Scale Modeling

# Table of Contents {#table-of-contents .TOC-Heading}

[MiGEMox: A Python Toolbox for Microbiome Genome-Scale Modeling
[1](#migemox-a-python-toolbox-for-microbiome-genome-scale-modeling)](#migemox-a-python-toolbox-for-microbiome-genome-scale-modeling)

[1. Introduction [1](#introduction)](#introduction)

[2. Overview of the Pipeline Workflow and Structure
[1](#overview-of-the-pipeline-workflow-and-structure)](#overview-of-the-pipeline-workflow-and-structure)

[3. Internal Mechanics of Community GEM Construction and Simulation
[3](#internal-mechanics-of-community-gem-construction-and-simulation)](#internal-mechanics-of-community-gem-construction-and-simulation)

[Stage 1: Personalized Community GEM Construction
[3](#stage-1-personalized-community-gem-construction)](#stage-1-personalized-community-gem-construction)

[Stage 2: Diet-Constrained FBA Simulations to Analyze the Metabolite
Produciton Potential of the Community
[6](#stage-2-diet-constrained-fba-simulations-to-analyze-the-metabolite-production-potential-of-the-community)](#stage-2-diet-constrained-fba-simulations-to-analyze-the-metabolite-production-potential-of-the-community)

[Stage 3. Predicting Microbe-Specific Contributions to the Secretion of
Metabolites of [11](#_Toc206363495)](#_Toc206363495)

[References [13](#references)](#references)

## 1. Introduction

This document provides a detailed description of the Microbiome
Genome-Scale Modeling Toolbox (MiGEMox), a Python-based workflow for
constructing and simulating personalized GEnome-scale Models (GEMs) of
microbiome metabolism. While this initial release of the pipeline
implements the core functionalities of the mgPipe workflow from The
Microbiome Modeling Toolbox in MATLAB ^1^, we plan to substantially
expand and extend its capabilities---beyond the original MATLAB-based
toolbox---to support new modeling features, scalable analyses, and
integration with modern computational technologies for microbiome
systems biology.

## 2. Overview of the Pipeline Workflow and Structure

The MiGEMox workflow is orchestrated by main.py, which integrates
several specialized modules to construct, simulate, and analyze
microbiome models. The pipeline is organized into the following key
stages and modules:

**Stage 1: Personalized Community GEM Construction**

This stage is managed by functions within community_gem_builder.py,
which handles the assembly of individual genome-scale models (GEMs) into
community models.

Core Functions:

- community_gem_builder(): The main driver function that orchestrates
  the model-building process for all samples.

- build_global_gem(): Loads all individual microbe GEMs found in the
  input abundance file from a directory containing all AGORA models and
  merges them into a single, global community model.

- build_sample_gem(): Takes the global model and customizes it for a
  specific sample by pruning absent organisms and adding an
  abundance-weighted community biomass reaction (see Section 3 for more
  details).

**Stage 2: Diet-Constrained Simulation and Analysis**

This stage applies environmental constraints (e.g., diet) and runs FBA
simulations. It is primarily handled by the community_fva_simulations.py
and diet_adapter.py modules.

Core Functions:

- run_community_fva(): The main driver that manages the simulation for
  all samples in parallel.

- adapt_vmh_diet_to_agora(): Pre-processes a standard diet file to make
  it compatible with the community GEMs, adding essential metabolites
  and adjusting bounds.

- run_single_fva(): The core simulation function that applies diet
  constraints to the microbiome GEM for a single sample, runs a
  feasibility check FBA, and performs Flux Variability Analysis (FVA) on
  fecal secretion reactions.

- collect_flux_profiles(): Gathers simulation results into structured
  Pandas DataFrames for analysis.

**Supporting Modules**

- constraints.py: Defines and applies relevant constraints for community
  FBA simulations, such as the biomass coupling constraints.

- io_utils.py: Handles file input/output, such as reading abundance data
  files and writing model files in the correct format. Also handles
  collection + organization of final simulation results into
  user-friendly CSV files.

MiGEMox is available on GitHub at:
https://github.com/aarushgarg1110/Microbiome-GEM-Modeling/tree/main

## 3. Internal Mechanics of Community GEM Construction and Simulation

In this section, we provide a comprehensive overview of the workflow and
implementation at each stage. Additional comments have been included in
the relevant sections of the code for greater clarity.

### Stage 1: Personalized Community GEM Construction

This stage, orchestrated by main.py calling functions from
community_gem_builder.py, constructs a unique community GEM for each
microbiome sample by integrating individual microbe GEMs according to
their abundance. For the remainder of this document, we use \"microbe\"
as a stand-in for any taxonomic level. One can substitute \"microbe\"
with the appropriate taxonomic category relevant to their work, such as
strain or genus.

***Step* 1.1*: Reformatting Individual Microbe GEMs for Integration into
a Community GEM* (**reformat_gem_for_community()**)**

The process begins with individual GEMs for all microbes present in the
input abundance file from the AGORA database.

First, the get_active_ex_mets() function is used to detect active
exchange metabolites. This function loads the individual GEMs, applies
coupling constraints to couple the flux of each reaction to the biomass
reaction (see step 1.3 for more details about this process), and runs
flux variability analysis with the biomass reaction set to be able to
have a fraction of the optimum of zero. Any metabolite whose exchange
reaction can carry a flux greater than a threshold of 1e-8 is considered
an active exchange metabolite.

In the MMT code, this step does not set the bounds on the exchange
reactions. It just uses whatever bounds are already in the model. For
the AGORA2 models, this is generally the complete medium (all exchange
bounds set to (-1000, 1000), but nothing forces the models to use this
medium. In MiGEMox, we plan to set the exchange bounds to (-1000, 1000)
explicitly in the code after loading the model and before running FVA,
but this is not currently implemented in order to maintain parity with
the MMT functionality.

The reformat_gem_for_community() function, called within
build_global_gem), reformats each microbe's GEM by changing reaction and
metabolites formatting as outlined below to allow its integration into a
larger community GEM.

- *Reaction & Metabolite Tagging*: To prevent overlap and track
  microbe-specific contributions, all intracellular reactions and
  metabolites are tagged with an organism-specific prefix. For example,
  an intracellular metabolite ade\[c\] in the *B. theta* model becomes
  B_theta_ade\[c\], and a reaction PGI becomes B_theta_PGI

- *Compartment Modification*: The extracellular space is redefined to
  enable inter-species communication in the human gut microbiome. The
  original extracellular compartment \[e\] is converted into a shared
  lumen compartment \[u\]. For example, a metabolite ade\[e\] that could
  be exchanged with the environment in an isolate microbial GEM now
  becomes B_theta_ade\[u\], a metabolite specific to that organism but
  residing in the shared lumen.

- *New Reactions - Internal Exchange (*IEX*)*: To allow microbes to
  share and compete for metabolites, the
  \_create_inter_microbe_exchange() helper function establishes a common
  lumen compartment \[u\]. Microbe-specific metabolites are connected to
  this shared pool via new IEX reactions of the format ade\[u\] \<=\>
  B_theta_ade\[u\]. These are named e.g., B_theta_IEX_ade\[u\]tr and are
  fully reversible, allowing a microbe to either secrete a metabolite
  into the common lumen or take it up.

The reformatted individual microbe GEMs are assembled into a first-draft
global community GEM comprising the GEMs of all microbes present in the
abundance input file, which encompasses all microbes found across all
samples within a given microbiome dataset.

An important difference between MiGEMox and MMT at this stage is that
MMT reformats the individual microbe GEMs and saves each reformatted GEM
as a separate .mat file to the hard drive. Next, when building the
sample GEMs, it reads in the needed reformatted individual GEMs and
combines them together, then it adds the diet and fecal compartments. In
MMT, by contrast, all the models are combined into a large global model
in memory and the diet and fecal compartments are added to this model.
Then, when building sample models, this large global model is pruned to
remove microbes not present in the sample. We eventually plan to
reimplement MiGEMox to follow MMT\'s strategy.

***Step* 1.2*: Incorporating Diet and Fecal Compartments and Associated
Reactions within the Global Community GEM*
(**add_diet_fecal_compartments())

After all microbe GEMs are assembled into a first-draft global community
model, add_diet_fecal_compartments() (called by build_global_gem())
processes the draft model further to add additional compartments
representing dietary intake and feces.

- *Removed Reactions*: All original exchange reactions from the base
  models (e.g., EX_ade(e)) are removed. This is a critical step to
  ensure all nutrient uptake and waste secretion occurs through the new
  lumen compartment\[u\].

- *New Compartments & Reactions*: This function then introduces the diet
  \[d\] and fecal \[fe\] compartments. It then creates a chain of new
  transport reactions for every distinct metabolite in the shared lumen
  \[u\], connecting the diet to the fecal output. The list of distinct
  metabolites in the lumen comprises all unique metabolites for which at
  least one microbe has an associated active exchange reaction in
  individual species GEMs. (An exchange reaction was considered active
  if it could carry flux in the FVA run by get_active_exchange_mets()
  earlier.). For each of these lumen metabolites, the following
  reactions are added:

  - Diet Exchange: EX_metabolite\[d\]: metabolite\[d\] \<=\> , LB =
    -1000, UB = 10000

  - Diet-to-Lumen Transport (DUt): DUt_metabolite: metabolite\[d\] -\>
    metabolite\[u\], LB = 0, UB = 10000

  - Lumen-to-Fecal Transport (UFEt): UFEt_metabolite: metabolite\[u\]
    -\> metabolite\[fe\], LB = 0, UB = 10000

  - Fecal Exchange: EX_metabolite\[fe\]: metabolite\[fe\] \<=\> , LB =
    -1000, UB = 10000

**Step 1.3: Formulating Biomass Coupling Constraints
(**build_global_coupling_constraints() **)**

A standard constraint-based model relies on the stoichiometric matrix
$S$ to enforce steady-state mass balance ($S.\mathbf{v} = \mathbf{0}$).
However, this alone is insufficient for community modeling. For example,
a microbial species may be unable to grow in a community (having zero
biomass flux) under certain conditions yet can carry non-zero fluxes for
exchange reactions that secrete metabolites into the lumen. This can
result in unrealistic solutions particularly for inter-species metabolic
interactions. To prevent this, one needs to couple the flux of each
reaction within the GEM for each microbe to the flux of the biomass
reaction in that microbe as outlined in Heinken et. al, (2013) ^2^. This
was implemented in the build_global_coupling_constraints() function in
constraints.py, which introduces additional linear constraints linking
the flux of every reaction within a species directly to that organism\'s
growth rate (biomass reaction flux).

- *Mathematical Formulation*: The formulation is based on the principles
  outlined in Heinken et al. (2013) ^2^---with parameter $u$ set to 0
  and $c$ to 400. The flux of each reaction $i\ $ belonging to a
  specific microbe is coupled to that microbe's biomass reaction flux
  ($v_{biomass}$) as follows:

  - For irreversible reactions:
    $v_{i} - \left( c \times v_{biomass} \right) \leq u$

  - For reversible reactions:

    - Forward Direction ($v_{j} \geq 0$):
      $v_{i} - \left( c \times v_{biomass} \right) \leq u$

    - Reverse Direction $(v_{j} \leq 0$):
      $v_{i} + \left( c \times v_{biomass} \right) \geq u$

  - *Implementation*: Here, $c$ is the coupling factor set to 400
    according to Heinken et al. Gut Microbes (2013) and $u$ is a
    parameter that accounts for the required flux needed to maintain
    cellular function when the cell is not dividing. While $u\ $ was set
    0.01 in Heinken et al. Gut Microbes (2013), it was set to 0 in the
    mgPipe implementation from the MMT. We follow the same practice in
    mgPipe for MiGEMox. Setting $u$ to 0 ensures that if $v_{biomass}$
    is 0, then $v_{i}\ $ is also forced to 0. This prevents
    microbe-specific internal exchange (IEX) reactions from carrying
    non-zero flux when the microbe is unable to growth thereby making
    the model predictions more biologically realistic. These rules are
    compiled into a sparse matrix equation $C*v \leq d\ $ where $C\ $ is
    the coupling matrix and $d\ $ is the associated right-hand side
    vector.

**Step 1.4: Building the Sample-Specific Microbiome GEMs
(**build_sample_gem())

Finally, build_sample_gem() (in community_gem_builder.py) customizes the
global model for each individual microbiome sample based on taxonomic
profiling results. This function contains the following steps:

- *Pruning*: The prune_zero_abundance_microbe() function is used to
  remove all reactions and metabolites associated with microbes whose
  abundance is below a defined threshold (1e-7). The corresponding rows
  in the coupling matrix are also removed via
  prune_coupling_constraints_by_microbe() (from constraints.py).

- *Community Biomass Reaction*: The com_biomass() creates a biomass
  reaction for the community for each microbiome sample. It consumes the
  individual biomass metabolites of each present organism in the
  microbiome sample with their stoichiometric coefficients being the
  organism\'s relative abundance from taxanomic profiling results. The
  only metabolite on the right-hand side of this reaction is the
  microbeBiomass\[u\] metabolite:

sum(I, abundance_i \* species_i_biomass\[c\]) \--\> microbeBiomass\[u\]

- Additionally, the UFEt_microbeBiomass transport reaction

microbeBiomass\[u\]−−\> microbeBiomass\[fe\]

and the EX_microbeBiomass\[fe\] fecal exchange reaction,

microbeBiomass\[fe\] \<=\>

are added to prevent blockage of the community biomass reaction.

- It is important to note that individual microbe GEMs in the AGORA
  dataset already include a biomass metabolite on the right-hand side of
  the biomass reactions and a biomass exchange reaction.

<!-- -->

- *Objective Function*: The model\'s objective is set to maximize the
  flux of the community biomass fecal exchange reaction: model.objective
  = EX_microbeBiomass\[fe\]. Note that this objective function is solely
  used for a solving a feasibility check FBA problem. Prediction of
  fecal secretions and microbe-microbe metabolite exchanges is achieved
  through solving other FBA problems, as outlined in the subsequent
  sections.

The final sample-specific GEM along with all relevant constraints, is
saved as a .mat file using the make_community_gem_dict() function from
io_utils.py.

### Stage 2: Diet-Constrained FBA Simulations to Analyze the Metabolite Production Potential of the Community

This stage, driven by main.py calling functions from
community_fva_simulation.py, applies dietary constraints and simulates
the metabolic production potential (metabolite secretions into feces) of
the microbiome using FBA simulation of the constructed microbiome GEM.

***Step* 2.1*: Dietary Adjustment* (**adapt_vmh_diet_to_agora()**)**

The adapt_vmh_diet_to_agora() function in diet_adapter.py pre-processes
an in-silico diet file to make it compatible with the community GEM. It
renames diet reaction IDs (e.g., EX_glc(e) becomes Diet_EX_glc_D\[d\])
and ensures feasibility by adding essential metabolites (ESSENTIAL_METS,
UNMAPPED_METS) and relaxing bounds for certain micronutrients
(MICRONUTRIENTS).

- ESSENTIAL_METS: This is a predefined list of essential metabolites
  that are critical for the growth (biomass production) of individual
  microbe or community growth. These metabolites include water, oxygen
  or alternative electron acceptors (depending on environment), Ions
  (e.g., Na⁺, K⁺, Cl⁻, phosphate, etc.), cofactors and trace elements,
  protons (H⁺), carbon sources (e.g., glucose, acetate, etc., depending
  on media). The full list can be found within the code. These
  metabolites are assumed to be available in the environment (e.g., gut
  lumen) and must be included in the dietary input, if not present in
  the in-silico diet to ensure feasibility and realistic behavior of
  flux simulations. Any missing essential metabolites are included in
  the diet with an uptake limit of 0.1 (LB = -0.1). Currently, if these
  metabolites are present in the diet, they are left as they are and are
  not updated. However, we may change it to update the bounds of these
  diets to ensure they are -0.1 or more lenient in the future, to
  prevent infeasibility when diets have too low bounds on essential
  metabolites.

- UNMAPPED_METS: The Diet Designer tool takes a diet file or nutritional
  table (daily consumption of different food items in g or mg per
  person) and translates it into exchange fluxes in AGORA models and
  bounds on their fluxes. Each dietary compound has to be manually or
  programmatically mapped to one or more model exchange reactions.
  UNMAPPED_METS is a predefined list of exchange reactions corresponding
  to dietary or host-derived compounds that exist in AGORA models but
  are not currently mapped in the Diet Designer (i.e., not included in
  the diet's nutrient table or composition data). However, they are
  known to be potentially available in the gut (via food, host
  secretions, or microbial metabolism). These metabolites should still
  be allowed in the lumen, despite not being mapped in silico diet as
  their uptake is still biologically plausible. The full list of these
  metabolites can be found in the code. The metabolites are added to the
  diet with an uptake limit of 50 mmol/gDW/h (LB = -50).

  - Choline, EX_chol(e) is added to the diet as well if not already
    present. Its uptake limit is set to 41.251 mmol/gDW/h (LB =
    -41.251).

- MICRONUTRIENTS: This is a predefined list of trace compounds (e.g.,
  vitamins, minerals, cofactors, and functional dietary fibers) that are
  required in very small amounts by microbes for growth or metabolic
  function and may be already present in the in silico diet, but are
  assigned extremely low uptake values (e.g., \< 1e-6 mol/day/person) in
  the original diet definitions. If not sufficiently available, these
  metabolites may artificially limit growth of individual microbes or
  communities in FBA simulation. The full list of these metabolites is
  available in the code.

  - If the uptake limit in the diet is equal to or below 0.1 mmol/gDW/h,
    their uptake limit is increased by a factor of 100 (LB is multiplied
    by 100).

  - If the micronutrient being considered is pnto_R and its uptake limit
    is still below 0.1 mmol/gDW/h, then its uptake limit is set to 0.1,
    otherwise it is left unchanged.

  - If the micronutrient being considered is fol, arab_L, xyl_D, amp,
    nh4, or cobalt2 and its uptake limit is less than 1 mmol/gDW/h, then
    its uptake limit is set to 1, otherwise it is left unchanged.

  - These relaxations of constraints are only applied to present
    micronutrients; none are manually added if not present.

***Step* 2.2*: The FBA Formulation and Implementation***

The run_single_fva() function in simulation.py sets up and solves a
constrained FBA problem to predict the set of metabolites that a given
microbiome sample can produce and secrete into the lumen.

*Building the FBA formulation:*

The helper function build_constraint_matrix() (in constraints.py)
illustrates how these components are assembled from the model file.

- *Objective Function*: Maximize $c \cdot v\ $, where *c* is a vector
  with 1 at the position of the objective reaction
  (EX_microbeBiomass\[fe\]) and 0 elsewhere.

- *Constraints*:

  - *Mass Balance*: $S.\mathbf{v} = \mathbf{b}$, where $S$ is the
    stoichiometric matrix and $b\ $ is a zero vector.

  - *Flux Bounds*: $LB_{j} \leq v_{j} \leq UB_{j}$. The lower bounds of
    diet reactions are set by the diet file.

  - *Coupling Constraints*: $C \cdot v \leq d\ $, where $C$ is the
    coupling matrix built in Stage 1, and $\mathbf{d}$ is a zero vector,
    meaning that the constraint is just a relationship between reaction
    fluxes.

*Flux Variability Analysis (FVA):*

The goal of FVA is to determine the metabolic flexibility of a GEM by
calculating the minimum and maximum flux of each reaction while the
system remains in a near-optimal state for a given objective function.
Here, FVA is used to determine the minimum and maximum flux of fecal and
diet exchange reactions while constraining the community biomass flux
between 0.4 and near (i.e., 99% of) its maximum. The implementation of
this step is crucial and is handled by \_perform_fva(), which is called
within simulate_single_sample(). This pipeline contains two parallel
implementations for this task, indicating both the underlying mechanics
and the practical, high-performance approach.

A)  The Manual, \"From-Scratch\" FVA Implementation:

This approach, preserved in commented-out code within constraints.py,
demonstrates how the FVA problem is constructed from its fundamental
components. It contains the following steps:

1.  *Assembling the Full Problem* (build_constraint_matrix()): This
    function loads all mathematical components from the .mat file: the
    stoichiometric matrix ($S$), the coupling matrix ($C$), their
    respective right-hand-side vectors ($\mathbf{b}$ and $\mathbf{d}$),
    constraint sense vectors (csense, dsense), which indicate if a
    constraint is ≤, ≥, or = to $\mathbf{b}$ and $\mathbf{d}$, and
    reaction bounds ($lb\ $, $ub\ $). It combines them into a single,
    large constraint matrix.

2.  *Creating a Solver-Ready Model* (build_optlang_model()): This
    function translates the raw matrices into a structured optimization
    model that a solver like CPLEX can understand. It explicitly defines
    each flux $v_{i}$ as a variable and builds each constraint equation
    from the rows of the A matrix.

3.  *Executing Sequential FVA* (run_sequential_fva()): This function
    performs the core FVA logic:

    a.  First, it solves the primary FBA problem to maximize community
        biomass, $Z_{opt}$. This is a check to see if the model is
        feasible

    b.  It then adds a new constraint to the model:
        $v_{biomass} \geq 0.9999*Z_{opt}$. This forces the model to only
        consider solutions that achieve near-optimal growth.

    c.  Note that the community biomass reaction flux is also
        constrained between 0.4 and 1 by specifying its lower and upper
        bounds as noted above.

    d.  Finally, it iterates through every exchange reaction of interest
        (EX_met\[fe\] reactions). For each iteration, it sets that
        exchange reaction as the model\'s objective and solves two FBA
        problems: one to minimize its flux and another to maximize it.
        This process identifies the full range of metabolic activity
        possible (min and max of all exchange reactions fluxes) while
        the community is thriving.

<!-- -->

B)  The Abstracted, High-Performance FVA Implementation

This is the active method used by the workflow for efficiency. It
leverages the highly optimized cobrapy library. It involves the
following steps:

1.  *Applying Coupling Constraints* (apply_couple_constraints()): This
    function is the critical bridge. It reads the coupling constraints
    ($C$, $d$, dsense) from the .mat file and, using the optlang API (a
    sympy-based optimization modeling language) and adds them directly
    to the cobra.Model object\'s internal solver interface. Once this is
    done, the COBRA model itself is inherently aware of the coupling
    rules for all subsequent calculations.

2.  Using cobra.flux_variability_analysis(): With the model now fully
    constrained, the pipeline can call COBRApy\'s built-in
    flux_variability_analysis() function. This function executes the
    same logic as the manual method (constraining the objective and
    iterating) but does so far more efficiently, using optimized code
    and parallel processing capabilities. Essentially, this method
    already accounts for the model\'s stochiometric matrix, and we only
    have to construct the additional coupling constraints, rather than
    reconstructing all constraints as demoed above.

Both methods solve the same biological problem, but the second approach
is used in practice for its performance advantage.

***Step* 2.3*: Reporting Final Fluxes***

The FVA results are used by \_analyze_metabolite_fluxes() to report the
following results for each metabolite, consistent with those in the
Microbiome Modeling Toolbox. The final values are collected by
collect_flux_profiles() in analysis.py.

C)  *Net Production*: NetProd\[met\] =
    abs(min_flux_diet\[Diet_EX_met\[d\]\] +
    max_flux_fecal\[EX_met\[fe\]\])

    a.  Net Production is defined as the absolute value of the
        difference between the maximum fecal secretion flux and the
        maximum dietary uptake flux for metabolite met. Keep in mind
        that when running FVA on Diet_EX_met\[d\] reactions, both
        min_flux_diet and max_flux_diet only contain values \<= 0. Since
        these are negative values, min_flux_diet represents maximum
        dietary uptake and max_flux_diet shows the minimum dietary
        uptake.

    b.  Note the directions of the exchange reactions: EX_met\[d\]:
        met\[d\] \<=\>

D)  *Net Uptake*: NetUptake\[met\] =
    abs(max_flux_diet\[Diet_EX_met\[d\]\] +
    min_flux_fecal\[EX_met\[fe\]\])

    a.  Net uptake is defined as the absolute value of the difference
        between the minimum fecal secretion flux and the minimum dietary
        uptake flux for the metabolite met.

    b.  Again, note that the minimum dietary uptake is determined by
        maximizing the flux dietary exchange reaction given its reaction
        direction. Also, this flux is always non-positive.

Additionally, we have decided to report two new outputs:

E)  *Min Net Fecal Excretion*: MinNetFeEx\[met\] =
    min_flux_fecal\[EX_met\[fe\]\] + min_flux_diet\[Diet_EX_met\[d\]\]

    a.  Represents the minimum net fecal secretion potential of the
        community (worst-case scenario for secretion) and is defined min
        fecal secretion -- max diet uptake for the metabolite met.

F)  *Raw FVA Results*: This simply holds the raw flux values computed
    from running FVA on the fecal and dietary exchange reactions for
    each metabolite. This will report the min_flux_diet, max_flux_diet,
    min_flux_fecal, max_flux_fecal as described above.

The Net Production value represents the community\'s de novo synthesis
capability for that metabolite. The final values are collected for all
lumen metabolites across all samples and written
to inputDiet_net_secretion_fluxes.csv and
inputDiet_net_uptake_fluxes.csv.

Notably, we believe that "Net Uptake" is a rather confusing term and the
way it is defined, does not provide any useful biological insights, but
we keep it as is for now, so our results can be compared directly with
those of the Microbiome Modeling Toolbox.

[]{#_Toc206363495 .anchor}Stage 3. Predicting Microbe-Specific
Contributions to the Secretion of Metabolites of Interest
(predict_microbe_contributions())

The predict_microbe_contribution.py script, through its core
predict_microbe_contributions() function, performs targeted analysis to
determine the contribution of individual microbes to the production of
individual metabolites secreted by the microbiome into feces. This is
particularly useful for metabolites where the community-wide secretion
potential shows significant differences between different conditions
(e.g., disease cases and controls). Even if the community-wide potential
does not differ, individual microbe contributions might still vary
significantly, providing crucial insights for metabolites with non-zero
net production (secretion) into feces.

The function operates on the diet-constrained models that are generated
and saved in the \"Diet\" subfolder during Stage 2. Note that in these
models the community biomass reaction flux is also constrained between
0.4 and 1 by specifying its lower and upper bounds as noted above.

**Workflow for Microbes** **Contributions Analysis:**

1.  *Input Models*: The function takes as input the personalized
    community GEM that have already been constrained with the simulated
    dietary regimes, found in a subfolder called \"Diet\" within the
    results folder.

2.  *Target Metabolites*: It allows defining a specific list of
    metabolites to analyze (e.g., acetate and formate). Often times,
    these are metabolites with non-zero net secretion into feces. By
    default, if no list is provided, it analyzes all metabolites that
    can be exchanged among microbes, i.e., metabolites with Internal
    Exchange (IEX) reactions within the models.

3.  Optional Net Production Constraint: A new feature allows users to
    pass an optional flag (or dictionary) mapping each target metabolite
    to its sample-specific net fecal production value. When this option
    is enabled, the model enforces a lower bound for each fecal exchange
    reaction (EX_metabolite\[fe\]) that is near (i.e., 0.99% of) its net
    production flux (see **Stages 2.2** and **2.3**) before performing
    FVA. This ensures that predicted individual microbe contributions
    are consistent with the observed net production (fecal secretion)
    for metabolites of interest. Uptake and secretion fluxes reported
    (in Microbe_Uptake.csv and Microbe_Secretion.csv) will reflect this
    additional constraint, potentially leading to different patterns
    compared to the unconstrained mode.

4.  *Flux Variability Analysis (FVA) on* IEX *Reactions*: For each
    diet-constrained model, FVA is performed on the internal exchange
    (IEX) reactions for all metabolites present in the lumen or for
    those provided in Target Metabolites list. These are reactions that
    connect microbe-specific metabolites to the same metabolites shared
    lumen (species_name_IEX_met\[u\]tr). The FVA determines the minimum
    and maximum possible flux for each of these IEX reactions while the
    overall community biomass production (EX_microbeBiomass\[fe\])
    remains at a near-optimal level (typically 99.99% of the optimal
    objective) and constrained between 0.4 and 1.

    a.  Crucially, before running FVA, the couple_constraints() function
        (from diet_adaptation.py) is called to add the biomass coupling
        constraints to the model, ensuring that the FVA results reflect
        biologically realistic flux distributions within each strain
        \[1\]\[3\].

5.  *Output Metrics*: The function produces three main outputs:

    a.  minFluxes: Shows the minimum fluxes of all analyzed internal
        exchange (IEX) reactions. This often corresponds to the
        potential for secretion by the microbe into the lumen given the
        direction of these reactions (from lumen to microbe).

    b.  maxFluxes: Shows the maximum fluxes of all analyzed internal
        exchange (IEX)reactions. This often corresponds to the potential
        for uptake by the microbe from the lumen given the direction of
        these reactions.

        i.  

    c.  fluxSpans: Represents the distance between the minimal and
        maximal fluxes for each internal exchange reaction, indicating
        the metabolic flexibility of that particular exchange.

        i.  If you are comparing the MMT and MiGEMox fluxSpans output
            files, you may notice minor discrepancies, where some
            metabolites are included in the MiGEMox files that are not
            present in the MMT files, or vice versa. These discrepancies
            will usually be on the order of 10\^-7 or 10\^-8. The reason
            you may see these discrepancies is MMT converts minFluxes
            and maxFluxes from floating point numbers to strings, which
            loses precision in the strings (e.g. 0.0008282938298832 -\>
            \"0.0008283\"). It then converts the strings back into
            numbers before calculating the flux spans. This results in
            the min and max fluxes losing precision. MiGEMox does not do
            this; it leaves the min and max fluxes as floating point
            numbers. When MMT and MiGEMox write these numbers to files,
            they filter out any flux span with a magnitude of less than
            10\^-7. Sometimes, these differences in precision result in
            a flux span produced by MMT being slightly above 10\^-7
            whereas the flux span by MiGEMox is slightly less than
            10\^-7 (or vice versa), resulting in the MMT flux spans file
            containing species-metabolite linkages not present in the
            MiGEMox flux spans file and vice versa. The MATLAB code uses
            the \`num2str\` function to convert the floating point
            numbers into strings, but does not specify a number of
            decimal places to keep. Therefore, the number of decimal
            places kept is difficult to replicate in Python since MATLAB
            is closed source and keeps different numbers of decimal
            places depending on the magnitude of the number. For this
            reason, in MiGEMox, we do not perform this same string
            conversion. As a result of this, there may be minor
            discrepancies between the results produced by MMT and the
            results produced by MiGEMox.

        ii. Also, there is currently a known bug in MMT where it fails
            to filter the flux spans to remove numbers with magnitude
            less than 1e-7. MMT was coded with the intent of filtering
            such numbers, but due to a minor bug this filtering does not
            occur. MiGEMox performs this filtering properly, without the
            bug. Until MMT fixes this bug, this will likely also cause
            discrepancies between the two. This bug is independent of
            the string precision issue mentioned above. Even if this bug
            is fixed, the string precision issue mentioned above may
            still create discrepancies between MMT and MiGEMox.

By performing this analysis, users can gain detailed insights into the
specific roles individual microbes play in the overall metabolic
production potential of the microbiome under a given dietary intake.

## References

1\. Baldini F, Heinken A, Heirendt L, Magnusdottir S, Fleming RMT,
Thiele I. The Microbiome Modeling Toolbox: from microbial interactions
to personalized microbial communities. Bioinformatics.
2019;35(13):2332-4. doi: 10.1093/bioinformatics/bty941. PubMed PMID:
30462168; PMCID: PMC6596895.

2\. Heinken A, Sahoo S, Fleming RM, Thiele I. Systems-level
characterization of a host-microbe metabolic symbiosis in the mammalian
gut. Gut Microbes. 2013;4(1):28-40. Epub 20120928. doi:
10.4161/gmic.22370. PubMed PMID: 23022739; PMCID: PMC3555882.
