# MiGEMox

MiGEMox is a computational framework for analyzing and simulating genome-scale metabolic models (GEMs) of microbial communities. This project enables researchers to perform in silico experiments to investigate microbiome function and interactions.

## Pre-Release Notes

This project is cuurrently under development and may contain bugs.

## Features

- Community-level metabolic modeling and simulation
- Support for constraint-based modeling with CPLEX
- Modular pipeline for reproducible analyses

## Getting Started

To get started, clone this repository and follow the instructions in [Summary.md](./Summary.md) for detailed workflow guidance.

The modeling pipeline is containerized for reproducibility. See the `Dockerfile` for environment details.

## Citing & Credits

This project builds on the work of the COBRApy community and leverages IBM CPLEX for optimization. Please cite relevant tools and datasets as appropriate for your research.

- COBRApy: [https://github.com/opencobra/cobrapy](https://github.com/opencobra/cobrapy)
- IBM CPLEX: [https://www.ibm.com/products/ilog-cplex-optimization-studio](https://www.ibm.com/products/ilog-cplex-optimization-studio)

Project lead: [Aarush Garg, Zomorrodi Lab]

For questions or contributions, please open an issue or pull request.

## How to Run This Project with Docker

This project requires IBM CPLEX with an Academic License, which cannot be redistributed directly. You must manually extract and prepare the CPLEX files before building or running the modeling scripts.

### 1. Install Docker

If you do not have Docker installed, download and install it from [https://www.docker.com/get-started/](https://www.docker.com/get-started/). Follow the platform-specific instructions for your operating system.

### 2. Prepare CPLEX Files

1. **Download the CPLEX installer** from IBM SkillsBuild or the IBM Academic Initiative and place it in a directory named `cplex` inside your project root.

2. **Extract CPLEX using a Debian container:**
    ```bash
    docker run -it --rm -v "${PWD}:/mnt" debian:bullseye bash
    apt update && apt install -y openjdk-11-jre-headless
    cd /mnt/cplex/
    ./cplex_studio2212.linux_x86_64.bin
    # Accept all license agreements and follow prompts
    cd ../opt
    tar czf /mnt/cplex.tar.gz ibm
    exit
    ```

3. **Unpack the CPLEX files on your host:**
    ```bash
    tar xzf cplex.tar.gz
    # You should now have Microbiome-GEM-Modeling/cplex/ibm/ILOG/CPLEX_Studio2212
    ```

### 3. Build and Run the Docker Image

1. **Build the Docker image:**
    ```bash
    docker build -t cobra .
    ```

2. **Run the modeling pipeline:**
    ```bash
    docker run --rm \
        -v "${PWD}/Results:/app/Results" \
        -v "${PWD}/Contributions:/app/Contributions" \
        cobra \
        --abun_filepath /test_data_input/normCoverageReduced.csv \
        --mod_filepath /test_data_input/AGORA103 \
        --contr_filepath /app/Contributions
        --res_filepath /app/Results \
        --diet_filepath /test_data_input/AverageEU_diet_fluxes.txt \
        --workers 2
        --solver cplex
        --biomass_lb 0.4
        --biomass_ub 1.0
        --analyze_contributions
        --use_net_production_dict
        --fresh_start
    ```
    

## How to Run This Project as a Python Package

You can use Microbiome-GEM-Modeling directly in your Python scripts:

1. **Clone and install the package:**
    ```bash
    # activate your environment where you want to use MiGEMox
    git clone https://github.com/Zomorrodi-Lab/MiGEMox.git
    cd MiGEMox
    pip install -e .
    ```

2. **Also clone and install the structural COBRApy model dependency:**
    ```bash
    # Do this outside the MiGEMox folder
    # activate your environment where you want to use MiGEMox
    git clone https://github.com/nsyousef/cobrapy.git
    cd cobrapy
    pip install -e .
    ```

2. **Import and run the pipeline in your script:**
    ```python
    import migemox as mgx

    mgx.pipeline.run_migemox_pipeline(
        abun_filepath="test_data_input/normCoverageReduced.csv",
        mod_filepath="test_data_input/AGORA103",
        res_filepath="Results",
        contr_filepath="Contributions",
        diet_filepath="test_data_input/AverageEU_diet_fluxes.txt",
        workers=2,
        solver="cplex"
        biomass_bounds=(0.4, 1,0),
        analyze_contributions=True,
        use_net_production_dict=False,
        fresh_start=False
    )
    ```

    You should only need to specify --abun_filepath (-a), --mod_filepath (-m), --diet_filepath (-d). The rest have default parameters.
    -  workers: int = 1
    -  solver: str = 'cplex',
    -  biomass_bounds: tuple = (0.4, 1.0)
    -  contr_filepath: str = 'Contributions,
    -  res_filepath: str = 'Results'
    -  analyze_contributions = 'False'
    -  fresh_start: bool = False
    -  use_net_production_dict: bool = False
   
For further details on workflow and usage, see [Summary.md](./Summary.md).
