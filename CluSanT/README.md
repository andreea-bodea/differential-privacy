# CluSanT: Differentially Private and Semantically Coherent Text Sanitization

## Overview

This repository contains the replication package for the paper **CluSanT: Differentially Private and Semantically Coherent Text Sanitization**. This package includes all necessary scripts and instructions to reproduce the experiments discussed in our paper.

## Experimental Setup

### Compute Canada Resources

Our experiments were conducted on Compute Canada using the following resources:

-   **Time Allocation**: 168 hours
-   **Memory**: 64GB
-   **CPUs**: 2 nodes
-   **GPUs**: 2 V100l GPUs with 32GB VRAM each

### Python Environment and Libraries

To set up the environment and install necessary libraries, follow these steps:

1. **Create and Activate a Virtual Environment**:

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

2. **Upgrade Pip**:

    ```bash
    pip install --upgrade pip
    ```

3. **Install Required Python Libraries**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Experiment

To reproduce the experiments, follow these steps:

1. **Download TAB Dataset**:
   Download the TAB dataset from `https://github.com/NorskRegnesentral/text-anonymization-benchmark/blob/master/echr_dev.json` and save it to the root directory.

2. **Create Clusters**:

    - Ensure that you create clusters in the same format as `clusters/gpt-4/ORG.json` and `clusters/gpt-4/LOC.json`. Each cluster should be disjoint (i.e., no value should be present in more than one cluster).

3. **Ensure the `run_experiments.sh` script has execute permissions**:

    ```bash
    chmod +x run_experiments.sh
    ```

4. **Run the Shell Script**:
    ```bash
    ./run_experiments.sh
    ```

The `run_experiments.sh` script will:

1. Create the required directories (`centroids/`, `embeddings/`, `inter/`, and `intra/`).
2. Activate the virtual environment.
3. Run the generate_and_save_embeddings function with `['clusters/gpt-4/LOC.json', 'clusters/gpt-4/ORG.json']` and `embeddings` as parameters. You can append more .json files of clusters of the same format into the list.
4. Execute `src/test.py`.

## Contact

For any questions or issues, please contact **Ahmed Musa** at `its.ahmed.musa@gmail.com`, **Alex Thomo** at `thomo@uvic.ca`, **Yun Lu** at `yunlu@uvic.ca`, or **Shera Potka** at `shera.potka@uvic.ca`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

