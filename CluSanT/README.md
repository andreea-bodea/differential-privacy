## Run a Batch Sentence Sanitization Example

You can quickly test the CluSanT mechanism on a batch of sentences using the provided script:

### 1. Install Dependencies
Make sure you have installed the required packages (see requirements.txt if available):
```sh
cd CluSanT
pip install -r requirements.txt  # or install the required packages manually
```

### 2. Prepare Data
Ensure the following files are present in the appropriate directories:
- `embeddings/all-mpnet-base-v2.txt` (precomputed embeddings)
- `clusters/gpt-4/LOC.json` and `clusters/gpt-4/ORG.json` (cluster files)

If these files are missing, generate or download them as required by your project setup.

### 3. Run the Script (from the CluSanT root directory)
**Important:** Run the script from the `CluSanT` root directory (not from `src/`) so that all file paths resolve correctly:
```sh
python src/clusant_sanitize.py
```
This will print the original and anonymized versions of several example sentences, along with timing for both serial and parallel processing.

### 4. Test with Your Own Sentences
To test different sentences, edit the `example_sentences` variable in `src/clusant_sanitize.py` and provide the corresponding list of sensitive words for each sentence.

For advanced usage (batch/parallel processing in your own code), see the section below: "Efficient Batch Sentence Sanitization".

#### Troubleshooting
If you see an error like:
```
FileNotFoundError: Embeddings file 'embeddings/all-mpnet-base-v2.txt' not found. Please generate it first.
```
make sure you are running the script from the `CluSanT` root directory, not from `src/`.

## Efficient Batch Sentence Sanitization

The `clusant_sanitize.py` script provides a `CluSanTBatchProcessor` class for efficient batch sanitization of sentences using the CluSanT mechanism. This class loads all resources (embeddings, clusters, etc.) only once per process and can be used for both serial and parallel processing.

### Example Usage: Serial and Parallel Processing

```python
from clusant_sanitize import CluSanTBatchProcessor

example_sentences = [
    ("I visited Paris and met with CompanyX in Berlin.", ['paris', 'companyx', 'berlin']),
    ("London is a city where OrgY has an office.", ['london', 'orgy']),
    ("The headquarters of OrgZ are in Rome.", ['orgz', 'rome'])
]

# Serial processing
processor = CluSanTBatchProcessor()
results = [processor.sanitize(sent, sensitive_words) for sent, sensitive_words in example_sentences]
for (orig, _), anon in zip(example_sentences, results):
    print("Original text:", orig)
    print("Anonymized text:", anon)
    print()

# Parallel processing with multiprocessing.Pool
import multiprocessing

def _sanitize_helper(args):
    sentence, sensitive_words, processor_args = args
    processor = CluSanTBatchProcessor(**processor_args)
    return processor.sanitize(sentence, sensitive_words)

processor_args = {}
pool_args = [(sent, sensitive_words, processor_args) for sent, sensitive_words in example_sentences]
with multiprocessing.Pool() as pool:
    parallel_results = pool.map(_sanitize_helper, pool_args)
for (orig, _), anon in zip(example_sentences, parallel_results):
    print("Original text:", orig)
    print("Anonymized text:", anon)
    print()
```

- **Serial processing** is simple and works well for small datasets.
- **Parallel processing** (using `multiprocessing.Pool`) is recommended for large datasets and will utilize multiple CPU cores for faster processing. Each process loads its own resources, so ensure your system has enough memory.

### Class Signature
```python
class CluSanTBatchProcessor:
    def __init__(self,
        embedding_file='all-mpnet-base-v2',
        embeddings_path=None,
        loc_clusters_path='clusters/gpt-4/LOC.json',
        org_clusters_path='clusters/gpt-4/ORG.json',
        epsilon=1.0,
        K=1,
        dp_type='metric'):
        ...
    def sanitize(self, sentence, sensitive_words):
        ...
```

### Parameters
- `sentence` (str): The input sentence to sanitize.
- `sensitive_words` (list of str): List of sensitive words to anonymize in the sentence.
- The other parameters control resource paths and privacy settings.

**For best performance on large datasets, use the batch processor class and parallelization as shown above.**

----------------------------------------------------------------------------------------

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

