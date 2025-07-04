## Run a Single Sentence Sanitization Example

You can quickly test the SanText and SanText+ algorithms on a single sentence using the provided script:

### 1. Install Dependencies
Make sure you have installed the required packages:
```sh
cd SanText
pip install -r requirements.txt
```

### 2. Prepare Data
Ensure the following files are present in the `SanText/data/` directory:
- `glove.840B.300d.txt` (GloVe embeddings, ~5.3GB)
- `SST-2/train.tsv` and `SST-2/dev.tsv` (SST-2 dataset)

If these files are missing, run:
```sh
./download.sh
```
This will download and extract the required datasets. For GloVe, download from [GloVe website](https://nlp.stanford.edu/projects/glove/) if not present.

### 3. Run the Script
Run the following command to execute the example:
```sh
python sanitize_one_sentence.py
```
This will print the original sentences and the outputs of both SanText and SanText+ sanitization methods, along with the time taken for each method.

To test different sentences, edit the `example_sentences` variable in `sanitize_one_sentence.py`.

## Efficient Batch Sentence Sanitization

If you want to sanitize many sentences (e.g., a whole dataset), use the `SanTextBatchProcessor` class in `sanitize_one_sentence.py`. This class loads all resources (vocab, embeddings, caches, etc.) only once and can efficiently process sentences in both serial and parallel modes.

### Example Usage: Serial and Parallel Processing

```python
from sanitize_one_sentence import SanTextBatchProcessor

sentences = [
    "The movie was absolutely wonderful and inspiring.",
    "I did not enjoy the film at all.",
    "The plot was predictable but the acting was great."
]

processor = SanTextBatchProcessor()

# Serial processing
results_serial = [processor.sanitize(sent, method="SanText") for sent in sentences]

# Parallel processing
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor() as executor:
    results_parallel = list(executor.map(lambda sent: processor.sanitize(sent, method="SanText"), sentences))

for sent, sanitized in zip(sentences, results_serial):
    print(f"Original: {sent}")
    print(f"Sanitized (serial): {sanitized}\n")

for sent, sanitized in zip(sentences, results_parallel):
    print(f"Original: {sent}")
    print(f"Sanitized (parallel): {sanitized}\n")
```

- You can use `method="SanText+"` for the SanText+ mechanism.
- This approach is much faster than calling the single-sentence function repeatedly.
- For very large datasets, you can further optimize by chunking or using more threads.

## Using the `SanTextBatchProcessor` Programmatically

The `SanTextBatchProcessor` class is the recommended way to sanitize sentences in bulk. It supports both the SanText and SanText+ methods.

### Class Signature
```python
class SanTextBatchProcessor:
    def __init__(self,
        glove_path="data/glove.840B.300d.txt",
        filtered_glove_path="data/glove.filtered.txt",
        data_dir="data/SST-2/",
        epsilon=15.0,
        p=0.2,
        sensitive_word_percentage=0.5,
        vocab_cache_path="vocab_cache.pkl",
        glove_cache_path="glove_filtered_cache.pkl"
    ):
        ...
    def sanitize(self, sentence, method="SanText"):
        ...
```

### Parameters
- `sentence` (str): The input sentence to sanitize.
- `method` (str): Which sanitization method to use. Must be either `'SanText'` or `'SanText+'`. Default is `'SanText'`.
- The other parameters are the same as before and control resource paths and privacy settings.

### Example: Sanitize a Single Sentence
```python
from sanitize_one_sentence import SanTextBatchProcessor

processor = SanTextBatchProcessor()
sentence = "The movie was absolutely wonderful and inspiring."
sanitized = processor.sanitize(sentence, method="SanText")
print("SanText output:", sanitized)
```

**For best performance on datasets, always use the batch processor class and parallelization as shown above.**

--------------------------------------------------------------------------------------------
# SanText
Code for Findings of ACL-IJCNLP 2021 **"[Differential Privacy for Text Analytics via Natural Text Sanitization](https://arxiv.org/pdf/2106.01221.pdf)"**

@@ -16,7 +125,7 @@ Please kindly cite the paper if you use the code or any resources in this repo:
The privacy issue is often overlooked in NLP. 
We address privacy from the root: 
directly producing sanitized text documents based on differential privacy.
We further propose sanitization-aware pretraining and finetuning to adapt the currently dominating LM (e.g., BERT) over sanitized texts. It “prepares” the model to work with sanitized texts, which leads to an increase in accuracy while additionally ensuring privacy.
We further propose sanitization-aware pretraining and finetuning to adapt the currently dominating LM (e.g., BERT) over sanitized texts. It "prepares" the model to work with sanitized texts, which leads to an increase in accuracy while additionally ensuring privacy.

<p align="center">
<img src="img.png" alt="SanText" width="500"/>
@@ -174,4 +283,4 @@ python run_language_modeling.py \
```

Note that if you enable full wiki training, it will be really time-consuming. 
The data preprocessing often takes more than 10 hours. We released our pretrained models 
The data preprocessing often takes more than 10 hours. We released our pretrained models 