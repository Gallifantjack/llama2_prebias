<!-- exclude_docs -->
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE.txt)
[![arXiv](https://img.shields.io/badge/arXiv-pending.svg)](https://www.overleaf.com/project/654bbc7fcc22efd04a6c63f6)
<!-- exclude_docs_end -->


# <img src = "assets/llama-emoji.png" height=25> llama2_prebiasing

<!-- exclude_docs -->
> **⚗️ Status:** This project is still in *alpha*, and may change without warning.  
<!-- exclude_docs_end -->
<!-- include_docs
:::{important}
**Status:** This project is still in *alpha*, and the API may change without warning.  
:::
include_docs_end -->

<div align="center">

<!-- exclude_docs -->
<img src="assets/dalle_llama_bias.png" alt="Dalle generated LLama image reading books" style="width: 500; height: 500; margin-right: 2%;">
<!-- exclude_docs_end -->
<!-- include_docs
<img src="assets/dalle_llama_bias.png.png" alt="Dalle generated LLama image reading books" style="width: 49%; margin-right: 2%;">
include_docs_end -->

</div>


## 📃 Overview
***llama2_prebiasing*** challenges traditional methods in large language model training. Utilizing the RedPajama dataset and Llama architecture, this project focuses on a ground-up approach to training language models. Throughout the training process, models are evaluated for coherence, objectivity, bias, and toxicity. Our repository enables the generation of performance curves across multiple metrics, identifying performance plateaus. The impact of training batch sequences on performance saturation and the resultant shifts in performance curves offers deep insights into the malleability of embedding spaces and attention layers during the pretraining phase. Furthermore, the project investigates the persistence of bias by experimenting with different training batch sequences, mapping embedding spaces, attention heads, and output metrics across checkpoints to provide a comprehensive view of model behavior.

### How is llama2_prebias unique?
* **📊 Data-Driven Development:** Emphasizes the importance of diverse and representative data in training robust language models.
* **🔍 Bias-aware Training:** Focused on identifying and mitigating biases during language model training, instead of relying on fine-tuning methods.
* **🌐 Community Collaboration:** Aims to develop datacards and model cards that can be reused and disseminated online to foster a more inclusive and comprehensive approach to language model development.


### Key Concepts

<div align="center">

<!-- exclude_docs -->
<img src="assets/Saturation and Influence Curves .png" alt="Saturation and Influence Curves" style="width: 48%;">
<img src="assets/Durability.png" alt="Durability" style="width: 48%;">
<!-- exclude_docs_end -->
<!-- include_docs
<img src="assets/Saturation and Influence Curves.png" alt="Saturation and Influence Curves" style="width: 49%; margin-right: 2%;">
<img src="assets/Durability.png" alt="Durability" style="width: 49%;">
include_docs_end -->

</div>

<br>
<br>

# Workpackages
## 1. Tinystories  ![Working](https://img.shields.io/badge/status-working-green)

### a. Download Dataset 

First we download tinystories dataset. 

```bash
python download/download_datasets.py tinystories
```

This creates a new folder called data in the root directory that contains 'TinyStories_all_data'.

### b. Tokenize and Evaluate Batches

We then tokenize the dataset using sentencepiece tokenizer model.
This creates a .bin file in the data/TinyStories_all_data folder that contains the tokenized dataset.

```bash
python tokenization/pretokenize.py pretokenize
```

During this process each shard is tokenized and global ids are created


### c. Metadata
```bash
python metadata/batch_metadata.py compute_metadata
```
Here each sample is evaluated using metrics such as perplexity and sentence length. These are the same metrics as what is used in model evaluation in 3.

Batch metrics are stored in out/tables/batch_metrics.csv


## 2. Pretraining  ![Working](https://img.shields.io/badge/status-working-green) |


```bash
python modelling/train.py
```
The Modelling folder contains the model configurations, training loop, and dataset class. It also contains the custom sampler and transformation functions that allow modification of the batch order.

During training checkpoint files contain embeddings, attention weights, and batches used to that point, and the model folder contains the bin files. These are all store in the out folder.


## 3. Model Evaluation  ![Working](https://img.shields.io/badge/status-working-green) |
There are three main scripts for checkpoint evaluation:

#### a. Attention Visualization
```bash
python visualize_attn.py
```

These take the stored attention weights from each checkpoint and plot the attention weights for each layer and head.

All plots from this section are stored in the model specific out dir/visualize

### b. Embedding Visualization
```bash
python visualize_embd.py
```

This takes the stored embeddings from each checkpoint and plots the embeddings using PCA.


### c. input/output Evaluation
```bash
python eval.py
```
To evaluate the output this loads the model and generates a sequence of 200 tokens. It then compares the generated sequence to the expected sequence and other metrics such as sentance length and outputs the results to a csv file in out/tables.

For the batch metrics, these metrics were already calculated in stage 1b. This script just loads the table and adds the batch metrics of those batches used at a given checkpoint to a summary table.


### d. Saturation curves
```bash
python visualize_sat_curves.py
```
This script takes the batch metadata (1c) and the model ouput metadata (3c) and plots the metrics for each checkpoint next to each other. This allows for easy comparison of the metrics of model inputs and outputs

## 4. Experiments  ![Working](https://img.shields.io/badge/status-working-green) |

### a. Durability
```bash
python experiments/durability/experiment_durability.py
```

This script takes the model input metadata (1c) and sorts the batches by a given metric. It then trains the model on the sorted batches and evaluates the model output (3c). 

Saturation curves are then plotted for each metric and compared to the saturation curves of the unsorted model. Comparisons across time as well as the end points are made to see if the same data in different orders has different effects on the end model.


## 5. Dashboard  ![Working](https://img.shields.io/badge/status-working-green) |
``` bash
python dash_llama/app.py
```

<div align="center">

<!-- exclude_docs -->
<img src="assets/dash_v1.png" alt="Current Dashboard">
<!-- exclude_docs_end -->
<!-- include_docs
<img src="assets/dash_v1.png" alt="Current Dashboard">
include_docs_end -->

</div>

<br>
<br>

# Planned Work

| Feature | Description | Status |
| ------- | ----------- | ------ |
| Dynamic Batch Sampling | Utilizing model output metrics at each checkpoint to update the probability of batch selection for subsequent training epochs. | ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow) |
| Batch-level Metrics Analysis | Implementing a range of metrics for model evaluation: [Better flags](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words), k-word accuracy, sophisticated co-occurrence, [Benchmarking Large Language Model Capabilities for Conditional Generation](https://aclanthology.org/2023.acl-long.511.pdf), [Resources and Benchmarks for NLP](https://slds-lmu.github.io/seminar_nlp_ss20/resources-and-benchmarks-for-nlp.html), [Dynabench](https://arxiv.org/pdf/2104.14337.pdf), [lexical metrics](https://aclanthology.org/2022.nlppower-1.6.pdf), [Towards explainable NLG metrics](https://arxiv.org/pdf/2203.11131.pdf). | ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow) |
| Dataset Expansion | Extending training to include larger datasets such as arXiv and Wikipedia for more comprehensive model training and evaluation. | ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow) |
| Fine Tuning Effects | Investigating the impacts of fine-tuning on model performance and behavior. | ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow) |
| In-context Learning Exploration | Examining the implications of in-context learning for enhancing model adaptability and understanding. | ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow) |
| Universal IDs for Resources | Implementing universal identifiers for dataset cards and model cards online to streamline resource management and accessibility. | ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow) |


## ✍️ Citing
The current manuscript is available on [Overleaf](https://www.overleaf.com/project/654bbc7fcc22efd04a6c63f6). 
