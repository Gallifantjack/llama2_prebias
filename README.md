## llama2_prebiasing

Fork from llama2.c 

## Workpackages
### 1. Tinystories

#### a. Download Dataset 

First we download tinystories dataset. 

```bash
python download/download_datasets.py tinystories
```

This creates a new folder called data in the root directory that contains 'TinyStories_all_data'.

#### b. Tokenize and Evaluate Batches

We then tokenize the dataset using sentencepiece tokenizer model.
This creates a .bin file in the data/TinyStories_all_data folder that contains the tokenized dataset.

```bash
python tokenization/pretokenize.py pretokenize
```

During this process each shard is tokenized and global ids are created


#### c. Metadata
```bash
python metadata/batch_metadata.py compute_metadata
```
Here each sample is evaluated using metrics such as perplexity and sentence length. These are the same metrics as what is used in model evaluation in 3.

Batch metrics are stored in out/tables/batch_metrics.csv


### 2. Pretraining


```bash
python modelling/train.py
```
The Modelling folder contains the model configurations, training loop, and dataset class. It also contains the custom sampler and transformation functions that allow modification of the batch order.

During training checkpoint files contain embeddings, attention weights, and batches used to that point, and the model folder contains the bin files. These are all store in the out folder.


### 3. Model Evaluation
There are three main scripts for checkpoint evaluation:

#### a. Attention Visualization
```bash
python visualize_attn.py
```

These take the stored attention weights from each checkpoint and plot the attention weights for each layer and head.

All plots from this section are stored in the model specific out dir/visualize

#### b. Embedding Visualization
```bash
python visualize_embd.py
```

This takes the stored embeddings from each checkpoint and plots the embeddings using PCA.


#### c. input/output Evaluation
```bash
python eval.py
```
To evaluate the output this loads the model and generates a sequence of 200 tokens. It then compares the generated sequence to the expected sequence and other metrics such as sentance length and outputs the results to a csv file in out/tables.

For the batch metrics, these metrics were already calculated in stage 1b. This script just loads the table and adds the batch metrics of those batches used at a given checkpoint to a summary table.


#### d. Saturation curves
```bash
python visualize_sat_curves.py
```
This script takes the batch metadata (1c) and the model ouput metadata (3c) and plots the metrics for each checkpoint next to each other. This allows for easy comparison of the metrics of model inputs and outputs

## 5. Experiments

### a. Durability
```bash
python experiments/durability/experiment_durability.py
```

This script takes the model input metadata (1c) and sorts the batches by a given metric. It then trains the model on the sorted batches and evaluates the model output (3c). 

Saturation curves are then plotted for each metric and compared to the saturation curves of the unsorted model. Comparisons across time as well as the end points are made to see if the same data in different orders has different effects on the end model.



## 6. Ideal Batch selection TBC
The goal here is to perform dynamic batch sampling based on model output. 

Output metrics are calculated for a model at each checkpoint. These metrics are then used to update the probability of a batch being selected for the next training epoch.



## Hardware used:
- **CPU**: intel i9 32 core
- **GPU**: 1x RTX 4090 
- **RAM**: 64GB
- **VRAM**: 24GB

## TODO
- wandb integration
- Metrics to add
  - k-word accuracy
  - Sophisticated co-occurrance
  - real toxicity flags

## License

MIT
