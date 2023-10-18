## llama2_prebiasing

Fork from llama2.c 

## Workpackages
### 1. Tinystories

#### a. Download Dataset 

First we download tinystories dataset. 

```bash
python tinystories.py download
```

This creates a new folder called data in the root directory that contains 'TinyStories_all_data'.

#### b. Tokenize and Evaluate Batches

We then tokenize the dataset using sentencepiece tokenizer model.
This creates a .bin file in the data/TinyStories_all_data folder that contains the tokenized dataset.

```bash
python tinystories.py pretokenize
```

During this process each shard is evaluated using metrics such as perplexity and sentence length. These are the same as what is used in model evaluation in 3.

Batch metrics are stored in out/tables/batch_metrics.csv


### 2. Pretraining


```bash
python train.py
```
This creates a model.bin file in the root directory that contains the trained model.

During this process the batch ids used in each checkpoint are stored in the checkpoint file.


### 3. Model Evaluation
There are three main scripts for checkpoint evaluation:

#### a. Attention Visualization
```bash
python visualize_attn.py
```

These take the stored attention weights from each checkpoint and plot the attention weights for each layer and head
python visualize_embd.py

#### b. Embedding Visualization
```bash
python visualize_embd.py
```

This takes the stored embeddings from each checkpoint and plots the embeddings using PCA.

All plots are stored in out/visualize

#### c. input/output Evaluation
```bash
python eval.py
```
To evaluate the output this loads the model and generates a sequence of 200 tokens. It then compares the generated sequence to the expected sequence and other metrics such as sentance length and outputs the results to a csv file in out/tables.

For the batch metrics, these metrics were already calculated in stage 1b. This script just loads the table and adds the batch metrics of those batches used at a given checkpoint to a summary table.



### 4. Saturation curves
#### a. Output curves
```bash
python visualize_sat_output.py
```
This script takes the summary table produced in 3. and plots the normalised score for each metric against the number of training steps. 

Plots are stored in out/visualize/sat_curves_output.png

#### b. Batch input curves

```bash
python visualize_sat_batch.py
```

This script takes summarised batch metrics for each checkpoint and produces the same plot as in 4a.

Plots are stored in out/visualize/sat_curves_batch.png

## 6. Batch play
- characterise each batch using same metrics as above 
- play with sequencing 
- add specific tag to seq for durability

## 6. Ideal Batch selection
- ideal batch selection


## TODO
- wandb integration

## License

MIT
