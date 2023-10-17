## llama2_prebiasing

Fork from llama2.c 

## Workpackages
### 1. Dataset 

First we download tinystories dataset and tokenize it. 

```bash
python tinystories.py download
```

This creates a new folder called data in the root directory that contains 'TinyStories_all_data'.

We then tokenize the dataset using sentencepiece tokenizer model

```bash
python tinystories.py pretokenize
```
This creates a .bin file in the data/TinyStories_all_data folder that contains the tokenized dataset.

### 2. Pretraining

The main training script lives here:

```bash
python train.py
```

This creates a model.bin file in the root directory that contains the trained model.

### 3. Model Evaluation
There are three main scripts for checkpoint evaluation:

1. Attention Visualization
```bash
python visualize_attn.py
```

These take the stored attention weights from each checkpoint and plot the attention weights for each layer and head
python visualize_embd.py

2. Embedding Visualization
```bash
python visualize_embd.py
```

This takes the stored embeddings from each checkpoint and plots the embeddings using PCA.

All plots are stored in out/visualize

3. Output Evaluation
```bash
eval.py
```
This loads the model and generates a sequence of 200 tokens. It then compares the generated sequence to the expected sequence and other metrics such as sentance length and ourputs the results to a csv file in out/tables.


### 5. Saturation curves
- plot each of the eval metrics against the number of training steps


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
