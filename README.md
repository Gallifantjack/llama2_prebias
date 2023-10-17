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

### 3. Evaluation
## unsorted todos
- pretend that the full training is only a checkpoint in early stages
    - pt 1- model specs
        - extract attention and embedding weights
        - visualise as in hf blog
    - pt 2- output eval
        - use python test script (200 toks output)
            - then bootstrap to get multiple outputs
        - adjust compare script
            - not just comp vs expected 
            - co-occurance etc 
            - plot comparison/performance
    - pt 3- throughout
        - add logging
        - add tensorboard
        - add checkpointing
        - repeat pt 1-2

## 4. Batch play
- sequencing and durability
- ideal batch selection


## TODO
- wandb integration

## License

MIT
