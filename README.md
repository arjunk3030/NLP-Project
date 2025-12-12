## How to Run

### 1. Install Dependencies
First, install all required Python packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## How to Run the Project

### Part 1: Mini Decoder Training

Before running the training script, make sure you are logged in to Hugging Face and Weights & Biases (to receive real time visualizations while and after training):

```bash
huggingface-cli login
wandb login
```

Once logged in, you can train the mini decoder using the following commands depending on model size and positional encoding:

Small models:
```bash
python train_decoder.py --output-dir checkpoints/sinusoidal_small --max-seq-len 128 --batch-size 32 --epochs 20
```

```bash
python train_decoder.py --use_rope --output-dir checkpoints/rope_small --max-seq-len 128 --batch-size 32 --epochs 20
```

```bash
python train_decoder.py --use_learned_pos --output-dir checkpoints/learned_small --max-seq-len 128 --batch-size 32 --epochs 20
```

Medium models:

```bash
python train_decoder.py --output-dir checkpoints/medium_sinusoidal --max-seq-len 128 --batch-size 32 --epochs 20 --embed-dim 256 --num-layers 6 --num-heads 8
```

```bash
python train_decoder.py --use_rope --output-dir checkpoints/medium_rope --max-seq-len 128 --batch-size 32 --epochs 20 --embed-dim 256 --num-layers 6 --num-heads 8
```

```bash
python train_decoder.py --use-learned-pos --output-dir checkpoints/medium_learned --max-seq-len 128 --batch-size 32 --epochs 20 --embed-dim 256 --num-layers 6 --num-heads 8
```

## Part 2: Pretrained Model Fine-Tuning

For part 2, we fine-tuned a pretrained DistilBERT model and also ran our medium mini decoder for comparison. This is done via the notebook file SentimentAnalysis.ipynb.

Before running the notebook, make sure to install the required packages, log into HuggingFace, and wandb:

To run the experiments:

For DistilBERT training, execute cell [53] in the notebook.

For our medium mini decoder training, execute cell [45].

Note: For the mini decoder, you may need to update the checkpoint path on line [45] depending on which weights you are using. Please note that the training takes a while to run and it is reccomended to run using a GPU.

## Part 3: Visualizations

Part 1 and Part 2 will indepedently print our their results and graphs directly to wandb (if logged in).

To get a visualization of the attention maps, please run the command below, replacing --checkpoint with the appropriate checkpoints as needed.

```bash
python visualize_attention.py --checkpoint checkpoints/rope/decoder.pt --use_rope --text "The army was defeated by the"
```
