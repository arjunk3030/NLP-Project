"""Training script for the decoder-only causal language model."""

import argparse
from pathlib import Path
from typing import List
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
import resource
import wandb
import math
import time

from decoder_only import DecoderConfig, MiniDecoder, build_few_shot_prompt
from utils import SimpleTokenizer, set_seed, BPETokenizer

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PACKAGE_DIR / "data" / "pattern_sequences.txt"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "checkpoints"


class PatternDataset(Dataset):
    """Dataset made from repeating pattern sequences."""

    def __init__(self, sequences: List[str], tokenizer: BPETokenizer):
        """Store tokenized pattern sequences."""
        self.sequences = [seq for seq in sequences if seq.strip()]
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.sequences)

    def __getitem__(self, index: int):
        """Tokenize one sequence and add BOS/EOS tokens."""
        tokens = self.tokenizer.encode(self.sequences[index], add_special_tokens=True)
        return {"input_ids": tokens}


def collate_fn(batch, tokenizer: BPETokenizer, max_length=None):
    """Pad sequences for training.

    Args:
        batch: List of dictionaries produced by `PatternDataset`.
        tokenizer: Tokenizer used for padding.

    Returns:
        Dictionary containing padded ids, attention mask, and labels.
    """
    input_ids = [item["input_ids"] for item in batch]
    padded, attention = tokenizer.pad(input_ids, max_length=max_length)
    return {
        "input_ids": padded,
        "attention_mask": attention,
        "labels": padded.clone(),
    }


def load_sequences(path: Path) -> List[str]:
    """Read newline-separated sequences from disk.

    Args:
        path: Path to the text file containing sequences.

    Returns:
        List of stripped sequence strings.
    """
    # Use strip to avoid empty trailing lines that would create short samples.
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def train(args):
    """Main training loop for the decoder-only model.

    Args:
        args: Parsed CLI arguments.
    """
    wandb.init(
        project="mini-transformer-research", 
        config=vars(args),
        name=f"run_rope_{args.use_rope}_lr_{args.lr}"
    )

    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and not args.cpu:
        device = torch.device("mps")
        print("Using MPS (Mac GPU) acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU (Warning: Slow)")

    train_hf = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
    train_seqs = [row["text"] for row in train_hf if row["text"].strip()]

    val_hf = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:200]")
    val_seqs = [row["text"] for row in val_hf if row["text"].strip()]

    tokenizer = BPETokenizer(vocab_size=2048)
    tokenizer.train(train_seqs)

    train_ds = PatternDataset(train_seqs, tokenizer)
    val_ds = PatternDataset(val_seqs, tokenizer)

    print("the length of training dataset is:", len(train_hf))
    print("the length of validation dataset is:", len(val_hf))
    print(f"Vocab size: {tokenizer.idx_offset + len(tokenizer.merges)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer, max_length=args.max_seq_len))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer, max_length=args.max_seq_len))

    config = DecoderConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_rope=args.use_rope,
        use_learned_pos=args.use_learned_pos
    )

    model = MiniDecoder(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Size: {num_params/1e6:.2f} Million Parameters")
    
    wandb.config.update({"num_params": num_params})
    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        steps = 0

        t0 = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.forward_train(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            tokens_processed = input_ids.numel() 
            tokens_per_sec = tokens_processed / dt

            epoch_loss += float(loss.item())

            if steps % 10 == 0:
                log_data = {
                    "train_loss": loss.item(),
                    "train_acc": outputs["accuracy"].item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "train_perplexity": math.exp(loss.item()),
                    "epoch": epoch,
                    "iter_time_ms": dt * 1000,      # Latency
                    "tokens_per_sec": tokens_per_sec, # Efficiency
                    "total_time_min": (time.time() - total_start_time) / 60,
                }
                if device.type == "mps":
                    # Current memory allocated in MB
                    mem = torch.mps.current_allocated_memory() / 1e6 
                    log_data["mps_memory_mb"] = mem

                wandb.log(log_data)

            epoch_acc += float(outputs["accuracy"].item())
            steps += 1

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model.forward_train(input_ids, attention_mask=attention_mask, labels=labels)
                
                val_loss += float(outputs["loss"].item())
                val_acc += float(outputs["accuracy"].item())
                val_steps += 1
        
        model.train()
        
        avg_val_loss = val_loss / max(1, val_steps)
        avg_val_acc = val_acc / max(1, val_steps)
        avg_val_ppl = math.exp(avg_val_loss)


        wandb.log({
            "val_loss": avg_val_loss, 
            "val_perplexity": avg_val_ppl,
            "val_acc": avg_val_acc,
            "epoch": epoch
        })
        print(f"Epoch {epoch:02d} - val loss: {avg_val_loss:.4f} - val_token_acc: {avg_val_acc:.4f}")

    prompt = build_few_shot_prompt(
        examples=[
            {"input": train_seqs[0], "output": train_seqs[0].split()[-1]},
            {"input": train_seqs[1], "output": train_seqs[1].split()[-1]},
        ],
        query="red blue red blue",
    )
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    max_input_len = args.max_seq_len - 5
    if len(tokens) > max_input_len:
        print(f"Warning: Truncating prompt from {len(tokens)} to {max_input_len} tokens.")
        # Keep the END of the prompt (where the query is)
        tokens = tokens[-max_input_len:]
    generated = model.generate(torch.tensor([tokens], device=device), max_new_tokens=5)
    prediction = tokenizer.decode(generated[0].tolist())
    print("Few-shot prompt:\n", prompt)
    print("Generated continuation:\n", prediction)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "tokenizer_merges": tokenizer.merges,
        },
        Path(args.output_dir) / "decoder.pt",
    )
    wandb.finish()


def parse_args():
    """Parse command-line flags for decoder training.

    Returns:
        Namespace of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train the decoder-only language model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=128) # TODO: used to be 64 btw
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-query", type=str, default="red blue red blue")
    parser.add_argument("--use_rope", action="store_true", help="Use Rotary Positional Embeddings")
    parser.add_argument("--use_learned_pos", action="store_true", help="Use Learned Positional Embeddings")
    return parser.parse_args()


if __name__ == "__main__":
    # MAX_MEMORY = 1024 * 1024 * 1024 * 14
    # resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY, MAX_MEMORY))
    args = parse_args()
    set_seed(args.seed)
    train(args)
