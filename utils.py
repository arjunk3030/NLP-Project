"""Shared utilities for Assignment 4 transformer models.

This module provides lightweight tokenization, batching helpers, and common
transformer building blocks that are reused across the encoder, decoder, and
seq2seq models. The goal is to keep the implementation accessible for
students while mirroring the components found in larger frameworks such as
PyTorch's `nn.Transformer`.
"""

import json
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<mask>"]


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducibility.

    Args:
        seed: Integer seed shared across libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class BPETokenizer:
    """Byte Pair Encoding (BPE) tokenizer implemented from scratch."""

    def __init__(self, vocab_size: int = 4096):
        """
        Args:
            vocab_size: Target vocabulary size. 
                        4096 is good for 'Mini' models (Wikidata).
                        GPT-4 uses ~100k.
        """
        self.vocab_size = vocab_size
        self.merges = {}  # (int, int) -> int
        self.vocab = {}   # int -> bytes
        self.special_tokens = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
        }
        # Reverse map for decoding
        self.special_ids = {v: k for k, v in self.special_tokens.items()}
        
        # Base vocabulary (0-255 are raw bytes)
        # We start our learned tokens after the special tokens and raw bytes
        self.idx_offset = 256 + len(self.special_tokens)

    def train(self, texts: List[str]):
        """Train the tokenizer on a list of strings."""
        print(f"Training BPE Tokenizer on {len(texts)} texts...")
        
        # 1. Convert all text to a massive list of integers (utf-8 bytes)
        # In a real large-scale scenario, you'd sample the data, not use all of it.
        raw_bytes = "".join(texts).encode("utf-8")
        ids = list(raw_bytes)
        
        # 2. Iteratively merge the most common pairs
        num_merges = self.vocab_size - 256 - len(self.special_tokens)
        
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break
                
            # Find the most frequent pair
            pair = max(stats, key=stats.get)
            
            # Mint a new token ID
            idx = self.idx_offset + i
            
            # Record the merge
            self.merges[pair] = idx
            
            # Apply merge to the training data (so next iteration sees the new token)
            ids = self._merge_ids(ids, pair, idx)
            
            if (i + 1) % 100 == 0:
                print(f"BPE Merge {i+1}/{num_merges}: {pair} -> {idx}")

        print("Tokenizer training complete.")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text into token IDs."""
        # 1. Start with raw bytes
        ids = list(text.encode("utf-8"))
        
        # 2. Apply merges in the order they were learned
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            # Find the pair in ids that has the lowest merge index (was learned earliest)
            # Use 'inf' if the pair isn't in our merge rules
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break # No more mergeable pairs
            
            idx = self.merges[pair]
            ids = self._merge_ids(ids, pair, idx)
        
        if add_special_tokens:
            ids = [self.special_tokens["<bos>"]] + ids + [self.special_tokens["<eos>"]]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text using recursive reconstruction."""
        # 1. Invert the merges to look up children: idx -> (p0, p1)
        vocab_map = {idx: pair for pair, idx in self.merges.items()}
        
        def decode_token(idx):
            # Base case: raw byte
            if idx < 256:
                return bytes([idx])
            # Special tokens
            if idx in self.special_ids:
                return b"" if skip_special_tokens else self.special_ids[idx].encode("utf-8")
            # Recursive case: BPE token
            if idx in vocab_map:
                p0, p1 = vocab_map[idx]
                return decode_token(p0) + decode_token(p1)
            # Fallback (shouldn't happen if trained correctly)
            return b"?"

        res = b""
        for idx in ids:
            res += decode_token(idx)
        
        return res.decode("utf-8", errors="replace")

    # --- Helpers ---
    def _get_stats(self, ids):
        """Count frequency of adjacent pairs."""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_ids(self, ids, pair, idx):
        """Replace all occurrences of `pair` with `idx`."""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    @property
    def pad_token_id(self): return self.special_tokens["<pad>"]
    
    @property
    def bos_token_id(self): return self.special_tokens["<bos>"]

    @property
    def eos_token_id(self): return self.special_tokens["<eos>"]
    
    def pad(self, sequences, max_length=None):
        # ... copy your existing pad logic here ...
        # reuse the pad_sequences helper you already have in utils.py
        from utils import pad_sequences 
        pad_id = self.pad_token_id
        return pad_sequences(sequences, pad_value=pad_id, max_length=max_length)

class SimpleTokenizer:
    """Whitespace tokenizer with a fixed vocabulary and special tokens."""

    def __init__(self, texts: Iterable[str], extra_tokens: Optional[Iterable[str]] = None):
        """Build a vocabulary from the provided corpus.

        Args:
            texts: Corpus used to derive the vocabulary.
            extra_tokens: Optional iterable of additional tokens to insert.
        """
        vocab = list(SPECIAL_TOKENS)
        if extra_tokens:
            vocab.extend(extra_tokens)
        seen = set(vocab)
        for text in texts:
            for token in text.strip().split():
                if token not in seen:
                    vocab.append(token)
                    seen.add(token)
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(vocab)}
        self.id_to_token: Dict[int, str] = {idx: tok for tok, idx in self.token_to_id.items()}
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.mask_token = "<mask>"

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.token_to_id)

    def token_id(self, token: str) -> int:
        """Map a token to its integer id."""
        return self.token_to_id[token]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text to token ids.

        Args:
            text: Input string to tokenize.
            add_special_tokens: If True, prepend `<bos>` and append `<eos>`.

        Returns:
            List of token ids.
        """
        tokens = text.strip().split()
        ids = [self.token_to_id.get(token, self.token_to_id[self.mask_token]) for token in tokens]
        if add_special_tokens:
            return [self.token_id(self.bos_token)] + ids + [self.token_id(self.eos_token)]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Convert ids back into text.

        Args:
            ids: Token ids to decode.
            skip_special_tokens: Whether to drop BOS/EOS/MASK/PAD tokens.

        Returns:
            Detokenized string.
        """
        tokens: List[str] = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), self.mask_token)
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def pad(self, sequences: Sequence[Sequence[int]], max_length: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """Pad a batch of token sequences.

        Args:
            sequences: Batch of variable-length token sequences.
            max_length: Optional maximum length to pad/truncate to.

        Returns:
            Tuple of `(padded_ids, attention_mask)` tensors.
        """
        pad_id = self.token_id(self.pad_token)
        return pad_sequences(sequences, pad_value=pad_id, max_length=max_length)


def pad_sequences(
    sequences: Sequence[Sequence[int]],
    pad_value: int = 0,
    max_length: Optional[int] = None,
    dtype: torch.dtype = torch.long,
) -> Tuple[Tensor, Tensor]:
    """Pad / truncate sequences to a uniform length.

    Args:
        sequences: Sequence batch to pad.
        pad_value: Value used to pad positions beyond sequence length.
        max_length: Optional target sequence length.
        dtype: Tensor dtype for the padded output.

    Returns:
        Tuple `(padded, attention_mask)` where the mask contains `1` for real
        tokens and `0` for padded positions.
    """
    if not sequences:
        raise ValueError("pad_sequences expects a non-empty sequence.")
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_length), pad_value, dtype=dtype)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        if length > 0:
            padded[i, :length] = torch.tensor(seq[:length], dtype=dtype)
    attention_mask = (padded != pad_value).long()
    return padded, attention_mask


def build_causal_mask(size: int, device: Optional[torch.device] = None) -> Tensor:
    """Construct an upper-triangular mask used for causal self-attention.

    Args:
        size: Sequence length.
        device: Optional target device for the mask.

    Returns:
        Attention mask tensor with `-inf` above the diagonal.
    """
    mask = torch.triu(torch.full((size, size), float("-inf")), diagonal=1)
    if device:
        mask = mask.to(device)
    return mask


def build_padding_mask(attention_mask: Tensor) -> Tensor:
    """Convert a binary attention mask to a boolean padding mask.

    Args:
        attention_mask: Tensor containing 1 for tokens and 0 for padding.

    Returns:
        Boolean tensor where `True` marks padding positions.
    """
    # Expected shape (batch, seq_len); helps guard against silent shape bugs.
    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must be 2D")
    return attention_mask == 0


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 2048):
        """Pre-compute sinusoidal encodings per `Attention Is All You Need`.

        Args:
            embed_dim: Transformer embedding dimension.
            max_len: Maximum sequence length supported.
        """
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional information to the input embeddings.

        Args:
            x: Input embeddings of shape `(batch, seq_len, embed_dim)`.

        Returns:
            Tensor with positional encodings added.
        """
        # Input shape (batch, seq_len, embed_dim).
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float):
        """Create the hidden MLP block used in transformer layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the feed-forward network.

        Args:
            x: Hidden states of shape `(batch, seq_len, embed_dim)`.

        Returns:
            Transformed tensor of the same shape.
        """
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention supporting self- and cross-attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """Initialise a multi-head attention layer.

        Args:
            embed_dim: Transformer embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout applied to attention weights and outputs.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        rotary_freqs=None
    ) -> Tensor:
        """Perform attention computation.

        Args:
            query: Query tensor (batch, seq_q, embed_dim).
            key: Key tensor (batch, seq_k, embed_dim).
            value: Value tensor (batch, seq_k, embed_dim).
            attn_mask: Optional additive mask applied to attention logits.
            key_padding_mask: Boolean mask for padded key/value positions.

        Returns:
            Attention output of shape `(batch, seq_q, embed_dim)`.
        """
        batch_size, query_len, _ = query.shape
        key_len = key.size(1)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        if rotary_freqs is not None:
            cos, sin = rotary_freqs
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        v = v.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            scores = scores + attn_mask

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, self.embed_dim)
        return self.out_proj(output)


class ResidualBlock(nn.Module):
    """Transformer-style residual block with attention and feed-forward."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        """Create a residual attention + MLP block."""
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        kv: Optional[Tensor] = None,
        rotary_freqs=None
    ) -> Tensor:
        """Apply attention, residual connection, and feed-forward network.

        Args:
            x: Query tensor.
            attn_mask: Optional additive mask for attention logits.
            key_padding_mask: Boolean mask for padded positions.
            kv: Optional key/value tensor for cross-attention.

        Returns:
            Tensor after attention + FFN processing.
        """
        # Use provided key/value tensor for cross-attention, default to self-attn.
        key_value = x if kv is None else kv
        attn_out = self.self_attn(x, key_value, key_value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, rotary_freqs=rotary_freqs)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


def load_json(path: str) -> List[Dict[str, str]]:
    """Load a JSON file containing a list of dictionaries.

    Args:
        path: Path to the JSON file on disk.

    Returns:
        Parsed list of dictionaries.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class Batch:
    """Lightweight batch container used by training utilities."""

    input_ids: Tensor
    attention_mask: Tensor
    labels: Optional[Tensor] = None
    aux: Optional[Dict[str, Tensor]] = None

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        # We need cosine and sine of these frequencies
        # Shape: (max_seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x shape: (batch, num_heads, seq_len, head_dim)
        if seq_len > self.cos_cached.shape[2]:
            raise ValueError("Sequence length exceeds precomputed RoPE limit")
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k shape: (batch, num_heads, seq_len, head_dim)
    # This function rotates the vectors
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed