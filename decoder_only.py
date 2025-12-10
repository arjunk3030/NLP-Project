"""Starter code for the decoder-only miniature language model."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from utils import PositionalEncoding, ResidualBlock, RotaryEmbedding, build_causal_mask, build_padding_mask


class CausalSelfAttentionBlock(nn.Module):
    """Causal self-attention block used inside :class:`MiniDecoder`."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.block = ResidualBlock(embed_dim, num_heads, ff_dim, dropout)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, rotary_freqs=None) -> Tensor:
        """Apply left-to-right self-attention.

        Args:
            hidden_states: Tensor `(batch, seq_len, embed_dim)`.
            attention_mask: Tensor `(batch, seq_len)` with padding markers.

        Returns:
            Tensor `(batch, seq_len, embed_dim)` after causal attention.
        """
        
        B, T = attention_mask.shape

        # 1. padding: True where pad
        padding_mask = build_padding_mask(attention_mask)   # (B, T)

        # 2. causal: -inf above diagonal
        causal = build_causal_mask(T, hidden_states.device)      # (T, T)

        # expand causal to (1, T, T)
        causal = causal.unsqueeze(0)   
        
        return self.block.forward(hidden_states, attn_mask=causal, key_padding_mask=padding_mask, rotary_freqs=rotary_freqs)
    # TODO: check this

@dataclass
class DecoderConfig:
    """Configuration for :class:`MiniDecoder`."""

    vocab_size: int
    embed_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    num_layers: int = 4
    max_seq_len: int = 2048 #TODO: use to be 64 btw
    dropout: float = 0.1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_rope: bool = False
    use_learned_pos: bool = False


class MiniDecoder(nn.Module):
    """Decoder-only transformer for causal language modelling."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id)
        self.pos_encoding = PositionalEncoding(config.embed_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                CausalSelfAttentionBlock(config.embed_dim, config.num_heads, config.ff_dim, config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.use_rope = config.use_rope
        self.use_learned_pos = config.use_learned_pos
        if self.use_rope:
            head_dim = config.embed_dim // config.num_heads
            self.rope = RotaryEmbedding(head_dim, config.max_seq_len)
            self.pos_encoding = None # Disable standard pos encoding
        elif self.use_learned_pos:
            # Learned: A trainable lookup table for positions 0..max_len
            self.pos_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
            self.pos_encoding = None
            self.rope = None
        else:
            self.pos_encoding = PositionalEncoding(config.embed_dim, config.max_seq_len)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Compute hidden states for a batch of prefix sequences.

        Args:
            input_ids: Tensor `(batch, seq_len)` with token ids.
            attention_mask: Tensor `(batch, seq_len)` marking valid tokens.

        Returns:
            Tensor `(batch, seq_len, embed_dim)` of decoder hidden states.
        """
        tok_output = self.token_embed(input_ids)

        rotary_freqs = None

        if self.use_rope:
            drop_output = self.dropout(tok_output)
            
            # Get rotary freqs
            seq_len = input_ids.shape[1]
            cos, sin = self.rope(tok_output, seq_len)
            rotary_freqs = (cos, sin)
        elif self.use_learned_pos:
            # Learned Mode: Add a vector for Position 0, Position 1, etc.
            seq_len = input_ids.size(1)
            # Create indices [0, 1, 2, ... seq_len-1] on the correct device
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            
            # Lookup the learned vector and add it
            pos_embeds = self.pos_embedding(positions)
            drop_output = self.dropout(tok_output + pos_embeds)
        else:
            # Sinusoidal way
            pos_output = self.pos_encoding(tok_output)
            drop_output = self.dropout(pos_output)
            rotary_freqs = None

        # causal = build_causal_mask(drop_output.size(1), drop_output.device)
        # padding = build_padding_mask(attention_mask)

        temp = drop_output
        for layer in self.layers:
            temp = layer(temp, attention_mask=attention_mask, rotary_freqs=rotary_freqs)
        return self.layer_norm(temp)
    

    def logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Convenience helper returning vocabulary logits.

        Args:
            input_ids: Tensor `(batch, seq_len)`.
            attention_mask: Tensor `(batch, seq_len)`.

        Returns:
            Logits `(batch, seq_len, vocab_size)`.
        """
        forward_result = self.forward(input_ids, attention_mask)
        return self.lm_head(forward_result)

    def forward_train(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Teacher-forced training forward pass.

        Args:
            input_ids: Tensor `(batch, seq_len)` containing BOS + targets.
            attention_mask: Optional tensor `(batch, seq_len)` marking valid tokens.
            labels: Optional tensor `(batch, seq_len)` for next-token prediction.

        Returns:
            Dictionary containing at least keys `loss`, `logits`, `accuracy`.
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
            
        output = {
            "loss": None,
            "logits": None,
            "accuracy": None,
        }
        output['logits'] = self.logits(input_ids, attention_mask)
        if labels is not None:
            shift_logits = output['logits'][..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            criterion = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)

            output['loss'] = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            preds = shift_logits.argmax(dim=-1)
            valid_mask = (shift_labels != self.config.pad_token_id)
            output['accuracy'] = ((preds == shift_labels) & valid_mask).sum().float() / valid_mask.sum()

        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 20,
        temperature: float = 0.0,
    ) -> Tensor:
        """Autoregressively generate continuations.

        Args:
            input_ids: Prompt tokens `(batch, seq_len_prompt)`.
            max_new_tokens: Maximum number of new tokens to append.
            temperature: Sampling temperature; `0.0` should correspond to greedy decoding.

        Returns:
            Tensor with prompt + generated tokens.
        """
        for _ in range(max_new_tokens):
            attention_mask = (input_ids != self.config.pad_token_id).long()
            logits = self.logits(input_ids, attention_mask)
            next_token_logits = logits[:, -1, :]

            if temperature == 0:
                next_token = next_token_logits.argmax(dim=-1)
            else:
                probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
        return input_ids

    def sequence_log_probs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        target_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return summed log-probabilities of the response tokens (used in RLHF).

        Args:
            input_ids: Token ids containing prompt + response.
            attention_mask: Padding mask for the input ids.
            target_mask: Optional mask selecting response tokens (same shape as input ids).

        Returns:
            Tensor `(batch,)` containing log-probabilities for each sequence.
        """
        raise NotImplementedError("Implement log-probability computation for sequences.")


def build_few_shot_prompt(examples: List[Dict[str, str]], query: str) -> str:
    """Helper used in the training script to form a few-shot prompt."""
    prompt_lines = []
    for example in examples:
        prompt_lines.append(f"Input: {example['input']}")
        prompt_lines.append(f"Output: {example['output']}")
    prompt_lines.append(f"Input: {query}")
    prompt_lines.append("Output:")
    return "\n".join(prompt_lines)