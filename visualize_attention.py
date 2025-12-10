import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

# Import your model definitions
from decoder_only import MiniDecoder, DecoderConfig
from utils import BPETokenizer

def load_model_and_tokenizer(checkpoint_path, config):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 1. Rebuild Tokenizer
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    if "tokenizer_merges" in checkpoint:
        tokenizer.merges = checkpoint["tokenizer_merges"]
        # Rebuild the vocab map from merges
        tokenizer.vocab_size = config.vocab_size # approximate
        tokenizer.idx_offset = 256 + len(tokenizer.special_tokens)
    else:
        print("Warning: No tokenizer merges found in checkpoint. Using untrained tokenizer.")

    # 2. Rebuild Model
    model = MiniDecoder(config)
    
    # Handle state dict keys (in case of 'module.' prefix from DataParallel)
    state_dict = checkpoint["model_state"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model, tokenizer

def visualize_attention(args):
    # --- CONFIGURATION (Must match your training!) ---
    config = DecoderConfig(
        vocab_size=2048,
        embed_dim=128,    # Change if you used Medium (256)
        num_layers=4,     # Change if you used Medium (6)
        num_heads=4,      # Change if you used Medium (8)
        max_seq_len=128,
        use_rope=args.use_rope,
        use_learned_pos=args.use_learned_pos
    )
    
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, config)
    
    # Text to visualize
    text = args.text
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    # Truncate if too long
    if len(input_ids) > config.max_seq_len:
        input_ids = input_ids[:config.max_seq_len]
        
    x = torch.tensor([input_ids]) # Add batch dim

    # --- THE HOOK ---
    # We want to grab the attention weights from the Last Layer
    # Path: model.layers[-1].block.self_attn
    attention_weights = {}

    def get_attn_hook(name):
        def hook(module, input, output):
            # The output of MultiHeadAttention in your utils.py is the Projected Output.
            # We cannot easily get the weights from the output.
            # INSTEAD: We will manually re-calculate the attention scores for visualization
            # inside this hook using the input (query/key/value).
            
            # Input to self_attn is (x, x, x, ...)
            hidden_states = input[0] # (Batch, Seq, Dim)
            
            # Re-run the projection logic from utils.py to get Q and K
            B, T, C = hidden_states.shape
            head_dim = module.head_dim
            
            q = module.q_proj(hidden_states)
            k = module.k_proj(hidden_states)
            
            q = q.view(B, T, module.num_heads, head_dim).transpose(1, 2)
            k = k.view(B, T, module.num_heads, head_dim).transpose(1, 2)
            
            # Apply RoPE if it was passed in the forward pass kwargs
            # (Fetching kwargs in a hook is hard, so we assume RoPE if configured)
            if config.use_rope:
                # We need to re-generate freqs
                from utils import RotaryEmbedding, apply_rotary_pos_emb
                # Creating a temporary rope instance to get cos/sin
                temp_rope = RotaryEmbedding(head_dim, config.max_seq_len)
                cos, sin = temp_rope(q, T) # T is seq_len
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # Calculate Scores
            scores = torch.matmul(q, k.transpose(-2, -1)) * module.scale
            
            # Apply Causal Mask
            mask = torch.triu(torch.ones(T, T), diagonal=1).to(hidden_states.device)
            scores = scores.masked_fill(mask == 1, float('-inf'))
            
            attn = torch.softmax(scores, dim=-1)
            attention_weights[name] = attn.detach()
            
        return hook

    # Register hook on the last layer
    last_layer_idx = config.num_layers - 1
    model.layers[last_layer_idx].block.self_attn.register_forward_hook(get_attn_hook("last_layer"))

    # Run Forward
    with torch.no_grad():
        # We don't care about output, just triggering the hook
        model(x, attention_mask=torch.ones_like(x))

    # --- PLOTTING ---
    if "last_layer" not in attention_weights:
        print("Error: Failed to capture attention weights.")
        return

    # Shape: (Batch, Num_Heads, Seq_Len, Seq_Len)
    attn = attention_weights["last_layer"][0] # Remove batch
    
    # Convert tokens to strings for labels
    tokens = [tokenizer.decode([t], skip_special_tokens=False) for t in input_ids]

    # Plot Head 0 (or average of all heads)
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn[0].numpy(), xticklabels=tokens, yticklabels=tokens, cmap="viridis", square=True)
    plt.title(f"Attention Heatmap (Layer {config.num_layers}, Head 0)\n'{text}'")
    plt.xlabel("Key (Attended To)")
    plt.ylabel("Query (Current Position)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    save_path = "attention_heatmap.png"
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Success! Heatmap saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to decoder.pt")
    parser.add_argument("--text", type=str, default="The quick brown fox jumps over the lazy dog", help="Text to visualize")
    parser.add_argument("--use_rope", action="store_true")
    parser.add_argument("--use_learned_pos", action="store_true")
    args = parser.parse_args()
    
    visualize_attention(args)