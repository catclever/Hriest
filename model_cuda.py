import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =========================================================================
# TEACHER: GodEncoder & SensoryFuser (PyTorch Port of Frozen Phase 0)
# =========================================================================

class AdapterCUDA(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class SensoryFuserCUDA(nn.Module):
    def __init__(self, emb_dims, z_dim):
        super().__init__()
        # We must use ModuleList to dynamically create the adapters
        self.adapters = nn.ModuleList([AdapterCUDA(dim, z_dim) for dim in emb_dims])
        self.z_dim = z_dim
        
    def forward(self, embs, weights=None):
        B = embs[0].shape[0] # PyTorch shape (B, in_dim)
        stacked_z = torch.stack([adapter(emb) for adapter, emb in zip(self.adapters, embs)], dim=1) # (B, 4, z_dim)
        
        if weights is None:
            # Centroid mean
            z_fused = torch.mean(stacked_z, dim=1) # (B, z_dim)
        else:
            w = weights.unsqueeze(-1) # (B, 4, 1)
            z_fused = torch.sum(stacked_z * w, dim=1)
            
        return z_fused

class GodEncoderCUDA(nn.Module):
    def __init__(self, d_model, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_model, z_dim)
        
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

def load_mlx_safetensors_into_torch(torch_module, safetensor_path):
    """
    Seamlessly loads an MLX-generated safetensors file into a PyTorch nn.Module.
    MLX and PyTorch nn.Linear weight shapes identically map to (out_features, in_features).
    """
    from safetensors.torch import load_file
    state_dict = load_file(safetensor_path)
    target_keys = set(torch_module.state_dict().keys())
    if not any(k in target_keys for k in state_dict.keys()):
        remapped = {}
        for k, v in state_dict.items():
            new_k = k
            new_k = new_k.replace(".net.layers.0.", ".fc1.")
            new_k = new_k.replace(".net.layers.2.", ".fc2.")
            new_k = new_k.replace("net.layers.0.", "fc1.")
            new_k = new_k.replace("net.layers.2.", "fc2.")
            remapped[new_k] = v
        state_dict = remapped
    # Add support for WeakDecoder if any key needs remapping (it perfectly matches)
    for k in list(state_dict.keys()):
        if k.endswith('.wq.weight'):
            state_dict[k.replace('.wq.', '.query_proj.')] = state_dict.pop(k)
        elif k.endswith('.wk.weight'):
            state_dict[k.replace('.wk.', '.key_proj.')] = state_dict.pop(k)
        elif k.endswith('.wv.weight'):
            state_dict[k.replace('.wv.', '.value_proj.')] = state_dict.pop(k)
        elif k.endswith('.wo.weight'):
            state_dict[k.replace('.wo.', '.out_proj.')] = state_dict.pop(k)

    torch_module.load_state_dict(state_dict, strict=False)
    torch_module.eval()
    for param in torch_module.parameters():
        param.requires_grad = False
    print(f"Successfully bridged MLX safetensors to PyTorch from: {safetensor_path}")


class WeakTransformerLayerCUDA(nn.Module):
    def __init__(self, d_model=128, n_heads=4, mlp_dims=512):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = nn.ModuleDict({
            'query_proj': nn.Linear(d_model, d_model, bias=False),
            'key_proj': nn.Linear(d_model, d_model, bias=False),
            'value_proj': nn.Linear(d_model, d_model, bias=False),
            'out_proj': nn.Linear(d_model, d_model, bias=False)
        })
        self.ln2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, mlp_dims)
        self.linear2 = nn.Linear(mlp_dims, d_model)
        self.n_heads = n_heads
        
    def forward(self, x, mask=None):
        h = self.ln1(x)
        B, L, D = h.shape
        H = self.n_heads
        D_h = D // H
        
        q = self.attention['query_proj'](h).view(B, L, H, D_h).transpose(1, 2)
        k = self.attention['key_proj'](h).view(B, L, H, D_h).transpose(1, 2)
        v = self.attention['value_proj'](h).view(B, L, H, D_h).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.attention['out_proj'](out)
        
        x = x + out
        h = self.ln2(x)
        x = x + self.linear2(F.relu(self.linear1(h)))
        return x

class WeakTransformerEncoderCUDA(nn.Module):
    def __init__(self, num_layers=2, dims=128, num_heads=4, mlp_dims=512):
        super().__init__()
        self.layers = nn.ModuleList([WeakTransformerLayerCUDA(dims, num_heads, mlp_dims) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(dims)
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.ln(x)

class WeakDecoderCUDA(nn.Module):
    """
    PyTorch instantiation of WeakDecoder.
    """
    def __init__(self, z_dim: int, vocab_size: int, d_model: int = 128, n_layers: int = 2):
        super().__init__()
        self.z_dim = z_dim
        self.vocab_size = vocab_size
        self.z_proj = nn.Linear(z_dim, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = WeakTransformerEncoderCUDA(n_layers, d_model, 4, d_model * 4)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, z_target: torch.Tensor, token_inputs: torch.Tensor):
        z_projected = self.z_proj(z_target) # (B, d_model)
        x = self.embedding(token_inputs) # (B, L, d_model)
        x = x + z_projected.unsqueeze(1)
        
        L = x.shape[1]
        causal_mask = torch.ones((L, L), dtype=torch.bool, device=x.device).tril()
        mask = causal_mask.unsqueeze(0).unsqueeze(0) # (1, 1, L, L)
        
        x = self.transformer(x, mask=mask)
        logits = self.out_proj(x)
        return logits
        
    def generate(self, z_target: torch.Tensor, start_token: int, eos_token: int = None, max_tokens: int = 50, temperature: float = 0.7):
        self.eval()
        z_projected = self.z_proj(z_target) # (1, d_model)
        device = z_target.device
        
        tokens = torch.tensor([[start_token]], device=device)
        result_tokens = [start_token]
        
        with torch.no_grad():
            for i in range(max_tokens):
                x = self.embedding(tokens) # (1, seq_len, d_model)
                x = x + z_projected.unsqueeze(1)
                
                L = x.shape[1]
                causal_mask = torch.ones((L, L), dtype=torch.bool, device=x.device).tril()
                mask = causal_mask.unsqueeze(0).unsqueeze(0)
                
                x = self.transformer(x, mask=mask)
                logits = self.out_proj(x[:, -1, :]) # (1, vocab_size)
                
                if temperature == 0:
                    next_token = torch.argmax(logits, dim=-1).item()
                else:
                    # Scale logits by temperature and sample
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    
                result_tokens.append(next_token)
                
                if eos_token is not None and next_token == eos_token:
                    break
                    
                tokens = torch.tensor([result_tokens], device=device)
                
        return result_tokens


# =========================================================================
# STUDENT: TinyCharEncoder (PyTorch Implementation with RoPE & FlashAttention)
# =========================================================================

def apply_rotary_pos_emb(x, d_head):
    """
    Applies rotary position embeddings (RoPE) to a tensor of shape (B, H, L, D).
    """
    B, H, L, D = x.shape
    positions = torch.arange(L, device=x.device, dtype=x.dtype).unsqueeze(1) # (L, 1)
    
    # frequencies: theta_i = 10000 ^ (-2(i-1)/d)
    div_term = torch.exp(torch.arange(0, D, 2, device=x.device, dtype=x.dtype) * -(math.log(10000.0) / D))
    freqs = positions * div_term # (L, D/2)
    
    # Duplicate to (L, D) matching the complex sinusoidal pairing
    emb = torch.cat((freqs, freqs), dim=-1) # (L, D)
    cos = emb.cos().view(1, 1, L, D)
    sin = emb.sin().view(1, 1, L, D)
    
    # traditional rotary slice interleaving
    x1, x2 = x[..., :D//2], x[..., D//2:]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    
    return (x * cos) + (x_rotated * sin)

class RoPETransformerLayerCUDA(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Scaling initialization for variance control (Gradient Explosion Prevention)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * 6))
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model * 4)
        self.ff2 = nn.Linear(d_model * 4, d_model)
        self.gelu = nn.GELU()
        
        nn.init.normal_(self.ff2.weight, mean=0.0, std=0.02 / math.sqrt(2 * 6))
        
    def forward(self, x, mask=None):
        # Pre-LN
        h = self.ln1(x)
        B, L, _ = h.shape
        
        q = self.q_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2) # (B, H, L, D)
        k = self.k_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply RoPE directly
        q = apply_rotary_pos_emb(q, self.d_head)
        k = apply_rotary_pos_emb(k, self.d_head)
        
        # Invoke PyTorch Native FlashAttention-2!
        # PyTorch scaled_dot_product_attention seamlessly routes to FlashAttention when on CUDA.
        # If mask is boolean 0/1 where 1 is valid (from MLX), transform to boolean True/False mask.
        if mask is not None:
             # Generates an additive float mask (0.0 for valid, -large for padding)
             # PyTorch SDPA on MPS natively handles explicit large negative floats far better than Booleans or -inf
             attn_mask = torch.zeros((B, 1, 1, L), dtype=q.dtype, device=q.device)
             attn_mask = attn_mask.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e4) # -10000.0 is sufficiently zero in Softmax
        else:
             attn_mask = None
             
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.out_proj(out)
        
        # Residual 1
        x = x + out
        
        # Residual 2
        x = x + self.ff2(self.gelu(self.ff1(self.ln2(x))))
        return x

class TinyCharEncoderCUDA(nn.Module):
    """
    The PyTorch Clone of the Teacher-Distilled Character Flow Encoder.
    Completely refactored for native CUDA performance, utilizing pure PyTorch primitives.
    """
    def __init__(self, vocab_size: int, d_model: int = 1024, n_heads: int = 8, n_layers: int = 6, max_seq_len: int = 512, z_dim: int = 1024):
        super().__init__()
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([RoPETransformerLayerCUDA(d_model, n_heads) for _ in range(n_layers)])
        
        self.final_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, z_dim)
        
        # Better initialization for embedding table
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=d_model**-0.5)

    def forward(self, x, attention_mask=None):
        h = self.tok_emb(x)
        
        for layer in self.layers:
            h = layer(h, attention_mask)
            
        h = self.final_ln(h)
        
        if attention_mask is not None:
            # attention_mask: (B, L) where 1 is valid, 0 is pad
            # Safely mask out padding without risking 0 * NaN = NaN propagation
            h_masked = torch.where(attention_mask.unsqueeze(-1) > 0, h, torch.zeros_like(h))
            
            # Safe avg pool
            sum_mask = attention_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            h_pool = h_masked.sum(dim=1) / sum_mask
        else:
            h_pool = h.mean(dim=1)
            
        return self.out_proj(h_pool)

class StrongDecoderCUDA(nn.Module):
    """
    A powerful AR Decoder with RoPE and sufficient capacity.
    Takes z_target as prefix or added bias to tokens.
    """
    def __init__(self, z_dim: int, vocab_size: int, d_model: int = 512, n_layers: int = 6, n_heads: int = 8):
        super().__init__()
        self.z_dim = z_dim
        self.vocab_size = vocab_size
        self.z_proj = nn.Linear(z_dim, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([RoPETransformerLayerCUDA(d_model, n_heads) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)
        
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model**-0.5)

    def forward(self, z_target: torch.Tensor, token_inputs: torch.Tensor):
        z_projected = self.z_proj(z_target) # (B, d_model)
        x = self.embedding(token_inputs) # (B, L, d_model)
        
        # Add Z to all token embeddings
        x = x + z_projected.unsqueeze(1)
        
        B, L, D = x.shape
        # Causal mask for PyTorch SDPA
        causal_mask = torch.ones((L, L), dtype=torch.bool, device=x.device).tril()
        
        for layer in self.layers:
            x = layer(x, mask=causal_mask)
            
        x = self.final_ln(x)
        logits = self.out_proj(x)
        return logits
        
    def generate(self, z_target: torch.Tensor, start_token: int, eos_token: int = None, max_tokens: int = 50, temperature: float = 0.7):
        self.eval()
        z_projected = self.z_proj(z_target) # (1, d_model)
        device = z_target.device
        
        tokens = torch.tensor([[start_token]], device=device)
        result_tokens = [start_token]
        
        with torch.no_grad():
            for i in range(max_tokens):
                x = self.embedding(tokens) # (1, seq_len, d_model)
                x = x + z_projected.unsqueeze(1)
                
                L = x.shape[1]
                causal_mask = torch.ones((L, L), dtype=torch.bool, device=device).tril()
                
                for layer in self.layers:
                    x = layer(x, mask=causal_mask)
                
                x_final = self.final_ln(x)
                logits = self.out_proj(x_final[:, -1, :]) # (1, vocab_size)
                
                if temperature == 0:
                    next_token = torch.argmax(logits, dim=-1).item()
                else:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    
                result_tokens.append(next_token)
                
                if eos_token is not None and next_token == eos_token:
                    break
                    
                tokens = torch.tensor([result_tokens], device=device)
                
        return result_tokens



