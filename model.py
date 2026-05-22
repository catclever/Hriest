import mlx.core as mx
import mlx.nn as nn

class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = nn.RoPE(self.d_head, traditional=True)

    def __call__(self, x: mx.array, mask: mx.array = None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.d_head)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.d_head)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.d_head)
        
        # Apple MLX 原生旋转复数注入 (RoPE)
        q = self.rope(q)
        k = self.rope(k)
        
        q = q.transpose(0, 2, 1, 3) # (B, H, L, D)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Apple MLX 原生底层 FlashAttention 核心 (极大节约显存，防止 Swap 暴降速)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v,
            scale=1.0 / (self.d_head ** 0.5),
            mask=mask
        )
        
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)

class RoPETransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attention = RoPEMultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model * 4)
        self.ff2 = nn.Linear(d_model * 4, d_model)
        self.gelu = nn.GELU()
        
    def __call__(self, x: mx.array, mask: mx.array = None):
        h = x + self.attention(self.ln1(x), mask)
        h = h + self.ff2(self.gelu(self.ff1(self.ln2(h))))
        return h

class TinyCharEncoder(nn.Module):
    """
    A lightweight, pure-native Text-to-Vector encoder.
    Distilled from the GodEncoder multi-view space.
    Upgraded: Uses RoPE (Rotary Position Embeddings) to allow infinite length extrapolation 
    without O(N^2) memory bloating during training!
    """
    def __init__(self, vocab_size: int, d_model: int = 1024, n_heads: int = 8, n_layers: int = 6, max_seq_len: int = 512, z_dim: int = 1024):
        super().__init__()
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        
        # A custom symmetric encoder explicitly built with RoPE logic
        self.layers = [RoPETransformerLayer(d_model, n_heads) for _ in range(n_layers)]
        self.final_ln = nn.LayerNorm(d_model)
        
        # Final projection from inner transformer dimension to the absolute GodEncoder semantic space dimension
        self.out_proj = nn.Linear(d_model, z_dim)
        
    def __call__(self, x: mx.array, attention_mask: mx.array = None, return_seq: bool = False) -> mx.array:
        """
        x: (B, L)
        attention_mask: (B, L) where 1 is valid, 0 is pad
        """
        B, L = x.shape
        
        # Pure Token Embeddings (No Abs POS Emb!)
        h = self.tok_emb(x) # (B, L, d_model)
        
        # Create non-causal additive padding mask
        mask = None
        if attention_mask is not None:
            mask = mx.where(attention_mask[:, None, None, :] == 0, mx.array(-1e9), mx.array(0.0))
            
        # Pumping through the RoPE Stack
        for layer in self.layers:
            h = layer(h, mask)
            
        h = self.final_ln(h)
        
        if return_seq:
            return self.out_proj(h) # (B, L, z_dim)
        
        # Average Pooling (Masked) to extract the holistic sentence representation
        if attention_mask is not None:
            h_masked = h * attention_mask[:, :, None] # zero out padded positions
            # Safe division by sum
            h_pool = h_masked.sum(axis=1) / mx.maximum(attention_mask.sum(axis=1, keepdims=True), 1e-8)
        else:
            h_pool = h.mean(axis=1) # (B, d_model)
            
        # Project into the 1024-D Z_target space!
        z_pred = self.out_proj(h_pool) # (B, z_dim)
        
        return z_pred
