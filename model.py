import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def round_ste(z: torch.Tensor) -> torch.Tensor:
    """
    Round with Straight-Through Estimator (STE).
    """
    z_hat = torch.round(z)
    return z + (z_hat - z).detach()

class FSQ(nn.Module):
    """
    Finite Scalar Quantization (FSQ) Module.
    Implementation mirrors the design in Appendix A.1 of the paper.
    """
    def __init__(self, levels: list[int]):
        super().__init__()
        self.levels = torch.tensor(levels, dtype=torch.float32)
        self.d = len(levels) # Number of dimensions
        self.codebook_size = np.prod(levels)
        
        # Basis for converting indices: [1, L1, L1*L2, ...]
        basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0)
        self.register_buffer('basis', basis.to(torch.int64))
        
        # Pre-calculate for bound function
        self._levels_np = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer('half_width', self._levels_np // 2)
        
        eps = 1e-3
        # [d]
        half_l = (self._levels_np - 1) * (1 - eps) / 2
        # [d]
        offset = torch.where(self._levels_np % 2 == 1, 0.0, 0.5)
        # [d]
        shift = torch.tan(offset / half_l)
        
        self.register_buffer('half_l', half_l)
        self.register_buffer('offset', offset)
        self.register_buffer('shift', shift)

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """
        Applies the bounding function f(z) before rounding.
        """
        return torch.tanh(z + self.shift) * self.half_l - self.offset

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Quantizes z, returns the quantized z_hat (normalized).
        Input shape: (Batch, Sequence, d) or (Batch, d)
        """
        # 1. Bound the input
        z_bounded = self.bound(z)
        
        # 2. Round with STE
        z_hat_integers = round_ste(z_bounded)
        
        # 3. Renormalize to [-1, 1] range
        z_hat_normalized = z_hat_integers / self.half_width
        
        return z_hat_normalized

    def codes_to_indexes(self, z_hat_normalized: torch.Tensor) -> torch.Tensor:
        """Converts normalized quantized vectors to single integer indices."""
        z_hat_indices = (z_hat_normalized * self.half_width) + self.half_width
        z_hat_indices = z_hat_indices.round().to(torch.uint32)
        return (z_hat_indices * self.basis).sum(dim=-1).to(torch.uint32)

    def indexes_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Converts single integer indices back to normalized quantized vectors."""
        indices = indices.unsqueeze(-1) # (..., 1)
        indices_long = indices.to(torch.int64)
        basis_long = self.basis.to(torch.int64)
        
        codes_non_centered = (indices_long // basis_long) % self._levels_np.to(torch.int64)
        
        z_hat_normalized = (codes_non_centered - self.half_width) / self.half_width
        return z_hat_normalized

class RotaryEmbedding(nn.Module):
    """
    1D Rotary Positional Embedding (RoPE).
    """
    def __init__(self, dim: int, max_seq_len: int = 10000, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        
        # Precompute cos and sin
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Shape: (seq_len, dim/2) -> (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """
        Args:
            x: (B, Num_Heads, Seq_Len, Head_Dim)
        Returns:
            cos, sin with shape (1, 1, Seq_Len, Head_Dim)
        """
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
            
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Applies RoPE to Q and K.
    """
    # Note: RoPE is usually applied in float32 for stability
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadSelfAttention(nn.Module):
    """
    Custom Multi-Head Self Attention with RoPE. No Dropout.
    """
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # Initialize RoPE
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, Seq_Len, Dim)
        """
        B, N, C = x.shape
        
        # 1. Compute QKV
        qkv = self.qkv(x) # (B, N, 3*Dim)
        
        # 2. Reshape to (B, N, 3, Num_Heads, Head_Dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        
        # 3. Permute to (3, B, Num_Heads, N, Head_Dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 4. Apply RoPE
        cos, sin = self.rope(v, seq_len=N)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # 5. Attention (No dropout)
        x = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=0.0,
            is_causal=False
        )
        
        # 6. Reshape back: (B, Num_Heads, N, Head_Dim) -> (B, N, Dim)
        x = x.transpose(1, 2).reshape(B, N, C)
        
        # 7. Output projection
        x = self.proj(x)
        
        return x


class FeedForward(nn.Module):
    """FeedForward without Dropout"""
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer Block without Dropout"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CustomTransformerEncoder(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class TransformerFSQ(nn.Module):
    """
    Transformer-based VAE with FSQ and RoPE Embeddings.
    No Dropout.
    """
    def __init__(
        self,
        levels: list[int],
        input_dim: int,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward_ratio: float = 4.0,
    ):
        super().__init__()
        
        self.fsq_dim = len(levels)
        self.d_model = d_model
        
        # Encoder
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.encoder = CustomTransformerEncoder(
            dim=d_model, 
            depth=num_encoder_layers, 
            num_heads=num_heads, 
            mlp_ratio=dim_feedforward_ratio
        )
        
        # Bottleneck
        self.pre_fsq_proj = nn.Linear(d_model, self.fsq_dim)
        
        # FSQ
        self.fsq = FSQ(levels)
        
        # Decoder
        self.post_fsq_proj = nn.Linear(self.fsq_dim, d_model)
        
        self.decoder = CustomTransformerEncoder(
            dim=d_model, 
            depth=num_decoder_layers, 
            num_heads=num_heads, 
            mlp_ratio=dim_feedforward_ratio
        )
        
        self.output_head = nn.Linear(d_model, input_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.encoder(x)
        z_e = self.pre_fsq_proj(x)
        return z_e

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        x = self.post_fsq_proj(z_q)
        x = self.decoder(x)
        x_rec = self.output_head(x)
        return x_rec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encode(x)
        z_q = self.fsq(z_e)
        x_hat = self.decode(z_q)
        return x_hat, z_e, z_q

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encode(x)
        z_q = self.fsq(z_e)
        indices = self.fsq.codes_to_indexes(z_q)
        return indices

    @torch.no_grad()
    def decompress(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.fsq.indexes_to_codes(indices)
        x_hat = self.decode(z_q)
        return x_hat