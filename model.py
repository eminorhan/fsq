import torch
import torch.nn as nn
import numpy as np


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """
    Round with Straight-Through Estimator (STE).
    """
    z_hat = torch.round(z)
    return z + (z_hat - z).detach()


class FSQ(nn.Module):
    """
    Finite Scalar Quantization (FSQ) Module
    
    This is a PyTorch implementation of the FSQ method from: https://arxiv.org/abs/2309.15505
    """
    def __init__(self, levels: list[int]):
        super().__init__()
        # TODO: check dtypes
        # [d]
        self.levels = torch.tensor(levels, dtype=torch.float32)
        self.d = len(levels) # Number of dimensions
        
        # [d], e.g., [1, L1, L1*L2, ...]
        basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0)
        self.register_buffer('basis', basis.to(torch.int64))
        
        self.codebook_size = np.prod(levels)
        
        # Pre-calculate for bound function
        self.register_buffer('_levels_np', torch.tensor(levels, dtype=torch.float32))
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
        # This function is a bit complex, but it's a general
        # way to map z to a range that, when rounded,
        # produces L distinct integer values.
        # A simpler version is f:z -> floor(L/2) * tanh(z)
        return torch.tanh(z + self.shift) * self.half_l - self.offset

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Quantizes z, returns the quantized z_hat (normalized).
        """
        # 1. Bound the input
        z_bounded = self.bound(z)
        
        # 2. Round with STE
        z_hat_integers = round_ste(z_bounded)
        
        # 3. Renormalize to [-1, 1] range for the decoder
        z_hat_normalized = z_hat_integers / self.half_width
        
        return z_hat_normalized

    def _scale_and_shift(self, z_hat_normalized: torch.Tensor) -> torch.Tensor:
        """Helper to convert normalized codes to {0, 1, ..., L-1} indices."""
        return (z_hat_normalized * self.half_width) + self.half_width

    def _scale_and_shift_inverse(self, z_hat_indices: torch.Tensor) -> torch.Tensor:
        """Helper to convert {0, 1, ..., L-1} indices to normalized codes."""
        return (z_hat_indices - self.half_width) / self.half_width

    def codes_to_indexes(self, z_hat_normalized: torch.Tensor) -> torch.Tensor:
        """
        Converts normalized quantized vectors to single integer indices.
        
        Args:
            z_hat_normalized (Tensor): Shape (..., d)
        Returns:
            indices (Tensor): Shape (...,)
        """
        # Convert from normalized e.g. [-1, 0, 1] to {0, 1, 2}
        z_hat_indices = self._scale_and_shift(z_hat_normalized)
        z_hat_indices = z_hat_indices.round().to(torch.uint32)
        
        # Project to 1D index
        return (z_hat_indices * self.basis).sum(dim=-1).to(torch.uint32)

    def indexes_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Converts single integer indices back to normalized quantized vectors.
        
        Args:
            indices (Tensor): Shape (...,)
        Returns:
            z_hat_normalized (Tensor): Shape (..., d)
        """
        indices = indices.unsqueeze(-1) # (..., 1)
        
        # Cast to int64 (Long) for floor division, as uint32 is not supported
        indices_long = indices.to(torch.int64)
        basis_long = self.basis.to(torch.int64)

        # (..., d)
        codes_non_centered = (indices_long // basis_long) % self._levels_np
        
        # Convert from {0, 1, 2} back to normalized e.g. [-1, 0, 1]
        z_hat_normalized = self._scale_and_shift_inverse(codes_non_centered)
        
        return z_hat_normalized


# Transformer like MLPBlock
class MLPBlock(nn.Module):
    """
    A Transformer-style FFN block with pre-normalization.
    mlp_expansion_factor determines the "bottleneck" size.
    """
    def __init__(self, hidden_dim: int, mlp_expansion_factor: int = 4):
        super().__init__()
        mlp_dim = hidden_dim * mlp_expansion_factor
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-normalization
        h = self.norm(x)
        
        # FFN
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        
        # Residual connection
        return h + x


class MLPEncoder(nn.Module):
    """MLP Encoder for FSQ"""
    def __init__(self, input_dim: int, hidden_dim: int, fsq_dim: int, depth: int):
        super().__init__()
                
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.proj_act = nn.GELU()

        # A stack of MLP blocks
        layers = []
        for _ in range(depth):
            layers.append(MLPBlock(hidden_dim=hidden_dim))

        self.blocks = nn.Sequential(*layers)
        
        # Final layer norm and projection head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, fsq_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Input layer
        x = self.proj_act(self.proj(x))
        
        # Pass through blocks
        x = self.blocks(x)
        
        # Normalize
        x = self.norm(x)
        
        # Project to FSQ's latent dimension 'd'
        z_e = self.head(x)
        return z_e


class MLPDecoder(nn.Module):
    """MLP Decoder for FSQ"""
    def __init__(self, fsq_dim: int, hidden_dim: int, output_dim: int, depth: int):
        super().__init__()
        
        # Project from FSQ's dim 'd' back to MLP hidden dim
        self.proj = nn.Linear(fsq_dim, hidden_dim)
        self.proj_act = nn.GELU()
        
        # A stack of MLP blocks
        layers = []
        for _ in range(depth):
            layers.append(MLPBlock(hidden_dim=hidden_dim))

        self.blocks = nn.Sequential(*layers)
        
        # Final norm and projection head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        
        # Project back to hidden_dim
        x = self.proj_act(self.proj(z_q))
                
        # Pass through decoder
        x = self.blocks(x)
        x = self.norm(x)
                
        # Pass through decoder head     
        x = self.head(x)
        return x


class FSQ_VAE(nn.Module):
    """
    An MLP-based autoencoder using FSQ.
    """
    def __init__(
        self, 
        levels: list[int],
        input_dim: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        encoder_depth: int,
        decoder_depth: int,
    ):
        super().__init__()
        
        self.fsq_dim = len(levels)
        
        # MLP Encoder
        self.encoder = MLPEncoder(input_dim, encoder_hidden_dim, self.fsq_dim, encoder_depth)
        
        # FSQ Module
        self.fsq = FSQ(levels)
        
        # MLP Decoder
        self.decoder = MLPDecoder(self.fsq_dim, decoder_hidden_dim, input_dim, decoder_depth)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full pass for training.
        x shape: (B, L), normalized to [0, 1].
        """
        
        # Encode
        z_e = self.encoder(x)  # (B, d)
        
        # Quantize
        # FSQ forward applies to the last dimension
        z_q_normalized = self.fsq(z_e)  # (B, d)
        
        # Decode
        x_hat = self.decoder(z_q_normalized)  # (B, L)
                
        return x_hat, z_e, z_q_normalized

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compresses the input array into a sequence of integer indices.
        x_in shape: (B, L), normalized to [0, 1]
        """        
        z_e = self.encoder(x)    # (B, d)
        
        # Quantize (no STE, but fsq.forward doesn't use it anyway)
        z_q_normalized = self.fsq(z_e)
        
        # Get indices (this is the compressed token indices)
        indices = self.fsq.codes_to_indexes(z_q_normalized)  # (B,)
        
        return indices

    @torch.no_grad()
    def decompress(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decompresses a sequence of integer indices back into an array.
        indices shape: (B,)
        """
        # (B,) -> (B, d)
        z_q_normalized = self.fsq.indexes_to_codes(indices)
        
        # Decode
        x_hat = self.decoder(z_q_normalized)  # (B, L)
        
        return x_hat
