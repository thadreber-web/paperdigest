"""Module docstring for the Transformer implementation.
See paper "Attention Is All You Need" (Vaswani et al., 2017).
"""
from __future__ import annotations

class TransformerEncoder:
    """
    Transformer encoder stack.
    See paper §3.1 Encoder and Decoder Stacks.
    """
    def __init__(self, d_model: int = 512, N: int = 6, d_k: int = 64, d_v: int = 64, P_drop: float = 0.1):
        # TODO(paper §3.1): Initialize encoder layers.
        raise NotImplementedError

    def forward(self, src, mask=None):
        # TODO(paper §3.1): Forward pass through encoder stack.
        raise NotImplementedError

class TransformerDecoder:
    """
    Transformer decoder stack.
    See paper §3.1 Encoder and Decoder Stacks.
    """
    def __init__(self, d_model: int = 512, N: int = 6, d_k: int = 64, d_v: int = 64, P_drop: float = 0.1):
        # TODO(paper §3.1): Initialize decoder layers.
        raise NotImplementedError

    def forward(self, memory, tgt, memory_key_padding_mask=None, tgt_mask=None, src_mask=None):
        # TODO(paper §3.1): Forward pass through decoder stack.
        raise NotImplementedError

class Transformer:
    """
    Complete Transformer model combining encoder, decoder, and embeddings.
    See paper §3.1 Encoder and Decoder Stacks.
    """
    def __init__(self, d_model: int = 512, N: int = 6, d_k: int = 64, d_v: int = 64, P_drop: float = 0.1, d_ff: int = 2048):
        # TODO(paper §3.1): Initialize Transformer model (encoder, decoder, embeddings).
        raise NotImplementedError

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # TODO(paper §3.1): Forward pass through Transformer model.
        raise NotImplementedError
