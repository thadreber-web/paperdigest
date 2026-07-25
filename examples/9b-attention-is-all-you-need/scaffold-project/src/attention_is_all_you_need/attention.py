"""
Implements scaled dot-product and multi-head attention mechanisms for sequence processing.
Based on the Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017).
"""

from __future__ import annotations

import math
from typing import Optional


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor = None
) -> Tensor:
    """
    Compute scaled dot-product attention.

    Computes attention weights by scaling dot products of query and key vectors
    to prevent vanishing gradients.

    See paper §3.2.1 Scaled Dot-Product Attention, Eq. 1

    Args:
        q: Query tensor of shape [batch_size, num_heads, seq_len, d_k]
        k: Key tensor of shape [batch_size, num_heads, seq_len, d_k]
        v: Value tensor of shape [batch_size, num_heads, seq_len, d_k]
        mask: Optional mask tensor to apply attention masking

    Returns:
        Output tensor of shape [batch_size, num_heads, seq_len, d_v]
    """

    # TODO(paper §3.2.1, Eq. 1): Implement scaled dot-product attention computation
    # q @ k.transpose(-2, -1) / sqrt(d_k)
    # softmax over heads
    raise NotImplementedError


def multi_head_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    num_heads: int,
    head_dim: int,
    mask: Tensor = None
) -> Tensor:
    """
    Compute multi-head attention by applying scaled dot-product attention multiple times in parallel.

    Applies scaled dot-product attention multiple times in parallel to jointly attend to information
    from different representation subspaces.

    See paper §3.2.2 Multi-Head Attention, Eq. 2

    Args:
        q: Query tensor of shape [batch_size, seq_len, d_model]
        k: Key tensor of shape [batch_size, seq_len, d_model]
        v: Value tensor of shape [batch_size, seq_len, d_model]
        num_heads: Number of attention heads (N in paper)
        head_dim: Dimension per head (d_k = d_model / N)
        mask: Optional mask tensor to apply attention masking

    Returns:
        Output tensor of shape [batch_size, seq_len, d_model]
    """

    # TODO(paper §3.2.2, Eq. 2): Implement multi-head attention computation
    # Split q, k, v into num_heads sub-queries, sub-keys, sub-values
    # Apply scaled_dot_product_attention for each head
    # Concatenate heads and linear projection
    raise NotImplementedError
