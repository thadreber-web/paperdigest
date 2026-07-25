"""
Transformers Layers Module

Provides feed-forward networks, layer normalization, and residual connection wrappers
for the Transformer architecture described in "Attention Is All You Need" (Vaswani et al., 2017).

This module implements the core building blocks used in the Transformer architecture,
including position-wise feed-forward networks, layer normalization, and residual connections.
See: Vaswani et al., "Attention Is All You Need", 2017
"""
from __future__ import annotations

__all__ = ["feed_forward", "layer_norm", "residual"]


def feed_forward(x: Tensor) -> Tensor:
    """
    Position-wise feed-forward network.

    Applies two linear transformations with a ReLU activation to each position
    separately and identically, as described in Section 3.3 "Position-wise
    Feed-Forward Networks".

    See paper §3.3, Eq. 6 for the mathematical formulation.

    Args:
        x: Input tensor of shape (batch, seq_len, d_model)

    Returns:
        Output tensor of shape (batch, seq_len, d_model)
    """
    # TODO(paper §3.3, Eq. 6): Implement position-wise feed-forward network
    # Two linear transformations with ReLU activation
    raise NotImplementedError


def layer_norm(x: Tensor) -> Tensor:
    """
    Layer normalization.

    Normalizes the input tensor per dimension, as described in Section 3.1
    "Encoder and Decoder Stacks".

    See paper §3.1, Eq. 2 for the mathematical formulation.

    Args:
        x: Input tensor of shape (batch, seq_len, d_model)

    Returns:
        Normalized tensor of shape (batch, seq_len, d_model)
    """
    # TODO(paper §3.1, Eq. 2): Implement layer normalization
    raise NotImplementedError


def residual(x: Tensor) -> Tensor:
    """
    Residual connection wrapper.

    Wraps a sub-layer with a residual connection followed by layer normalization,
    as described in Section 3.1 "Encoder and Decoder Stacks".

    See paper §3.1 for the mathematical formulation.

    Args:
        x: Input tensor of shape (batch, seq_len, d_model)

    Returns:
        Residual connection output of shape (batch, seq_len, d_model)
    """
    # TODO(paper §3.1): Implement residual connection wrapper
    # residual = x + sublayer(x)
    raise NotImplementedError
