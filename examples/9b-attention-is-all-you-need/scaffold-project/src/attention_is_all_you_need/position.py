"""
Positional encoding module.

Generates sinusoidal positional encodings to inject sequence order information.
See paper §3.5 Positional Encoding.
"""

from __future__ import annotations


def positional_encoding(pos: int, dim: int) -> Tensor:
    """
    Generates sinusoidal positional encoding for a given position and dimension.
    
    See paper §3.5 Positional Encoding, Eq. 2-3.
    
    The positional encoding uses sinusoidal functions of different frequencies
    to encode absolute positional information. For each dimension j,
    the encoding at position i is computed as:
    
    pos_encoding(i, 2j) = sin(i / 10000^(2j/d_model))
    pos_encoding(i, 2j+1) = cos(i / 10000^(2j/d_model))
    
    Args:
        pos: Position index.
        dim: Dimension of the output vector.
    
    Returns:
        Positional encoding vector at position `pos` with dimension `dim`.
    """
    # TODO(paper §3.5): Implement sinusoidal positional encoding formula
    # See paper Eq. 2-3 for the mathematical formulation
    raise NotImplementedError
