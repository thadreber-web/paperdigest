Attention Is All You Need

arXiv Link: https://arxiv.org/abs/1706.03762

Method Summary:
This repository implements the Transformer architecture described in "Attention Is All You Need" (Vaswani et al., 2017). The model utilizes multi-head attention mechanisms to jointly attend to information from different representation subspaces via scaled dot-product attention. Position-wise feed-forward networks with residual connections and layer normalization are applied to each sub-layer and embedding layer output. Sinusoidal positional encodings are injected to inform the model of token positions within the sequence without learning relative positions from data.

Repo Map:
- `attention.py`: Implements scaled dot-product and multi-head attention mechanisms for sequence processing.
- `layers.py`: Provides feed-forward networks, layer normalization, and residual connection wrappers.
- `position.py`: Generates sinusoidal positional encodings to inject sequence order information.
- `transformer.py`: Combines encoder, decoder, and embedding layers into the complete Transformer model.

Setup Instructions:
Run `pip install -e '.[dev]'` to install the package with development dependencies.

Smoke Experiment:
Run `python scripts/run_smoke.py` to verify the basic transformer instantiation and forward pass.
