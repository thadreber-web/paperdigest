from attention_is_all_you_need import attention, layers, position, transformer


class TestAttention:
    def test_scaled_dot_product_attention_exists(self):
        assert hasattr(attention, 'scaled_dot_product_attention')
        assert callable(attention.scaled_dot_product_attention)

    def test_multi_head_attention_exists(self):
        assert hasattr(attention, 'multi_head_attention')
        assert callable(attention.multi_head_attention)


class TestLayers:
    def test_feed_forward_exists(self):
        assert hasattr(layers, 'feed_forward')
        assert callable(layers.feed_forward)

    def test_layer_norm_exists(self):
        assert hasattr(layers, 'layer_norm')
        assert callable(layers.layer_norm)

    def test_residual_exists(self):
        assert hasattr(layers, 'residual')
        assert callable(layers.residual)


class TestPosition:
    def test_positional_encoding_exists(self):
        assert hasattr(position, 'positional_encoding')
        assert callable(position.positional_encoding)


class TestTransformer:
    def test_Transformer_exists(self):
        assert hasattr(transformer, 'Transformer')
        assert issubclass(transformer.Transformer, object)

    def test_TransformerEncoder_exists(self):
        assert hasattr(transformer, 'TransformerEncoder')
        assert issubclass(transformer.TransformerEncoder, object)

    def test_TransformerDecoder_exists(self):
        assert hasattr(transformer, 'TransformerDecoder')
        assert issubclass(transformer.TransformerDecoder, object)
