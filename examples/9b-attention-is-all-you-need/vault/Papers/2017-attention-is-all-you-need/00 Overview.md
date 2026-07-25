---
source: https://arxiv.org/abs/1706.03762
date: 2026-07-25
model: qwen35-9b
level: intermediate
tags: [paper-digest]
---

# Attention Is All You Need

**TLDR:** This paper introduces the Transformer, a new model architecture for machine translation that uses only attention mechanisms instead of recurrent or convolutional networks. It trains faster and achieves better results than previous methods, including on parsing tasks.

## Why it matters

This paper shifted the field of sequence modeling away from recurrent networks towards attention-based models, enabling massive parallelization during training. It established the foundation for modern large language models like GPT and BERT.

## Concepts

- [[01 Scaled Dot-Product Attention]]
- [[02 Multi-Head Attention]]
- [[03 Positional Encoding]]
- [[04 Transformer Architecture]]
- [[05 Why Self-Attention]]
- [[06 Self-Attention]]

## Glossary terms

[[attention]] · [[encoder]] · [[decoder]] · [[recurrent neural network]] · [[convolutional neural network]] · [[self-attention]] · [[multi-head attention]] · [[scaled dot-product attention]] · [[positional encoding]] · [[residual connection]] · [[layer normalization]] · [[softmax]] · [[query]] · [[key]] · [[value]] · [[embedding]] · [[transformer]] · [[bleu score]] · [[word-piece]] · [[byte-pair encoding]] · [[dropout]] · [[adam optimizer]] · [[perplexity]] · [[auto-regressive]] · [[attention mechanism]]

## Check your understanding

1. What fundamental limitation of recurrent neural networks does the Transformer aim to solve?
2. Explain how the model handles sequence order without using recurrent layers.
3. Describe the role of the decoder's masking mechanism during training.
4. Why was the scaling factor applied to the dot products in scaled dot-product attention?
5. How did the authors evaluate the model's performance on translation tasks?


## Figure: Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of the verb ‘making’, completing the phrase ‘making…more difficult’. Attentions here shown only for the word ‘making’. Different colors represent different heads. Best viewed in color.

![[fig3.png]]

This figure visualizes the attention mechanism within the Transformer architecture, a core innovation of this paper. It displays a specific "attention head" from the encoder's layer 5, tracking how the model processes the word "making."

The diagram shows colored lines connecting "making" to other words in the sequence, such as "be," "more," and "difficult." Each color represents a distinct attention head, showing how different sub-modules focus on different parts of the input sentence simultaneously.

Crucially, the figure demonstrates the model's ability to capture "long-distance dependencies"—connections between words far apart in a sentence. Here, "making" attends strongly to "more difficult," completing the grammatical phrase "making…more difficult." This visual evidence is vital because it proves that the Transformer, which lacks the sequential memory of previous RNNs or LSTMs, can effectively understand context and syntax across a whole sentence at once. It confirms the model learns meaningful representations rather than just local word patterns.


## Figure: Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word ‘its’ for attention heads 5 and 6. Note that the attentions are very sharp for this word.

![[fig4.png]]

This figure visualizes the "attention" mechanism inside the Transformer model, specifically for two distinct "heads" in layer 5.

Visually, the dense web of purple lines represents connections between words in a sentence. The thickness or density of these lines indicates how strongly the model links two specific words together. The caption notes these heads are involved in "anaphora resolution"—solving which noun a pronoun (like "its") refers to.

Why it matters: It provides concrete proof that the Transformer works as intended. By showing sharp, focused connections (lines converging on specific words), it demonstrates the model is actively linking words based on context rather than just using random weights. This validates that attention can effectively replace complex recurrent networks, allowing the model to solve "dependency" problems (like connecting a pronoun to its antecedent) efficiently.


## Figure: Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks.

![[fig5.png]]

This figure visualizes the attention weights of specific neural network "heads" within the Transformer model. Visually, it displays a dense web of green lines connecting positions on the left (input words) to positions on the right (output positions). These lines represent the model's focus: a denser connection indicates that the model is paying significant attention to that specific relationship between words.

The caption highlights that these specific heads (from layer 5 of 6) exhibit behavior related to sentence structure. This matters because it provides transparency into the "black box." It proves the model isn't just randomly connecting words; distinct heads learn to perform specific tasks, such as matching words or tracking syntactic dependencies. This visual evidence supports the paper's claim that the Transformer architecture is effective at replacing recurrent networks (like RNNs or LSTMs) for machine translation, demonstrating that the model is learning meaningful, structured representations of the text rather than relying on sequential processing.
