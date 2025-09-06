# VideoPrism Project Overview

VideoPrism is a JAX/Flax-based foundational video encoder designed to solve a wide range of video understanding tasks such as classification, retrieval, localization, captioning, and question answering. The models are pre-trained on billions of image-text and video-text pairs to learn strong visual representations that transfer across tasks with minimal adaptation.

## Repository Layout

- `videoprism/`
  - `encoders.py` – core video and video-text encoder architectures.
  - `layers.py` – custom Flax layers (attention, transformer blocks, normalization, pooling).
  - `models.py` – high level builders and checkpoint loaders for released variants.
  - `tokenizers.py` – lightweight SentencePiece-based tokenizer utilities.
  - `utils.py` – helpers for checkpoint I/O and text processing.
  - `*_test.py` – unit tests demonstrating encoder behaviors.
- `requirements.txt` – minimal dependency list.
- `setup.py` / `pyproject.toml` – packaging metadata.
- `README.md` – introduction, usage examples, and model checkpoints.

## Core Concepts

### Factorized Vision Transformer

The `FactorizedEncoder` implements the architecture from the ViViT paper, factorizing spatial and temporal attention:
1. **Patch Projection** – Frames are divided into patches and linearly projected.
2. **Spatial Encoding** – A stack of spatial transformer layers processes each frame independently using 2D positional embeddings.
3. **Temporal Encoding** – Tokens are regrouped across time and passed through temporal transformer layers with 1D positional embeddings.
4. **Outputs** – The encoder returns a sequence of spatiotemporal embeddings that can be pooled for downstream tasks.

### Video CLIP and Classification Heads

- `FactorizedVideoCLIP` combines a video encoder with a text encoder to produce normalized embeddings for cross-modal retrieval.
- `FactorizedVideoClassifier` adds an attention pooling layer and projection head for action or video classification tasks.

### Text Tokenization

`tokenizers.py` wraps SentencePiece to tokenize query text. A default C4 tokenizer (`c4_en`) is provided and referenced in `models.py` when building video‑text models.

## Using Pretrained Models

`models.py` exposes factory functions and checkpoint URLs:
```python
from videoprism import models as vp

model_name = "videoprism_public_v1_base"
flax_model = vp.get_model(model_name)
params = vp.load_pretrained_weights(model_name)
```
The `CHECKPOINTS` dictionary maps model names to Hugging Face repositories hosting `.npz` weight files, which are loaded into the Flax modules at runtime.

## Tokenizing Text Queries

```python
text_tokenizer = vp.load_text_tokenizer('c4_en')
queries = ["a person riding a bike", "dogs playing"]
ids, paddings = vp.tokenize_texts(text_tokenizer, queries)
```
The tokenizer returns integer IDs and padding masks compatible with the text encoder.

## Example Forward Pass

```python
import jax
from videoprism import models as vp

# Build model and load weights
model_name = 'videoprism_lvt_public_v1_base'
model = vp.get_model(model_name)
state = vp.load_pretrained_weights(model_name)

@jax.jit
def forward(video_inputs, text_ids, text_paddings):
    return model.apply(state, video_inputs, text_ids, text_paddings, train=False)
```

## Additional Resources

- [Paper](https://arxiv.org/abs/2402.13217) – technical details on training data and architecture.
- [Blog post](https://research.google/blog/videoprism-a-foundational-visual-encoder-for-video-understanding/) – high level overview and results.
- [Colab notebooks](videoprism/colabs) – interactive demos for video encoders and video‑text retrieval.

VideoPrism demonstrates that a single frozen encoder can achieve state‑of‑the‑art results on many public benchmarks, making it a powerful starting point for new video understanding applications.
