# Using VideoPrism for Video Understanding

This guide provides a minimal end-to-end example for loading a pretrained VideoPrism model, preparing inputs, and running inference. The snippets build on top of the utilities shipped with this repository so you can adapt them to your own projects.

## 1. Install and Import

Install the package from source and import the helper APIs:

```bash
pip install -e .
```

```python
import jax
from videoprism import models as vp
```

## 2. Load a Model and Checkpoint

Choose a configuration and load its pretrained weights:

```python
model_name = "videoprism_lvt_public_v1_base"  # video-text encoder
model = vp.get_model(model_name)
state = vp.load_pretrained_weights(model_name)
```

`get_model` builds the Flax module while `load_pretrained_weights` downloads the published `.npz` checkpoint and returns the model parameters.

## 3. Prepare Inputs

### Video

Video inputs must be float32 RGB values in `[0, 1]` with shape `(batch, frames, 288, 288, 3)`. Any video loader can be used; below we show `tensorflow-io` for simplicity:

```python
import tensorflow_io as tfio

path = "./sample.mp4"
video = tfio.videoread(path)            # shape (num_frames, height, width, 3)
video = tf.image.resize(video, (288, 288))
video = tf.cast(video, jax.numpy.float32) / 255.0
video = video[jax.numpy.newaxis, ...]   # add batch dimension
```

### Text

For video-text models, tokenize query strings using the provided SentencePiece model:

```python
text_tokenizer = vp.load_text_tokenizer('c4_en')
queries = ["a dog catching a frisbee"]
text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, queries)
```

## 4. Run Inference

```python
@jax.jit
def forward(video_inputs, text_ids, text_paddings):
    return model.apply(state, video_inputs, text_ids, text_paddings, train=False)

video_emb, text_emb, _ = forward(video, text_ids, text_paddings)
```

`video_emb` and `text_emb` are `(batch, feature_channels)` embeddings that can be compared with cosine similarity for retrieval tasks. To use the video-only encoder, omit the text-related arguments and the model will return spatial-temporal tokens instead of global embeddings.

## 5. Next Steps

- Plug the embeddings into a downstream classifier.
- Compute cosine similarity between a query text and a database of video embeddings.
- Fine-tune the model by adding task-specific heads on top of the frozen encoder.

This simple workflow demonstrates how to integrate VideoPrism into custom pipelines for classification, retrieval, or other video understanding tasks.
