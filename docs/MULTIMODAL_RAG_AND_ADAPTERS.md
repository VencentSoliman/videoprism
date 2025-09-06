# Multimodal RAG and Adapters for Video Q&A

Goal: assemble multiple pretrained feature extractors (VideoPrism, CLIP, CLAP, OCR, detectors, etc.) to power accurate video Q&A. Use embeddings to retrieve and assemble evidence; let an LLM reason over compact, time‑coded text context. Optionally add trainable adapters to feed non‑text features directly into an LLM later.

## Mental Model

- Tokens vs. embeddings: LLMs accept text tokens. Internally, tokens are mapped to token embeddings via a learned matrix. Arbitrary external vectors are not “tokens” the LLM understands.
- No raw‑vector input: Serializing floats (e.g., JSON/base64) yields random tokens that don’t align with the LLM’s embedding space and won’t work without training.
- Two routes:
  - RAG (plug‑and‑play): Use embeddings to select and summarize evidence; feed text to the LLM.
  - Adapters (trainable): Learn a projector that maps external embeddings into the LLM’s hidden space as soft tokens or attention memory.

## VideoPrism Basics (from USAGE_EXAMPLE.md)

- Global embeddings: `video_emb` and `text_emb` are dense vectors suited for retrieval/similarity.
- Token mode: Omit text args to get spatial‑temporal tokens (fine‑grained cues across time/space). See the note in USAGE_EXAMPLE.md where the video‑only encoder returns tokens instead of global embeddings.
- Shapes: Video inputs `(B, T, 288, 288, 3)` in `[0,1]`; global embeddings `(B, D)`; video‑only tokens `(B, N, D)`.

## Plug‑and‑Play Multimodal RAG

- Normalize: L2‑normalize each model’s embeddings; compute cosine similarities.
- Late fusion (recommended):
  - RRF: Rank‑based, robust; sum 1/(k + rank) across models.
  - Weighted z‑sum: z‑score each model’s sims; sum with small weights.
- Align IDs: Chunk videos into windows (e.g., 1–8s). Keep shared window IDs and timecodes across models.
- Evidence packs:
  - For top‑K windows, assemble concise, time‑coded facts from: ASR (Whisper), OCR, object/action tags, audio events (CLAP/YAMNet), optional keyframe captions.
  - Keep each line short and factual; include timecodes.
- LLM prompt: Provide the question + the evidence pack (top‑K lines) + an instruction to answer strictly from evidence and cite timecodes.

### Minimal Fusion Recipe (NumPy)

```python
import numpy as np

# E[name]: (N, d_m) L2-normalized clip embeddings per model; shared window order
# q[name]: (d_m,) L2-normalized query embedding per model

def zscore(x):
    mu, sd = x.mean(), x.std() + 1e-8
    return (x - mu) / sd

def fuse_scores(E, q, weights=None, use_rrf=False, rrf_k=60, topk=20):
    sims = {name: E[name] @ q[name] for name in E.keys() if name in q}
    N = next(iter(E.values())).shape[0]
    if use_rrf:
        agg = np.zeros(N, dtype=np.float32)
        for s in sims.values():
            ranks = np.argsort(np.argsort(-s))  # 0 = best
            agg += 1.0 / (rrf_k + ranks)
        scores = agg
    else:
        weights = weights or {k: 1.0 for k in sims.keys()}
        scores = sum(weights[k] * zscore(sims[k]) for k in sims.keys())
    top = np.argpartition(-scores, topk)[:topk]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]
```

### Prompt Template (Evidence‑Grounded)

```
You are answering a question about a video using retrieved, time-coded evidence. 
Answer concisely and cite timecodes.

Question: {question}

Evidence:
{evidence_lines}

Rules:
- Base your answer only on the evidence above.
- If missing info, say you don’t know; suggest the closest timecodes.
```

## Preserving Spatiotemporal Detail

- Fine windows: Use 1–2s windows (possibly overlapping) to avoid averaging away moments.
- Token scoring: With VideoPrism tokens, compute zero‑shot similarities against a small label bank (objects/actions/attributes). Pool tokens into time bins, pick top labels per bin.
- Time‑coded context: Convert top labels into succinct lines like “`[12.0–13.5s] dog catches frisbee; crowd cheering`”.

### Token‑Level Concept Tagging (Sketch)

```python
# 1) Get video-only tokens (omit text args per the usage guide)
	okens = model.apply(state, video_inputs, train=False)  # (B, N, D)

# 2) Build a small label bank (objects/actions)
labels = ["a red frisbee", "a person jumping", "a dog running", "crowd cheering"]

# 3) Get text embeddings for labels (reuse your forward path)
text_ids, pads = vp.tokenize_texts(tokenizer, labels)
_, label_emb, _ = forward(video_inputs[:1], text_ids, pads)  # (L, D)
label_emb = label_emb / (np.linalg.norm(label_emb, axis=-1, keepdims=True) + 1e-8)

# 4) Cosine sim: tokens x labels
Tok = tokens / (np.linalg.norm(tokens, axis,-1, keepdims=True) + 1e-8)   # (B, N, D)
sims = Tok @ label_emb.T                                                 # (B, N, L)

# 5) Map token indices to time bins; take top labels per time
#    Emit lines like: "[12.0-14.0s] dog running; red frisbee"
```

## Adapters vs. LoRA (LLM Conditioning)

- LoRA: Efficiently fine‑tunes existing weights. It does not create a bridge from foreign embeddings by itself.
- Adapter/projector: A small network that maps external embeddings into the LLM’s hidden space, producing k “soft tokens” the LLM can attend to.
- Cross‑attention adapters: Feed features as a memory via added cross‑attn blocks (larger change, often better).
- Q‑Former/Resampler: Compress many tokens (e.g., spatiotemporal tokens) into a small set of latent tokens.

### Minimal Adapter Sketch (PyTorch‑like)

```python
class ModalityAdapter(nn.Module):
    def __init__(self, d_src, d_model, k=16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_src, 4*d_model), nn.GELU(), nn.Linear(4*d_model, k*d_model)
        )
        self.k, self.d_model = k, d_model

    def forward(self, x):            # x: (B, d_src) or pooled tokens
        soft = self.proj(x)          # (B, k*d_model)
        return soft.view(-1, self.k, self.d_model)  # (B, k, d_model)
```

- Training signal: Supervised next‑token loss on (video/audio/etc., question → answer). Freeze most of the LLM; train adapter (and optionally small LoRA ranks on the LLM).
- Multiple modalities: One adapter per modality (VP global or pooled tokens, CLIP, CLAP, OCR). Concatenate their soft tokens or gate/weight them.
- Token budget: Start with k=8–32 per modality; adjust for detail vs. cost.

## Suggested Pipeline

- Indexing:
  - Chunk each video into 1–4s windows with metadata (video_id, start, end).
  - Compute per‑window embeddings: VideoPrism (global; optionally tokens), CLIP on keyframes (average), CLAP/YAMNet for audio, OCR text.
  - L2‑normalize and store matrices per model + metadata.
- Query:
  - Embed the question with the matching text encoders (VP text, CLIP text, CLAP text).
  - Fuse scores (RRF or weighted z‑sum); select top‑K windows.
  - Build evidence: ASR lines for those windows; add top token‑concept tags and audio/vision events; keep it concise and time‑coded.
  - Prompt LLM with the question + evidence; instruct to cite timecodes and stay grounded.

## Practical Tips

- Calibration: Always L2‑normalize embeddings; z‑score or temperature‑scale per model before fusion. Learn 2–5 fusion weights on a small dev set (optimize nDCG@K).
- Re‑rankers: A small cross‑encoder or the LLM itself can re‑rank top‑50 evidence snippets for better precision.
- Hubness: On very large corpora, RRF or whitening/PCA can mitigate hubness in cosine spaces.
- Coverage: Use complementary models (vision, audio, OCR). If a model lacks a text encoder, skip it for text→embedding retrieval or train a projector.

## FAQ

- Can I tokenize embeddings and send them to the LLM?
  - No. Tokenizers map strings to learned token IDs; external floats don’t map meaningfully. Use RAG or train adapters.
- Is LoRA the bridge?
  - LoRA adapts the LLM efficiently; the projector/adapter is the actual bridge from external embedding spaces.
- What about VideoPrism captioning?
  - If the captioning model isn’t available, use VP embeddings/tokens for retrieval and concept tagging, and pair with ASR/OCR/keyframe captions from other models.

## Next Steps

- Start with multimodal RAG and evidence packs for immediate gains.
- If you need direct conditioning, add adapters per modality and lightly LoRA‑tune using Q&A/caption supervision.
