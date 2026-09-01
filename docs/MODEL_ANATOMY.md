# Model Anatomy — What's Inside & What Can Be Selectively Loaded
> **Context**: this is the model-structure analysis behind Joule's storage
> design. See [docs/USAGE.md](USAGE.md) (how to run), [docs/JOULE_PAPER.md](JOULE_PAPER.md)
> (the retrospective), [results/VALIDATION_LOG.md](../results/VALIDATION_LOG.md) (evidence).

---

> Date: 2026-08-29 | Measured on Qwen2.5-1.5B-Instruct (fp16, safetensors)
> This answers: "what's inside a downloaded model, and what can be
> selectively loaded piece by piece"

## 1. What's Inside a Downloaded Model

A model file is NOT one blob. It is a **named tensor database**. Every weight has an address:

```
model.embed_tokens.weight                    (151936, 1536)   token → vector table
model.layers.{0..27}.self_attn.q_proj.weight (1536, 1536)     attention: query
model.layers.{0..27}.self_attn.k_proj.weight (256, 1536)      attention: key (GQA)
model.layers.{0..27}.self_attn.v_proj.weight (256, 1536)      attention: value (GQA)
model.layers.{0..27}.self_attn.o_proj.weight (1536, 1536)     attention: output
model.layers.{0..27}.mlp.gate_proj.weight    (8960, 1536)     FFN: neuron keys (gate)
model.layers.{0..27}.mlp.up_proj.weight      (8960, 1536)     FFN: neuron content (up)
model.layers.{0..27}.mlp.down_proj.weight    (1536, 8960)     FFN: neuron values (down)
model.layers.{0..27}.input_layernorm.weight  (1536,)          norm (tiny)
model.layers.{0..27}.post_attention_layernorm.weight (1536,)  norm (tiny)
model.norm.weight                            (1536,)          final norm
```

## 2. Size Breakdown (Qwen2.5-1.5B = 1543M params = 2944 MB fp16)

| Component | Params | Size | % | Role | Load policy |
|---|---|---|---|---|---|
| embed_tokens | 233M | 445 MB | 15.1% | token↔vector | **always** (small) |
| Attention ×28 | 154M | 294 MB | 10.1% | token mixing | **always** (cheap) |
| FFN ×28 | 1157M | 2208 MB | **76.6%** | knowledge memory | **selective ← THE database** |
| norms | 0.1M | 0.2 MB | ~0% | scaling | always (negligible) |

**FFN is 77% of the model.** FFN = where knowledge lives (Geva et al. 2021: FFN layers are
key-value memories). Per layer: 8960 neurons. Neuron i =
- key: `gate_proj[i,:]`, `up_proj[i,:]` (what activates it)
- value: `down_proj[:,i]` (what it writes)

Total addressable neurons: 28 × 8960 = **250,880 rows** — a literal row-store.

## 3. Proof: Partial Loading Works at Row Granularity

safetensors is mmap-based → random access by tensor name AND by row:

| Operation | Data read | RAM delta | Time |
|---|---|---|---|
| Load full model | 2944 MB | ~3000 MB | ~3 s |
| Load ONE tensor (L14 down_proj) | 26 MB | 1 MB | **2.4 ms** |
| Load ONE neuron row | 28 KB | ~0 | **<0.1 ms** |

**The database analogy is not an analogy — the file format already supports it.**

Granularity ladder:
1. whole model → 2944 MB
2. per layer → ~105 MB
3. per component (attn/ffn) → 10.5 / 78.8 MB
4. **per neuron row → 28 KB** ← DB row-level access, zero training needed

## 4. Measured Neuron Activity (real forward pass, query: "What is the capital of France?")

SwiGLU activation magnitude per neuron, last token, threshold = 1% of layer max:

| Layer | Active neurons | Neurons carrying 90% of mass |
|---|---|---|
| 0 | **30.3%** | 52.8% |
| 4 | 61.2% | 55.1% |
| 8 | 69.5% | 57.3% |
| 12 | 63.5% | 55.7% |
| 16 | 52.1% | 53.8% |
| 20 | 69.5% | 54.4% |
| 24 | 52.7% | 56.5% |
| 27 | **11.7%** | 48.6% |
| **mean** | **55.7%** | **53.8%** |

Findings:
- **Boundary layers are extremely sparse** (L0: 30%, L27: 12%) — matches ShortGPT's BI finding
- Middle layers ~52-70% active
- ~54% of neurons carry 90% of activation mass
- Consistent with Deja Vu (2303.17101) contextual sparsity (~30-50%)

## 5. RAM Consequence

| Policy | FFN loaded | Total RAM |
|---|---|---|
| Full model | 100% | 2944 MB |
| Activity-threshold (55%) | 55% | ~1930 MB |
| Mass-based (40%) | 40% | ~2042 MB → with aggressive boundary-layer policy lower |
| Cache hit (no model load) | 0% | ~50 MB |

## 6. The Database Mapping

| Database concept | Model equivalent | Size |
|---|---|---|
| database file | model.safetensors | 2944 MB |
| table | tensor (e.g. L14 down_proj) | 26 MB |
| row | FFN neuron (gate/up/down slice) | 28 KB |
| index | sense-point router (τ_s threshold) | 0 MB (computed) |
| query | user prompt → hidden state | — |
| SELECT ... WHERE | load neurons with activation > τ_s | row-level |

## 7. Honest Physics / Caveats

1. **Just-in-time dependency**: which neurons fire depends on the hidden state produced by
   lower layers → selection must happen layer-by-layer during the forward pass
   (predict-then-load per layer), not once upfront.
2. **OS page cache**: mmap reads are fast because the OS caches pages. On machines where the
   model fits in RAM anyway, savings show up as *addressed* memory, not disk reads. The real
   win: models LARGER than RAM (e.g. 70B on 32GB), or pinned/locked pages.
3. **Predictor cost**: threshold-based selection needs the activation value first → either
   (a) cheap probe forward with gate/up only (down_proj is the big value part), or
   (b) per-layer lightweight router (our sense-point predictor, training-free).
4. **Quality gate**: skipped neurons' contribution must be verified — Theorem 1 verify pass
   or Top-k mass coverage ≥ 90% rule.

## 8. Next Step

Prototype `SenseWeightStore`: open safetensors by mmap, expose
`load_rows(layer, neuron_indices)` + `load_tensor(name)`, and a `LayerPredictor`
(training-free, activation-magnitude based) that decides per layer which neuron rows
to pull. Measure: RAM peak, tokens/s, output drift vs full model.
