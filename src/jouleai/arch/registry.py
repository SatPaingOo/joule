"""ArchRegistry — config-driven model architecture specs.

One declarative spec covers the flag matrix across families; true outliers
(DeepSeek MLA attention, Gemma norm style) are flagged so the engine can
dispatch to the right kernel path.

Supported model_type values:
  qwen2, qwen3, llama, mistral, gemma, phi, gpt_oss   (dense)
  olmoe, qwen3_moe, mixtral, deepseek                 (MoE)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArchSpec:
    model_type: str
    n_layers: int
    d: int
    n_heads: int
    n_kv: int
    head_dim: int
    eps: float
    theta: float
    tied: bool
    vocab: int
    # attention flags
    bias_qkv: bool = False
    qk_norm: str = "none"          # none | per_head | whole
    clip_qkv: float | None = None
    rope_scaling: dict | None = None
    mla: bool = False              # DeepSeek MLA attention (latent KV)
    gemma_norm: bool = False       # Gemma norm style
    # ffn
    moe: bool = False
    intermediate: int = 0
    n_experts: int = 0
    top_k: int = 0
    norm_topk_prob: bool = False
    # expert tensor naming (Qwen-style vs Mixtral/DeepSeek block_sparse_moe)
    expert_naming: str = "qwen"    # qwen | block_sparse_moe
    # lm head
    separate_lm_head: bool = False

    def gqa_rep(self) -> int:
        return self.n_heads // self.n_kv


_DENSE = ("qwen2", "qwen3", "llama", "mistral", "gemma", "phi", "gpt_oss")
_MOE = ("olmoe", "qwen3_moe", "mixtral", "deepseek")
SUPPORTED = _DENSE + _MOE


def get_spec(cfg: dict) -> ArchSpec:
    mt = cfg.get("model_type", "?")
    if mt not in SUPPORTED:
        raise ValueError(
            f"unsupported architecture '{mt}'. Supported: {', '.join(SUPPORTED)}")
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    kv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    spec = ArchSpec(
        model_type=mt,
        n_layers=cfg["num_hidden_layers"],
        d=cfg["hidden_size"],
        n_heads=cfg["num_attention_heads"],
        n_kv=kv,
        head_dim=head_dim,
        eps=cfg.get("rms_norm_eps", cfg.get("layer_norm_eps", 1e-5)),
        theta=cfg.get("rope_theta", 1e4),
        tied=cfg.get("tie_word_embeddings", False),
        vocab=cfg["vocab_size"],
        bias_qkv=bool(cfg.get("attention_bias", mt == "qwen2")),
        qk_norm="none",
        clip_qkv=cfg.get("clip_qkv"),
        rope_scaling=cfg.get("rope_scaling") or (
            cfg.get("rope_parameters") if isinstance(cfg.get("rope_parameters"), dict)
            else None),
        mla=bool(cfg.get("q_lora_rank") or cfg.get("kv_lora_rank")),
        gemma_norm=(mt == "gemma"),
        moe=mt in _MOE,
        intermediate=cfg.get("moe_intermediate_size") or cfg.get("intermediate_size", 0),
        n_experts=cfg.get("num_experts", 0),
        top_k=cfg.get("num_experts_per_tok", 0),
        norm_topk_prob=cfg.get("norm_topk_prob", False),
        expert_naming=("block_sparse_moe" if mt in ("mixtral", "deepseek") else "qwen"),
    )
    if mt in ("qwen3", "qwen3_moe"):
        spec.qk_norm = "per_head"
    elif mt == "olmoe":
        spec.qk_norm = "whole"
    spec.separate_lm_head = not spec.tied
    return spec
