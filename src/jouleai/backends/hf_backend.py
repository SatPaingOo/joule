"""HuggingFace backend — the only place that touches `transformers` directly.

Single Responsibility: wrap model/tokenizer mechanics (load, tokenize,
generate, forward, embed). Everything else in the package depends on the
`Backend` protocol (interfaces.py), never on transformers.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..interfaces import Backend


class HuggingFaceBackend:
    def __init__(self, model_path: str, dtype: str = "auto",
                 exact_verify: bool = False):
        self.model_path = model_path
        self.exact_verify = exact_verify
        resolved = self._resolve_dtype(dtype)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=resolved,
            device_map="cuda" if self.device == "cuda" else "cpu",
            low_cpu_mem_usage=True)
        self.model.eval()

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype:
        if dtype == "auto":
            # bf16: native on GPU + AVX512-BF16 CPUs (measured 9x batched
            # forwards vs fp16, Entry 11)
            return torch.bfloat16
        return getattr(torch, dtype)

    # ---------------------------------------------------------- Backend api
    def chat_ids(self, question: str) -> torch.Tensor:
        messages = [{"role": "user", "content": question}]
        try:
            text = self.tok.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)
        except Exception:
            text = question
        return self.tok(text, return_tensors="pt").input_ids.to(
            self.model.device)

    def generate(self, prompt_ids: torch.Tensor, max_new: int) -> torch.Tensor:
        out = self.model.generate(
            prompt_ids.to(self.model.device), max_new_tokens=max_new,
            do_sample=False, pad_token_id=self.tok.eos_token_id)
        return out[0][prompt_ids.size(1):].unsqueeze(0)

    def full_decode(self, question: str, max_new: int = 64):
        """Greedy decode from a chat-formatted question.
        Returns (text, new_token_ids [1, N])."""
        ids = self.chat_ids(question)
        out = self.model.generate(
            ids, max_new_tokens=max_new, do_sample=False,
            pad_token_id=self.tok.eos_token_id)
        new_ids = out[0][ids.size(1):].unsqueeze(0)
        text = self.tok.decode(new_ids[0], skip_special_tokens=True)
        return text, new_ids

    def decode_tokens(self, ids: torch.Tensor) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model(ids.to(self.model.device)).logits

    def forward_with_hidden(self, ids: torch.Tensor):
        """Full logits + all hidden states (M(x) readiness signal needs
        layers 1→2 saturation)."""
        return self.model(ids.to(self.model.device), output_hidden_states=True)

    def embed(self, question: str) -> torch.Tensor:
        ids = self.chat_ids(question)
        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True)
        v = out.hidden_states[-1][0][-1].float()   # last-token state (Entry 6 fix)
        v = v - v.mean()
        return v / (v.norm() + 1e-9)

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        self._device = value
