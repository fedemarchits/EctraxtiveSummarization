"""Generation backends behind one interface.

    backend.generate_batch(prompts, system=None) -> list[str]

`local`    -> HuggingFace transformers on GPU (small/medium tiers).
`endpoint` -> OpenAI-compatible chat API (large tiers), from env:
                OPENAI_BASE_URL, OPENAI_API_KEY.

get_backend(alias, models_yaml) reads models.yaml, finds the model by alias,
and builds the backend named in its `backend:` field. Heavy deps (torch,
transformers, openai) import lazily so this module loads without them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class GenConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    seed: int = 42


# --------------------------------------------------------------------------
# Local HuggingFace
# --------------------------------------------------------------------------

class LocalHFBackend:
    def __init__(self, hf_id: str, gen: GenConfig, device_map=None):
        self.hf_id = hf_id
        self.gen = gen
        # Placement: default single-GPU {"":0}. `device_map="auto"` (set per
        # model in models.yaml) shards across all visible GPUs — use for models
        # too big for one 3090 (e.g. gemma4_12b on 2x3090). Single-GPU avoids
        # the multimodal Gemma-4 meta-offload crash, so keep it as the default.
        self.device_map = device_map if device_map is not None else {"": 0}
        self._model = None
        self._tok = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.hf_id, trust_remote_code=True)
        model_kwargs = dict(
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=self.device_map,
        )
        # When sharding ("auto"), cap memory to the GPUs only so nothing spills
        # to CPU/disk (which reintroduces the meta-device crash).
        if self.device_map == "auto":
            n = torch.cuda.device_count()
            model_kwargs["max_memory"] = {i: "23GiB" for i in range(n)}
        model = AutoModelForCausalLM.from_pretrained(self.hf_id, **model_kwargs)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        self._model, self._tok = model.eval(), tok

    def _gen_chats(self, chats: List[str]) -> List[str]:
        import torch
        tok, model = self._tok, self._model
        enc = tok(chats, return_tensors="pt", padding=True).to("cuda:0")
        plens = enc["attention_mask"].sum(dim=1)
        do_sample = self.gen.temperature > 0
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=self.gen.max_new_tokens,
                do_sample=do_sample,
                temperature=self.gen.temperature if do_sample else None,
                use_cache=True,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        texts = [tok.decode(row[int(pl):], skip_special_tokens=True)
                 for row, pl in zip(out, plens)]
        del enc, out
        return texts

    def generate_batch(self, prompts: List[str], system: Optional[str] = None) -> List[str]:
        import torch

        self._ensure()
        tok = self._tok
        # Qwen3.x is a hybrid reasoning model: with thinking on it can spend the
        # whole budget in <think> and emit no/short JSON. This task wants a direct
        # answer, so disable thinking. The kwarg is ignored by templates that
        # don't use it (e.g. Gemma), so it's safe to pass unconditionally.
        tmpl_kwargs = {}
        if "qwen" in self.hf_id.lower():
            tmpl_kwargs["enable_thinking"] = False
        chats = []
        for p in prompts:
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": p}
            ]
            chats.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, **tmpl_kwargs))

        # Try the whole batch; on OOM (tight-fit models + long prompts) fall back
        # to one prompt at a time, clearing the cache between, so a peak spike on
        # one long variant doesn't kill the run.
        try:
            return self._gen_chats(chats)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            texts: List[str] = []
            for c in chats:
                texts.extend(self._gen_chats([c]))
                torch.cuda.empty_cache()
            return texts

    def free_memory(self) -> None:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# OpenAI-compatible endpoint
# --------------------------------------------------------------------------

class EndpointBackend:
    def __init__(self, model_id: str, gen: GenConfig, disable_reasoning: bool = True):
        self.model_id = model_id
        self.gen = gen
        # Reasoning models (Qwen3.5) spend the token budget in a hidden reasoning
        # channel and return EMPTY content at small max_tokens. This task wants a
        # direct answer, so disable thinking by default. Harmless for non-reasoning
        # models (Gemma) — OpenRouter ignores the flag.
        self.disable_reasoning = disable_reasoning
        self._client = None

    def _ensure(self):
        if self._client is not None:
            return
        import os
        from openai import OpenAI
        # Bound each request so a stalled/queued endpoint fails fast instead of
        # hanging for hours (large MoE endpoints are often throttled).
        self._client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=float(os.environ.get("OPENROUTER_TIMEOUT", "120")),
            max_retries=2,
        )

    def generate_batch(self, prompts: List[str], system: Optional[str] = None) -> List[str]:
        self._ensure()
        out = []
        for p in prompts:
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": p}
            ]
            extra = {"reasoning": {"enabled": False}} if self.disable_reasoning else None
            resp = self._client.chat.completions.create(
                model=self.model_id,
                messages=msgs,
                max_tokens=self.gen.max_new_tokens,
                temperature=self.gen.temperature,
                seed=self.gen.seed,
                extra_body=extra,
            )
            out.append(resp.choices[0].message.content or "")
        return out

    def free_memory(self) -> None:  # no-op; nothing local to free
        pass


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

def _find_model(alias: str, models_yaml: str) -> Dict:
    cfg = yaml.safe_load(open(models_yaml))
    for m in cfg["models"]:
        if m["alias"] == alias:
            return m
    raise KeyError(f"model alias not found in {models_yaml}: {alias}")


def get_backend(alias: str, models_yaml: str, gen: Optional[GenConfig] = None):
    m = _find_model(alias, models_yaml)
    gen = gen or GenConfig()
    kind = m.get("backend", "local")
    if kind == "local":
        return LocalHFBackend(m["hf_id"], gen, device_map=m.get("device_map"))
    if kind == "endpoint":
        return EndpointBackend(m.get("endpoint_model", m["hf_id"]), gen)
    raise ValueError(f"unknown backend '{kind}' for model {alias}")
