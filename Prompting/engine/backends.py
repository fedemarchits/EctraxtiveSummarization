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
    def __init__(self, hf_id: str, gen: GenConfig):
        self.hf_id = hf_id
        self.gen = gen
        self._model = None
        self._tok = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.hf_id, trust_remote_code=True)
        # Pin the whole model to GPU 0. `device_map="auto"` can silently offload
        # parts to CPU/meta (esp. multimodal Gemma-4), which then crashes at
        # generate with "Tensor on device meta". These local models fit on one
        # 3090, so force single-GPU placement.
        model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        self._model, self._tok = model.eval(), tok

    def generate_batch(self, prompts: List[str], system: Optional[str] = None) -> List[str]:
        import torch

        self._ensure()
        tok, model = self._tok, self._model
        chats = []
        for p in prompts:
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": p}
            ]
            chats.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))

        # Model is pinned to GPU 0; place inputs there explicitly (model.device
        # is unreliable with a device_map).
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
        texts = []
        for row, plen in zip(out, plens):
            texts.append(tok.decode(row[int(plen):], skip_special_tokens=True))
        return texts


# --------------------------------------------------------------------------
# OpenAI-compatible endpoint
# --------------------------------------------------------------------------

class EndpointBackend:
    def __init__(self, model_id: str, gen: GenConfig):
        self.model_id = model_id
        self.gen = gen
        self._client = None

    def _ensure(self):
        if self._client is not None:
            return
        import os
        from openai import OpenAI
        self._client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def generate_batch(self, prompts: List[str], system: Optional[str] = None) -> List[str]:
        self._ensure()
        out = []
        for p in prompts:
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": p}
            ]
            resp = self._client.chat.completions.create(
                model=self.model_id,
                messages=msgs,
                max_tokens=self.gen.max_new_tokens,
                temperature=self.gen.temperature,
                seed=self.gen.seed,
            )
            out.append(resp.choices[0].message.content or "")
        return out


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
        return LocalHFBackend(m["hf_id"], gen)
    if kind == "endpoint":
        return EndpointBackend(m.get("endpoint_model", m["hf_id"]), gen)
    raise ValueError(f"unknown backend '{kind}' for model {alias}")
