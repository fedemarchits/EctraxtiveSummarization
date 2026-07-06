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
    def __init__(self, hf_id: str, gen: GenConfig, device_map=None, load_in_4bit: bool = False):
        self.hf_id = hf_id
        self.gen = gen
        # Placement: default single-GPU {"":0}. `device_map="auto"` (set per
        # model in models.yaml) shards across all visible GPUs — use for models
        # too big for one 3090 (e.g. gemma4_12b on 2x3090). Single-GPU avoids
        # the multimodal Gemma-4 meta-offload crash, so keep it as the default.
        self.device_map = device_map if device_map is not None else {"": 0}
        # 4-bit (NF4 + double-quant) shrinks a 12B to ~7GB so it fits ONE 3090,
        # avoiding the slow 2-GPU shard. Set `load_in_4bit: true` in models.yaml.
        self.load_in_4bit = load_in_4bit
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
            device_map=self.device_map,
        )
        if self.load_in_4bit:
            # Quantized: fits one GPU, so keep device_map single-GPU. compute
            # dtype stays bf16; do NOT also pass a top-level `dtype`.
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            model_kwargs["dtype"] = torch.bfloat16
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
# vLLM (fast local inference: fused quant kernels + continuous batching)
# --------------------------------------------------------------------------

class VLLMBackend:
    # Signals the runner to hand this backend EVERY prompt of a variant at once
    # (100 docs x 3 aspects) so continuous batching kicks in. HF backends leave
    # this False and get the per-doc path.
    wants_full_batch = True

    def __init__(self, hf_id: str, gen: GenConfig, quantization: Optional[str] = None,
                 tensor_parallel_size: int = 1, max_model_len: Optional[int] = None,
                 gpu_memory_utilization: float = 0.90, disable_reasoning: bool = True):
        self.hf_id = hf_id
        self.gen = gen
        self.quantization = quantization
        self.tp = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_mem = gpu_memory_utilization
        self.disable_reasoning = disable_reasoning
        self._llm = None
        self._tok = None
        self._sp = None

    def _ensure(self):
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.hf_id, trust_remote_code=True)
        kw = dict(
            model=self.hf_id, trust_remote_code=True,
            tensor_parallel_size=self.tp,
            gpu_memory_utilization=self.gpu_mem,
            dtype="bfloat16",
        )
        if self.max_model_len:
            kw["max_model_len"] = self.max_model_len
        if self.quantization:
            kw["quantization"] = self.quantization
            # in-flight bitsandbytes needs the matching load_format; it's also
            # single-GPU only (tp must be 1).
            if self.quantization == "bitsandbytes":
                kw["load_format"] = "bitsandbytes"
        self._llm = LLM(**kw)
        # temperature 0 -> greedy/deterministic (seed then irrelevant).
        self._sp = SamplingParams(
            temperature=self.gen.temperature,
            max_tokens=self.gen.max_new_tokens,
            seed=self.gen.seed if self.gen.temperature > 0 else None,
        )

    def generate_batch(self, prompts: List[str], system: Optional[str] = None) -> List[str]:
        self._ensure()
        tok = self._tok
        tmpl_kwargs = {}
        if "qwen" in self.hf_id.lower() and self.disable_reasoning:
            tmpl_kwargs["enable_thinking"] = False
        chats = []
        for p in prompts:
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": p}
            ]
            chats.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, **tmpl_kwargs))
        # vLLM returns RequestOutputs in input order.
        outs = self._llm.generate(chats, self._sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def free_memory(self) -> None:  # vLLM manages its own KV cache
        pass


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
        return LocalHFBackend(m["hf_id"], gen, device_map=m.get("device_map"),
                              load_in_4bit=bool(m.get("load_in_4bit", False)))
    if kind == "vllm":
        return VLLMBackend(
            m["hf_id"], gen,
            quantization=m.get("quantization"),
            tensor_parallel_size=int(m.get("tensor_parallel_size", 1)),
            max_model_len=m.get("max_model_len"),
            gpu_memory_utilization=float(m.get("gpu_memory_utilization", 0.90)),
        )
    if kind == "endpoint":
        return EndpointBackend(m.get("endpoint_model", m["hf_id"]), gen)
    raise ValueError(f"unknown backend '{kind}' for model {alias}")
