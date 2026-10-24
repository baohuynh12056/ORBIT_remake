from __future__ import annotations
import importlib.metadata
import os
import shutil
import time
from pathlib import Path
from .config import MODELS, JUDGE
from .storage import write_json, resources


def check_access(c):
    """Small metadata requests only; authorize gated weights before from_pretrained."""
    from huggingface_hub import HfApi, hf_hub_url, get_hf_file_metadata
    token = os.getenv("HF_TOKEN") or None
    api = HfApi(token=token)
    result = {}
    for model_id in MODELS + [JUDGE]:
        rev = c["revisions"][model_id]
        info = api.model_info(model_id, revision=rev, files_metadata=True)
        if info.id != model_id or info.sha != rev:
            raise ValueError(f"Model identity mismatch: {model_id}")
        files = [s.rfilename for s in info.siblings]
        weight = next((f for f in files if f.endswith(".safetensors")), None)
        if not weight:
            raise ValueError(f"No safetensors weights: {model_id}")
        # HEAD the config AND one gated weight; a public model card alone is insufficient.
        for filename in ("config.json", weight):
            get_hf_file_metadata(hf_hub_url(model_id, filename, revision=rev), token=token)
        result[model_id] = {"revision": info.sha, "gated": info.gated, "weight_head": "ok",
                             "weights_bytes": sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors"))}
    return result


def preflight(c, root, require_gpu=True):
    import importlib.util
    import torch
    import transformers
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    info = {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu_checks_executed": False, "resources": resources(), "gpu": [],
            "ram": {},"openai_sdk_installed":importlib.util.find_spec("openai") is not None,
            "inference_backend":"in_process_local_transformers_no_api"}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal:", "MemAvailable:")):
            k,v,*_ = line.split(); info["ram"][k.rstrip(":")] = int(v) * 1024
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        info["gpu"].append({"id": i, "name": p.name, "vram": total, "free": free,
                             "capability": list(torch.cuda.get_device_capability(i))})
    info["packages"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    write_json(root / "preflight.json", info)
    if require_gpu and (torch.cuda.device_count() != 2 or not torch.cuda.is_available()):
        raise RuntimeError("Default workflow requires two visible CUDA GPUs. CPU checks are separate; select GPU T4 x2 in Kaggle.")
    if require_gpu:
        if any("T4" not in g["name"] for g in info["gpu"]):
            raise RuntimeError("This protocol targets 2xT4; actual GPU names are in preflight.json. Select T4 x2 before comparing results.")
        for i in range(2):
            a = torch.randn(32,32,device=f"cuda:{i}",dtype=torch.float16)
            if not torch.isfinite(a @ a).all():
                raise RuntimeError("FP16 CUDA matmul failed")
        info["gpu_checks_executed"] = True
    info["access"] = check_access(c)
    write_json(root / "preflight.json", info)
    return info


class LocalModel:
    def __init__(self, c, judge=False):
        import torch
        from transformers import AutoConfig, AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
        self.c, self.is_judge = c, judge
        self.model_id = JUDGE if judge else c["target_model_id"]
        self.revision = c["revisions"][self.model_id]
        self.precision = c["judge"]["precision"] if judge else c["precision"]
        if not torch.cuda.is_available():
            raise RuntimeError("Real local LLM stages require CUDA. No substitute model is allowed.")
        # Verify exact ID/revision and gated weight access on every fresh worker.
        from huggingface_hub import HfApi, hf_hub_url, get_hf_file_metadata
        api = HfApi()
        info = api.model_info(self.model_id, revision=self.revision)
        weight = next(s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors"))
        get_hf_file_metadata(hf_hub_url(self.model_id, weight, revision=self.revision), token=os.getenv("HF_TOKEN"))
        if info.sha != self.revision or info.id != self.model_id:
            raise ValueError("Unexpected model revision")
        config = AutoConfig.from_pretrained(self.model_id, revision=self.revision, trust_remote_code=False)
        memory = {}
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            limit = min(free, total) - int(c["hardware"]["reserve_gib_per_gpu"] * 2**30)
            if limit < 2**30:
                raise RuntimeError(f"Not enough free memory on GPU {i}")
            memory[i] = int(limit)
        # A mapping of independent GPU budgets; these are never summed into a fictitious GPU.
        kwargs = dict(revision=self.revision, dtype=torch.float16, device_map="balanced",
                      max_memory=memory, attn_implementation="eager", trust_remote_code=False)
        if self.precision == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        elif self.precision == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        if judge:
            if config.model_type != "qwen3_5" or "Qwen3_5ForConditionalGeneration" not in config.architectures:
                raise RuntimeError(f"Unsupported judge architecture: {config.model_type} {config.architectures}")
            self.processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision, trust_remote_code=False)
            self.tokenizer = self.processor.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)
        else:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, trust_remote_code=False)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        mapping = getattr(self.model, "hf_device_map", {})
        if any(str(v) in ("cpu", "disk") for v in mapping.values()):
            raise RuntimeError(f"Model did not fit configured GPU budgets: {mapping}. Choose an explicit low-memory profile in a NEW run.")
        self.model.eval().requires_grad_(False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not self.tokenizer.chat_template:
            raise RuntimeError("Exact model chat template required; no guessed template fallback")
        for k in ("temperature", "top_p", "top_k"):
            setattr(self.model.generation_config, k, None)
        self.device = self.model.get_input_embeddings().weight.device

    def generate(self, prompt, seed, max_new_tokens, sampling=False, system=None):
        import torch
        from transformers import set_seed
        set_seed(seed)
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role":"user", "content":prompt}]
        if self.is_judge:
            inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                     enable_thinking=self.c["judge"]["thinking"], return_dict=True, return_tensors="pt")
        else:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=False)
        inputs = inputs.to(self.device)
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=sampling, use_cache=True,
                      pad_token_id=self.tokenizer.pad_token_id)
        if sampling:
            kwargs.update(temperature=self.c["build"]["temperature"], top_p=self.c["build"]["top_p"], top_k=0)
        t = time.monotonic()
        # no_grad, not inference_mode: K-Steer hooks re-enable autograd on hidden states.
        with torch.no_grad():
            output = self.model.generate(**inputs, **kwargs)
        ids = output[0, inputs["input_ids"].shape[1]:]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        return {"text": self.tokenizer.decode(ids, skip_special_tokens=True), "generated_tokens": len(ids),
                "truncated": len(ids) >= max_new_tokens and (not len(ids) or int(ids[-1]) not in eos),
                "prompt_tokens": inputs["input_ids"].shape[1], "elapsed_s": time.monotonic() - t,
                "seed": seed, "model_id": self.model_id, "revision": self.revision, "precision": self.precision,
                "resources": resources(), "effective_batch_size": 1}


def retry_oom_batches(items, batch_size, fn, store):
    """Reduce batch only. Never change token budget, precision or sample count."""
    import gc
    import torch
    index, size = 0, batch_size
    while index < len(items):
        part = items[index:index+size]
        try:
            result = fn(part)
        except torch.cuda.OutOfMemoryError:
            if size == 1:
                store.log("oom_unrecoverable", batch_size=1)
                raise
            old, size = size, max(1, size // 2)
            gc.collect(); torch.cuda.empty_cache()
            store.log("oom_reduce_batch", previous=old, effective=size)
            continue
        yield part, result
        index += len(part)
