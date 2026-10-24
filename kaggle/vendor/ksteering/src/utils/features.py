# Vai trò file: Trích hidden state theo layer thành đặc trưng NumPy cho classifier và CAA.
from __future__ import annotations

from typing import List

import numpy as np
import torch


# Tokenize văn bản theo batch và lấy hidden state tại layer_idx từ backbone model.
# Dùng attention_mask.sum()-1 để chọn token, trả ma trận NumPy [số mẫu, hidden_dim]; không chạy LM head hay tạo KV cache.
def get_hidden_cached(
    texts: List[str],
    *,
    tokenizer,
    model,
    layer_idx: int,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    all_vectors: list[np.ndarray] = []
    # Avoid unnecessary LM-head logits and KV cache allocation:
    # we only need hidden states from the transformer backbone.
    #
    # For CausalLM wrappers (Qwen/Llama/etc.), using `model.model` directly
    # reliably bypasses lm_head. `get_base_model()` may still return a wrapper
    # in some model classes/versions.
    backbone = getattr(model, "model", None)
    if backbone is None:
        backbone = model.get_base_model() if hasattr(model, "get_base_model") else model
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.inference_mode():
            outputs = backbone(
                **tokenized,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden = outputs.hidden_states[layer_idx]
        mask = tokenized["attention_mask"]
        lengths = mask.sum(dim=1) - 1

        for i, idx in enumerate(lengths):
            all_vectors.append(hidden[i, int(idx), :].detach().cpu().float().numpy())

    return np.stack(all_vectors, axis=0)

