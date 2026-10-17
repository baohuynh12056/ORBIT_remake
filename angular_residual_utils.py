# Vai trò file: Thu activation contrastive tại các hook site và tính các hướng, mặt phẳng, thống kê dùng xây vector ORBIT.
"""Activation extraction utilities for ORBIT vector construction."""

from __future__ import annotations

import contextlib
import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

HookKey = Tuple[int, str]

# Mã phiên bản riêng cho mỗi layout extraction, để vector và hook generation dùng cùng vị trí activation.
ORBIT_EXTRACTION_LAYOUT_VERSIONS = {
    "norm": "block_input_and_post_attention_norm_v2",
    "residual": "raw_resid_mid_and_resid_post_v2",
}
# Phiên bản layout norm mặc định, được lưu và kiểm tra trong metadata vector.
ORBIT_EXTRACTION_LAYOUT_VERSION = ORBIT_EXTRACTION_LAYOUT_VERSIONS["norm"]


# Chuẩn hóa tên layout trích activation và kiểm tra các layout được hỗ trợ.
# Giữ cùng quy ước vị trí hook giữa bước xây vector và bước can thiệp.
def normalize_orbit_extraction_layout(layout: str) -> str:
    value = str(layout or "norm").strip().lower().replace("-", "_")
    aliases = {
        "norm": "norm",
        "normalized": "norm",
        "normalization": "norm",
        "norm_output": "norm",
        "residual": "residual",
        "raw": "residual",
        "raw_residual": "residual",
        "resid_mid_post": "residual",
    }
    if value not in aliases:
        raise ValueError(
            "ORBIT extraction layout must be 'norm' or 'residual'; "
            f"got {layout!r}."
        )
    return aliases[value]


# Trả mã phiên bản của layout trích activation đã chuẩn hóa.
# Mã được lưu trong metadata để phát hiện vector tạo bằng vị trí hook khác.
def orbit_extraction_layout_version(layout: str) -> str:
    return ORBIT_EXTRACTION_LAYOUT_VERSIONS[
        normalize_orbit_extraction_layout(layout)
    ]


# Mô tả một vị trí hook gồm layer, vị trí, module và kiểu can thiệp; dùng liên kết activation với metadata.
@dataclass
class ResidualHook:
    layer_idx: int
    position: str  # "mid" or "post"
    module_name: str
    module: nn.Module
    hook_type: str = "output"  # "output" or "pre"

    # Trả khóa (layer_idx, position) nhận diện một vị trí trích activation.
    # Khóa này liên kết hook với thống kê positive/negative của cùng vị trí.
    @property
    def key(self) -> HookKey:
        return (self.layer_idx, self.position)


# Tìm thuộc tính module đầu tiên tồn tại trong danh sách tên ứng viên.
# Trả tên và module tìm được, hoặc cặp None khi kiến trúc không có tên phù hợp.
def _find_module_attr(
    module: nn.Module, candidates: Iterable[str]
) -> Tuple[Optional[str], Optional[nn.Module]]:
    for name in candidates:
        if hasattr(module, name):
            return name, getattr(module, name)
    return None, None


# Tìm danh sách decoder layers theo các cấu trúc model được hỗ trợ.
# Trả đường dẫn và module chứa layers để khám phá hook mà không gắn cứng một họ model.
def _resolve_layers_module(model: nn.Module) -> Tuple[str, nn.Module]:
    candidates = [
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
    ]
    for path in candidates:
        cur = model
        for attr in path.split("."):
            if not hasattr(cur, attr):
                break
            cur = getattr(cur, attr)
        else:
            if isinstance(cur, (nn.ModuleList, list, tuple)):
                return path, cur
    raise ValueError("Could not locate transformer block list on model")


# Khám phá hai vị trí hook riêng biệt trong mỗi decoder block theo extraction_layout.
# Với layout norm, chọn các normalization site phù hợp kiến trúc; báo lỗi nếu không đủ vị trí hợp lệ.
def discover_residual_hooks(
    model: nn.Module,
    extraction_layout: str = "norm",
) -> List[ResidualHook]:
    """Return two configured intervention sites for every decoder block."""
    extraction_layout = normalize_orbit_extraction_layout(extraction_layout)
    layer_path, layers = _resolve_layers_module(model)
    hooks: List[ResidualHook] = []

    input_candidates = [
        "input_layernorm",
        "input_layer_norm",
        "pre_attention_layer_norm",
        "pre_layernorm",
        "pre_lnorm",
        "ln_1",
    ]
    post_attn_candidates = [
        "post_attention_layernorm",
        "post_attention_layer_norm",
        "post_attention_ln",
        "post_layernorm",
        "post_lnorm",
        "post_attention_layernorm_1",
        "ln_2",
    ]
    post_ff_candidates = [
        "post_feedforward_layernorm",
        "post_feedforward_layer_norm",
        "post_ff_layernorm",
        "post_ffn_layernorm",
        "ffn_layernorm",
    ]

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        mid_attr, mid_module = _find_module_attr(layer, post_attn_candidates)
        if mid_module is None:
            raise ValueError(
                f"Unable to find post-attention layernorm on layer {layer_idx}."
            )
        hooks.append(
            ResidualHook(
                layer_idx=layer_idx,
                position="mid",
                module_name=f"{layer_path}.{layer_idx}.{mid_attr}",
                module=mid_module,
                # Raw resid_mid is the tensor entering the MLP norm. The norm
                # layout instead captures that module's normalized output.
                hook_type=("pre" if extraction_layout == "residual" else "output"),
            )
        )

        if extraction_layout == "residual":
            # Raw resid_post is the first tensor returned by the entire block.
            hooks.append(
                ResidualHook(
                    layer_idx=layer_idx,
                    position="post",
                    module_name=f"{layer_path}.{layer_idx}",
                    module=layer,
                    hook_type="output",
                )
            )
            continue

        # Associate the attention-input norm with its own block.
        post_attr, post_module = _find_module_attr(layer, input_candidates)
        post_module_name = (
            f"{layer_path}.{layer_idx}.{post_attr}"
            if post_module is not None
            else None
        )

        # OLMo-2 is post-norm and has no attention-input norm. Its second
        # normalized intervention site is the current block's FF branch norm.
        if post_module is None:
            post_attr, post_module = _find_module_attr(layer, post_ff_candidates)
            if post_module is not None:
                post_module_name = f"{layer_path}.{layer_idx}.{post_attr}"

        if post_module is not None:
            hooks.append(
                ResidualHook(
                    layer_idx=layer_idx,
                    position="post",
                    module_name=post_module_name,
                    module=post_module,
                )
            )
        else:
            child_names = [n for n, _ in layer.named_children()]
            raise ValueError(
                f"Unable to find a second normalization hook on layer {layer_idx}. "
                f"Tried current-layer input norms {input_candidates} and current-layer "
                f"post-FF norms {post_ff_candidates}. Layer children: {child_names}"
            )

    expected = 2 * len(layers)
    if len(hooks) != expected or len({id(h.module) for h in hooks}) != expected:
        raise ValueError(
            "ORBIT requires two distinct hook modules per decoder "
            f"block; discovered {len(hooks)} hooks ({len({id(h.module) for h in hooks})} "
            f"distinct) for {len(layers)} blocks in layout={extraction_layout!r}."
        )
    return hooks


# Đếm số token của văn bản mà không thêm special token.
# Dùng xác định ranh giới giữa instruction, câu hỏi và response.
def _token_length(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


# Tìm độ dài hậu tố token chung giữa các chuỗi token.
# Giúp nhận diện token kết thúc do chat template thêm vào.
def _common_suffix_len(token_lists: List[List[int]]) -> int:
    """Length of the common suffix across token lists (ignores padding)."""
    if not token_lists:
        return 0
    # Use the shortest list length as cap.
    min_len = min(len(toks) for toks in token_lists)
    if min_len == 0:
        return 0
    suffix_len = 0
    for i in range(1, min_len + 1):
        tok = token_lists[0][-i]
        if all(toks[-i] == tok for toks in token_lists[1:]):
            suffix_len += 1
        else:
            break
    return suffix_len


# Tìm chỉ số token cuối không phải padding cho mỗi hàng attention_mask.
# Cách quét từ cuối hỗ trợ cả padding trái và padding phải.
def _last_nonpad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the last non-pad index for each row in attention_mask."""
    seq_len = attention_mask.shape[1]
    last_from_end = torch.flip(attention_mask, dims=[1]).to(torch.long).argmax(dim=1)
    return seq_len - 1 - last_from_end


# Tính khoảng chỉ số [start, end] gồm hai đầu của response trong batch đã padding.
# Dựa vào độ dài prompt/response và phía padding khi token của hai phần được nối riêng.
def _response_spans(
    prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
    seq_len: int,
    *,
    padding_side: str = "left",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute inclusive [start, end] spans for response tokens in a padded batch.
    Assumes prompt_ids and response_ids were concatenated per sample.
    """
    padding_side = (padding_side or "left").lower()
    if padding_side not in {"left", "right"}:
        raise ValueError(f"padding_side must be 'left' or 'right'. Got: {padding_side}")

    if padding_side == "left":
        pad_len = seq_len - (prompt_lens + response_lens)
        resp_start = pad_len + prompt_lens
    else:
        resp_start = prompt_lens
    resp_end = resp_start + response_lens - 1
    return resp_start, resp_end


# Xác định token response từ offset ký tự khi tokenize prompt+response cùng nhau.
# Báo lỗi nếu có response nhưng không tìm được token, chẳng hạn response bị truncation.
def _joint_response_span_from_offsets(
    offsets: torch.Tensor,
    boundary: int,
    *,
    has_response: bool,
) -> Tuple[int, int]:
    """Locate response tokens in one jointly tokenized prompt+response row."""
    response_token_indices = [
        i
        for i, (start_char, end_char) in enumerate(offsets.tolist())
        if int(end_char) > int(boundary) and int(end_char) > int(start_char)
    ]
    if has_response and not response_token_indices:
        raise ValueError(
            "Tokenizer offsets did not identify response tokens; the response "
            "may have been truncated."
        )
    if not response_token_indices:
        return (0, -1)
    return response_token_indices[0], response_token_indices[-1]


# Đổi khoảng token response thành khoảng activation cần lấy theo token_mode.
# Chế độ causal dịch lùi một vị trí vì hidden state trước token là trạng thái dự đoán token đó.
def _response_activation_span(
    response_start: int,
    response_end: int,
    token_mode: str,
) -> Tuple[int, int]:
    """Map an inclusive answer-token span to an activation span.

    ``response_*`` modes inspect answer-token representations. The causal
    modes inspect the immediately preceding states whose logits predict those
    answer tokens, matching answer-causal intervention during MC scoring.
    """
    start, end = int(response_start), int(response_end)
    if end < start:
        return start, end
    if token_mode == "response_last":
        return end, end
    if token_mode == "response_mean":
        return start, end
    if token_mode == "response_causal_last":
        return end - 1, end - 1
    if token_mode == "response_causal_mean":
        return start - 1, end - 1
    raise ValueError(f"unsupported response token mode: {token_mode!r}")


# Định dạng câu hỏi theo chat template của tokenizer, có nhánh dự phòng theo họ model.
# Trả về prompt đã định dạng và cờ cho biết đã có special token, tránh thêm token hai lần.
def _format_prompts_for_model(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    system_prompts: Optional[List[str]] = None,
) -> Tuple[List[str], bool]:
    """Return formatted prompts and whether they already include special/chat tokens."""
    if system_prompts is None:
        resolved_systems = [""] * len(prompts)
    else:
        if len(system_prompts) != len(prompts):
            raise ValueError("system_prompts length must match prompts length")
        resolved_systems = [str(value or "").strip() for value in system_prompts]

    if getattr(tokenizer, "chat_template", None):
        out = []
        for p, system_prompt in zip(prompts, resolved_systems):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": p})
            out.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return out, True
    name = str(getattr(tokenizer, "name_or_path", "")).lower()
    if "qwen" in name:
        return [
            (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
            )
            if system_prompt
            else f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
            for p, system_prompt in zip(prompts, resolved_systems)
        ], True
    if "llama" in name and "instruct" in name:
        return (
            [
                (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )
                if system_prompt
                else (
                    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )
                for p, system_prompt in zip(prompts, resolved_systems)
            ],
            True,
        )
    if "gemma" in name and "it" in name:
        return (
            [
                (
                    "<bos><start_of_turn>user\n"
                    f"{system_prompt}\n\n{p}<end_of_turn>\n<start_of_turn>model\n"
                )
                if system_prompt
                else (
                    "<bos><start_of_turn>user\n"
                    f"{p}<end_of_turn>\n<start_of_turn>model\n"
                )
                for p, system_prompt in zip(prompts, resolved_systems)
            ],
            True,
        )
    # Preserve the behavioral system instruction for models without chat roles.
    return [
        (
            f"System instruction:\n{system_prompt}\n\n"
            f"User question:\n{prompt}\n\nAssistant answer:\n"
        )
        if system_prompt
        else prompt
        for prompt, system_prompt in zip(prompts, resolved_systems)
    ], False


# Tokenize instruction theo định dạng prompt dành cho angular extraction.
# Dùng cùng quy ước special token với các bước xác định vị trí activation.
def _tokenize_instructions_angular(
    tokenizer: AutoTokenizer, instructions: List[str]
) -> Dict[str, torch.Tensor]:
    """Match angular-steering repo tokenization for instructions."""
    if getattr(tokenizer, "chat_template", None):
        convos = [
            [{"role": "user", "content": instruction}] for instruction in instructions
        ]
        # Mirror angular-steering: first build strings with chat_template, then tokenize.
        rendered = tokenizer.apply_chat_template(
            convos,
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(rendered, return_tensors="pt", padding=True, truncation=False)
    return tokenizer(instructions, padding=True, truncation=False, return_tensors="pt")


# Đo số token hậu tố do chat template thêm sau nội dung user.
# Giúp tìm đúng token nội dung thay vì chọn nhầm token điều khiển của template.
def _template_suffix_len(tokenizer: AutoTokenizer) -> int:
    """Compute template suffix length using two dummy instructions (angular-steering)."""
    prompts = ["a", "b"]
    prompts_fmt, prompts_have_special = _format_prompts_for_model(tokenizer, prompts)
    enc = tokenizer(
        prompts_fmt,
        add_special_tokens=not prompts_have_special,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask")
    token_lists: List[List[int]] = []
    if attn is None:
        token_lists = [row.tolist() for row in input_ids]
    else:
        for row, mask in zip(input_ids, attn):
            tokens = row[mask.bool()].tolist()
            token_lists.append(tokens)
    suffix_len = _common_suffix_len(token_lists)
    return max(1, suffix_len)


# Xác định độ dài hậu tố template theo cách tokenize dùng trong angular extraction.
# Giữ vị trí token được trích nhất quán với định dạng instruction của extractor.
def _template_suffix_len_angular(tokenizer: AutoTokenizer) -> int:
    """Compute template suffix length using angular-steering tokenization."""
    enc = _tokenize_instructions_angular(tokenizer, ["a", "b"])
    if isinstance(enc, torch.Tensor):
        input_ids = enc
        token_lists = [row.tolist() for row in input_ids]
    else:
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask")
        if attn is None:
            token_lists = [row.tolist() for row in input_ids]
        else:
            token_lists = [
                row[mask.bool()].tolist() for row, mask in zip(input_ids, attn)
            ]
    suffix_len = _common_suffix_len(token_lists)
    return max(1, suffix_len)


# Quản lý việc khám phá hook, thu activation theo token và tính thống kê contrastive để xây vector ORBIT.
class ResidualActivationExtractor:
    # Lưu model/tokenizer, kích thước batch, giới hạn mẫu PCA và layout extraction.
    # Khám phá các hook rồi tạo mapping vị trí dùng xuyên suốt việc thu thập và tính vector.
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size: int = 4,
        max_pca_samples: int = 2048,
        extraction_layout: str = "norm",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.extraction_layout = normalize_orbit_extraction_layout(
            extraction_layout
        )
        self.extraction_layout_version = orbit_extraction_layout_version(
            self.extraction_layout
        )
        self.hooks = discover_residual_hooks(model, self.extraction_layout)
        self._hook_map = {hook.key: hook for hook in self.hooks}
        self.batch_size = max(1, batch_size)
        self.max_pca_samples = max_pca_samples

    # Đăng ký forward hook hoặc pre-hook theo thông tin từng vị trí extraction.
    # Trả các handle để tháo hook; cache giữ activation thu được ở lần forward hiện tại.
    @contextlib.contextmanager
    def _register_extraction_hooks(self, cache: Dict[HookKey, torch.Tensor]):
        """Capture the exact tensors defined by the selected layout."""
        handles = []

        # Tạo forward hook đóng gói khóa của một vị trí extraction.
        # Mỗi hook ghi activation vào đúng phần tử cache của module tương ứng.
        def make_hook(key: HookKey):
            # Đọc tensor activation tại vị trí hook và lưu vào cache theo khóa đã đóng gói.
            # Hook chỉ thu thập dữ liệu, không thay đổi output hay input của model.
            def hook(module, input, output):
                # Capture either a normalization output or a decoder-block
                # output, depending on the selected layout.
                if isinstance(output, tuple):
                    cache[key] = output[0].detach()
                else:
                    cache[key] = output.detach()

            return hook

        # Tạo pre-hook để thu hidden state đầu vào của module.
        # Dùng cho layout cần lấy activation trước khi module thực hiện phép biến đổi.
        def make_pre_hook(key: HookKey):
            # Đọc tensor activation tại vị trí hook và lưu vào cache theo khóa đã đóng gói.
            # Hook chỉ thu thập dữ liệu, không thay đổi output hay input của model.
            def hook(module, inputs):
                if not inputs or not torch.is_tensor(inputs[0]):
                    raise RuntimeError(
                        f"ORBIT pre-hook {key} did not receive a tensor input."
                    )
                cache[key] = inputs[0].detach()

            return hook

        for hook_def in self.hooks:
            if hook_def.hook_type == "pre":
                handles.append(
                    hook_def.module.register_forward_pre_hook(
                        make_pre_hook(hook_def.key)
                    )
                )
            else:
                handles.append(
                    hook_def.module.register_forward_hook(
                        make_hook(hook_def.key)
                    )
                )
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    # Tokenize prompt/response theo cấu hình rồi chạy forward theo batch để thu activation.
    # Chọn token theo token_mode, tích lũy thống kê và mẫu giới hạn cho PCA; trả dữ liệu theo vị trí hook và tháo hook sau khi dùng.
    def collect(
        self,
        prompts: List[str],
        responses: List[str],
        desc: str = "collect",
        token_mode: str = "prompt_last",
        format_prompts: bool = True,
        system_prompts: Optional[List[str]] = None,
        preformatted_add_special_tokens: Optional[bool] = None,
        response_tokenization: str = "joint",
    ):
        assert len(prompts) == len(responses)
        if not prompts:
            return {}

        device = next(self.model.parameters()).device
        hidden_dim = self.model.config.hidden_size

        tokenizer = self.tokenizer
        token_mode = token_mode.lower().strip()
        response_tokenization = response_tokenization.lower().strip()
        if response_tokenization not in {"joint", "separate"}:
            raise ValueError(
                "response_tokenization must be 'joint' or 'separate'. "
                f"Got: {response_tokenization}"
            )
        if token_mode not in {
            "prompt_last",
            "response_last",
            "response_mean",
            "response_causal_last",
            "response_causal_mean",
            "template_suffix_last",
            "template_suffix_mean",
            "template_suffix_last_exact",
            "template_suffix_mean_exact",
        }:
            raise ValueError(
                "token_mode must be one of prompt_last, response_last, response_mean, "
                "response_causal_last, response_causal_mean, "
                "template_suffix_last, template_suffix_mean, "
                f"template_suffix_last_exact, template_suffix_mean_exact. Got: {token_mode}"
            )

        # Prepare tokenization depending on token_mode
        prompt_ids = None
        response_ids = None
        response_spans = None
        if token_mode in {"template_suffix_last_exact", "template_suffix_mean_exact"}:
            encodings = _tokenize_instructions_angular(tokenizer, prompts)
        elif token_mode in {"prompt_last", "template_suffix_last", "template_suffix_mean"}:
            # Format prompts exactly as they are formatted for generation.
            if format_prompts:
                prompts_fmt, prompts_have_special = _format_prompts_for_model(
                    tokenizer, prompts, system_prompts=system_prompts
                )
            else:
                prompts_fmt = list(prompts)
                add_special = (
                    True
                    if preformatted_add_special_tokens is None
                    else bool(preformatted_add_special_tokens)
                )
                prompts_have_special = not add_special
            encodings = tokenizer(
                prompts_fmt,
                return_tensors="pt",
                padding=True,
                add_special_tokens=not prompts_have_special,
            )
        else:
            # Match the scoring pipeline's token stream. Normal Hugging Face
            # sequence scoring tokenizes prompt+response jointly; the imported
            # PIXEL evaluators intentionally concatenate separately tokenized
            # prompt and stripped candidate ids, so that protocol requests
            # response_tokenization="separate".
            if format_prompts:
                prompts_fmt, prompts_have_special = _format_prompts_for_model(
                    tokenizer, prompts, system_prompts=system_prompts
                )
            else:
                prompts_fmt = list(prompts)
                add_special = (
                    True
                    if preformatted_add_special_tokens is None
                    else bool(preformatted_add_special_tokens)
                )
                prompts_have_special = not add_special
            if response_tokenization == "separate":
                prompt_ids = tokenizer(
                    prompts_fmt,
                    add_special_tokens=not prompts_have_special,
                    padding=False,
                    truncation=False,
                )["input_ids"]
                response_ids = tokenizer(
                    [str(response).strip() for response in responses],
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                )["input_ids"]
                combined = [
                    p_ids + r_ids for p_ids, r_ids in zip(prompt_ids, response_ids)
                ]
                encodings = tokenizer.pad(
                    {"input_ids": combined},
                    padding=True,
                    return_tensors="pt",
                )
            else:
                combined_texts = [p + r for p, r in zip(prompts_fmt, responses)]
                boundaries = [len(p) for p in prompts_fmt]
                try:
                    encodings = tokenizer(
                        combined_texts,
                        add_special_tokens=not prompts_have_special,
                        padding=True,
                        truncation=False,
                        return_offsets_mapping=True,
                        return_tensors="pt",
                    )
                except (NotImplementedError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "Exact joint response extraction requires a fast "
                        "tokenizer with offset mappings. Use the model's fast "
                        "tokenizer, or request response_tokenization='separate' "
                        "only when the evaluation scorer also concatenates "
                        "separately tokenized IDs."
                    ) from exc
                offsets = encodings.pop("offset_mapping")
                response_spans = [
                    _joint_response_span_from_offsets(
                        row_offsets,
                        boundary,
                        has_response=bool(response),
                    )
                    for row_offsets, boundary, response in zip(
                        offsets, boundaries, responses
                    )
                ]

        total = len(prompts)
        batch_size = min(self.batch_size, total)
        if batch_size <= 0:
            batch_size = 1

        accumulator: Dict[HookKey, Dict[str, torch.Tensor]] = {}

        indices = range(0, total, batch_size)
        for start in tqdm(indices, total=len(indices), desc=desc, leave=False):
            end = min(start + batch_size, total)
            batch_inputs = {key: value[start:end].to(device) for key, value in encodings.items()}
            batch_attention = batch_inputs.get("attention_mask")
            if batch_attention is None:
                pad_id = tokenizer.pad_token_id
                if pad_id is None:
                    raise ValueError("Tokenizer has no pad_token_id; cannot infer attention mask.")
                batch_attention = (batch_inputs["input_ids"] != pad_id).to(torch.long)

            # Find the last non-pad token index for each row.
            seq_len = batch_attention.shape[1]
            last_idx = _last_nonpad_indices(batch_attention)
            valid_rows = batch_attention.any(dim=1)

            cache: Dict[HookKey, torch.Tensor] = {}
            with torch.no_grad():
                with self._register_extraction_hooks(cache):
                    _ = self.model(**batch_inputs)

            consistent_device = next(iter(cache.values())).device
            last_idx = last_idx.to(device=consistent_device)
            valid_rows = valid_rows.to(device=consistent_device)

            for key, acts in cache.items():
                entry = accumulator.get(key)
                acts = acts.to(device=consistent_device, dtype=torch.float32)
                if entry is None:
                    acc_device = acts.device
                    accumulator[key] = entry = {
                        "prompt_last_sum": torch.zeros(hidden_dim, dtype=torch.float32, device=acc_device),
                        "prompt_last_count": torch.zeros((), dtype=torch.long, device=acc_device),
                        "prompt_last_samples": [],
                    }

                # Extract tokens per token_mode.
                if token_mode in {
                    "prompt_last",
                    "template_suffix_last",
                    "template_suffix_mean",
                    "template_suffix_last_exact",
                    "template_suffix_mean_exact",
                }:
                    # Extract the last non-padding prompt token.
                    if not bool(valid_rows.any()):
                        # Entire batch is padding (should be rare); skip.
                        continue

                    # Filter invalid rows BEFORE indexing to avoid ever reading PAD activations.
                    if bool(valid_rows.all()):
                        sel_rows = torch.arange(acts.shape[0], device=acts.device)
                        sel_last_idx = last_idx.to(device=acts.device)
                    else:
                        sel = valid_rows.nonzero(as_tuple=False).squeeze(1)
                        sel_rows = sel.to(device=acts.device)
                        sel_last_idx = last_idx.index_select(0, sel).to(device=acts.device)

                    if token_mode == "prompt_last":
                        last_token_act = acts[sel_rows, sel_last_idx, :]  # [n_valid, hidden]
                        entry["prompt_last_sum"] += last_token_act.sum(dim=0)
                        entry["prompt_last_count"] += int(last_token_act.shape[0])
                        entry["prompt_last_samples"].extend(last_token_act.detach().cpu().unbind(0))
                    else:
                        # Use the configured template-suffix token selection.
                        if token_mode in {"template_suffix_last_exact", "template_suffix_mean_exact"}:
                            if not hasattr(self, "_template_suffix_len_exact"):
                                self._template_suffix_len_exact = None
                            if self._template_suffix_len_exact is None:
                                self._template_suffix_len_exact = _template_suffix_len_angular(tokenizer)
                            suffix_len = int(self._template_suffix_len_exact)
                        else:
                            if not hasattr(self, "_template_suffix_len"):
                                self._template_suffix_len = None
                            if self._template_suffix_len is None:
                                self._template_suffix_len = _template_suffix_len(tokenizer)
                            suffix_len = int(self._template_suffix_len)
                        # indices for suffix tokens ending at sel_last_idx
                        start_idx = sel_last_idx - (suffix_len - 1)
                        if token_mode in {"template_suffix_last", "template_suffix_last_exact"}:
                            token_act = acts[sel_rows, sel_last_idx, :]
                        else:
                            # template_suffix_mean
                            token_act = []
                            for row_i, s_idx, e_idx in zip(
                                sel_rows.tolist(), start_idx.tolist(), sel_last_idx.tolist()
                            ):
                                s_idx = max(0, int(s_idx))
                                e_idx = int(e_idx)
                                token_act.append(acts[row_i, s_idx : e_idx + 1, :].mean(dim=0))
                            token_act = torch.stack(token_act, dim=0)
                        entry["prompt_last_sum"] += token_act.sum(dim=0)
                        entry["prompt_last_count"] += int(token_act.shape[0])
                        entry["prompt_last_samples"].extend(token_act.detach().cpu().unbind(0))
                else:
                    # Response-based extraction: use prompt+response lengths to locate response tokens.
                    if response_spans is not None:
                        batch_spans = response_spans[start:end]
                        resp_start = torch.tensor(
                            [span[0] for span in batch_spans], device=acts.device
                        )
                        resp_end = torch.tensor(
                            [span[1] for span in batch_spans], device=acts.device
                        )
                        valid_rows = resp_end >= resp_start
                    else:
                        assert prompt_ids is not None and response_ids is not None
                        batch_prompt_lens = torch.tensor(
                            [len(x) for x in prompt_ids[start:end]], device=acts.device
                        )
                        batch_resp_lens = torch.tensor(
                            [len(x) for x in response_ids[start:end]], device=acts.device
                        )
                        seq_len = acts.shape[1]
                        resp_start, resp_end = _response_spans(
                            batch_prompt_lens,
                            batch_resp_lens,
                            seq_len,
                            padding_side=getattr(tokenizer, "padding_side", "left"),
                        )
                        valid_rows = batch_resp_lens > 0
                    if not bool(valid_rows.any()):
                        continue

                    if token_mode in {"response_last", "response_causal_last"}:
                        sel = valid_rows.nonzero(as_tuple=False).squeeze(1)
                        sel_rows = sel.to(device=acts.device)
                        sel_last_idx = resp_end.index_select(0, sel).to(device=acts.device)
                        if token_mode == "response_causal_last":
                            # Hidden state j predicts token j+1.  Use the state
                            # immediately before the final response token so
                            # vector extraction is temporally symmetric with
                            # answer-causal intervention during MC scoring.
                            sel_last_idx = sel_last_idx - 1
                            causal_valid = sel_last_idx >= 0
                            sel_rows = sel_rows[causal_valid]
                            sel_last_idx = sel_last_idx[causal_valid]
                            if sel_rows.numel() == 0:
                                continue
                        last_token_act = acts[sel_rows, sel_last_idx, :]
                        entry["prompt_last_sum"] += last_token_act.sum(dim=0)
                        entry["prompt_last_count"] += int(last_token_act.shape[0])
                        entry["prompt_last_samples"].extend(last_token_act.detach().cpu().unbind(0))
                    else:  # response_mean or response_causal_mean
                        # Mean over response tokens per row (variable length).
                        sel = valid_rows.nonzero(as_tuple=False).squeeze(1)
                        for idx in sel.tolist():
                            s, e = _response_activation_span(
                                int(resp_start[idx].item()),
                                int(resp_end[idx].item()),
                                token_mode,
                            )
                            if e < s:
                                continue
                            if s < 0:
                                continue
                            mean_act = acts[idx, s : e + 1, :].mean(dim=0)
                            entry["prompt_last_sum"] += mean_act
                            entry["prompt_last_count"] += 1
                            entry["prompt_last_samples"].append(mean_act.detach().cpu())

            cache.clear()

        for entry in accumulator.values():
            entry["prompt_last_sum"] = entry["prompt_last_sum"].cpu()
            entry["prompt_last_count"] = entry["prompt_last_count"].cpu()

        return accumulator

    # Từ thống kê positive/negative, tính hướng chênh lệch trung bình và trục PCA trực giao.
    # Tạo vector chiếu/boost cùng thống kê hiệu chuẩn; tùy chọn cân bằng theo câu hỏi để biến thể không làm lệch trọng số.
    def compute_directional_differences(
        self,
        positive_stats: Dict[HookKey, Dict[str, torch.Tensor]],
        negative_stats: Dict[HookKey, Dict[str, torch.Tensor]],
        *,
        delta_proj_mode: str = "plane_mean",
        delta_proj_scale: float = 1.0,
        pair_group_ids: Optional[List[str]] = None,
        pair_positive_ids: Optional[List[str]] = None,
        pair_negative_ids: Optional[List[str]] = None,
        aggregate_by_pair_group: bool = False,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        module_vectors: Dict[str, Dict[str, torch.Tensor]] = {}
        delta_proj_mode = delta_proj_mode.lower().strip()
        if delta_proj_mode not in {"plane_mean", "diff_mean", "b1_only", "none"}:
            raise ValueError(
                f"delta_proj_mode must be one of plane_mean, diff_mean, b1_only, none. Got: {delta_proj_mode}"
            )

        for key, hook in self._hook_map.items():
            pos_data = positive_stats.get(key)
            neg_data = negative_stats.get(key)
            if not pos_data or pos_data["prompt_last_count"] == 0:
                continue
            if not neg_data or neg_data["prompt_last_count"] == 0:
                continue

            pos_samples = positive_stats[key].get("prompt_last_samples", [])
            neg_samples = negative_stats[key].get("prompt_last_samples", [])
            raw_n_pairs = min(len(pos_samples), len(neg_samples))
            direction_pos_samples = list(pos_samples[:raw_n_pairs])
            direction_neg_samples = list(neg_samples[:raw_n_pairs])
            stability_group_ids = (
                [str(value) for value in pair_group_ids[:raw_n_pairs]]
                if pair_group_ids is not None
                else None
            )

            if aggregate_by_pair_group:
                required_lists = (
                    pair_group_ids,
                    pair_positive_ids,
                    pair_negative_ids,
                )
                if any(values is None for values in required_lists):
                    raise ValueError(
                        "Question-balanced vector construction requires group, "
                        "positive-candidate, and negative-candidate IDs."
                    )
                if any(len(values) < raw_n_pairs for values in required_lists):
                    raise ValueError(
                        "Question-balanced IDs must cover every paired activation."
                    )
                group_order: List[str] = []
                group_indices: Dict[str, List[int]] = {}
                for index, value in enumerate(pair_group_ids[:raw_n_pairs]):
                    group_id = str(value)
                    if group_id not in group_indices:
                        group_order.append(group_id)
                        group_indices[group_id] = []
                    group_indices[group_id].append(index)

                grouped_positive: List[torch.Tensor] = []
                grouped_negative: List[torch.Tensor] = []
                for group_id in group_order:
                    indices = group_indices[group_id]
                    positive_seen = set()
                    negative_seen = set()
                    positive_indices: List[int] = []
                    negative_indices: List[int] = []
                    for index in indices:
                        positive_id = str(pair_positive_ids[index])
                        negative_id = str(pair_negative_ids[index])
                        if positive_id not in positive_seen:
                            positive_seen.add(positive_id)
                            positive_indices.append(index)
                        if negative_id not in negative_seen:
                            negative_seen.add(negative_id)
                            negative_indices.append(index)
                    grouped_positive.append(
                        torch.stack(
                            [pos_samples[index] for index in positive_indices]
                        ).mean(dim=0)
                    )
                    grouped_negative.append(
                        torch.stack(
                            [neg_samples[index] for index in negative_indices]
                        ).mean(dim=0)
                    )
                direction_pos_samples = grouped_positive
                direction_neg_samples = grouped_negative
                stability_group_ids = group_order

            # Aggregate paired candidate rows by the configured grouping mode.
            pos_mean = torch.stack(direction_pos_samples).mean(dim=0)
            neg_mean = torch.stack(direction_neg_samples).mean(dim=0)

            diff = pos_mean - neg_mean
            first_dir = diff.to(torch.float32).cpu()
            first_norm = float(first_dir.norm().item())
            first_is_degenerate = (
                not math.isfinite(first_norm) or first_norm <= 1e-8
            )
            # Keep one vector entry for every configured hook site.
            first_unit = (
                torch.zeros_like(first_dir)
                if first_is_degenerate
                else first_dir / first_dir.norm().clamp_min(1e-8)
            )

            # Gate calibration retains the full candidate-level distribution.
            r_typical_stats = self._compute_r_typical_stats(
                pos_samples, neg_samples, first_unit
            )
            r_typical_stats["norm_first_dir"] = float(first_dir.norm().item())

            # Compute b2 from paired differences (pos_i - neg_i), using selected mode.
            second_dir = None
            vector_quality_stats = {
                "n_paired_differences": 0,
                "pair_aggregation": (
                    "question_class_means_v1"
                    if aggregate_by_pair_group
                    else "candidate_rows_v1"
                ),
                "b1_split_half_cosine": None,
                "b1_stability_split_version": (
                    "question_group_disjoint_sha256_v1"
                    if pair_group_ids is not None
                    else "row_interleaved_v1"
                ),
                "n_stability_groups": None,
                "b2_explained_variance_ratio": None,
                "paired_difference_residual_rms": None,
                "local_b1_degenerate": bool(first_is_degenerate),
            }
            n_pairs = min(
                len(direction_pos_samples), len(direction_neg_samples)
            )
            if n_pairs >= 2:
                from sklearn.decomposition import PCA

                diffs = []
                for i in range(n_pairs):
                    diffs.append(
                        direction_pos_samples[i] - direction_neg_samples[i]
                    )

                data = torch.stack(diffs)
                vector_quality_stats["n_paired_differences"] = int(
                    data.shape[0]
                )
                centered = data - data.mean(dim=0, keepdim=True)
                vector_quality_stats["paired_difference_residual_rms"] = float(
                    centered.norm(dim=1).square().mean().sqrt().item()
                )
                if stability_group_ids is not None:
                    if len(stability_group_ids) < n_pairs:
                        raise ValueError(
                            "pair_group_ids must cover every paired activation."
                        )
                    group_ids = stability_group_ids[:n_pairs]
                    unique_groups = sorted(
                        set(group_ids),
                        key=lambda value: hashlib.sha256(
                            value.encode("utf-8")
                        ).hexdigest(),
                    )
                    vector_quality_stats["n_stability_groups"] = len(
                        unique_groups
                    )
                    first_groups = set(unique_groups[::2])
                    second_groups = set(unique_groups[1::2])
                    first_indices = [
                        index
                        for index, group_id in enumerate(group_ids)
                        if group_id in first_groups
                    ]
                    second_indices = [
                        index
                        for index, group_id in enumerate(group_ids)
                        if group_id in second_groups
                    ]
                    stable_halves = bool(first_indices and second_indices)
                    if stable_halves:
                        first_half = data[first_indices].mean(dim=0)
                        second_half = data[second_indices].mean(dim=0)
                else:
                    stable_halves = data.shape[0] >= 4
                    if stable_halves:
                        first_half = data[::2].mean(dim=0)
                        second_half = data[1::2].mean(dim=0)
                if stable_halves:
                    denom = first_half.norm() * second_half.norm()
                    if float(denom.item()) > 1e-8:
                        vector_quality_stats["b1_split_half_cosine"] = float(
                            ((first_half @ second_half) / denom).item()
                        )
                max_samples = self.max_pca_samples
                if max_samples and data.shape[0] > max_samples:
                    # Stable, coverage-preserving subsampling. A global RNG
                    # here made b2 depend on unrelated work performed before
                    # vector construction.
                    idx = torch.linspace(
                        0,
                        data.shape[0] - 1,
                        steps=max_samples,
                    ).round().to(torch.long)
                    data = data[idx]
                data_np = data.numpy()
                # A zero-variance matrix makes sklearn return an arbitrary PC
                # with a NaN explained-variance ratio. A degenerate b1 also
                # cannot define a meaningful local 2-D plane. Preserve the
                # hook entry, but leave its local b2 empty; the global-plane
                # construction does not depend on stored local b2 values.
                residual_rms = float(
                    vector_quality_stats["paired_difference_residual_rms"]
                    or 0.0
                )
                if not first_is_degenerate and residual_rms > 1e-8:
                    try:
                        pca = PCA(n_components=1)
                        pca.fit(data_np)
                        vector_quality_stats[
                            "b2_explained_variance_ratio"
                        ] = float(pca.explained_variance_ratio_[0])
                        component = torch.from_numpy(pca.components_[0]).to(torch.float32)
                        # Orthogonalize against b1 (the mean of these same differences)
                        component = component - (component @ first_unit) * first_unit
                        comp_norm = component.norm()
                        if comp_norm > 1e-8:
                            second_dir = (component / comp_norm).cpu()
                    except Exception:
                        second_dir = None

            # Compute mean projection vector of positive/negative samples onto the plane (b1, b2)
            pos_proj_vector_mean = None
            neg_proj_vector_mean = None
            if direction_pos_samples or direction_neg_samples:
                b1 = first_unit
                b2 = second_dir
                b2_unit = None
                if b2 is not None:
                    b2 = b2.to(torch.float32)
                    b2 = b2 - (b2 @ b1) * b1
                    if b2.norm() > 1e-8:
                        b2_unit = b2 / b2.norm()

                # Chiếu các activation lên mặt phẳng đã chọn rồi lấy vector trung bình.
                # Dùng tạo các hướng positive/negative và hướng boost trong cùng mặt phẳng.
                def _proj_mean(samples):
                    if not samples:
                        return None
                    proj_vecs = []
                    for s in samples:
                        s_f = s.to(torch.float32)
                        comp1 = (s_f @ b1) * b1
                        if b2_unit is not None:
                            comp2 = (s_f @ b2_unit) * b2_unit
                            proj = comp1 + comp2
                        else:
                            proj = comp1
                        proj_vecs.append(proj)
                    if proj_vecs:
                        return torch.stack(proj_vecs).mean(dim=0).cpu()
                    return None

                pos_proj_vector_mean = _proj_mean(direction_pos_samples)
                neg_proj_vector_mean = _proj_mean(direction_neg_samples)

            delta_proj_vector = None
            if delta_proj_mode != "none":
                if delta_proj_mode == "plane_mean":
                    if pos_proj_vector_mean is not None and neg_proj_vector_mean is not None:
                        delta_proj_vector = (pos_proj_vector_mean - neg_proj_vector_mean).cpu()
                elif delta_proj_mode == "b1_only":
                    if pos_proj_vector_mean is not None and neg_proj_vector_mean is not None:
                        # Project difference only onto b1 direction.
                        diff = pos_proj_vector_mean - neg_proj_vector_mean
                        scalar = float(diff @ first_unit)
                        delta_proj_vector = (scalar * first_unit).cpu()
                elif delta_proj_mode == "diff_mean":
                    # Mean of (pos_i - neg_i) projected onto the plane.
                    n_pairs = min(
                        len(direction_pos_samples),
                        len(direction_neg_samples),
                    )
                    if n_pairs >= 1:
                        diffs = []
                        for i in range(n_pairs):
                            diffs.append(
                                direction_pos_samples[i]
                                - direction_neg_samples[i]
                            )
                        data = torch.stack(diffs).to(torch.float32)
                        # Project onto (b1, b2) plane.
                        b1 = first_unit
                        b2 = second_dir
                        b2_unit = None
                        if b2 is not None:
                            b2 = b2.to(torch.float32)
                            b2 = b2 - (b2 @ b1) * b1
                            if b2.norm() > 1e-8:
                                b2_unit = b2 / b2.norm()
                        comp1 = (data @ b1)[:, None] * b1
                        if b2_unit is not None:
                            comp2 = (data @ b2_unit)[:, None] * b2_unit
                            proj = comp1 + comp2
                        else:
                            proj = comp1
                        delta_proj_vector = proj.mean(dim=0).cpu()

            if delta_proj_vector is not None and not math.isclose(float(delta_proj_scale), 1.0, abs_tol=1e-8):
                delta_proj_vector = (delta_proj_vector * float(delta_proj_scale)).cpu()

            module_vectors[hook.module_name] = {
                "first_direction": first_dir,
                "second_direction": second_dir,
                "hook_point": (
                    f"resid_{hook.position}"
                    if self.extraction_layout == "residual"
                    else f"norm_{hook.position}"
                ),
                "hook_type": hook.hook_type,
                "extraction_layout_version": self.extraction_layout_version,
                "r_typical_stats": r_typical_stats,
                # Store class means used by global-plane construction.
                "pos_mean_vector": pos_mean.to(torch.float32).cpu(),
                "neg_mean_vector": neg_mean.to(torch.float32).cpu(),
                "r_typical_vector": pos_proj_vector_mean,
                "pos_proj_vector": pos_proj_vector_mean,
                "neg_proj_vector": neg_proj_vector_mean,
                "delta_proj_vector": delta_proj_vector,
                "vector_quality_stats": vector_quality_stats,
            }

        return module_vectors

    # Chiếu mẫu positive/negative lên trục thuộc tính và tính mean, median, độ lệch chuẩn, phân vị.
    # Trả thống kê độ lớn điển hình và khoảng cách hai nhóm để hiệu chuẩn gate/boost.
    def _compute_r_typical_stats(
        self,
        pos_samples: List[torch.Tensor],
        neg_samples: List[torch.Tensor],
        first_unit: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute projection statistics for the contrastive samples."""
        stats = {
            "norm_first_dir": 0.0,
            "pos_proj_mean": 0.0,
            "pos_proj_median": 0.0,
            "neg_proj_mean": 0.0,
            "neg_proj_median": 0.0,
            "pos_proj_std": 0.0,
            "neg_proj_std": 0.0,
            "gap": 0.0,
            "pos_proj_p75": 0.0,
            "pos_proj_p90": 0.0,
        }
        
        first_unit_cpu = first_unit.cpu()
        
        # Project positive samples
        if pos_samples:
            pos_projs = torch.tensor([
                (s.cpu() @ first_unit_cpu).item() for s in pos_samples
            ], dtype=torch.float32)
            stats["pos_proj_mean"] = float(pos_projs.mean())
            stats["pos_proj_median"] = float(pos_projs.median())
            stats["pos_proj_std"] = float(pos_projs.std()) if len(pos_projs) > 1 else 0.0
            if len(pos_projs) >= 4:
                stats["pos_proj_p75"] = float(torch.quantile(pos_projs, 0.75))
                stats["pos_proj_p90"] = float(torch.quantile(pos_projs, 0.90))
            else:
                stats["pos_proj_p75"] = stats["pos_proj_mean"]
                stats["pos_proj_p90"] = stats["pos_proj_mean"]
        
        # Project negative samples
        if neg_samples:
            neg_projs = torch.tensor([
                (s.cpu() @ first_unit_cpu).item() for s in neg_samples
            ], dtype=torch.float32)
            stats["neg_proj_mean"] = float(neg_projs.mean())
            stats["neg_proj_median"] = float(neg_projs.median())
            stats["neg_proj_std"] = float(neg_projs.std()) if len(neg_projs) > 1 else 0.0
        
        # Compute gap (positive - negative mean projection)
        stats["gap"] = stats["pos_proj_mean"] - stats["neg_proj_mean"]
        
        return stats
