#!/usr/bin/env python3
# Vai trò file: Endpoint generation local cho lm-eval; dùng vector và calibration từ thí nghiệm chính để đo năng lực TinyBench.
"""
TinyBench local-completions endpoint backed by this repo's angular steering hooks.

Routes:
  POST /orbit/{variant}
  POST /orbit/{variant}/completions
  GET  /health

Variants:
  - baseline
  - orbit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure repository modules are importable when the endpoint is launched directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from angular_steering_utils import (  # type: ignore
    load_global_layernorm_vectors,
    register_layernorm_additive_hooks,
    remove_hooks,
)
from angular_residual_utils import ORBIT_EXTRACTION_LAYOUT_VERSION  # type: ignore


# Schema request generation cho endpoint TinyBench: prompt, token budget, sampling và tùy chọn logprobs.
class CompletionRequest(BaseModel):
    # Cho phép request chứa trường bổ sung để tương thích với client lm-eval.
    class Config:
        extra = "allow"

    model: str
    prompt: Union[str, List[str]]
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    logprobs: int = 0
    echo: bool = False


# Tách chuỗi phân cách bằng dấu phẩy thành danh sách chuỗi.
# Loại khoảng trắng và bỏ phần tử rỗng để xử lý các tùy chọn CLI như labels, methods hoặc targets.
def _parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


# Định dạng câu hỏi theo chat template của tokenizer, có nhánh dự phòng theo họ model.
# Trả về prompt đã định dạng và cờ cho biết đã có special token, tránh thêm token hai lần.
def _format_prompts_for_model(tokenizer, prompts: Sequence[str]) -> Tuple[List[str], bool]:
    if getattr(tokenizer, "chat_template", None):
        out = []
        for p in prompts:
            out.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return out, True
    name = str(getattr(tokenizer, "name_or_path", "")).lower()
    if "qwen" in name:
        return [f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n" for p in prompts], True
    if "llama" in name and "instruct" in name:
        return (
            [
                (
                    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )
                for p in prompts
            ],
            True,
        )
    return list(prompts), False


# Thay dấu / trong tên model bằng __ để tạo tên dùng trong đường dẫn.
# Trả chuỗi định danh model thuận tiện cho lưu artifact.
def _slug_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


# Nạp tokenizer và causal language model từ Hugging Face hoặc thư mục model cục bộ.
# Áp dụng device_map, dtype và giới hạn bộ nhớ GPU; trả về cặp model/tokenizer cho các bước sau.
def _load_hf_model(
    model_name: str,
    *,
    local_files_only: bool,
    hf_device_map: str,
    hf_torch_dtype: str,
    hf_max_memory_per_gpu_gib: int,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs: Dict[str, object] = {
        "local_files_only": local_files_only,
    }
    dm = str(hf_device_map).strip()
    if dm:
        load_kwargs["device_map"] = dm

    dt = str(hf_torch_dtype).strip().lower()
    if dt == "float16":
        load_kwargs["torch_dtype"] = torch.float16
    elif dt == "bfloat16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    elif dt == "float32":
        load_kwargs["torch_dtype"] = torch.float32

    if hf_max_memory_per_gpu_gib > 0 and torch.cuda.is_available():
        n_visible = torch.cuda.device_count()
        if n_visible > 0:
            load_kwargs["max_memory"] = {
                i: f"{hf_max_memory_per_gpu_gib}GiB" for i in range(n_visible)
            }
            if "device_map" not in load_kwargs:
                load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return model, tokenizer


# Giải mã một token, kể cả special token, thành chuỗi.
# Nếu tokenizer không decode được thì dùng token ID dạng chuỗi làm giá trị dự phòng.
def _safe_decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return str(int(token_id))


# Chọn k token có log-probability cao nhất từ vector logp.
# Trả mapping văn bản token → log-probability theo định dạng completions endpoint.
def _top_logprobs_dict(tokenizer, logp_vec: torch.Tensor, k: int) -> Dict[str, float]:
    if k <= 0:
        return {}
    vals, idx = torch.topk(logp_vec, k=min(k, int(logp_vec.shape[-1])))
    out: Dict[str, float] = {}
    for v, i in zip(vals.tolist(), idx.tolist()):
        out[_safe_decode_token(tokenizer, int(i))] = float(v)
    return out


# Nạp vector ORBIT của từng target và kiểm tra phiên bản ghép cặp/layout.
# Dựng mặt phẳng global cho mỗi trait; trả các mapping vector và độ lớn điển hình dùng khi generation.
def _load_target_vectors(
    *,
    vector_dir: Path,
    targets: Sequence[str],
    r_typical_method: str,
) -> Tuple[List[Dict[str, Dict[str, torch.Tensor]]], List[Dict[str, float]]]:
    mv_list: List[Dict[str, Dict[str, torch.Tensor]]] = []
    rt_list: List[Dict[str, float]] = []
    for tgt in targets:
        vec_path = vector_dir / f"{tgt}_angular_instruct.pt"
        if not vec_path.exists():
            raise FileNotFoundError(f"Missing vector file for target '{tgt}': {vec_path}")
        payload = torch.load(vec_path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        if metadata.get("pairing_version") != "question_aligned_v3":
            raise ValueError(
                f"ORBIT vector metadata mismatch for {vec_path}: expected "
                "pairing_version='question_aligned_v3'. Rebuild the vectors with "
                "the question-aligned pipeline before TinyBench."
            )
        if (
            metadata.get("extraction_layout_version")
            != ORBIT_EXTRACTION_LAYOUT_VERSION
        ):
            raise ValueError(
                f"ORBIT vector metadata mismatch for {vec_path}: expected "
                f"extraction_layout_version={ORBIT_EXTRACTION_LAYOUT_VERSION!r}. "
                "Rebuild the vectors with the current two-sites-per-block extractor."
            )
        mv, rt = load_global_layernorm_vectors(
            str(vec_path),
            global_b1_select="max_sim",
            r_typical_method=r_typical_method,
        )
        mv_list.append(mv)
        rt_list.append(rt)
    return mv_list, rt_list


# Tìm additive_coeff đã calibration trong CSV cho đúng tập target và số K.
# Không phụ thuộc thứ tự nhãn; báo lỗi nếu không có hệ số cho tổ hợp yêu cầu.
def _lookup_calibrated_coeff(
    calib_csv: Path,
    targets: Sequence[str],
) -> float:
    if not calib_csv.exists():
        raise FileNotFoundError(f"Calibration CSV not found: {calib_csv}")
    target_set = {t.strip() for t in targets if t.strip()}
    k = len(target_set)
    best = None
    with calib_csv.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        cols = {c: i for i, c in enumerate(header)}
        for line in f:
            parts = []
            cur = ""
            in_quote = False
            for ch in line.rstrip("\n"):
                if ch == '"':
                    in_quote = not in_quote
                elif ch == "," and not in_quote:
                    parts.append(cur)
                    cur = ""
                else:
                    cur += ch
            parts.append(cur)
            if len(parts) < len(header):
                continue
            try:
                row_k = int(parts[cols["k"]])
            except Exception:
                continue
            if row_k != k:
                continue
            row_targets = {x.strip() for x in parts[cols["targets"]].split(",") if x.strip()}
            if row_targets != target_set:
                continue
            try:
                coeff = float(parts[cols["additive_coeff"]])
            except Exception:
                continue
            best = coeff
            break
    if best is None:
        raise ValueError(
            f"No calibrated coefficient found for targets={sorted(target_set)} in {calib_csv}"
        )
    return float(best)


# Giữ model, tokenizer, vectors và calibration để phục vụ generation baseline/ORBIT qua HTTP.
class TinyBenchOrbitServer:
    # Nạp model/tokenizer, danh sách targets, vector và hệ số calibration cho server.
    # Các tài nguyên được giữ lại để phục vụ nhiều request benchmark mà không load model mỗi lần.
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model, self.tokenizer = _load_hf_model(
            args.model_name,
            local_files_only=bool(args.local_files_only),
            hf_device_map=str(args.hf_device_map),
            hf_torch_dtype=str(args.hf_torch_dtype),
            hf_max_memory_per_gpu_gib=int(args.hf_max_memory_per_gpu_gib),
        )
        self.targets = _parse_str_list(args.targets)
        if not self.targets:
            raise ValueError("--targets must contain at least one label.")
        self.mv_list, self.rt_list = _load_target_vectors(
            vector_dir=Path(args.vector_dir),
            targets=self.targets,
            r_typical_method="pos_mean",
        )
        self.calib_coeff = _lookup_calibrated_coeff(
            Path(args.calib_coeff_csv),
            self.targets,
        )

    # Chuyển tên variant baseline hoặc orbit thành cấu hình bật steering và boost.
    # Trả các cờ cùng additive_coeff; báo lỗi cho variant không được hỗ trợ.
    def _variant_params(self, variant: str) -> Tuple[bool, bool, float]:
        v = str(variant).strip().lower()
        if v == "baseline":
            return False, False, 0.0
        if v == "orbit":
            return True, True, float(self.calib_coeff)
        raise ValueError(f"Unsupported variant '{variant}'.")

    # Sinh completion cho batch prompt ở chế độ baseline hoặc ORBIT.
    # Áp dụng chat template, gắn/tháo hook và tạo payload text/logprobs tương thích công cụ benchmark.
    def _generate_batch(
        self,
        prompts: List[str],
        *,
        variant: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        logprobs: int,
        echo: bool,
    ) -> List[dict]:
        if bool(self.args.apply_chat_template):
            prompts_fmt, prompts_have_special = _format_prompts_for_model(self.tokenizer, prompts)
        else:
            prompts_fmt, prompts_have_special = list(prompts), False
        enc = self.tokenizer(
            prompts_fmt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=(not prompts_have_special) and bool(self.args.add_special_tokens),
        )
        input_ids = enc["input_ids"].to(self.model.device)
        attn = enc["attention_mask"].to(self.model.device)
        in_len = input_ids.shape[1]

        do_steer, additive_boost, add_coeff = self._variant_params(variant)
        handles = []
        if do_steer:
            handles = register_layernorm_additive_hooks(
                model=self.model,
                module_vectors=None,
                coeff=1.0,
                positions="response",
                add_module_vectors_list=self.mv_list,
                add_r_typical_list=self.rt_list,
                parallel_combine=False,
                adaptive_mode=4,
                adaptive_threshold=0.0,
                parallel_method="sum",
                sequential_mode="full",
                target_angle=0.0,
                hybrid_multitrait=True,
                trait_weights=[1.0] * len(self.targets),
                hybrid_gate_style="margin",
                hybrid_soft_temperature=0.5,
                hybrid_min_gate=0.0,
                additive_boost=bool(additive_boost),
                additive_coeff=float(add_coeff),
                boost_gate_mode="adaptive",
                boost_abs_tau=0.0,
                boost_min_gate=0.0,
                boost_clamp_ratio=0.0,
                boost_delta_clamp_mode="off",
                boost_approach=1,
                boost_fill_fraction=0.4,
                hybrid_gate_reference="zero",
                hybrid_rotation_gate_mode="full",
                hybrid_rotation_gate_scale=1.0,
            )

        prompt_token_logprobs_batch: List[List[float]] = []
        prompt_top_logprobs_batch: List[List[Dict[str, float]]] = []
        prompt_tokens_batch: List[List[str]] = []
        prompt_text_batch: List[str] = []

        try:
            # For lm_eval loglikelihood compatibility with local-completions:
            # when echo=True we must provide prompt token logprobs (not None),
            # otherwise lm_eval's parser fails while summing continuation logprobs.
            if echo:
                with torch.no_grad():
                    fw = self.model(
                        input_ids=input_ids,
                        attention_mask=attn,
                        use_cache=False,
                        return_dict=True,
                    )
                p_logits = fw.logits  # [B, L, V]
                for b in range(input_ids.shape[0]):
                    p_ids_all = input_ids[b].tolist()
                    p_mask_all = attn[b].tolist()
                    valid_idx = [i for i, m in enumerate(p_mask_all) if int(m) == 1]
                    if not valid_idx:
                        prompt_tokens_batch.append([])
                        prompt_token_logprobs_batch.append([])
                        prompt_top_logprobs_batch.append([])
                        prompt_text_batch.append("")
                        continue

                    p_ids = [int(p_ids_all[i]) for i in valid_idx]
                    p_tokens = [_safe_decode_token(self.tokenizer, tid) for tid in p_ids]
                    p_text = self.tokenizer.decode(p_ids, skip_special_tokens=False)

                    p_tok_lp: List[float] = []
                    p_top_lp: List[Dict[str, float]] = []
                    first_abs = valid_idx[0]
                    for local_j, abs_j in enumerate(valid_idx):
                        tok_id = int(p_ids_all[abs_j])
                        if abs_j <= first_abs:
                            # First token has no previous token distribution.
                            lp_val = 0.0
                            top_map = {_safe_decode_token(self.tokenizer, tok_id): float(lp_val)}
                        else:
                            lp_vec = torch.log_softmax(p_logits[b, abs_j - 1].float(), dim=-1)
                            lp_val = float(lp_vec[tok_id].item())
                            if int(logprobs) > 0:
                                top_map = _top_logprobs_dict(self.tokenizer, lp_vec, int(logprobs))
                                if not top_map:
                                    top_map = {_safe_decode_token(self.tokenizer, tok_id): float(lp_val)}
                            else:
                                top_map = {_safe_decode_token(self.tokenizer, tok_id): float(lp_val)}
                        p_tok_lp.append(float(lp_val))
                        p_top_lp.append(top_map)

                    prompt_tokens_batch.append(p_tokens)
                    prompt_token_logprobs_batch.append(p_tok_lp)
                    prompt_top_logprobs_batch.append(p_top_lp)
                    prompt_text_batch.append(p_text)

            gen_kwargs: Dict[str, object] = {
                "input_ids": input_ids,
                "attention_mask": attn,
                "max_new_tokens": max(0, int(max_new_tokens)),
                "do_sample": bool(float(temperature) > 0.0),
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            if float(temperature) > 0.0:
                gen_kwargs["temperature"] = float(temperature)
                gen_kwargs["top_p"] = float(top_p)

            with torch.no_grad():
                out = self.model.generate(**gen_kwargs)
        finally:
            if handles:
                remove_hooks(handles)

        sequences = out.sequences
        gen_ids = sequences[:, in_len:]
        gen_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        scores = out.scores if out.scores is not None else []

        rows: List[dict] = []
        for b in range(gen_ids.shape[0]):
            g_ids = gen_ids[b].tolist()
            g_tokens = [_safe_decode_token(self.tokenizer, tid) for tid in g_ids]
            g_logprobs: List[float] = []
            g_top_logprobs: List[Dict[str, float]] = []
            for j, tid in enumerate(g_ids):
                if j < len(scores):
                    lp = torch.log_softmax(scores[j][b].float(), dim=-1)
                    lp_val = float(lp[int(tid)].item())
                    g_logprobs.append(lp_val)
                    if logprobs > 0:
                        top_map = _top_logprobs_dict(self.tokenizer, lp, int(logprobs))
                        if not top_map:
                            top_map = {_safe_decode_token(self.tokenizer, int(tid)): lp_val}
                        g_top_logprobs.append(top_map)
                    else:
                        g_top_logprobs.append({_safe_decode_token(self.tokenizer, int(tid)): lp_val})
                else:
                    lp_val = 0.0
                    g_logprobs.append(lp_val)
                    g_top_logprobs.append({_safe_decode_token(self.tokenizer, int(tid)): lp_val})

            text_out = gen_texts[b]
            tokens = g_tokens
            token_logprobs = g_logprobs
            top_lp = g_top_logprobs
            if echo:
                p_tokens = prompt_tokens_batch[b] if b < len(prompt_tokens_batch) else []
                p_text = prompt_text_batch[b] if b < len(prompt_text_batch) else ""
                p_tok_lp = prompt_token_logprobs_batch[b] if b < len(prompt_token_logprobs_batch) else []
                p_top_lp = prompt_top_logprobs_batch[b] if b < len(prompt_top_logprobs_batch) else []
                text_out = p_text + text_out
                tokens = p_tokens + tokens
                token_logprobs = p_tok_lp + token_logprobs
                top_lp = p_top_lp + top_lp

            rows.append(
                {
                    "id": f"cmpl-{int(time.time()*1000)}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": self.args.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": text_out,
                            "finish_reason": "stop",
                            "logprobs": {
                                "tokens": tokens,
                                "token_logprobs": token_logprobs,
                                "top_logprobs": top_lp,
                            },
                        }
                    ],
                }
            )
        return rows


# Tạo ứng dụng FastAPI và đăng ký health/completions routes.
# Các route dùng chung server đã nạp model và trả kết quả cho lm-eval local-completions.
def build_app(server: TinyBenchOrbitServer) -> FastAPI:
    app = FastAPI()

    # Trả trạng thái sẵn sàng cùng model, targets và hệ số calibration.
    # Launcher dùng route này để chờ endpoint khởi động xong trước khi chạy benchmark.
    @app.get("/health")
    async def health():
        return {
            "ok": True,
            "model": server.args.model_name,
            "targets": server.targets,
            "calib_coeff": server.calib_coeff,
        }

    # Chuẩn hóa request prompt thành danh sách rồi gọi generation của server.
    # Trả một kết quả hoặc danh sách theo batch; đổi lỗi xử lý thành HTTP 500.
    async def _handle(variant: str, request: CompletionRequest):
        try:
            prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
            rows = server._generate_batch(
                [str(p) for p in prompts],
                variant=variant,
                max_new_tokens=int(request.max_tokens),
                temperature=float(request.temperature),
                top_p=float(request.top_p),
                logprobs=int(request.logprobs),
                echo=bool(request.echo),
            )
            return rows[0] if len(rows) == 1 else rows
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    # Xử lý POST /orbit/{variant} bằng handler dùng chung.
    # Variant chọn baseline hoặc orbit cho request completion.
    @app.post("/orbit/{variant}")
    async def completions_variant(variant: str, request: CompletionRequest):
        return await _handle(variant, request)

    # Xử lý POST /orbit/{variant}/completions bằng cùng handler generation.
    # Route này hỗ trợ client tự thêm hậu tố /completions vào base_url.
    @app.post("/orbit/{variant}/completions")
    async def completions_variant_completions(variant: str, request: CompletionRequest):
        return await _handle(variant, request)

    return app


# Khai báo các tham số dòng lệnh và giá trị mặc định của chương trình.
# Trả về ArgumentParser; bước parse_args ở hàm điều phối mới đọc cấu hình người dùng.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TinyBench local-completions endpoint for ORBIT.")
    p.add_argument("--model_name", required=True)
    p.add_argument("--vector_dir", required=True)
    p.add_argument("--targets", required=True, help="Comma-separated labels, e.g. 'expert,concise'")
    p.add_argument("--calib_coeff_csv", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9910)

    p.add_argument("--hf_device_map", default="auto")
    p.add_argument("--hf_torch_dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--hf_max_memory_per_gpu_gib", type=int, default=0)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--apply_chat_template", action="store_true", help="Wrap incoming prompts with chat template before tokenization.")
    p.add_argument(
        "--add_special_tokens",
        action="store_true",
        help="Add tokenizer special tokens when tokenizing incoming prompts.",
    )
    return p


# Đọc cấu hình endpoint, tạo server và ứng dụng FastAPI.
# Khởi chạy uvicorn tại host/port yêu cầu để lm-eval gửi request generation.
def main() -> int:
    args = build_parser().parse_args()
    server = TinyBenchOrbitServer(args)
    app = build_app(server)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
