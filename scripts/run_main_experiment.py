#!/usr/bin/env python3
# Vai trò file: Launcher cấu hình thí nghiệm chính; chuyển tùy chọn đơn giản thành lệnh gọi pipeline chi tiết.
"""Run an ORBIT main experiment."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


# Năm tone được launcher chọn cho thí nghiệm chính; pipeline duyệt các tổ hợp K=1,2,3.
TONE_LABELS = "expert,cautious,empathetic,casual,concise"
# Năm trait được launcher chọn trong số các trait định nghĩa ở TraitFactory.
TRAIT_LABELS = "deception,evil,humorous,empathy,sycophantic"


# Khai báo cấu hình cấp launcher: model, task, đường dẫn, GPU và quy mô thí nghiệm.
# Các tùy chọn này được chuyển thành bộ tham số chi tiết cho pipeline.
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", required=True, help="Hugging Face model id")
    p.add_argument("--task", required=True, choices=("tones", "traits"))
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument(
        "--questions-file",
        type=Path,
        help=(
            "External ToneBank or TraitFactory questions JSONL file. ToneBank is "
            "downloaded from Hugging Face when omitted; required for TraitFactory."
        ),
    )
    p.add_argument(
        "--trait-definitions-file",
        type=Path,
        help="External TraitFactory TRAIT_DEFINITIONS Python file (required for --task traits)",
    )
    p.add_argument("--device-ids", default="0", help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--device-map", default="balanced")
    p.add_argument("--max-memory-per-gpu-gib", type=int, default=36)
    p.add_argument("--max-questions", type=int, default=200)
    p.add_argument("--max-pairs", type=int, default=150)
    p.add_argument("--calibration-samples", type=int, default=100)
    p.add_argument("--n-generate", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument(
        "--split-mode",
        choices=("shared_pool", "question_holdout"),
        default="shared_pool",
    )
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--negative-seed", type=int, default=0)
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument(
        "--build-temperature",
        type=float,
        default=0.7,
        help="Temperature for contrastive-response generation",
    )
    p.add_argument(
        "--build-top-p",
        type=float,
        default=0.9,
        help="Top-p for contrastive-response generation",
    )
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


# Chọn float16 cho model có tên bắt đầu bằng qwen/, bfloat16 cho trường hợp còn lại.
# Đây là quy tắc mặc định của launcher khi người dùng chọn dtype=auto.
def inferred_dtype(model_name: str) -> str:
    return "float16" if model_name.lower().startswith("qwen/") else "bfloat16"


# Kiểm tra input rồi dựng lệnh chạy pipeline với cấu hình thí nghiệm chính.
# Dry-run chỉ in lệnh; chạy thật kiểm tra API key, lưu launcher_config và gọi subprocess với GPU được chọn.
def main() -> int:
    args = parser().parse_args()
    questions_file = args.questions_file.expanduser().resolve() if args.questions_file is not None else None
    trait_definitions_file = (
        args.trait_definitions_file.expanduser().resolve()
        if args.trait_definitions_file is not None
        else None
    )
    if questions_file is not None and not questions_file.is_file():
        raise FileNotFoundError(questions_file)
    if args.task == "traits":
        if questions_file is None:
            raise ValueError("--questions-file is required for --task traits")
        if trait_definitions_file is None:
            raise ValueError("--trait-definitions-file is required for --task traits")
        if not trait_definitions_file.is_file():
            raise FileNotFoundError(trait_definitions_file)
    root = Path(__file__).resolve().parents[1]
    pipeline = Path("scripts/run_experiment_pipeline.py")
    if not (root / pipeline).is_file():
        raise FileNotFoundError(root / pipeline)

    output_root = args.output_root.expanduser().resolve()
    model_slug = args.model_name.replace("/", "-")
    run_dir = output_root / args.task / model_slug
    vector_dir = output_root / "vectors" / args.task / model_slug / "prompt_last" / "global_plane_mean"
    shared_dir = output_root / "build_data" / args.task / model_slug
    labels = TONE_LABELS if args.task == "tones" else TRAIT_LABELS
    dtype = inferred_dtype(args.model_name) if args.dtype == "auto" else args.dtype

    cmd = [
        args.python_bin,
        str(pipeline),
        "--model_name", args.model_name,
        "--task", args.task,
        "--labels", labels,
        "--split_mode", args.split_mode,
        "--split_seed", str(args.split_seed),
        "--layers", "-6",
        "--eval_layer", "-1",
        "--k_values", "1,2,3",
        "--max_questions", str(args.max_questions),
        "--calibration_samples", str(args.calibration_samples),
        "--max_pairs", str(args.max_pairs),
        "--alpha_min", "1.0",
        "--alpha_max", "32.0",
        "--alpha_iters", "8",
        "--coherence_check_max_new_tokens", "50",
        "--n_generate", str(args.n_generate),
        "--max_new_tokens", str(args.max_new_tokens),
        "--hidden_extraction_batch_size", "8",
        "--methods", "baseline,ksteering,caa,orbit",
        "--judge_enabled",
        "--judge_model", args.judge_model,
        "--judge_coherence_eval_mode", "text_number",
        "--use_definition_judge",
        "--build_filter_enabled",
        "--build_filter_threshold", "60",
        "--build_filter_coherence_threshold", "50",
        "--negatives_mode", "neg_prompt",
        "--negatives_seed", str(args.negative_seed),
        "--shared_build_generation_for_all_methods",
        "--shared_build_use_for_classifiers",
        "--shared_build_generations_dir", str(shared_dir),
        "--angular_vector_dir", str(vector_dir),
        "--angular_auto_build_from_train",
        "--angular_build_dir", str(vector_dir),
        "--angular_build_token_mode", "prompt_last",
        "--angular_build_delta_proj_mode", "plane_mean",
        "--angular_build_delta_proj_scale", "1.0",
        "--caa_build_max_new_tokens", "2048",
        "--caa_build_generation_max_cap", "2048",
        "--angular_build_max_new_tokens", "2048",
        "--angular_build_generation_max_cap", "2048",
        "--angular_plane_mode", "global",
        "--angular_mode", "hybrid",
        "--angular_target_angle_deg", "0",
        "--angular_variants", "adaptive",
        "--angular_positions", "response",
        "--angular_adaptive_mode", "4",
        "--angular_adaptive_threshold", "0",
        "--angular_hybrid_gate_style", "margin",
        "--angular_hybrid_gate_reference", "zero",
        "--angular_hybrid_rotation_gate_mode", "full",
        "--angular_additive_boost",
        "--angular_additive_coeff", "1.0",
        "--angular_calibrate_additive_coeff",
        "--angular_additive_coeff_min", "0.0",
        "--angular_additive_coeff_max", "3.0",
        "--angular_additive_coeff_iters", "8",
        "--angular_additive_coeff_calib_max_new_tokens", "50",
        "--angular_no_compare_boost",
        "--angular_boost_approach", "1",
        "--angular_boost_gate_mode", "adaptive",
        "--angular_boost_fill_fraction", "0.4",
        "--angular_boost_abs_tau", "0.0",
        "--angular_boost_min_gate", "0.0",
        "--angular_boost_clamp_ratio", "0.0",
        "--angular_boost_delta_clamp_mode", "off",
        "--angular_r_typical_method", "pos_mean",
        "--caa_build_generation_source", "local",
        "--angular_build_generation_source", "local",
        "--local_build_do_sample",
        "--local_build_temperature", str(args.build_temperature),
        "--local_build_top_p", str(args.build_top_p),
        "--output_dir", str(run_dir),
        "--hf_device_map", args.device_map,
        "--hf_torch_dtype", dtype,
        "--hf_max_memory_per_gpu_gib", str(args.max_memory_per_gpu_gib),
    ]
    if questions_file is not None:
        cmd.extend(("--questions_file", str(questions_file)))
    if args.task == "tones":
        cmd.append("--use_tone_judge")
    if args.task == "traits":
        cmd.extend((
            "--traits_variants", "all",
            "--trait_definitions_file", str(trait_definitions_file),
        ))
    if args.local_files_only:
        cmd.append("--local_files_only")

    snapshot = {
        "command": cmd,
        "cuda_visible_devices": args.device_ids,
    }
    print(json.dumps(snapshot, indent=2))
    if args.dry_run:
        print("\n" + shlex.join(cmd))
        return 0
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for calibration and judging")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "launcher_config.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.device_ids
    return subprocess.run(cmd, cwd=root, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
