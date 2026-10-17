#!/usr/bin/env python3
# Vai trò file: Chấm lại response đã lưu bằng rubric thuộc tính; tách bước judge khỏi bước load LLM và generation.
"""Score saved ToneBank or TraitFactory responses with an attribute judge."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

from run_experiment_pipeline import (
    TONE_DEFINITIONS,
    TRAIT_EVAL_DEFINITIONS,
    judge_definition_scores_batch,
    judge_target_scores_batch,
)


# Khai báo các tham số dòng lệnh và giá trị mặc định của chương trình.
# Trả về ArgumentParser; bước parse_args ở hàm điều phối mới đọc cấu hình người dùng.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=("tones", "traits"))
    parser.add_argument(
        "--judge-type",
        choices=("target", "detailed", "both"),
        default="target",
        help="Use the paper's target judge, the detailed rubric judge, or both.",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    return parser


# Tách cột targets dạng nhãn phân cách bằng dấu phẩy thành danh sách.
# Bỏ khoảng trắng và mục rỗng trước khi gửi danh sách thuộc tính cho judge.
def parse_targets(value: object) -> list[str]:
    return [target.strip() for target in str(value).split(",") if target.strip()]


# Khởi tạo các cột điểm, response gốc và trạng thái cho loại judge đã chọn.
# Sửa trực tiếp DataFrame; detailed judge có thêm cột giải thích điểm và coherence.
def add_result_columns(frame: pd.DataFrame, judge_type: str) -> None:
    if judge_type == "target":
        defaults = {
            "judge_target_mean_0_10": None,
            "judge_target_coherence_0_100": None,
            "judge_target_scores_json": "{}",
            "judge_target_raw_response": "",
            "judge_target_status": "",
        }
    else:
        defaults = {
            "detailed_target_mean_0_10": None,
            "detailed_coherence_0_100": None,
            "detailed_target_scores_json": "{}",
            "detailed_score_reasoning_json": "{}",
            "detailed_coherence_reasoning": "",
            "detailed_raw_response": "",
            "detailed_status": "",
        }
    for column, default in defaults.items():
        frame[column] = default


# Ghi kết quả target judge vào đúng index của các hàng response.
# Lưu điểm từng nhãn dạng JSON, điểm trung bình, coherence, response gốc và trạng thái.
def store_target_results(
    frame: pd.DataFrame,
    indices: list[int],
    judgments: list[dict],
) -> None:
    for row_index, judgment in zip(indices, judgments):
        frame.at[row_index, "judge_target_mean_0_10"] = judgment.get("target_mean")
        frame.at[row_index, "judge_target_coherence_0_100"] = judgment.get("coherence")
        frame.at[row_index, "judge_target_scores_json"] = json.dumps(
            judgment.get("target_scores", {})
        )
        frame.at[row_index, "judge_target_raw_response"] = judgment.get("__raw_response", "")
        frame.at[row_index, "judge_target_status"] = judgment.get("__status", "")


# Ghi kết quả detailed judge và giải thích vào các hàng tương ứng.
# Sửa trực tiếp DataFrame để có thể lưu cả điểm số và bằng chứng judge dùng chấm.
def store_detailed_results(
    frame: pd.DataFrame,
    indices: list[int],
    judgments: list[dict],
) -> None:
    for row_index, judgment in zip(indices, judgments):
        frame.at[row_index, "detailed_target_mean_0_10"] = judgment.get("target_mean")
        frame.at[row_index, "detailed_coherence_0_100"] = judgment.get("coherence")
        frame.at[row_index, "detailed_target_scores_json"] = json.dumps(
            judgment.get("target_scores", {})
        )
        frame.at[row_index, "detailed_score_reasoning_json"] = json.dumps(
            judgment.get("score_reasoning", {})
        )
        frame.at[row_index, "detailed_coherence_reasoning"] = judgment.get(
            "coherence_reasoning", ""
        )
        frame.at[row_index, "detailed_raw_response"] = judgment.get("__raw_response", "")
        frame.at[row_index, "detailed_status"] = judgment.get("__status", "")


# Tổng hợp điểm theo các cột nhóm có sẵn và xuất CSV.
# Đồng thời tạo bảng dài có một hàng cho mỗi điểm thuộc tính, thuận tiện vẽ biểu đồ và so sánh.
def write_judge_artifacts(frame: pd.DataFrame, output_dir: Path, judge_type: str) -> None:
    if judge_type == "target":
        mean_column = "judge_target_mean_0_10"
        coherence_column = "judge_target_coherence_0_100"
        scores_column = "judge_target_scores_json"
        mean_summary_column = "judge_target_mean_avg"
        coherence_summary_column = "judge_target_coherence_avg"
    else:
        mean_column = "detailed_target_mean_0_10"
        coherence_column = "detailed_coherence_0_100"
        scores_column = "detailed_target_scores_json"
        mean_summary_column = "detailed_target_mean_avg"
        coherence_summary_column = "detailed_coherence_avg"

    group_columns = [column for column in ("k", "method", "targets") if column in frame.columns]
    if not group_columns:
        group_columns = ["targets"]
    summary = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            **{
                "n": ("response", "count"),
                mean_summary_column: (mean_column, "mean"),
                coherence_summary_column: (coherence_column, "mean"),
            }
        )
        .reset_index()
    )
    summary.to_csv(output_dir / f"{judge_type}_summary.csv", index=False)

    long_rows = []
    for row_index, row in frame.iterrows():
        raw_scores = row.get(scores_column, "{}")
        try:
            scores = json.loads(raw_scores) if isinstance(raw_scores, str) else {}
        except json.JSONDecodeError:
            scores = {}
        for target, score in scores.items():
            long_rows.append(
                {
                    "row_index": row_index,
                    "k": row.get("k"),
                    "method": row.get("method"),
                    "targets": row["targets"],
                    "target": target,
                    "score_0_10": score,
                    "coherence_0_100": row[coherence_column],
                }
            )
    columns = (
        "row_index",
        "k",
        "method",
        "targets",
        "target",
        "score_0_10",
        "coherence_0_100",
    )
    pd.DataFrame(long_rows, columns=columns).to_csv(
        output_dir / f"{judge_type}_scores_long.csv",
        index=False,
    )


# Đọc CSV câu trả lời, kiểm tra cột/nhãn và chọn rubric tones hoặc traits.
# Chấm theo từng nhóm targets bằng API, cập nhật kết quả và lưu bảng response cùng các bảng tổng hợp.
async def run() -> int:
    args = build_parser().parse_args()
    if not args.api_key:
        raise RuntimeError("Set OPENAI_API_KEY or pass --api-key")
    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)

    frame = pd.read_csv(args.input_csv)
    required = {"question", "response", "targets"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    judge_types = ("target", "detailed") if args.judge_type == "both" else (args.judge_type,)
    for judge_type in judge_types:
        add_result_columns(frame, judge_type)

    definitions = TONE_DEFINITIONS if args.task == "tones" else TRAIT_EVAL_DEFINITIONS
    client = AsyncOpenAI(api_key=args.api_key)
    for target_text, group in frame.groupby("targets", sort=False, dropna=False):
        targets = parse_targets(target_text)
        if not targets:
            continue
        unknown = sorted(set(targets).difference(definitions))
        if unknown:
            raise ValueError(f"Unknown targets for {args.task}: {', '.join(unknown)}")
        questions = group["question"].fillna("").astype(str).tolist()
        answers = group["response"].fillna("").astype(str).tolist()
        indices = group.index.tolist()
        if "target" in judge_types:
            target_results = await judge_target_scores_batch(
                client=client,
                model_name=args.model,
                questions=questions,
                answers=answers,
                targets=targets,
                definition_map=definitions,
            )
            store_target_results(frame, indices, target_results)
        if "detailed" in judge_types:
            detailed_results = await judge_definition_scores_batch(
                client=client,
                model_name=args.model,
                questions=questions,
                answers=answers,
                targets=targets,
                definition_map=definitions,
            )
            store_detailed_results(frame, indices, detailed_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "responses_scored.csv", index=False)

    for judge_type in judge_types:
        write_judge_artifacts(frame, args.output_dir, judge_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
