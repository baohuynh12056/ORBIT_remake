# Vai trò file: Nạp câu hỏi từ Hugging Face/JSONL và dựng dataset có instruction theo task.
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from datasets import load_dataset

from .tasks import (
    configure_trait_definitions,
    debates_prompts,
    tones_prompts,
    traits_prompts,
)


# Đọc all hoặc danh sách chỉ số biến thể prompt TraitFactory trong khoảng 0–4.
# Loại trùng, sắp xếp và dùng biến thể 0 khi danh sách rỗng.
def _parse_trait_variants(spec: str) -> list[int]:
    s = str(spec).strip().lower()
    if s == "all":
        return [0, 1, 2, 3, 4]
    out = sorted({int(x.strip()) for x in str(spec).split(",") if x.strip()})
    if not out:
        return [0]
    for v in out:
        if v < 0 or v > 4:
            raise ValueError(f"Invalid traits variant index {v}. Supported range is 0..4.")
    return out


# Đọc file JSONL và giữ các hàng có trường text.
# Trả danh sách text/id, tự tạo ID khi thiếu; các trường khác không được chuyển vào loader output.
def _load_local_questions(jsonl_path: Path) -> list[dict]:
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj:
                continue
            rows.append(
                {
                    "text": obj["text"],
                    "id": str(obj.get("id", len(rows))),
                }
            )
    return rows


# Nạp câu hỏi và ghép instruction theo task tones, debates hoặc traits.
# Trả (dataset có nhãn, danh sách nhãn, câu hỏi gốc); pipeline chính lấy câu hỏi gốc rồi tự xây dữ liệu training.
def load_task(
    task: str,
    traits_variants: str = "0",
    questions_path: str | Path | None = None,
    trait_definitions_path: str | Path | None = None,
):
    external_questions = Path(questions_path).expanduser().resolve() if questions_path else None
    if external_questions is not None and not external_questions.is_file():
        raise FileNotFoundError(f"Missing questions file: {external_questions}")
    if task == "tones":
        if external_questions is not None:
            rows = _load_local_questions(external_questions)
        else:
            ds = load_dataset("withmartian/tone_agnostic_questions", split="train")
            rows = [{"text": r["text"], "id": r["id"]} for r in ds]
        steered_prompts = tones_prompts()
        unique_labels = sorted({t for t, _ in steered_prompts})
        dataset = []
        for row in rows:
            q_text, q_id = row["text"], row["id"]
            for lbl, sys_prompt in steered_prompts:
                dataset.append(
                    {
                        "id": f"{q_id}_{lbl}",
                        "original_question": q_text,
                        "text": f"{sys_prompt}\n{q_text}",
                        "label": lbl,
                    }
                )
        eval_prompts = [r["text"] for r in rows]
        return dataset, unique_labels, eval_prompts
    if task == "debates":
        ds = load_dataset("Narmeen07/debate_style_agnostic_questions", split="train")
        steered_prompts = debates_prompts()
        unique_labels = sorted({t for t, _ in steered_prompts})
        dataset = []
        for row in ds:
            q_text, q_id = row["text"], row.get("id", None) or str(hash(row["text"]))
            for lbl, sys_prompt in steered_prompts:
                dataset.append(
                    {
                        "id": f"{q_id}_{lbl}",
                        "original_question": q_text,
                        "text": f"{sys_prompt}\n{q_text}",
                        "label": lbl,
                    }
                )
        eval_prompts = list(ds["text"])
        return dataset, unique_labels, eval_prompts
    if task == "traits":
        if external_questions is None:
            raise FileNotFoundError(
                "TraitFactory questions are external to the code package. "
                "Pass --questions_file /path/to/questions.jsonl."
            )
        if not trait_definitions_path:
            raise FileNotFoundError(
                "TraitFactory prompt definitions are external to the code package. "
                "Pass --trait_definitions_file /path/to/trait_definitions.py."
            )
        configure_trait_definitions(trait_definitions_path)
        rows = _load_local_questions(external_questions)

        variant_ids = _parse_trait_variants(traits_variants)
        steered_prompts = []
        for vi in variant_ids:
            steered_prompts.extend([(lbl, prompt, vi) for lbl, prompt in traits_prompts(variant=vi, polarity="pos")])
        unique_labels = sorted({lbl for lbl, _, _ in steered_prompts})

        dataset = []
        for row in rows:
            q_text, q_id = row["text"], row["id"]
            for lbl, sys_prompt, vi in steered_prompts:
                dataset.append(
                    {
                        "id": f"{q_id}_{lbl}_v{vi}",
                        "original_question": q_text,
                        "text": f"{sys_prompt}\n{q_text}",
                        "label": lbl,
                    }
                )
        eval_prompts = [r["text"] for r in rows]
        return dataset, unique_labels, eval_prompts
    raise ValueError(f"Unknown task {task}")
