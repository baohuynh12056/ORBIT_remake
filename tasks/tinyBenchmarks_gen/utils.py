# Vai trò file: Định dạng câu hỏi trắc nghiệm, tiền xử lý dataset và chấm toán cho các task TinyBench dạng generation.
"""Helper functions for generative tinyBenchmarks task configs."""

import re
from typing import Any, Dict


# Đổi một mẫu ARC thành prompt chứa câu hỏi và các lựa chọn có nhãn.
# Yêu cầu model chỉ trả chữ cái để exact-match chấm đáp án.
def arc_doc_to_text(doc: Dict[str, Any]) -> str:
    choices = doc["choices"]
    labels = choices["label"]  # e.g. ['A','B','C','D']
    texts = choices["text"]
    lines = [f"Question: {doc['question'].strip()}"]
    for lbl, txt in zip(labels, texts):
        lines.append(f"{lbl}. {txt}")
    lines.append("Answer with just the letter:")
    return "\n".join(lines)


# Làm sạch văn bản HellaSwag bằng cách bỏ marker, nội dung trong ngoặc vuông và khoảng trắng dư.
# Dùng chung cho ngữ cảnh và các phương án kết thúc.
def _preprocess_hellaswag(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


# Áp dụng tiền xử lý cho toàn bộ dataset HellaSwag qua dataset.map.
# Trả dataset bổ sung query và choices_text để template generation sử dụng.
def hellaswag_process_docs(dataset):
    # Ghép activity_label, ctx_a và ctx_b thành ngữ cảnh đã làm sạch.
    # Ghi query và choices_text vào một mẫu rồi trả mẫu đã cập nhật.
    def _process(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        doc["query"] = _preprocess_hellaswag(doc["activity_label"] + ": " + ctx)
        doc["choices_text"] = [_preprocess_hellaswag(e) for e in doc["endings"]]
        return doc
    return dataset.map(_process)


# Định dạng ngữ cảnh và các kết thúc HellaSwag thành lựa chọn A, B, C, D.
# Yêu cầu chỉ trả chữ cái để bộ lọc benchmark lấy đáp án.
def hellaswag_doc_to_text(doc: Dict[str, Any]) -> str:
    choices = doc.get("choices_text", doc.get("endings", []))
    lines = [doc.get("query", doc.get("ctx", "")).strip()]
    for i, c in enumerate(choices):
        lines.append(f"{chr(65 + i)}. {c.strip()}")
    lines.append("Answer with just the letter:")
    return "\n".join(lines)


# Định dạng câu có chỗ trống và hai lựa chọn WinoGrande.
# Yêu cầu trả số 1 hoặc 2 thay vì toàn bộ câu đã điền.
def winogrande_doc_to_text(doc: Dict[str, Any]) -> str:
    sentence = doc["sentence"]
    return (
        f"{sentence.strip()}\n"
        f"1. {doc['option1']}\n"
        f"2. {doc['option2']}\n"
        f"Answer with just the number (1 or 2):"
    )


# Tạo prompt trắc nghiệm từ question và choices của mc1_targets.
# Đánh nhãn các phương án bằng chữ cái để chấm lựa chọn đúng duy nhất.
def truthfulqa_doc_to_text(doc: Dict[str, Any]) -> str:
    question = doc["question"].strip()
    choices = doc["mc1_targets"]["choices"]
    lines = [f"Q: {question}"]
    for index, choice in enumerate(choices):
        lines.append(f"{chr(65 + index)}. {choice.strip()}")
    lines.append("Answer with just the letter:")
    return "\n".join(lines)


# Tìm phương án có label=1 trong mc1_targets.
# Trả chữ cái tương ứng làm đáp án chuẩn cho exact-match.
def truthfulqa_doc_to_target(doc: Dict[str, Any]) -> str:
    labels = doc["mc1_targets"]["labels"]
    return chr(65 + labels.index(1))


# Parse đáp án chuẩn và response đầu tiên rồi so sánh bằng math_verify.
# Trả metric 1 nếu đúng và 0 nếu sai hoặc parse/verify phát sinh lỗi.
def gsm8k_process_results(doc: Dict[str, Any], results):
    from math_verify import parse, verify

    try:
        correct = verify(parse(str(doc["answer"])), parse(str(results[0])))
    except Exception:
        correct = False
    return {"math_verify": int(bool(correct))}
