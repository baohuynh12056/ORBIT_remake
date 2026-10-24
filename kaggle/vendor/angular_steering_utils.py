# Vai trò file: Nạp vector ORBIT, dựng mặt phẳng và can thiệp activation qua hook khi model chạy forward/generation.
from typing import Dict, Iterable, Optional, Tuple, List, Set

import os
import math
import torch

from angular_residual_utils import (
    normalize_orbit_extraction_layout,
    orbit_extraction_layout_version,
)

# Enable with: EVAL_DEBUG=1 or STEER_DEBUG=1
# Bật log chẩn đoán từ các biến môi trường EVAL_DEBUG hoặc STEER_DEBUG.
DEBUG = os.getenv("EVAL_DEBUG", "0") == "1" or os.getenv("STEER_DEBUG", "0") == "1"

# Ngưỡng norm để nhận diện trục b2 suy biến sau trực giao hóa.
B2_ORTH_EPS = 1e-6
# Mã nhận diện thuật toán dựng mặt phẳng global từ hướng ứng viên và PCA.
GLOBAL_PLANE_VERSION = "global_max_sim_pca_v1"

# Valid ADD approach names
# Tên các cách steering được hỗ trợ để diễn giải cấu hình can thiệp.
ADD_APPROACHES = {
    1: "rotation",           # Pure rotation, preserves projection magnitude r
    2: "set_r_typical",      # Set to coef*r_typical, no normalize
    3: "set_r_typical_norm", # Set to coef*r_typical, then normalize to preserve ||h||
    4: "max_r_typical",      # Use coef*max(r, r_typical), no normalize
}

# Sequential application modes for multi-trait steering
# Các chế độ kết hợp steering tuần tự giữa nhiều thuộc tính.
SEQUENTIAL_MODES = {
    "full": "Cascade full activations (h1 = steer(h0, p1), h2 = steer(h1, p2), etc.)",
    "perpendicular": "Cascade only perpendicular parts, accumulate steering deltas",
}

# Methods for computing r_typical
# Các cách chọn độ lớn chiếu điển hình từ thống kê contrastive để hiệu chuẩn boost.
R_TYPICAL_METHODS = {
    "norm": "||first_direction|| - magnitude of mean difference",
    "pos_mean": "mean projection of positive samples onto b1",
    "pos_median": "median projection of positive samples onto b1",
    "gap": "pos_proj_mean - neg_proj_mean (trait distinguishability)",
    "pos_p75": "75th percentile of positive projections",
    "pos_p90": "90th percentile of positive projections",
}


# Suy ra tâm và thang đo của gate từ thống kê chiếu positive/negative.
# Trả giá trị dự phòng khi metadata thiếu, để gate có thể xử lý các file vector được hỗ trợ.
def _gate_calibration_from_stats(stats: dict) -> Tuple[Optional[float], float]:
    """Return a class-boundary center and projection scale."""
    if not isinstance(stats, dict):
        return None, 1.0
    required = ("pos_proj_mean", "neg_proj_mean")
    if any(k not in stats for k in required):
        return None, 1.0
    pos_mean = float(stats["pos_proj_mean"])
    neg_mean = float(stats["neg_proj_mean"])
    if not math.isfinite(pos_mean) or not math.isfinite(neg_mean):
        return None, 1.0
    pos_std = max(0.0, float(stats.get("pos_proj_std", 0.0)))
    neg_std = max(0.0, float(stats.get("neg_proj_std", 0.0)))
    pooled_std = math.sqrt(0.5 * (pos_std * pos_std + neg_std * neg_std))
    gap = abs(pos_mean - neg_mean)
    # Use the zero-margin gate when the build distributions are not separated.
    if gap <= 1e-8 and pooled_std <= 1e-8:
        return None, 1.0
    scale = max(pooled_std, 0.05 * gap, 1e-6)
    return 0.5 * (pos_mean + neg_mean), scale


# Tìm module con từ đường dẫn chấm, gồm cả chỉ số trong danh sách layers.
# Dùng gắn hook đúng vị trí module mà file vector mô tả.
def _resolve_module(root, path: str):
    module = root
    for attr in path.split("."):
        if not hasattr(module, attr):
            raise AttributeError(f"Module '{path}' not found on model")
        module = getattr(module, attr)
    return module


# Đọc file vector trên CPU và chuẩn hóa các trường theo từng module.
# Trả mapping hướng/mặt phẳng/thông tin hook cùng r_typical dùng điều khiển cường độ boost.
def load_layernorm_vectors(
    path: str,
    r_typical_method: str = "pos_mean",
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, float]]:
    """Load per-module steering vectors and projection statistics from a .pt file."""
    if r_typical_method not in R_TYPICAL_METHODS:
        raise ValueError(f"Invalid r_typical_method: {r_typical_method}. "
                        f"Must be one of: {list(R_TYPICAL_METHODS.keys())}")
    
    payload = torch.load(path, map_location="cpu")
    vectors = payload.get("vectors")
    if not vectors:
        raise ValueError("Vector file missing 'vectors' section")
    formatted = {}
    r_typical = {}
    
    for module, entry in vectors.items():
        if isinstance(entry, dict):
            if "first_direction" not in entry:
                raise ValueError(f"Module {module} entry missing 'first_direction'")
            first = torch.as_tensor(entry["first_direction"], dtype=torch.float32)
            second = entry.get("second_direction")
            if second is not None:
                second = torch.as_tensor(second, dtype=torch.float32)
            r_typical_vector = entry.get("r_typical_vector")
            if r_typical_vector is not None:
                r_typical_vector = torch.as_tensor(r_typical_vector, dtype=torch.float32)
            pos_proj_vector = entry.get("pos_proj_vector")
            if pos_proj_vector is not None:
                pos_proj_vector = torch.as_tensor(pos_proj_vector, dtype=torch.float32)
            neg_proj_vector = entry.get("neg_proj_vector")
            if neg_proj_vector is not None:
                neg_proj_vector = torch.as_tensor(neg_proj_vector, dtype=torch.float32)
            pos_mean_vector = entry.get("pos_mean_vector")
            if pos_mean_vector is not None:
                pos_mean_vector = torch.as_tensor(pos_mean_vector, dtype=torch.float32)
            neg_mean_vector = entry.get("neg_mean_vector")
            if neg_mean_vector is not None:
                neg_mean_vector = torch.as_tensor(neg_mean_vector, dtype=torch.float32)
            delta_proj_vector = entry.get("delta_proj_vector")
            if delta_proj_vector is not None:
                delta_proj_vector = torch.as_tensor(delta_proj_vector, dtype=torch.float32)
            
            # Get r_typical based on method
            r_stats = entry.get("r_typical_stats", {})
            gate_center, gate_scale = _gate_calibration_from_stats(r_stats)
            hook_point = entry.get("hook_point")
            hook_type = entry.get("hook_type")
            extraction_layout_version = entry.get("extraction_layout_version")
            if r_typical_method == "norm":
                r_val = first.norm().item()
            elif r_typical_method == "pos_mean":
                r_val = r_stats.get("pos_proj_mean", first.norm().item())
            elif r_typical_method == "pos_median":
                r_val = r_stats.get("pos_proj_median", r_stats.get("pos_proj_mean", first.norm().item()))
            elif r_typical_method == "gap":
                r_val = r_stats.get("gap", first.norm().item())
            elif r_typical_method == "pos_p75":
                r_val = r_stats.get("pos_proj_p75", first.norm().item())
            elif r_typical_method == "pos_p90":
                r_val = r_stats.get("pos_proj_p90", first.norm().item())
            else:
                r_val = first.norm().item()
            
            # Ensure r_typical is positive (use absolute value for gap)
            r_typical[module] = abs(r_val) if r_val != 0 else first.norm().item()
            
        else:
            first = torch.as_tensor(entry, dtype=torch.float32)
            second = None
            r_typical_vector = None
            pos_proj_vector = None
            neg_proj_vector = None
            pos_mean_vector = None
            neg_mean_vector = None
            delta_proj_vector = None
            gate_center = None
            gate_scale = 1.0
            hook_point = None
            hook_type = None
            extraction_layout_version = None
            r_typical[module] = first.norm().item()

        formatted[module] = {
            "first_direction": first,
            "second_direction": second,
            "r_typical_vector": r_typical_vector,
            "pos_proj_vector": pos_proj_vector,
            "neg_proj_vector": neg_proj_vector,
            "pos_mean_vector": pos_mean_vector,
            "neg_mean_vector": neg_mean_vector,
            "delta_proj_vector": delta_proj_vector,
            "gate_center": gate_center,
            "gate_scale": gate_scale,
            "r_typical_stats": r_stats if isinstance(entry, dict) else {},
            "global_r_typical_stats": (
                entry.get("global_r_typical_stats", {})
                if isinstance(entry, dict)
                else {}
            ),
            "hook_point": hook_point,
            "hook_type": hook_type,
            "extraction_layout_version": extraction_layout_version,
        }
    
    return formatted, r_typical


# Lấy hướng từ layer tham chiếu rồi dùng chung hướng đó cho các vị trí can thiệp.
# Giữ thông tin từng module và trả cấu hình phục vụ biến thể dùng hướng chung.
def load_layernorm_vectors_single_direction(
    path: str,
    reference_layer: int = 16,
    r_typical_method: str = "pos_mean",
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, float]]:
    """Broadcast one reference-layer direction to every intervention module."""
    all_vectors, all_r_typical = load_layernorm_vectors(path, r_typical_method)

    ref_module = f"model.layers.{reference_layer}.post_attention_layernorm"
    if ref_module not in all_vectors:
        raise ValueError(f"Reference module {ref_module} not found in vectors")

    ref_entry = all_vectors[ref_module]
    ref_first = ref_entry["first_direction"]
    ref_second = ref_entry["second_direction"]
    ref_r_typical_vector = ref_entry.get("r_typical_vector")
    ref_r_typical = all_r_typical[ref_module]

    formatted = {}
    r_typical = {}
    for module in all_vectors.keys():
        formatted[module] = {
            "first_direction": ref_first,
            "second_direction": ref_second,
            "r_typical_vector": ref_r_typical_vector,
            "delta_proj_vector": all_vectors[module].get("delta_proj_vector"),
            "gate_center": ref_entry.get("gate_center"),
            "gate_scale": ref_entry.get("gate_scale", 1.0),
        }
        r_typical[module] = ref_r_typical

    return formatted, r_typical


# Dựng một mặt phẳng chung từ các hướng ứng viên của mọi module trong file vector.
# Chọn b1 theo tiêu chí cấu hình, lấy b2 bằng PC0 rồi trực giao hóa; chiếu vector từng module vào mặt phẳng và broadcast hai trục.
def load_global_layernorm_vectors(
    path: str,
    global_b1_select: str = "max_sim",
    r_typical_method: str = "pos_mean",
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, float]]:
    """Construct and broadcast one cross-layer plane for an attribute."""
    if global_b1_select not in {"max_sim", "max_norm", "mean"}:
        raise ValueError("global_b1_select must be one of: max_sim, max_norm, mean")

    all_vectors, all_r_typical = load_layernorm_vectors(path, r_typical_method)

    # Chuẩn hóa một vector về độ dài 1, với epsilon để tránh chia cho 0.
    # Dùng xây các trục mặt phẳng và tính cosine similarity.
    def _unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        v = v.to(torch.float32)
        return v / v.norm().clamp_min(eps)

    # Tạo khóa sắp xếp module theo layer và vị trí extraction.
    # Giữ thứ tự ứng viên ổn định khi chọn trục của mặt phẳng global.
    def _extraction_order_key(module_name: str):
        parts = module_name.split(".")
        try:
            layer_idx = int(parts[2])
        except Exception:
            return (10**9, 0, module_name)
        is_input_ln = "input_layernorm" in module_name
        block_idx = layer_idx - 1 if is_input_ln else layer_idx
        within = 1 if is_input_ln else 0  # mid first then post
        return (block_idx, within, module_name)

    # Build the candidate axes.
    cand_names = sorted(all_vectors.keys(), key=_extraction_order_key)
    cand_dirs = []
    cand_norms = []
    for name in cand_names:
        d = all_vectors[name].get("first_direction")
        if d is None:
            continue
        d = d.to(torch.float32).cpu()
        if not torch.isfinite(d).all():
            continue
        norm = float(d.norm().item())
        if (not math.isfinite(norm)) or norm < 1e-8:
            continue
        cand_dirs.append((name, _unit(d)))
        cand_norms.append((name, norm))

    if len(cand_dirs) < 3:
        raise ValueError(
            f"Not enough candidate directions to build a global plane (have {len(cand_dirs)})."
        )

    cand_mat = torch.stack([v for _, v in cand_dirs], dim=0).to(torch.float32)  # [M, d] unit

    if global_b1_select == "mean":
        b1 = _unit(cand_mat.mean(dim=0))
        selected_name = "mean"
        selected_mean_cosine = float((cand_mat @ b1).mean().item())
    elif global_b1_select == "max_norm":
        selected_name, _best_norm = max(cand_norms, key=lambda t: t[1])
        idx = [n for n, _ in cand_dirs].index(selected_name)
        b1 = _unit(cand_mat[idx])
        selected_mean_cosine = float((cand_mat @ b1).mean().item())
    else:  # max_sim
        pairwise = cand_mat @ cand_mat.T  # [M,M] cosine since unit
        mean_cos = pairwise.mean(dim=-1)
        idx = int(torch.argmax(mean_cos).item())
        selected_name = cand_dirs[idx][0]
        selected_mean_cosine = float(mean_cos[idx].item())
        b1 = _unit(cand_mat[idx])

    x = cand_mat - cand_mat.mean(dim=0, keepdim=True)
    try:
        _, s, vT = torch.linalg.svd(x, full_matrices=False)
        b2_pc0 = vT[0].contiguous()
    except Exception as e:
        raise RuntimeError("Failed to compute PCA PC0 over candidate directions") from e

    b2 = b2_pc0.to(torch.float32)
    b2 = b2 - (b2 @ b1) * b1
    if b2.norm().item() < B2_ORTH_EPS:
        b2 = None
    else:
        b2 = _unit(b2)

    residual_energy = float((s.square().sum()).item()) if s.numel() else 0.0
    b2_candidate_evr = (
        float((s[0].square() / residual_energy).item())
        if residual_energy > 0.0
        else 0.0
    )
    if cand_mat.shape[0] > 1:
        pairwise = cand_mat @ cand_mat.T
        mask = ~torch.eye(cand_mat.shape[0], dtype=torch.bool)
        mean_pairwise_cosine = float(pairwise[mask].mean().item())
    else:
        mean_pairwise_cosine = 1.0

    # Chiếu một vector lên b1 và b2 của mặt phẳng global.
    # Trả vector trên CPU, hoặc None nếu đầu vào không có vector.
    def _project_global(vector: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if vector is None:
            return None
        value = vector.to(torch.float32)
        projected = (value @ b1) * b1
        if b2 is not None:
            projected = projected + (value @ b2) * b2
        return projected.cpu()

    formatted: Dict[str, Dict[str, torch.Tensor]] = {}
    r_typical: Dict[str, float] = {}
    for module_name, entry in all_vectors.items():
        local_gap = entry.get("first_direction")
        global_delta = _project_global(local_gap)
        pos_mean = entry.get("pos_mean_vector")
        neg_mean = entry.get("neg_mean_vector")
        global_pos_vector = _project_global(
            pos_mean if pos_mean is not None else entry.get("pos_proj_vector")
        )
        global_neg_vector = _project_global(
            neg_mean if neg_mean is not None else entry.get("neg_proj_vector")
        )
        global_stats = entry.get("global_r_typical_stats") or {}
        gate_center, gate_scale = _gate_calibration_from_stats(global_stats)
        formatted[module_name] = {
            "first_direction": b1,
            "second_direction": b2,
            "r_typical_vector": global_pos_vector,
            "pos_proj_vector": global_pos_vector,
            "neg_proj_vector": global_neg_vector,
            "delta_proj_vector": global_delta,
            "gate_center": gate_center,
            "gate_scale": gate_scale,
            "hook_point": entry.get("hook_point"),
            "hook_type": entry.get("hook_type"),
            "extraction_layout_version": entry.get(
                "extraction_layout_version"
            ),
            "plane_mode": GLOBAL_PLANE_VERSION,
            "global_b1_source_module": selected_name,
            "global_b1_mean_cosine": selected_mean_cosine,
            "global_candidate_mean_pairwise_cosine": mean_pairwise_cosine,
            "global_b2_candidate_evr": b2_candidate_evr,
            "global_b2_residual_evr": b2_candidate_evr,
            "global_candidate_count": len(cand_dirs),
        }
        if global_stats and "pos_proj_mean" in global_stats:
            r_typical[module_name] = abs(float(global_stats["pos_proj_mean"]))
        elif global_pos_vector is not None:
            projected_pos = float(global_pos_vector @ b1)
            r_typical[module_name] = abs(projected_pos)
        else:
            r_typical[module_name] = all_r_typical[module_name]

    return formatted, r_typical


# Chọn top_n module theo metric trong metadata vector.
# Dùng giới hạn vị trí can thiệp hoặc chọn các module có hướng thuộc tính mạnh.
def select_top_modules_by_metric(
    all_vectors: Dict[str, Dict[str, torch.Tensor]],
    *,
    metric: str = "b1_norm",
    top_n: int = 0,
) -> Set[str]:
    """Select top-N module names by metric from raw per-module vectors."""
    metric = metric.lower().strip()
    if top_n <= 0:
        return set(all_vectors.keys())
    scores: List[Tuple[str, float]] = []
    for name, entry in all_vectors.items():
        if not isinstance(entry, dict):
            continue
        if metric == "gap":
            stats = entry.get("r_typical_stats") or {}
            val = float(stats.get("gap", 0.0))
        else:  # "b1_norm"
            b1 = entry.get("first_direction")
            val = float(b1.norm().item()) if b1 is not None else 0.0
        scores.append((name, val))
    scores.sort(key=lambda x: x[1], reverse=True)
    return set([n for n, _ in scores[:top_n]])


# Đọc chỉ số decoder layer từ đường dẫn module.
# Trả None nếu đường dẫn không khớp cấu trúc được hỗ trợ.
def _layer_for_module(module_path: str) -> Optional[int]:
    parts = module_path.split(".")
    try:
        layer_idx = int(parts[2])
    except (IndexError, ValueError):
        return None
    return layer_idx


_ANSWER_CAUSAL_TOKEN_RANGE = None


# Cập nhật khoảng token causal cần can thiệp khi chấm câu trả lời.
# Cho phép hook chọn trạng thái dự đoán answer token thay vì sửa mọi token của chuỗi.
def set_answer_causal_token_range(start: Optional[int], end: Optional[int]) -> None:
    """Set the token range used by ``positions='answer_causal'``."""
    global _ANSWER_CAUSAL_TOKEN_RANGE
    if start is None or end is None:
        _ANSWER_CAUSAL_TOKEN_RANGE = None
    else:
        _ANSWER_CAUSAL_TOKEN_RANGE = (int(start), int(end))


# Tạo mặt nạ token cần sửa theo positions: all, prompt hoặc response.
# Có xử lý khoảng answer causal và giai đoạn generation để tránh can thiệp sai vị trí.
def _select_tokens(resid: torch.Tensor, positions: str) -> Optional[torch.Tensor]:
    if positions == "response":
        if resid.shape[1] == 0:
            return None
        return resid[:, -1:, :]
    if positions == "prompt":
        if resid.shape[1] <= 1:
            return None
        return resid[:, :-1, :]
    if positions == "answer_causal":
        if resid.shape[1] == 0:
            return None
        if _ANSWER_CAUSAL_TOKEN_RANGE is None:
            return resid[:, -1:, :]
        start, end = _ANSWER_CAUSAL_TOKEN_RANGE
        start = max(0, min(start, resid.shape[1]))
        end = max(start, min(end, resid.shape[1]))
        if end <= start:
            return None
        return resid[:, start:end, :]
    return resid


# Đưa b1/b2 về cùng device và dtype tính toán của activation rồi trực chuẩn hóa.
# Xử lý b2 suy biến để các phép chiếu/xoay vẫn có hướng cơ sở hợp lệ.
def _prepare_plane_basis(
    h: torch.Tensor,
    b1: torch.Tensor,
    b2: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Prepare orthonormal basis vectors for the steering plane.
    
    Returns:
        b1_unit: unit vector in trait direction
        b2_unit: unit vector orthogonal to b1 (or None if 1D)
        h_float: h converted to float32
    """
    device = h.device
    
    # Normalize b1 to unit vector
    b1 = b1.to(device=device, dtype=torch.float32)
    b1_norm = b1.norm().clamp_min(1e-8)
    b1_unit = b1 / b1_norm
    
    # Prepare b2 (ensure orthogonal to b1 via Gram-Schmidt)
    if b2 is not None:
        b2 = b2.to(device=device, dtype=torch.float32)
        # Remove any component parallel to b1
        b2 = b2 - (b2 @ b1_unit) * b1_unit
        b2_norm = b2.norm()
        if b2_norm < B2_ORTH_EPS:
            # b2 was parallel to b1 or zero, fall back to 1D
            b2_unit = None
        else:
            b2_unit = b2 / b2_norm
    else:
        b2_unit = None
    
    h_float = h.to(torch.float32)
    return b1_unit, b2_unit, h_float


# Tách activation thành phần ngoài mặt phẳng, vector chiếu trong mặt phẳng và độ lớn vector chiếu.
# Trả (h_perp, proj, r) để phép steering sửa phần thuộc tính và giữ phần activation còn lại.
def _compute_projection(
    h_float: torch.Tensor,
    b1_unit: torch.Tensor,
    b2_unit: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute projection of h onto the plane and its magnitude.
    
    Returns:
        h_perp: out-of-plane component (h - proj)
        proj: in-plane projection
        r: magnitude of projection (scalar per position)
    """
    h_dot_b1 = (h_float * b1_unit).sum(dim=-1, keepdim=True)
    
    if b2_unit is None:
        # 1D case
        proj = h_dot_b1 * b1_unit
    else:
        # 2D case
        h_dot_b2 = (h_float * b2_unit).sum(dim=-1, keepdim=True)
        proj = h_dot_b1 * b1_unit + h_dot_b2 * b2_unit
    
    h_perp = h_float - proj
    r = proj.norm(dim=-1, keepdim=True)
    
    return h_perp, proj, r


# Chiếu vector lên không gian sinh bởi b1 và b2 nếu b2 tồn tại.
# Trả phần vector nằm trong mặt phẳng để hướng boost phù hợp với hướng steering.
def _project_vector_to_plane(
    vec: torch.Tensor,
    b1_unit: torch.Tensor,
    b2_unit: Optional[torch.Tensor],
) -> torch.Tensor:
    """Project a single vector onto the plane spanned by b1/b2."""
    v = vec.to(device=b1_unit.device, dtype=torch.float32)
    comp1 = (v * b1_unit).sum(dim=-1, keepdim=True) * b1_unit
    if b2_unit is None:
        return comp1
    comp2 = (v * b2_unit).sum(dim=-1, keepdim=True) * b2_unit
    return comp1 + comp2


# Xoay tọa độ activation trong mặt phẳng b1/b2 theo góc theta.
# Giữ thành phần ngoài mặt phẳng và đưa tensor kết quả về dtype ban đầu.
def _rotate_in_plane(
    h: torch.Tensor,
    b1: torch.Tensor,
    b2: Optional[torch.Tensor],
    theta: float,
) -> torch.Tensor:
    """Rotate the in-plane component to ``theta`` while preserving its magnitude."""
    dtype = h.dtype
    b1_unit, b2_unit, h_float = _prepare_plane_basis(h, b1, b2)
    h_perp, proj, r = _compute_projection(h_float, b1_unit, b2_unit)
    
    # Target direction at angle theta from b1
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    if b2_unit is None:
        v_theta = cos_theta * b1_unit
    else:
        v_theta = cos_theta * b1_unit + sin_theta * b2_unit
    
    # h_steered = (out-of-plane) + (rotated in-plane with same magnitude)
    h_steered = h_perp + r * v_theta
    
    return h_steered.to(dtype)


# Áp dụng một cách steering cộng/xoay dựa trên hai trục và độ lớn điển hình.
# Chọn hướng hoặc vector boost theo approach, hệ số và góc mục tiêu; trả activation đã chỉnh.
def _apply_add_steering(
    h: torch.Tensor,
    b1: torch.Tensor,
    b2: Optional[torch.Tensor],
    r_typical: float,
    coeff: float,
    approach: int,
    r_override: Optional[torch.Tensor] = None,
    r_typical_vector: Optional[torch.Tensor] = None,
    delta_proj_vector: Optional[torch.Tensor] = None,
    additive_coeff: float = 0.0,
    log_label: Optional[str] = None,
    target_angle: float = 0.0,
) -> torch.Tensor:
    """Apply the selected single-attribute steering transformation."""
    dtype = h.dtype
    b1_unit, b2_unit, h_float = _prepare_plane_basis(h, b1, b2)
    if delta_proj_vector is not None and not math.isclose(float(additive_coeff), 0.0, abs_tol=1e-8):
        delta_proj = _project_vector_to_plane(delta_proj_vector, b1_unit, b2_unit)
        h_float = h_float + float(additive_coeff) * delta_proj
        # Use r from the boosted activation (ignore r_override when boosting).
        r_override = None
    h_perp, proj, r = _compute_projection(h_float, b1_unit, b2_unit)
    if r_override is not None:
        r_mag = r_override.to(h_float.dtype).to(h_float.device)
    else:
        r_mag = r
    r = r_mag
    h_norm_orig = h_float.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # Project r_typical_vector onto the plane if provided
    r_typ_vec_proj = None
    r_typ_vec_norm = None
    if r_typical_vector is not None:
        r_typ_vec = r_typical_vector.to(device=h_float.device, dtype=torch.float32)
        comp1 = (r_typ_vec * b1_unit).sum(dim=-1, keepdim=True) * b1_unit
        if b2_unit is not None:
            comp2 = (r_typ_vec * b2_unit).sum(dim=-1, keepdim=True) * b2_unit
            r_typ_vec_proj = comp1 + comp2
        else:
            r_typ_vec_proj = comp1
        r_typ_vec_norm = r_typ_vec_proj.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if r_typ_vec_norm.mean().item() < 1e-8:
            r_typ_vec_proj = None
            r_typ_vec_norm = None
    
    if approach == 1:
        # Rotate toward b1 while preserving in-plane magnitude.
        if b2_unit is None:
            # 1D case: just use b1
            v_target = b1_unit
        elif target_angle == 0.0:
            v_target = b1_unit
        else:
            # Rotate BY target_angle toward b1 (angle 0)
            # First compute current angle in the plane
            h_dot_b1 = (h_float * b1_unit).sum(dim=-1, keepdim=True)
            h_dot_b2 = (h_float * b2_unit).sum(dim=-1, keepdim=True)
            current_angle = torch.atan2(h_dot_b2, h_dot_b1)  # angle from b1 in [-π, π]

            # Rotate toward b1 (angle 0):
            # - If current_angle > 0: subtract target_angle (e.g., 150° - 90° = 60°)
            # - If current_angle < 0: add target_angle (e.g., -150° + 90° = -60°)
            new_angle = torch.where(
                current_angle >= 0,
                current_angle - target_angle,
                current_angle + target_angle
            )

            # Clamp so we don't overshoot past b1 (angle 0)
            # If current > 0 and new < 0, clamp to 0
            # If current < 0 and new > 0, clamp to 0
            new_angle = torch.where(
                (current_angle > 0) & (new_angle < 0),
                torch.zeros_like(new_angle),
                new_angle
            )
            new_angle = torch.where(
                (current_angle < 0) & (new_angle > 0),
                torch.zeros_like(new_angle),
                new_angle
            )

            cos_new = torch.cos(new_angle)
            sin_new = torch.sin(new_angle)
            v_target = cos_new * b1_unit + sin_new * b2_unit

        h_rotated = h_perp + r * v_target
        # Interpolate: h_new = h + coeff * (h_rotated - h)
        h_steered = h_float + coeff * (h_rotated - h_float)
        
    elif approach == 2:
        # Approach 2: Set to coeff * r_typical in b1 direction, no normalize
        # Uses r_typical magnitude (ignores current r), steers toward b1_unit
        r_target = coeff * r_typical
        h_steered = h_perp + r_target * b1_unit

    elif approach == 3:
        # Approach 3: Set to coeff * r_typical in b1 direction, then normalize to preserve ||h||
        # Uses r_typical magnitude (ignores current r), steers toward b1_unit, then normalizes
        r_target = coeff * r_typical
        h_new = h_perp + r_target * b1_unit
        h_new_norm = h_new.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        h_steered = h_new * (h_norm_orig / h_new_norm)
        
    elif approach == 4:
        # Approach 4: Use coeff * max(r, r_typical) in b1 direction, no normalize
        # ALWAYS steer toward b1_unit direction (cleaner for multi-trait)
        # Use r_typical scalar as the magnitude threshold
        r_typical_tensor = torch.as_tensor(r_typical, device=r_mag.device, dtype=r_mag.dtype)
        r_max = torch.maximum(r_mag, r_typical_tensor)
        # Always steer toward b1_unit with magnitude coeff * max(r, r_typical)
        h_steered = h_perp + coeff * r_max * b1_unit
        
    else:
        raise ValueError(f"Invalid add_approach: {approach}. Must be 1, 2, 3, or 4.")
    
    return h_steered.to(dtype)


# Gom các trục của nhiều thuộc tính thành cơ sở trực chuẩn của không gian chung.
# Cơ sở này dùng để tách phần activation liên quan đến các trait trong hybrid steering.
def _build_joint_subspace_basis(
    trait_configs: list,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build ORBIT's effective-rank joint basis once for fixed trait planes."""
    basis_vectors = []
    for config in trait_configs:
        b1, b2 = config[0], config[1]
        target_device = device if device is not None else b1.device
        b1 = b1.to(device=target_device, dtype=torch.float32)
        b1_unit = b1 / b1.norm().clamp_min(1e-8)
        basis_vectors.append(b1_unit)
        if b2 is not None:
            b2 = b2.to(device=target_device, dtype=torch.float32)
            b2_orth = b2 - (b2 @ b1_unit) * b1_unit
            if b2_orth.norm() > 1e-6:
                basis_vectors.append(b2_orth / b2_orth.norm())
    if not basis_vectors:
        raise ValueError("Cannot build a joint subspace without a nonzero b1.")
    V = torch.stack(basis_vectors, dim=1)
    U, S, _ = torch.linalg.svd(V, full_matrices=False)
    threshold = 1e-6 * S[0] if S[0] > 0 else 1e-6
    effective_rank = max(1, int((S > threshold).sum().item()))
    return U[:, :effective_rank].contiguous()


# Điều chỉnh độ lớn tương đối của các vector boost theo chế độ cân bằng.
# Trả cấu hình đã cân bằng để một thuộc tính có vector lớn không lấn át các thuộc tính còn lại.
def _balance_hybrid_delta_vectors(
    add_configs: list,
    mode: str,
) -> list:
    """Balance additive trait scales without changing their directions."""
    mode = str(mode).strip().lower()
    if mode not in {"raw", "unit", "module_median"}:
        raise ValueError(
            "hybrid boost vector balance must be raw, unit, or module_median"
        )
    if mode == "raw" or not add_configs:
        return add_configs
    norms = [
        float(config[4].norm().item())
        for config in add_configs
        if config[4] is not None and float(config[4].norm().item()) > 1e-8
    ]
    if not norms:
        return add_configs
    target_norm = (
        1.0
        if mode == "unit"
        else float(torch.tensor(norms, dtype=torch.float32).median().item())
    )
    balanced = []
    for config in add_configs:
        values = list(config)
        delta = values[4]
        if delta is not None:
            delta_norm = float(delta.norm().item())
            if delta_norm > 1e-8:
                values[4] = delta * (target_norm / delta_norm)
        balanced.append(tuple(values))
    return balanced


# Điều chỉnh đồng thời nhiều trait bằng rotation và additive boost trong không gian chung.
# Gate thích nghi quyết định mức sửa mỗi trait/token; trọng số, clamp và góc mục tiêu kiểm soát độ lớn thay đổi.
def _apply_multitrait_hybrid_steering(
    h: torch.Tensor,
    trait_configs: list,
    coeff: float = 1.0,
    target_angle: float = 0.0,  # Final angle from target (0 = fully at target)
    E_basis: torch.Tensor = None,  # Pre-computed orthonormal basis (optional)
    adaptive_mode: int = 0,
    adaptive_threshold: float = 0.0,
    gate_style: str = "margin",
    soft_temperature: float = 0.5,
    min_gate: float = 0.0,
    additive_coeff: float = 0.0,
    boost_trait_weights: Optional[list] = None,
    boost_gate_mode: str = "adaptive",
    boost_abs_tau: float = 0.0,
    boost_min_gate: float = 0.0,
    boost_clamp_ratio: float = 0.0,
    boost_delta_clamp_mode: str = "r_typical",
    boost_approach: int = 1,
    boost_fill_fraction: float = 1.0,
    gate_reference: str = "zero",
    rotation_gate_mode: str = "full",
    rotation_gate_scale: float = 1.0,
    debug_stats: Optional[dict] = None,
) -> torch.Tensor:
    """Apply gated multi-attribute rotation in the joint subspace."""
    if not trait_configs:
        return h
    normalized_trait_configs = []
    for config in trait_configs:
        if len(config) == 5:
            b1, b2, weight, r_typical, delta_proj = config
            gate_center, gate_scale = None, 1.0
        elif len(config) >= 7:
            b1, b2, weight, r_typical, delta_proj, gate_center, gate_scale = config[:7]
        else:
            raise ValueError(
                "Each trait config must contain 5 or 7 fields."
            )
        normalized_trait_configs.append(
            (b1, b2, weight, r_typical, delta_proj, gate_center, gate_scale)
        )
    trait_configs = normalized_trait_configs
    if boost_trait_weights is None:
        resolved_boost_trait_weights = [
            float(config[2]) for config in trait_configs
        ]
    else:
        if len(boost_trait_weights) != len(trait_configs):
            raise ValueError(
                "boost_trait_weights must match the number of trait configs: "
                f"{len(boost_trait_weights)} != {len(trait_configs)}"
            )
        resolved_boost_trait_weights = [
            float(value) for value in boost_trait_weights
        ]
    if boost_approach not in {1, 2, 3, 4, 5, 6}:
        raise ValueError(f"boost_approach must be one of {{1,2,3,4,5,6}}. Got: {boost_approach}")
    if boost_delta_clamp_mode not in {"r_typical", "absolute", "off"}:
        raise ValueError(
            "boost_delta_clamp_mode must be one of {'r_typical','absolute','off'}. "
            f"Got: {boost_delta_clamp_mode}"
        )
    if gate_reference not in {"zero", "class_boundary"}:
        raise ValueError("gate_reference must be one of {'zero','class_boundary'}.")
    if rotation_gate_mode not in {"full", "severity"}:
        raise ValueError("rotation_gate_mode must be one of {'full','severity'}.")
    if float(rotation_gate_scale) <= 0.0:
        raise ValueError("rotation_gate_scale must be > 0.")

    dtype = h.dtype
    device = h.device
    h_float = h.to(torch.float32)
    batch, seq, hidden = h_float.shape

    # Step 1: Build orthonormal basis E from all (b1, b2) pairs using SVD
    # E spans the trait subspace - includes b2s for "room to rotate"
    if E_basis is None:
        E = _build_joint_subspace_basis(trait_configs, device=device)
    else:
        E = E_basis.to(device=device, dtype=torch.float32)

    m = E.shape[1]  # Subspace dimension

    # Step 2: For each trait, compute sign mismatch and adaptive gate.
    b1_units = []
    b2_units = []
    gate_strengths = []
    boost_gate_strengths = []

    if debug_stats is not None:
        traits = debug_stats.setdefault("traits", [])
        if len(traits) != len(trait_configs):
            traits[:] = []
            for _ in trait_configs:
                traits.append(
                    {
                        "tokens": 0,
                        "gate_sum": 0.0,
                        "gate_nonzero": 0,
                        "boost_gate_sum": 0.0,
                        "boost_gate_nonzero": 0,
                        "delta_l2_sum": 0.0,
                        "delta_l2_count": 0,
                        "delta_l2_max": 0.0,
                    }
                )
        debug_stats["forward_calls"] = int(debug_stats.get("forward_calls", 0)) + 1

    for i, (
        b1,
        b2,
        weight,
        r_typical,
        delta_proj,
        gate_center,
        gate_scale,
    ) in enumerate(trait_configs):
        b1 = b1.to(device=device, dtype=torch.float32)
        b1_unit = b1 / b1.norm().clamp_min(1e-8)
        b1_units.append(b1_unit)

        b2_unit = None
        if b2 is not None:
            b2 = b2.to(device=device, dtype=torch.float32)
            b2 = b2 - (b2 @ b1_unit) * b1_unit
            if b2.norm() > 1e-6:
                b2_unit = b2 / b2.norm()
        b2_units.append(b2_unit)

        # Primary projection for sign mismatch: h · b1  [batch, seq, 1]
        h_dot_b1 = (h_float * b1_unit).sum(dim=-1, keepdim=True)

        # Sign mismatch is always decided on b1 alignment (trait axis).
        desired_sign = 1.0 if weight > 0 else -1.0
        calibrated_b1_proj = h_dot_b1
        if gate_reference == "class_boundary" and gate_center is not None:
            center = float(gate_center)
            scale = max(float(gate_scale), 1e-6)
            calibrated_b1_proj = (h_dot_b1 - center) / scale
        signed_proj_b1 = desired_sign * calibrated_b1_proj
        needs_steering = (signed_proj_b1 < 0)

        # Adaptive feature for gating:
        # mode 2 can gate using b2 alignment if available; otherwise falls back to b1.
        feature_signed_proj = signed_proj_b1
        if adaptive_mode == 2 and b2_unit is not None:
            h_dot_b2 = (h_float * b2_unit).sum(dim=-1, keepdim=True)
            feature_signed_proj = desired_sign * h_dot_b2

        tau = float(adaptive_threshold)
        if gate_style == "binary":
            # Mode 0 accepts any mismatch; other modes apply the threshold.
            if adaptive_mode == 0:
                gate = needs_steering.to(torch.float32)
            else:
                gate = (needs_steering & (feature_signed_proj < -tau)).to(torch.float32)
        elif gate_style == "soft":
            # Smooth gate in (0,1): larger negative margin => stronger steering.
            temp = max(float(soft_temperature), 1e-4)
            margin = (-feature_signed_proj - tau) / temp
            gate = torch.sigmoid(margin) * needs_steering.to(torch.float32)
        else:
            # "margin": linear severity gate (0 when not mismatched/weakly mismatched).
            gate = torch.relu(-feature_signed_proj - tau) * needs_steering.to(torch.float32)

        if min_gate > 0.0:
            # Keep min-gate only on tokens that already need steering.
            # Do not force always-on steering for matched tokens.
            gate = torch.where(
                needs_steering,
                torch.clamp(gate, min=float(min_gate)),
                torch.zeros_like(gate),
            )
        gate_strengths.append(gate)

        # Boost gate can be decoupled from rotation gate so boost can inject
        # missing signal even when rotation gate is intentionally strict.
        if boost_gate_mode == "adaptive":
            boost_gate = gate
        elif boost_gate_mode == "steered_only":
            # Apply boost only to tokens selected for angular steering, binary mask.
            boost_gate = (gate > 0).to(torch.float32)
        elif boost_gate_mode == "unconditional":
            boost_gate = torch.ones_like(gate)
        elif boost_gate_mode == "near_zero_abs":
            tau_boost = max(float(boost_abs_tau), 1e-8)
            # With class-boundary calibration, "near zero" means near the
            # learned decision boundary rather than near raw projection zero.
            strength = (tau_boost - calibrated_b1_proj.abs()) / tau_boost
            boost_gate = torch.clamp(strength, min=0.0, max=1.0).to(torch.float32)
        elif boost_gate_mode == "below_tau":
            tau_boost = max(float(boost_abs_tau), 1e-8)
            strength = (tau_boost - signed_proj_b1) / tau_boost
            boost_gate = torch.clamp(strength, min=0.0, max=1.0).to(torch.float32)
        elif boost_gate_mode == "magnitude_fill":
            boost_gate = torch.ones_like(gate)
        else:
            raise ValueError(
                f"Unknown boost_gate_mode='{boost_gate_mode}'. "
                "Expected one of: adaptive, steered_only, unconditional, near_zero_abs, below_tau, magnitude_fill."
            )

        if boost_min_gate > 0.0:
            boost_gate = torch.clamp(boost_gate, min=float(boost_min_gate), max=1.0)

        boost_gate_strengths.append(boost_gate)

        if debug_stats is not None:
            traits = debug_stats["traits"]
            st = traits[i]
            n_tok = int(gate.numel())
            st["tokens"] += n_tok
            st["gate_sum"] += float(gate.sum().item())
            st["gate_nonzero"] += int((gate > 0).sum().item())
            st["boost_gate_sum"] += float(boost_gate.sum().item())
            st["boost_gate_nonzero"] += int((boost_gate > 0).sum().item())

    # Optional additive boost in the trait subspace (per-token, per-trait).
    # Boost is computed here but applied AFTER rotation to prevent cross-trait
    # interference (boosting trait A should not corrupt rotation of trait B).
    _pending_boost = None
    if not math.isclose(float(additive_coeff), 0.0, abs_tol=1e-8):
        delta_total = torch.zeros_like(h_float)
        for i, (
            _,
            _,
            weight,
            r_typical,
            delta_proj,
            _gate_center,
            _gate_scale,
        ) in enumerate(trait_configs):
            b1_unit = b1_units[i]
            b2_unit = b2_units[i]
            delta_proj_proj = None
            if delta_proj is not None:
                delta_proj_proj = _project_vector_to_plane(delta_proj, b1_unit, b2_unit)
            # Approaches 1/2/4 need delta direction; 3/5/6 can run without it.
            if boost_approach in {1, 2, 4} and delta_proj_proj is None:
                continue
            delta_i = None
            if boost_approach == 3:
                # Magnitude-fill in the b1 direction.
                h_proj_b1 = (h_float * b1_unit).sum(dim=-1, keepdim=True)
                r_typ_val = max(float(abs(r_typical)), 1e-8)
                deficit = torch.relu(r_typ_val - h_proj_b1)  # [batch, seq, 1]
                if boost_clamp_ratio > 0.0:
                    # Cap fill amount per token relative to r_typical.
                    fill_cap = float(boost_clamp_ratio) * r_typ_val
                    deficit = torch.clamp(deficit, max=fill_cap)
                ff = max(0.0, min(1.0, float(boost_fill_fraction)))
                delta_i = boost_gate_strengths[i] * ff * deficit * b1_unit
            elif boost_approach == 4:
                # Approach 4: use magnitude-fill deficit, but move in learned delta direction.
                # This avoids hard snapping to pure b1 (which can cause repetitive phrasing),
                # while still injecting missing trait signal when projection is weak.
                h_proj_b1 = (h_float * b1_unit).sum(dim=-1, keepdim=True)
                r_typ_val = max(float(abs(r_typical)), 1e-8)
                deficit = torch.relu(r_typ_val - h_proj_b1)  # [batch, seq, 1]
                if boost_clamp_ratio > 0.0:
                    deficit = torch.clamp(deficit, max=float(boost_clamp_ratio) * r_typ_val)

                d_norm = float(delta_proj_proj.norm().item())
                if d_norm > 1e-8:
                    delta_dir = delta_proj_proj / d_norm
                else:
                    delta_dir = b1_unit
                ff = max(0.0, min(1.0, float(boost_fill_fraction)))
                delta_i = boost_gate_strengths[i] * ff * deficit * delta_dir
            elif boost_approach == 5:
                # Approach 5: variance-preserving sequence-level mean shift.
                # Compute one shift per sequence and apply it uniformly across tokens.
                h_proj_b1 = (h_float * b1_unit).sum(dim=-1, keepdim=True)  # [batch, seq, 1]
                seq_mean = h_proj_b1.mean(dim=1, keepdim=True)  # [batch, 1, 1]
                r_typ_val = max(float(abs(r_typical)), 1e-8)
                shift = torch.relu(r_typ_val - seq_mean)
                if boost_clamp_ratio > 0.0:
                    shift = torch.clamp(shift, max=float(boost_clamp_ratio) * r_typ_val)
                seq_gate = boost_gate_strengths[i].mean(dim=1, keepdim=True)  # [batch, 1, 1]
                ff = max(0.0, min(1.0, float(boost_fill_fraction)))
                delta_i = seq_gate * ff * shift * b1_unit
            elif boost_approach == 6:
                # Approach 6: radial magnitude boost in trait plane.
                # Increase in-plane magnitude while preserving each token's current in-plane angle.
                h_dot_b1 = (h_float * b1_unit).sum(dim=-1, keepdim=True)
                if b2_unit is not None:
                    h_dot_b2 = (h_float * b2_unit).sum(dim=-1, keepdim=True)
                    proj = h_dot_b1 * b1_unit + h_dot_b2 * b2_unit
                else:
                    proj = h_dot_b1 * b1_unit

                r_raw = proj.norm(dim=-1, keepdim=True)
                proj_dir = proj / r_raw.clamp_min(1e-8)
                near_zero = r_raw < 1e-8
                if near_zero.any():
                    proj_dir = torch.where(near_zero, b1_unit.view(1, 1, -1), proj_dir)

                r_typ_val = max(float(abs(r_typical)), 1e-8)
                deficit = torch.relu(r_typ_val - r_raw)
                if boost_clamp_ratio > 0.0:
                    deficit = torch.clamp(deficit, max=float(boost_clamp_ratio) * r_typ_val)
                ff = max(0.0, min(1.0, float(boost_fill_fraction)))
                delta_i = boost_gate_strengths[i] * ff * deficit * proj_dir
            else:
                # Approaches 1/2: projected pos-neg delta boost.
                if boost_clamp_ratio > 0.0 and boost_delta_clamp_mode != "off":
                    d_norm = float(delta_proj_proj.norm().item())
                    if boost_delta_clamp_mode == "r_typical":
                        max_norm = float(boost_clamp_ratio) * max(float(abs(r_typical)), 1e-8)
                    else:
                        # "absolute": interpret ratio as an absolute norm cap.
                        max_norm = float(boost_clamp_ratio)
                    if d_norm > max_norm:
                        delta_proj_proj = delta_proj_proj * (max_norm / max(d_norm, 1e-8))
                delta_i = boost_gate_strengths[i] * delta_proj_proj
            if delta_i is not None:
                if debug_stats is not None and "traits" in debug_stats and i < len(debug_stats["traits"]):
                    st = debug_stats["traits"][i]
                    delta_l2 = delta_i.norm(dim=-1)
                    st["delta_l2_sum"] += float(delta_l2.sum().item())
                    st["delta_l2_count"] += int(delta_l2.numel())
                    st["delta_l2_max"] = max(float(st.get("delta_l2_max", 0.0)), float(delta_l2.max().item()))
            # Rotation and boost weights are independent.
            delta_total = (
                delta_total
                + resolved_boost_trait_weights[i] * delta_i
            )
        if delta_total is not None:
            _pending_boost = float(additive_coeff) * delta_total

    # Approach 2: boost only; skip angular rotation entirely.
    if boost_approach == 2:
        h_out = h_float
        if _pending_boost is not None:
            h_out = h_out + _pending_boost
        return h_out.to(dtype)

    # Step 3: Compute per-token target from gated trait contributions
    # t_token = normalize(Σ wᵢ·gᵢ·b1ⁱ)
    c_token = torch.zeros_like(h_float)  # [batch, seq, hidden]

    for i, (
        b1,
        b2,
        weight,
        r_typical,
        _delta_proj,
        _gate_center,
        _gate_scale,
    ) in enumerate(trait_configs):
        mask_i = gate_strengths[i]  # [batch, seq, 1]
        b1_unit = b1_units[i]  # [hidden]
        c_token = c_token + mask_i * weight * b1_unit

    c_norm = c_token.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    t_token = c_token / c_norm  # [batch, seq, hidden] - per-token target direction

    # Check if any steering is needed
    no_steering_needed = (c_norm < 1e-6).squeeze(-1)  # [batch, seq]

    # Step 4: Decompose h into trait subspace and perpendicular
    # h_coords = Eᵀh  [batch, seq, m]
    # h_flat: [batch*seq, hidden]
    h_flat = h_float.view(-1, hidden)
    h_coords = h_flat @ E  # [batch*seq, m]

    # h_in = E @ h_coords^T -> [hidden, batch*seq] -> transpose
    h_in_flat = h_coords @ E.T  # [batch*seq, hidden]

    # h_perp = h - h_in (component OUTSIDE trait subspace - PRESERVE THIS)
    h_perp_flat = h_flat - h_in_flat  # [batch*seq, hidden]

    # Magnitude in subspace: μ = ||h_coords||
    mu = h_coords.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # [batch*seq, 1]

    # Step 5: Project target onto subspace and compute new coords
    # t_coords = Eᵀt  [batch, seq, m]
    t_flat = t_token.view(-1, hidden)  # [batch*seq, hidden]
    t_coords = t_flat @ E  # [batch*seq, m]
    t_coords_norm = t_coords.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    t_coords_unit = t_coords / t_coords_norm  # [batch*seq, m]

    # Compute current angle in subspace between h_coords and t_coords
    h_coords_unit = h_coords / mu
    cos_angle = (h_coords_unit * t_coords_unit).sum(dim=-1, keepdim=True).clamp(-1, 1)
    current_angle = torch.acos(cos_angle)  # [batch*seq, 1]

    # Determine new angle (don't rotate away if already closer). ``coeff`` is
    # an angular fraction, not a linear interpolation between two activation
    # vectors: linear interpolation shrinks the in-subspace norm and violates
    # ORBIT's norm-preserving rotation guarantee for every 0 < coeff < 1.
    target_angle_tensor = torch.full_like(current_angle, target_angle)
    full_new_angle = torch.where(
        current_angle > target_angle_tensor,
        target_angle_tensor,  # Rotate to target_angle
        current_angle  # Already closer, don't move
    )
    angular_coeff = max(0.0, min(1.0, float(coeff)))
    if rotation_gate_mode == "severity":
        max_gate = torch.stack(gate_strengths, dim=0).amax(dim=0)
        rotation_strength = torch.clamp(
            max_gate / float(rotation_gate_scale), min=0.0, max=1.0
        ).view(-1, 1) * angular_coeff
    else:
        rotation_strength = torch.full_like(current_angle, angular_coeff)
    new_angle = current_angle - rotation_strength * (
        current_angle - full_new_angle
    )

    # Compute new coords by rotating within subspace
    # Need perpendicular component within subspace for rotation
    h_parallel_coords = (h_coords_unit * t_coords_unit).sum(dim=-1, keepdim=True) * t_coords_unit
    h_perp_coords = h_coords_unit - h_parallel_coords
    h_perp_coords_norm_raw = h_perp_coords.norm(dim=-1, keepdim=True)
    h_perp_coords_unit = h_perp_coords / h_perp_coords_norm_raw.clamp_min(1e-8)
    if m > 1:
        # At exactly 180 degrees the rotation plane is not unique and the
        # perpendicular component is zero. Choose a deterministic orthogonal
        # coordinate direction so intermediate rotations still preserve norm.
        fallback_idx = torch.argmin(t_coords_unit.abs(), dim=-1)
        fallback = torch.nn.functional.one_hot(
            fallback_idx, num_classes=m
        ).to(t_coords_unit.dtype)
        fallback = fallback - (
            fallback * t_coords_unit
        ).sum(dim=-1, keepdim=True) * t_coords_unit
        fallback = fallback / fallback.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        h_perp_coords_unit = torch.where(
            h_perp_coords_norm_raw > 1e-7,
            h_perp_coords_unit,
            fallback,
        )

    # New coords in subspace: rotate from current to new_angle. A genuinely
    # one-dimensional fallback has no continuous rotation path; choose the
    # nearer of its two norm-preserving signs rather than shrinking magnitude.
    if m == 1:
        choose_target = (
            (full_new_angle < current_angle) & (rotation_strength >= 0.5)
        )
        h_coords_new = mu * torch.where(
            choose_target, t_coords_unit, h_coords_unit
        )
    else:
        h_coords_new = mu * (
            torch.cos(new_angle) * t_coords_unit +
            torch.sin(new_angle) * h_perp_coords_unit
        )  # [batch*seq, m]

    # Step 6: Reconstruct h
    # h_in_new = E @ h_coords_new^T
    h_in_new_flat = h_coords_new @ E.T  # [batch*seq, hidden]

    # h_new = h_in_new + h_perp (add back preserved perpendicular)
    h_new_flat = h_in_new_flat + h_perp_flat  # [batch*seq, hidden]

    # Reshape back
    h_new = h_new_flat.view(batch, seq, hidden)

    # Where no steering needed, keep original h
    no_steering_mask = no_steering_needed.unsqueeze(-1)  # [batch, seq, 1]
    h_steered = torch.where(no_steering_mask, h_float, h_new)

    # Apply deferred boost AFTER rotation (prevents cross-trait interference
    # in multi-trait scenarios where boosting one trait corrupts rotation of another).
    if _pending_boost is not None:
        h_steered = h_steered + _pending_boost

    return h_steered.to(dtype)


# Tạo closure kết hợp các cấu hình thêm/bớt thuộc tính thành một forward hook.
# Hook hỗ trợ chọn token, chế độ nhiều trait, rotation, boost và tùy chọn giữ norm.
def _make_combined_hook(
    add_entries: Optional[list],
    remove_entry: Optional[Dict[str, torch.Tensor]],
    coeff: float,
    positions: str,
    zero_override: bool,
    add_approach: int = 1,
    add_r_typicals: Optional[list] = None,
    module_name: str = "",
    parallel_combine: bool = False,
    adaptive_mode: int = 0,
    adaptive_threshold: float = 0.0,
    parallel_method: str = "sum",
    sequential_mode: str = "full",
    target_angle: float = 0.0,
    hybrid_multitrait: bool = False,
    trait_weights: Optional[list] = None,
    boost_trait_weights: Optional[list] = None,
    hybrid_gate_style: str = "margin",
    hybrid_soft_temperature: float = 0.5,
    hybrid_min_gate: float = 0.0,
    final_norm_match: bool = False,
    additive_boost: bool = False,
    additive_coeff: float = 1.0,
    boost_gate_mode: str = "adaptive",
    boost_abs_tau: float = 0.0,
    boost_min_gate: float = 0.0,
    boost_clamp_ratio: float = 0.0,
    boost_delta_clamp_mode: str = "r_typical",
    boost_approach: int = 1,
    boost_fill_fraction: float = 1.0,
    hybrid_gate_reference: str = "zero",
    hybrid_rotation_gate_mode: str = "full",
    hybrid_rotation_gate_scale: float = 1.0,
    hybrid_boost_vector_balance: str = "raw",
    debug_stats: Optional[dict] = None,
):
    """Create a hook that applies the configured ORBIT intervention."""
    positions = positions.lower()
    allowed = {"all", "prompt", "response", "answer_causal"}
    if positions not in allowed:
        raise ValueError(f"positions must be one of {allowed}")

    coeff_scalar = float(coeff)

    # Prepare ADD plane basis vectors for each entry
    add_configs = []
    add_config_source_indices = []
    if add_entries is not None and not zero_override:
        r_typicals = add_r_typicals if add_r_typicals else [1.0] * len(add_entries)
        for i, add_entry in enumerate(add_entries):
            if add_entry is not None:
                b1 = add_entry.get("first_direction")
                b2 = add_entry.get("second_direction")
                r_typ_vec = add_entry.get("r_typical_vector")
                delta_proj_vec = add_entry.get("delta_proj_vector")
                gate_center = add_entry.get("gate_center")
                gate_scale = add_entry.get("gate_scale", 1.0)
                if b1 is None:
                    continue
                b1_norm = b1.norm().item()
                if b1_norm < 1e-8:
                    if DEBUG:
                        print(
                            f"[STEER][{module_name}] skipping ADD#{i+1}: ||b1||={b1_norm:.2e}",
                            flush=True,
                        )
                    continue
                r_typ = r_typicals[i] if i < len(r_typicals) else 1.0
                add_configs.append(
                    (b1, b2, r_typ, r_typ_vec, delta_proj_vec, gate_center, gate_scale)
                )
                add_config_source_indices.append(i)
    
    # Prepare REMOVE plane basis vectors
    remove_b1 = None
    remove_b2 = None
    if remove_entry is not None and not zero_override:
        remove_b1 = remove_entry.get("first_direction")
        remove_b2 = remove_entry.get("second_direction")
        if remove_b1 is not None and remove_b1.norm().item() < 1e-8:
            if DEBUG:
                print(
                    f"[STEER][{module_name}] skipping REMOVE: ||b1||={remove_b1.norm().item():.2e}",
                    flush=True,
                )
            remove_b1 = None

    # Construct the fixed joint basis once per registered hook.
    hybrid_configs_template = []
    hybrid_basis_cpu = None
    if hybrid_multitrait and add_configs:
        add_configs = _balance_hybrid_delta_vectors(
            add_configs, hybrid_boost_vector_balance
        )
        weights = [
            (
                trait_weights[source_index]
                if trait_weights is not None
                and source_index < len(trait_weights)
                else 1.0
            )
            for source_index in add_config_source_indices
        ]
        boost_weights = [
            (
                boost_trait_weights[source_index]
                if boost_trait_weights is not None
                and source_index < len(boost_trait_weights)
                else weights[index]
            )
            for index, source_index in enumerate(add_config_source_indices)
        ]
        hybrid_boost_weights_template = []
        for i, (
            b1,
            b2,
            r_typ,
            _r_typ_vec,
            delta_proj_vec,
            gate_center,
            gate_scale,
        ) in enumerate(add_configs):
            weight = weights[i] if i < len(weights) else 1.0
            if math.isclose(float(weight), 0.0, abs_tol=1e-8):
                continue
            hybrid_configs_template.append(
                (
                    b1,
                    b2,
                    weight,
                    r_typ,
                    delta_proj_vec,
                    gate_center,
                    gate_scale,
                )
            )
            hybrid_boost_weights_template.append(float(boost_weights[i]))
        if hybrid_configs_template:
            hybrid_basis_cpu = _build_joint_subspace_basis(
                hybrid_configs_template, device=torch.device("cpu")
            ).cpu()
    hybrid_basis_device = None

    # Lấy activation output của module, chọn token và áp dụng các phép steering đã cấu hình.
    # Trả output có activation đã sửa, giữ cấu trúc tuple/tensor và các phần output còn lại.
    def hook(module, input, output):
        nonlocal hybrid_basis_device
        # Output hook: we modify the output of the LayerNorm module
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output

        if activation.shape[1] == 0:
            return output

        activation = activation.clone()
        token_slice = _select_tokens(activation, positions)
        if token_slice is None:
            if isinstance(output, tuple):
                return (activation, *output[1:])
            return activation

        # A zero angular coefficient is a valid boost-only configuration.
        # Only skip when neither the rotation nor the additive branch can act.
        boost_is_active = (
            bool(additive_boost)
            and not math.isclose(float(additive_coeff), 0.0, abs_tol=1e-8)
        )
        if math.isclose(coeff_scalar, 0.0, abs_tol=1e-8) and not boost_is_active:
            if isinstance(output, tuple):
                return (activation, *output[1:])
            return activation

        # Compute adaptive masking independently for each trait.
        # Tính mặt nạ can thiệp cho một trait từ tọa độ activation trong mặt phẳng.
        # Chỉ các token thỏa điều kiện adaptive mới nhận thay đổi theo chế độ gate đã chọn.
        def _adaptive_mask_for_trait(
            h_ref: torch.Tensor,
            add_b1: torch.Tensor,
            add_b2: Optional[torch.Tensor],
        ) -> Optional[torch.Tensor]:
            """Return the token mask for one trait."""
            if adaptive_mode == 0:
                return None

            feature_dir = None
            if adaptive_mode in {1, 3, 4}:
                feature_dir = add_b1
            elif adaptive_mode == 2:
                feature_dir = add_b2

            if feature_dir is None:
                return None

            # Use float32 for stable projection; mask multiplies deltas so dtype doesn't matter.
            feature_dir = feature_dir.to(device=h_ref.device, dtype=torch.float32)
            feature_dir = feature_dir / feature_dir.norm().clamp_min(1e-8)
            proj_to_feature = (h_ref.to(torch.float32) * feature_dir).sum(dim=-1, keepdim=True)
            thr = float(adaptive_threshold)
            # Apply the configured projection threshold.
            if adaptive_mode == 4:
                return (proj_to_feature < -thr).to(dtype=h_ref.dtype)
            return (proj_to_feature > thr).to(dtype=h_ref.dtype)

        # Record the pre-steering norm for optional normalization.
        base_norm = token_slice.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        # Hybrid multi-trait mode: per-token adaptive selection
        if hybrid_multitrait and add_configs:
            hybrid_configs = hybrid_configs_template
            if not hybrid_configs:
                return
            if (
                hybrid_basis_device is None
                or hybrid_basis_device.device != token_slice.device
            ):
                hybrid_basis_device = hybrid_basis_cpu.to(
                    device=token_slice.device, dtype=torch.float32
                )

            h_steered = _apply_multitrait_hybrid_steering(
                token_slice,
                hybrid_configs,
                coeff=coeff_scalar,
                target_angle=target_angle,
                E_basis=hybrid_basis_device,
                adaptive_mode=adaptive_mode,
                adaptive_threshold=adaptive_threshold,
                gate_style=hybrid_gate_style,
                soft_temperature=hybrid_soft_temperature,
                min_gate=hybrid_min_gate,
                additive_coeff=(additive_coeff if additive_boost else 0.0),
                boost_trait_weights=hybrid_boost_weights_template,
                boost_gate_mode=boost_gate_mode,
                boost_abs_tau=boost_abs_tau,
                boost_min_gate=boost_min_gate,
                boost_clamp_ratio=boost_clamp_ratio,
                boost_delta_clamp_mode=boost_delta_clamp_mode,
                boost_approach=boost_approach,
                boost_fill_fraction=boost_fill_fraction,
                gate_reference=hybrid_gate_reference,
                rotation_gate_mode=hybrid_rotation_gate_mode,
                rotation_gate_scale=hybrid_rotation_gate_scale,
                debug_stats=debug_stats,
            )
            token_slice.copy_(h_steered)

            # Skip other modes and go to REMOVE
        elif parallel_combine:
            # Compute deltas from original h0 for each trait, then combine once (order-invariant)
            h0 = token_slice.clone()
            deltas = []
            masks_for_count = []
            for add_idx, (
                add_b1,
                add_b2,
                r_typical,
                r_typ_vec,
                delta_proj_vec,
                _gate_center,
                _gate_scale,
            ) in enumerate(add_configs, start=1):
                steered = _apply_add_steering(
                    h0,
                    add_b1,
                    add_b2,
                    r_typical=r_typical,
                    coeff=coeff_scalar,
                    approach=add_approach,
                    r_typical_vector=r_typ_vec,
                    delta_proj_vector=(delta_proj_vec if additive_boost else None),
                    additive_coeff=(additive_coeff if additive_boost else 0.0),
                    log_label=f"{module_name} add#{add_idx} parallel",
                    target_angle=target_angle,
                )
                delta = steered - h0
                mask = _adaptive_mask_for_trait(h0, add_b1, add_b2)
                if mask is not None:
                    delta = mask * delta
                    masks_for_count.append(mask.to(torch.float32))
                else:
                    # Unconditional for this trait: count as active everywhere.
                    masks_for_count.append(torch.ones_like(delta[..., :1], dtype=torch.float32))
                deltas.append(delta)
            if deltas:
                stacked = torch.stack(deltas)
                # Combine deltas:
                # - "sum": sum over traits
                # - "mean": per-token mean over *active* traits (adaptive-aware)
                sum_deltas = stacked.sum(dim=0)
                if parallel_method == "sum":
                    delta_combined = sum_deltas
                else:  # "mean"
                    if masks_for_count:
                        active_counts = torch.stack(masks_for_count).sum(dim=0)  # [batch, tokens, 1]
                        denom = active_counts.clamp_min(1.0).to(dtype=sum_deltas.dtype)
                        delta_combined = sum_deltas / denom
                    else:
                        delta_combined = sum_deltas

                # Note: adaptive_mode masking is already applied per-trait above.
                token_slice.copy_(h0 + delta_combined)
        elif sequential_mode == "perpendicular":
            # Perpendicular sequential mode (decomposition approach):
            # Decompose h0 into orthogonal components across trait planes, steer each, reassemble.
            #
            # h0 = inplane1 + perp1
            # perp1 = inplane2 + perp2
            # perp2 = inplane3 + perp3
            # ...
            #
            # Final result: steered_inplane1 + steered_inplane2 + ... + final_perp
            #
            # This ensures each trait steers only its own subspace without interference.
            current_h = token_slice.clone()
            steered_components = []

            for add_idx, (
                add_b1,
                add_b2,
                r_typical,
                r_typ_vec,
                delta_proj_vec,
                _gate_center,
                _gate_scale,
            ) in enumerate(add_configs, start=1):
                # Prepare orthonormal basis for this plane
                b1_unit = add_b1.to(device=current_h.device, dtype=torch.float32)
                b1_unit = b1_unit / b1_unit.norm().clamp_min(1e-8)
                b2_unit = add_b2.to(device=current_h.device, dtype=torch.float32) if add_b2 is not None else None
                if b2_unit is not None:
                    b2_unit = b2_unit - (b2_unit @ b1_unit) * b1_unit
                    if b2_unit.norm() > B2_ORTH_EPS:
                        b2_unit = b2_unit / b2_unit.norm()
                    else:
                        b2_unit = None

                # Decompose current_h: current_h = inplane + perp
                h_perp, inplane_proj, r = _compute_projection(current_h.to(torch.float32), b1_unit, b2_unit)

                # Apply steering to current_h (which steers the in-plane component)
                steered = _apply_add_steering(
                    current_h,
                    add_b1,
                    add_b2,
                    r_typical=r_typical,
                    coeff=coeff_scalar,
                    approach=add_approach,
                    # Use the current r (from current_h), so later traits reflect earlier changes.
                    r_override=r,
                    r_typical_vector=r_typ_vec,
                    delta_proj_vector=(delta_proj_vec if additive_boost else None),
                    additive_coeff=(additive_coeff if additive_boost else 0.0),
                    log_label=f"{module_name} add#{add_idx} perp_seq",
                    target_angle=target_angle,
                )

                # Extract the steered in-plane component: steered = h_perp + steered_inplane
                # So steered_inplane = steered - h_perp
                steered_inplane = (steered.to(torch.float32) - h_perp).to(current_h.dtype)

                # Apply adaptive mode masking
                mask = _adaptive_mask_for_trait(current_h, add_b1, add_b2)
                if mask is not None:
                    # If masked out, keep original in-plane instead of steered
                    original_inplane = inplane_proj.to(current_h.dtype)
                    steered_inplane = mask * steered_inplane + (1 - mask) * original_inplane

                steered_components.append(steered_inplane)

                # Update current_h to be the perpendicular part for next iteration
                current_h = h_perp.to(current_h.dtype)

            # Final result: sum of steered in-plane components + final perpendicular
            # current_h is now the part orthogonal to ALL planes
            final_perp = current_h
            result = final_perp
            for comp in steered_components:
                result = result + comp
            token_slice.copy_(result)

        else:
            # Full sequential mode (default):
            # Apply steering to h, get h1, then apply next trait to h1, etc.
            for add_idx, (
                add_b1,
                add_b2,
                r_typical,
                r_typ_vec,
                delta_proj_vec,
                _gate_center,
                _gate_scale,
            ) in enumerate(add_configs, start=1):
                # Magnitudes before steering
                b1_unit = add_b1.to(device=token_slice.device, dtype=torch.float32)
                b1_unit = b1_unit / b1_unit.norm().clamp_min(1e-8)
                b2_unit = add_b2.to(device=token_slice.device, dtype=torch.float32) if add_b2 is not None else None
                if b2_unit is not None:
                    b2_unit = b2_unit - (b2_unit @ b1_unit) * b1_unit
                    if b2_unit.norm() > B2_ORTH_EPS:
                        b2_unit = b2_unit / b2_unit.norm()
                    else:
                        b2_unit = None
                _, proj_before, r_before = _compute_projection(token_slice.to(torch.float32), b1_unit, b2_unit)
                h_norm_before = token_slice.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                h_perp_before = (token_slice.to(torch.float32) - proj_before).norm(dim=-1, keepdim=True)

                h_before = token_slice.clone()
                steered = _apply_add_steering(
                    token_slice,
                    add_b1,
                    add_b2,
                    r_typical=r_typical,
                    coeff=coeff_scalar,
                    approach=add_approach,
                    r_typical_vector=r_typ_vec,
                    delta_proj_vector=(delta_proj_vec if additive_boost else None),
                    additive_coeff=(additive_coeff if additive_boost else 0.0),
                    log_label=f"{module_name} add#{add_idx}",
                    target_angle=target_angle,
                )
                delta = steered - h_before

                # Apply adaptive mode masking
                mask = _adaptive_mask_for_trait(h_before, add_b1, add_b2)
                if mask is None:
                    # Unconditional: apply to all tokens
                    token_slice.copy_(steered)
                else:
                    # Conditional: steer only where this trait's projection is negative.
                    token_slice.copy_(h_before + mask * delta)

        # Step 2: Apply REMOVE rotation (rotate away from trait direction)
        # theta_remove = π means opposite to b1 (180° away from trait)
        # REMOVE always uses rotation approach, interpolated by coeff
        if remove_b1 is not None:
            theta_remove = math.pi  # Target: opposite to trait direction
            h_before_remove = token_slice.clone()
            rotated = _rotate_in_plane(token_slice, remove_b1, remove_b2, theta_remove)
            # Scale the change by coeff
            delta_remove = rotated - h_before_remove

            # REMOVE is always unconditional (adaptive_mode only applies to ADD)
            token_slice.copy_(h_before_remove + coeff_scalar * delta_remove)

        # Optional: enforce that the FINAL steered token norms match the ORIGINAL norms.
        # This is applied once at the end (after all ADD and REMOVE), so it does not
        # destroy inter-trait interactions the way per-trait renorming can.
        if final_norm_match:
            cur_norm = token_slice.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            token_slice.mul_(base_norm / cur_norm)

        # Return modified activation (output hook style)
        if isinstance(output, tuple):
            return (activation, *output[1:])
        return activation

    return hook


# Gắn hook vào các normalization/residual site khớp vector và bộ lọc layer/module.
# Kiểm tra metadata layout, cấu hình steering từng vị trí và trả danh sách handle để tháo hook sau generation.
def register_layernorm_additive_hooks(
    model,
    module_vectors: Optional[Dict[str, Dict[str, torch.Tensor]]],
    coeff: float,
    positions: str = "all",
    allowed_layers: Optional[Iterable[int]] = None,
    allowed_modules: Optional[Set[str]] = None,
    zero_override: bool = False,
    remove_module_vectors: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    add_approach: int = 1,
    add_r_typical: Optional[Dict[str, float]] = None,
    add_module_vectors_list: Optional[list] = None,
    add_r_typical_list: Optional[list] = None,
    parallel_combine: bool = False,
    adaptive_mode: int = 0,
    adaptive_threshold: float = 0.0,
    parallel_method: str = "sum",
    sequential_mode: str = "full",
    target_angle: float = 0.0,
    hybrid_multitrait: bool = False,
    trait_weights: Optional[list] = None,
    boost_trait_weights: Optional[list] = None,
    hybrid_gate_style: str = "margin",
    hybrid_soft_temperature: float = 0.5,
    hybrid_min_gate: float = 0.0,
    final_norm_match: bool = False,
    additive_boost: bool = False,
    additive_coeff: float = 1.0,
    additive_boost_layers: Optional[Set[int]] = None,
    boost_gate_mode: str = "adaptive",
    boost_abs_tau: float = 0.0,
    boost_min_gate: float = 0.0,
    boost_clamp_ratio: float = 0.0,
    boost_delta_clamp_mode: str = "r_typical",
    boost_approach: int = 1,
    boost_fill_fraction: float = 1.0,
    hybrid_gate_reference: str = "zero",
    hybrid_rotation_gate_mode: str = "full",
    hybrid_rotation_gate_scale: float = 1.0,
    hybrid_boost_vector_balance: str = "raw",
    debug_stats: Optional[dict] = None,
    intervention_layout: str = "norm",
    require_layout_metadata: bool = False,
):
    """Register ORBIT hooks using the vector extraction layout."""
    intervention_layout = normalize_orbit_extraction_layout(intervention_layout)
    expected_layout_version = orbit_extraction_layout_version(
        intervention_layout
    )
    handles = []
    allowed_set = set(allowed_layers) if allowed_layers is not None else None
    allowed_modules_set = set(allowed_modules) if allowed_modules is not None else None

    # Build list of ADD module_vectors
    if add_module_vectors_list is not None:
        add_vectors_list = add_module_vectors_list
        r_typical_list = add_r_typical_list if add_r_typical_list else [None] * len(add_vectors_list)
    elif module_vectors is not None:
        add_vectors_list = [module_vectors]
        r_typical_list = [add_r_typical]
    else:
        add_vectors_list = []
        r_typical_list = []

    # Use additive_coeff directly. Downscaling by K under-boosts each trait in
    # multi-trait settings (K>1), especially when coeff was calibrated at K=1.
    effective_additive_coeff = additive_coeff
    
    # Collect all module paths from all ADD and REMOVE vectors
    module_keys = set()
    for mv in add_vectors_list:
        if mv:
            module_keys.update(mv.keys())
    if remove_module_vectors:
        module_keys.update(remove_module_vectors.keys())

    for module_path in sorted(module_keys):
        if allowed_modules_set is not None and module_path not in allowed_modules_set:
            continue
        if allowed_set is not None:
            logical_layer = _layer_for_module(module_path)
            if logical_layer is None or logical_layer not in allowed_set:
                continue
        module = _resolve_module(model, module_path)

        # Collect ADD entries for this module
        add_entries = []
        add_r_typicals = []
        add_trait_weights = []
        add_boost_trait_weights = []
        for i, mv in enumerate(add_vectors_list):
            entry = mv.get(module_path) if mv else None
            if entry is not None:
                add_entries.append(entry)
                # Keep weights aligned with the compacted per-module entry
                # list. A direction can be absent at one site (for example a
                # numerically zero class difference); using the original
                # unfiltered weight list would then assign every later
                # direction the preceding direction's weight.
                add_trait_weights.append(
                    trait_weights[i]
                    if trait_weights is not None and i < len(trait_weights)
                    else 1.0
                )
                add_boost_trait_weights.append(
                    boost_trait_weights[i]
                    if boost_trait_weights is not None
                    and i < len(boost_trait_weights)
                    else (
                        trait_weights[i]
                        if trait_weights is not None and i < len(trait_weights)
                        else 1.0
                    )
                )
                # Get r_typical for this module from corresponding dict
                r_typ_dict = r_typical_list[i] if i < len(r_typical_list) else None
                r_typ_val = 1.0
                if r_typ_dict is not None and module_path in r_typ_dict:
                    r_typ_val = r_typ_dict[module_path]
                add_r_typicals.append(r_typ_val)
        
        remove_entry = remove_module_vectors.get(module_path) if remove_module_vectors else None

        if not add_entries and remove_entry is None:
            continue

        entries_to_validate = list(add_entries)
        if remove_entry is not None:
            entries_to_validate.append(remove_entry)
        expected_hook_type = None
        strict_layout = bool(require_layout_metadata) or intervention_layout == "residual"
        for entry in entries_to_validate:
            entry_layout_version = entry.get("extraction_layout_version")
            entry_hook_type = entry.get("hook_type")
            if entry_layout_version is None and not strict_layout:
                # Normalized-site vectors use output hooks.
                entry_layout_version = expected_layout_version
                entry_hook_type = "output"
            if entry_layout_version != expected_layout_version:
                raise ValueError(
                    f"ORBIT vector for {module_path!r} has layout fingerprint "
                    f"{entry_layout_version!r}; requested {intervention_layout!r} "
                    f"({expected_layout_version!r}). Rebuild vectors in the "
                    "requested activation layout."
                )
            if entry_hook_type not in {"pre", "output"}:
                raise ValueError(
                    f"ORBIT vector for {module_path!r} is missing a valid "
                    f"hook_type; got {entry_hook_type!r}. Rebuild the vector."
                )
            if expected_hook_type is None:
                expected_hook_type = entry_hook_type
            elif entry_hook_type != expected_hook_type:
                raise ValueError(
                    f"ORBIT traits disagree on the hook type for "
                    f"{module_path!r}: {expected_hook_type!r} versus "
                    f"{entry_hook_type!r}."
                )

        layout_expected_hook = (
            "pre"
            if intervention_layout == "residual"
            and str(entries_to_validate[0].get("hook_point")) == "resid_mid"
            else "output"
        )
        if expected_hook_type != layout_expected_hook:
            raise ValueError(
                f"ORBIT vector for {module_path!r} declares hook_type="
                f"{expected_hook_type!r}, but hook_point="
                f"{entries_to_validate[0].get('hook_point')!r} in layout="
                f"{intervention_layout!r} requires {layout_expected_hook!r}."
            )

        boost_here = additive_boost
        if boost_here and additive_boost_layers is not None:
            logical_layer = _layer_for_module(module_path)
            if logical_layer is None or logical_layer not in additive_boost_layers:
                boost_here = False

        hook = _make_combined_hook(
            add_entries if add_entries else None,
            remove_entry,
            coeff,
            positions,
            zero_override,
            add_approach=add_approach,
            add_r_typicals=add_r_typicals if add_entries else None,
            module_name=module_path,
            parallel_combine=parallel_combine,
            adaptive_mode=adaptive_mode,
            adaptive_threshold=adaptive_threshold,
            parallel_method=parallel_method,
            sequential_mode=sequential_mode,
            target_angle=target_angle,
            hybrid_multitrait=hybrid_multitrait,
            trait_weights=add_trait_weights if add_entries else None,
            boost_trait_weights=(
                add_boost_trait_weights if add_entries else None
            ),
            hybrid_gate_style=hybrid_gate_style,
            hybrid_soft_temperature=hybrid_soft_temperature,
            hybrid_min_gate=hybrid_min_gate,
            final_norm_match=final_norm_match,
            additive_boost=boost_here,
            additive_coeff=effective_additive_coeff,
            boost_gate_mode=boost_gate_mode,
            boost_abs_tau=boost_abs_tau,
            boost_min_gate=boost_min_gate,
            boost_clamp_ratio=boost_clamp_ratio,
            boost_delta_clamp_mode=boost_delta_clamp_mode,
            boost_approach=boost_approach,
            boost_fill_fraction=boost_fill_fraction,
            hybrid_gate_reference=hybrid_gate_reference,
            hybrid_rotation_gate_mode=hybrid_rotation_gate_mode,
            hybrid_rotation_gate_scale=hybrid_rotation_gate_scale,
            hybrid_boost_vector_balance=hybrid_boost_vector_balance,
            debug_stats=(debug_stats.setdefault(module_path, {}) if debug_stats is not None else None),
        )
        if expected_hook_type == "pre":
            # Dùng phép steering trên hidden state đầu vào thay vì output module.
            # Giữ các input còn lại nguyên vẹn để can thiệp đúng vị trí của layout pre-hook.
            def pre_hook(_module, inputs, *, _hook=hook, _path=module_path):
                if not inputs or not torch.is_tensor(inputs[0]):
                    raise RuntimeError(
                        f"ORBIT residual pre-hook {_path!r} did not receive a "
                        "tensor as its first input."
                    )
                steered = _hook(_module, inputs, inputs[0])
                if not torch.is_tensor(steered):
                    raise RuntimeError(
                        f"ORBIT residual pre-hook {_path!r} returned a "
                        f"non-tensor value of type {type(steered).__name__}."
                    )
                # In a pre-norm decoder (including Llama), the block stores
                # ``residual = hidden_states`` immediately before calling the
                # post-attention norm. Merely returning a replacement norm
                # argument would steer the MLP branch while leaving that skip
                # reference untouched, so it would *not* be an intervention on
                # raw resid_mid. Mutate the shared tensor before the norm runs;
                # this updates both the MLP input and the residual skip. ORBIT
                # evaluation is inference-only, so no backward graph relies on
                # the pre-mutation value.
                with torch.no_grad():
                    inputs[0].copy_(steered)
                return inputs

            handles.append(module.register_forward_pre_hook(pre_hook))
        else:
            handles.append(module.register_forward_hook(hook))
    return handles


# Tháo toàn bộ hook từ danh sách handle.
# Cần gọi sau generation/extraction để lần chạy kế tiếp không bị can thiệp chồng lên nhau.
def remove_hooks(handles: Iterable):
    for handle in handles:
        handle.remove()
