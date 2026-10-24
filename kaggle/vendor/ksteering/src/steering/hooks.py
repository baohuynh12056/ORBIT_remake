# Vai trò file: Forward hook K-Steering/CAA, dùng gắn phép sửa activation vào decoder layer trong quá trình generation.
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


# Tạo forward hook K-Steering từ classifier, các nhãn mục tiêu/cần tránh và alpha.
# Mỗi lần forward dùng gradient theo hidden state để giảm steering loss rồi trả activation đã sửa.
def get_gradient_hook(
    *,
    steer_module,
    target_labels: List[int] | torch.Tensor | None = None,
    avoid_labels: List[int] | torch.Tensor | None = None,
    alpha: float = 1.0,
    steps: int = 1,
    step_size_decay: float = 1.0,
):
    target_labels = torch.as_tensor(target_labels or [], device=steer_module.device)
    avoid_labels = torch.as_tensor(avoid_labels or [], device=steer_module.device)

    # Lấy hidden state output, tính gradient loss thuộc tính và cập nhật activation.
    # Đưa classifier về cùng GPU với layer, giữ dtype/device và cấu trúc output để hỗ trợ model phân bố nhiều GPU.
    @torch.inference_mode(False)
    def fwd_hook(module, inp, out):
        # Keep return dtype/device aligned with the model stream. For decoder
        # layers, inp[0] is the incoming hidden states tensor.
        stream_ref = inp[0] if isinstance(inp, tuple) and len(inp) > 0 and torch.is_tensor(inp[0]) else None

        # Support both tuple outputs and tensor outputs from decoder layers
        if isinstance(out, tuple):
            h_out = out[0]
            rest = out[1:]
        else:
            h_out = out
            rest = None

        out_dtype = stream_ref.dtype if stream_ref is not None else h_out.dtype
        out_device = stream_ref.device if stream_ref is not None else h_out.device

        B, S, D = h_out.shape
        h_current = h_out.reshape(-1, D).float()

        # In model-parallel runs, decoder layers may execute on a different GPU
        # than steer_module.classifier was trained on. Keep classifier and label
        # tensors colocated with the current hidden state device.
        cls_dev = next(steer_module.classifier.parameters()).device
        if cls_dev != h_current.device:
            steer_module.classifier.to(h_current.device)
            steer_module.device = str(h_current.device)
        local_target_labels = target_labels.to(h_current.device)
        local_avoid_labels = avoid_labels.to(h_current.device)

        for step in range(steps):
            h_step = h_current.clone()
            h_step.requires_grad_(True)
            logits = steer_module.classifier(h_step)
            logits = logits.view(B, S, -1).mean(dim=1)
            loss_vec = _compute_steering_loss(
                logits, target_idx=local_target_labels, avoid_idx=local_avoid_labels
            )
            if loss_vec.numel() > 0:
                grad = torch.autograd.grad(
                    outputs=loss_vec,
                    inputs=h_step,
                    grad_outputs=torch.ones_like(loss_vec),
                    retain_graph=False,
                    create_graph=False,
                )[0]
                current_alpha = alpha * (step_size_decay ** step)
                grad = grad.view(B * S, D)
                h_current = (h_step - current_alpha * grad).detach()
            else:
                h_current = h_step.detach()
        h_new = h_current.reshape(B, S, D).to(device=out_device, dtype=out_dtype)
        if rest is None:
            return h_new
        return (h_new,) + rest

    return fwd_hook


# Tính loss bằng logit trung bình của nhãn cần tránh trừ logit của nhãn mục tiêu.
# Giảm loss sẽ tăng thuộc tính mong muốn và giảm thuộc tính cần tránh; trả về loss cho từng mẫu.
def _compute_steering_loss(
    logits: torch.Tensor,
    *,
    target_idx,
    avoid_idx,
) -> torch.Tensor:
    if not torch.is_tensor(target_idx):
        target_idx = torch.as_tensor(target_idx, device=logits.device)
    else:
        target_idx = target_idx.to(logits.device)
    if not torch.is_tensor(avoid_idx):
        avoid_idx = torch.as_tensor(avoid_idx, device=logits.device)
    else:
        avoid_idx = avoid_idx.to(logits.device)
    B, _ = logits.shape
    avoid_term = logits[:, avoid_idx].mean(dim=1) if avoid_idx.numel() > 0 else torch.zeros(B, device=logits.device)
    target_term = logits[:, target_idx].mean(dim=1) if target_idx.numel() > 0 else torch.zeros(B, device=logits.device)
    return avoid_term - target_term


# Tạo forward hook CAA thực hiện h_mới = h + alpha * vector.
# Vector được chuyển về device/dtype của hidden state trong mỗi lần hook chạy.
def get_additive_hook(vector: torch.Tensor | np.ndarray, *, alpha: float = 1.0):
    if not torch.is_tensor(vector):
        vector = torch.as_tensor(vector, dtype=torch.float32)

    # Cộng vector steering vào hidden state output của module.
    # Giữ các thành phần còn lại nếu output là tuple, hoặc trả tensor mới nếu output là tensor.
    def fwd_hook(module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            rest = out[1:]
            v = vector.to(device=h.device, dtype=h.dtype)
            return (h + alpha * v,) + rest
        else:
            h = out
            v = vector.to(device=h.device, dtype=h.dtype)
            return h + alpha * v

    return fwd_hook

