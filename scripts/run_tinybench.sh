#!/usr/bin/env bash
# Vai trò file: chạy lm-eval cho baseline và ORBIT trên mọi tổ hợp thuộc tính K=1,2,3.
# Mỗi tổ hợp dùng endpoint local nạp vector và hệ số calibration từ thí nghiệm chính.
set -euo pipefail

# In cách gọi script và các tùy chọn cần cung cấp.
# Hàm chỉ hiển thị hướng dẫn, không nạp model hay chạy benchmark.
usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_tinybench.sh \
    --model-name MODEL \
    --vector-dir DIR \
    --calibration-csv FILE \
    --labels LABEL_1,LABEL_2,... \
    [--k-values 1,2,3] \
    [--output-dir DIR] \
    [--tasks TASK_1,TASK_2,...] \
    [--gpu ID] \
    [--port PORT] \
    [--dry-run]

Runs baseline and ORBIT on every unordered label combination for each K.
GSM8K is scored with math_verify by its task definition.
EOF
}

# Chạy từ root của repo để import module và tìm cấu hình task bằng đường dẫn tương đối.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME=""
VECTOR_DIR=""
CALIBRATION_CSV=""
LABELS=""
K_VALUES="1,2,3"
OUTPUT_DIR="outputs/tinybench"
TASKS="tinyMMLU_gen,tinyHellaswag_gen,tinyArc_gen,tinyGSM8k_gen,tinyTruthfulQA_gen,tinyWinogrande_gen"
GPU="0"
PORT="9910"
DRY_RUN=0
ENDPOINT_PID=""

# Dừng tiến trình endpoint của tổ hợp hiện tại và thu hồi trạng thái PID.
# Được gọi khi xong tổ hợp và qua trap EXIT để không để server chạy sau khi script kết thúc.
cleanup_endpoint() {
  if [[ -n "$ENDPOINT_PID" ]]; then
    kill "$ENDPOINT_PID" >/dev/null 2>&1 || true
    wait "$ENDPOINT_PID" >/dev/null 2>&1 || true
    ENDPOINT_PID=""
  fi
}
trap cleanup_endpoint EXIT

# Đọc tham số CLI; shift bỏ các đối số đã xử lý khỏi danh sách.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --vector-dir) VECTOR_DIR="$2"; shift 2 ;;
    --calibration-csv) CALIBRATION_CSV="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --k-values) K_VALUES="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

# Kiểm tra cấu hình bắt buộc trước khi khởi động model hoặc tạo output.
for value in MODEL_NAME VECTOR_DIR CALIBRATION_CSV LABELS; do
  if [[ -z "${!value}" ]]; then
    echo "Missing required argument for $value" >&2
    usage
    exit 1
  fi
done

if [[ "$DRY_RUN" == "0" ]]; then
  [[ -d "$VECTOR_DIR" ]] || { echo "Vector directory not found: $VECTOR_DIR" >&2; exit 1; }
  [[ -f "$CALIBRATION_CSV" ]] || { echo "Calibration file not found: $CALIBRATION_CSV" >&2; exit 1; }
fi

# Dùng executable lm_eval nếu có; nếu không thì gọi module trong Python đã chọn.
if command -v lm_eval >/dev/null 2>&1; then
  LM_EVAL_CMD=(lm_eval)
else
  LM_EVAL_CMD=("$PYTHON_BIN" -m lm_eval)
fi

IFS=',' read -r -a LABEL_ARRAY <<<"$LABELS"
IFS=',' read -r -a TASK_ARRAY <<<"$TASKS"

# In từng tổ hợp K nhãn, không xét thứ tự và không chọn trùng nhãn.
# Chỉ hỗ trợ K=1,2,3; mỗi dòng là danh sách target phân cách bằng dấu phẩy.
combinations() {
  local k="$1"
  local n="${#LABEL_ARRAY[@]}"
  local i j m
  if [[ "$k" == "1" ]]; then
    for ((i=0; i<n; i++)); do echo "${LABEL_ARRAY[$i]}"; done
  elif [[ "$k" == "2" ]]; then
    for ((i=0; i<n; i++)); do
      for ((j=i+1; j<n; j++)); do echo "${LABEL_ARRAY[$i]},${LABEL_ARRAY[$j]}"; done
    done
  elif [[ "$k" == "3" ]]; then
    for ((i=0; i<n; i++)); do
      for ((j=i+1; j<n; j++)); do
        for ((m=j+1; m<n; m++)); do echo "${LABEL_ARRAY[$i]},${LABEL_ARRAY[$j]},${LABEL_ARRAY[$m]}"; done
      done
    done
  else
    echo "Unsupported K: $k" >&2
    return 1
  fi
}

# Đánh giá một tổ hợp target: khởi động endpoint, chờ health rồi chạy mọi task ở hai mode.
# Lưu output theo K/targets/task/mode và dừng endpoint khi hoàn thành; dry-run chỉ in cấu hình.
run_combination() {
  local targets="$1"
  local k slug task mode out_dir
  k="$(awk -F',' '{print NF}' <<<"$targets")"
  slug="${targets//,/-}"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "K=$k targets=$targets tasks=$TASKS modes=baseline,orbit"
    return
  fi

  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/tinybench_endpoint.py \
    --model_name "$MODEL_NAME" \
    --vector_dir "$VECTOR_DIR" \
    --calib_coeff_csv "$CALIBRATION_CSV" \
    --targets "$targets" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --apply_chat_template \
    >"$OUTPUT_DIR/endpoint.log" 2>&1 &
  ENDPOINT_PID=$!

  # Chờ model và vector nạp xong; health thất bại sau vòng chờ sẽ làm script dừng.
  for _ in {1..120}; do
    curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null

  # Gửi generation qua HTTP để cùng model được dùng cho baseline và ORBIT của tổ hợp này.
  for task in "${TASK_ARRAY[@]}"; do
    for mode in baseline orbit; do
      out_dir="$OUTPUT_DIR/k${k}/${slug}/${task}/${mode}"
      mkdir -p "$out_dir"
      "${LM_EVAL_CMD[@]}" \
        --model local-completions \
        --tasks "$task" \
        --include_path "$ROOT_DIR/tasks/tinyBenchmarks_gen" \
        --batch_size 1 \
        --model_args "model=${MODEL_NAME},base_url=http://127.0.0.1:${PORT}/orbit/${mode},num_concurrent=1,max_retries=3,tokenized_requests=False,max_gen_toks=4096,max_length=4096" \
        --output_path "$out_dir" \
        --log_samples
    done
  done
  cleanup_endpoint
}

if [[ "$DRY_RUN" == "0" ]]; then
  mkdir -p "$OUTPUT_DIR"
fi
# Duyệt từng K, đọc các tổ hợp do combinations xuất ra rồi chạy lần lượt.
for k in ${K_VALUES//,/ }; do
  while IFS= read -r targets; do
    [[ -n "$targets" ]] && run_combination "$targets"
  done < <(combinations "$k")
done
