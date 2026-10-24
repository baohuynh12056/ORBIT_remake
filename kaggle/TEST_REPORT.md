# Báo cáo kiểm chứng bàn giao

## Sửa bundle legacy lồng nhau — 23/10/2026

Kaggle đã giải nén ZIP legacy bên trong release paper, làm auto-discovery thấy
hai bundle và có thể làm sai file manifest. Bundle phát hành mới không chứa
legacy ZIP; bản legacy vẫn giữ riêng trong repo. Discovery chọn bundle ngoài
cùng khi có archive lồng nhau, vẫn báo lỗi nếu hai Code Dataset riêng trùng.
Đã chạy 7 tests stdlib bootstrap, gồm case nested legacy mới. Cần upload Code
Dataset mới; chỉ đổi đường dẫn không sửa được archive bị Kaggle biến đổi.

## Workflow paper v2 — kiểm tra 23/10/2026

Release hiện tại có **6 notebooks**. Bỏ judge/filter build, sinh POS/NEG bằng
system prompt và tối đa 96 tokens; thu ORBIT prompt_last ngay trong generation
prefill. CAA/K-Steer/eval vẫn replay raw prompt+response đúng feature repo trong
cùng worker; không tuyên bố không có thêm forward nào. Artifact CPU không load
model; calibration và final Qwen judge còn giữ nguyên. Release 7 notebooks
filtered trước đó được lưu riêng trong legacy/orbit_kaggle_repo_filtered_bundle.zip.
Các mục ngày cũ bên dưới là lịch sử kiểm chứng release trước.

Môi trường kiểm tra mới `/tmp/orbit-paper-cpu`: Python 3.12.3,
torch 2.6.0+cpu, torchvision 0.21.0+cpu, transformers 5.20.1 và runtime lock.
`pip check` không có dependency hỏng. **41 tests CPU đã qua**, gồm 36 tests
contract/bootstrap/runtime/filesystem hiện có và 5 tests fused workflow mới:

* Tiny Llama và Qwen thật (random weights, không tải pretrained weights):
  activation ở prefill system prompt khớp source collect(system_prompts) tại
  đủ 12 LayerNorm sites của 6 blocks với atol 2e-6; hook được gỡ sau generation.
* Cleanup hook khi generation bị exception.
* Ngắt giữa POS và NEG rồi resume: 20 cặp chỉ gọi generate đúng 40 lần kể cả
  nhiều lần resume; không tạo filter, không load judge; feature CAA/K-Steer/eval
  bằng replay adapter; dựng CAA/ORBIT và train cả hai classifier trên CPU.
* Ngắt sau cặp hoàn chỉnh, ghi system/user provenance; token budget khác bị
  chặn cache mismatch trước khi sinh thêm.
* Budget notebook được chia sẻ qua các stage: 90 rồi 60 giây còn lại, hết budget
  trả checkpoint mà không khởi động stage tiếp theo.

Report fixture kiểm filter_policy=none, counts lọc null/not applied, không tạo
threshold sensitivity build giả và không nhân baseline thành quan sát độc lập.
Notebook schema nbformat 4.5 và code-cell AST qua suite; ZIP có source numerical,
config, helper, tests và notebook, kiểm integrity riêng sau build.

**Chưa chạy workflow v2 trên 2×T4**, chưa tải pretrained model weights hoặc có
điểm ToneBank thực nghiệm mới. CPU FP32 tiny model không xác nhận parity FP16/NF4
trên T4. GPU smoke phải kiểm: đủ 6 notebook, real Qwen judge calibration/final,
norm capture generation_prefill_last, intervention cả ba phương pháp, baseline
không hook, mid-run resume và output version có file. Không giả định giảm cap
2048→96 làm runtime nhanh đúng 21 lần; chưa đo throughput GPU v2.

Lệnh kiểm chứng:

```bash
PYTHONPATH=kaggle /tmp/orbit-paper-cpu/bin/python -m unittest discover -s kaggle/tests -v
/tmp/orbit-paper-cpu/bin/python -m pip check
```

## Cập nhật xác minh snapshot và ZIP dễ tải — 20/10/2026

`snapshot_run` mới kiểm size/SHA-256 của archive trước và sau chuyển thư mục,
đồng thời kiểm lại snapshot cũ khi gọi lặp. Cả bảy notebook nhúng helper kiểm
checkpoint độc lập để dùng được cả bundle cũ. Cell cuối tự tạo ZIP có index và
archive parts ngay trong `/kaggle/working` khi tổng checkpoint <=64 MiB, in đường
dẫn và kích thước. Checkpoint lớn không tạo bản ZIP bổ sung. ZIP không tự upload
hay làm Kaggle saved Output có file.

Đã chạy **6 tests stdlib filesystem** trong `test_checkpoint_tools.py`: export
ZIP và restore từ Dataset giải nén, checksum/file thiếu khi gọi snapshot lặp,
khôi phục từ raw backup trong /tmp khi working checkpoint mất, size cap, atomic
export lỗi giữ bản trước, và chặn tên archive trùng/path traversal. Đã chạy lại
12 tests bootstrap/runtime. Chỉ kiểm trên CPU; chưa kiểm Save Version, file
browser hoặc download FileLink thật trên Kaggle. Chưa có kiểm tra GPU mới.

## Cập nhật hướng dẫn Output: 0 files — 20/10/2026

Cell snapshot của cả bảy notebook đã bổ sung kiểm tra và in đường dẫn/kích thước
index cùng archive trong phiên. Markdown phân biệt rõ log cell, file phiên và
file saved version. README có cell chẩn đoán không cần cài môi trường và cách
tải ZIP dự phòng cho checkpoint nhỏ của 00 rồi dùng làm Dataset input.

Đã kiểm AST code cells cả bảy notebook, thực thi cell snapshot mới bằng fixture
CPU và round-trip snapshot → ZIP → thư mục Dataset → restore, cùng 12 tests
stdlib bootstrap/runtime đã qua. Không đổi thuật toán hoặc định dạng checkpoint.
Không kiểm tra được giao diện Save Version, FileLink/download hoặc saved Output
trên tài khoản Kaggle của người dùng; không có kiểm tra GPU mới.

## Cập nhật môi trường Python — 20/10/2026

Traceback Kaggle cho thấy `venv` của kernel Python 3.13 thất bại tại ensurepip.
Bootstrap mới không gọi venv/ensurepip của kernel: tải wheel uv 0.8.22 có checksum
cố định, dùng managed CPython 3.12.11 và cài pip 25.2 vào worker riêng. Giữ nguyên
dependency khoa học và thuật toán. Bộ cài được nhúng trong cả bảy notebook để
dùng được cả Code Dataset cũ; môi trường legacy tạo dở không được tái sử dụng.

Đã chạy **6 tests stdlib mới** trong `tests/test_runtime_env.py`: chọn worker
3.12 kể cả host 3.13, CPU/GPU pin và fingerprint lock riêng, marker completed chỉ
sau pip check, failed/resume, kiểm phiên bản interpreter thật, loại PYTHONPATH
của host khi cài đặt, và checksum wheel/executable uv. Đây là các test mock về
contract bộ cài, không phải kiểm tra tải đủ dependency hoặc GPU.

Đã kiểm tra tích hợp **thật trên CPU Linux x86_64**: tải đúng wheel uv qua HTTPS,
kiểm SHA-256, tải managed CPython 3.12.11, tạo venv bằng uv, cài pip 25.2 và
NumPy 1.26.4 rồi import NumPy thành công trong worker. Không dùng ensurepip/pip
của host; không cài lại toàn bộ torch/dependency lock trong lần cập nhật này.
Chưa tái hiện kernel Python 3.13 thật của Kaggle; nhánh host 3.13 được kiểm bằng
test mock. Không có kiểm tra T4 mới hoặc generation LLM thật.

Lệnh kiểm tra stdlib (không cần torch):

```bash
python3 -m unittest discover -s kaggle/tests -p test_runtime_env.py -v
python3 -m unittest discover -s kaggle/tests -p test_bootstrap.py -v
```

## Cập nhật bootstrap — 20/10/2026

Đã sửa cả bảy notebook để nhận ZIP và bundle đã được Kaggle giải nén. Bundle mới
có `bundle_files.sha256.json` kiểm checksum từng file; bundle cũ được nhận với
thông báo giới hạn kiểm chứng vendor/prompt. Không đổi thuật toán hay cấu hình thí nghiệm.

Đã chạy **6 tests stdlib mới** bằng `python3 -m unittest discover -s kaggle/tests
-p test_bootstrap.py -v`: auto-detect thư mục nhiều cấp/đường dẫn explicit, ZIP +
outer checksum, input thiếu hoặc trùng, checksum thư mục/cache hỏng, compatibility
bundle cũ, và chặn ZIP path traversal. Đã kiểm tra syntax/schema JSON notebook
và round-trip bundle thật ở cả hai dạng. Không chạy lại 18 tests tensor của lần
bàn giao trước trong cập nhật này; môi trường venv CPU trước không còn ở phiên hiện tại.

Các kiểm tra GPU/T4 bên dưới vẫn chưa chạy.

Ngày kiểm tra: **19/10/2026**. Không có CUDA/GPU khả dụng trong môi trường kiểm
tra này. **Chưa chạy model weights hoặc thí nghiệm trên NVIDIA T4; chưa có điểm
ToneBank thực nghiệm của bốn phương pháp.** Không có số liệu giả để điền bảng paper.

## Đã thực hiện trên CPU

Môi trường venv `/tmp/orbit-kaggle-cpu`: Python 3.12.3, PyTorch 2.6.0+cpu,
Torchvision 0.21.0+cpu, Transformers 5.20.1; dependency lock được tạo từ môi
trường kiểm thử, bỏ CPU-specific torch wheel khỏi lock dùng chung.
`pip check`: không có dependency hỏng. Import loader Qwen3.5, Llama, Qwen2,
source angular utilities và toàn bộ adapter thành công. `compileall` thành công.

Suite `kaggle/tests/test_cpu.py`: **18 tests**, gồm:

* 25 unordered combos full, config model ID/precision và seed ổn định.
* Từ chối artifact khác backbone/precision/split; thay riêng evaluator config vẫn
  dùng lại được saved generation/activation.
* JSON shard/tensor checksum, shape, dtype/finiteness; config mismatch dừng.
* Ngắt sau shard đã commit, trạng thái pending và resume không sinh duplicate ID.
* Archive parts, giải nén fresh root và từ chối archive hỏng checksum.
* Parse JSON/final_json/thinking, đủ nhãn, ranges, missing/null/NaN/boolean,
  coherence=5 vẫn là 5; retries có giới hạn, cache hit, lỗi kỹ thuật giữ null.
* Ngưỡng POS ≥60, NEG <40, coherence ≥50 và tách unscored.
* Calibration update thật theo input judge score, history/cache key, restore
  trạng thái vòng và dừng khi technical_error theo policy mặc định.
* Question holdout không giao ID; duplicate detection và giữ thứ tự câu hỏi đầu
  tiên ở shared_pool; paired prompt generation khớp source helper.
* Classifier Adam state + minibatch cursor: ngắt giữa training rồi tiếp tục cho
  trọng số **bằng chính xác** với training liền mạch trong test CPU.
* Source hooks CAA, gradient K-Steer, ORBIT trên tensor nhỏ: can thiệp khác zero,
  finite, ORBIT response selector chỉ sửa token cuối, cleanup khôi phục baseline;
  rotation-only ORBIT giữ norm.
* Tiny random Llama/Qwen models (không tải weights): feature adapter khớp
  hidden_states[-6] và final hidden trên batch left-padding; source
  ResidualActivationExtractor thu đúng 12 site của 6 blocks, đúng shape/finite.
* Dựng source candidates/global plane từ activation shard CPU, orthogonality,
  CAA shape, đủ 2 classifier, chạy lại từ cache không cần LLM.
* Report fixture có missing/errors, paired delta, số baseline vật lý không nhân
  theo combo; bootstrap CI không đổi khi lặp lại alias của cùng question.
* Schema nbformat 4.5 và AST code cells của bảy notebook; import runtime trong
  môi trường chặn module `openai` (SDK không được cài).

Các tiny models, score fixtures và câu hỏi fixture chỉ dùng trong unit tests,
lưu ở temporary directories; không thay ToneBank và không được xuất thành kết
quả thí nghiệm.

Lệnh đã chạy:

```bash
PYTHONPATH=kaggle /tmp/orbit-kaggle-cpu/bin/python -m unittest discover -s kaggle/tests -v
PYTHONPATH=kaggle /tmp/orbit-kaggle-cpu/bin/python -m compileall -q kaggle
/tmp/orbit-kaggle-cpu/bin/python -m pip check
```

## Kiểm tra dữ liệu và processor thật, không có generation LLM

Đã gọi metadata công khai Hugging Face cho cả bốn model ID, ghi exact commit ở
configs/models_verified.json. Hai Llama gated=manual; Qwen2.5 và Qwen3.5 không
gated. Đây **không** xác nhận token của người dùng đã được Meta cấp quyền đọc
weight; notebook 00 phải HEAD một weight bằng HF_TOKEN trước tải.

Đã tải ToneBank thật từ revision
`c789fd10483cb6a2bb3703234d56168ce468cb93`: **1.184 rows**, schema
`id: string, text: string, category: string`. HF fingerprint
`35bfb384b456a7e9`; content fingerprint JSON snapshot
`452201cafb27df646f4bce8b006218a106286b273cd7a13ef2377740bf55619e`.
Hai duplicate groups (index 0-based): `[108,421]`, `[162,1173]` sau chuẩn hóa
case/whitespace. Snapshot giữ mọi row, split dedup theo stable content ID.
Full question_holdout config tạo train=50, calibration=75, report=16 trên 200
selected questions. Chưa sinh POS/NEG bằng bất kỳ model thật nào.

Đã tải config + processor/tokenizer **thật** Qwen3.5 tại
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, không tải weight. Auto mapping đúng
`Qwen3_5ForConditionalGeneration`. `processor.apply_chat_template` với
enable_thinking=False/True trả input_ids/attention_mask/mm_token_type_ids hợp lệ,
suffix thinking đúng. Lần thử không có Torchvision lỗi ở video processor; đã
xác nhận cần và pin Torchvision cùng PyTorch. Kết quả này chỉ xác nhận processor,
**không** xác nhận kernel Gated DeltaNet, FP16, quantization hoặc throughput T4.

Evidence máy đọc được: [checks/cpu_evidence.json](checks/cpu_evidence.json).
Snapshot dataset đầy đủ được tạo ở thư mục kiểm tra tạm; notebook 00 tái tải đúng
revision và lưu snapshot vào output run của người dùng. Không đính kèm weight,
token hoặc dataset giả trong bundle.

## Phải thực hiện trên Kaggle trước full

| Kiểm tra | Cell/lệnh có sẵn | Trạng thái hiện tại |
|---|---|---|
| Đúng 2×T4, VRAM riêng từng GPU, RAM/disk, CUDA FP16 | 00 preflight | Chưa chạy GPU |
| Token đọc được weight gated ở exact revisions | 00 preflight HEAD | Chưa xác minh tài khoản người dùng |
| Qwen3.5 load + real generate + JSON rubric | 00 model-check judge | Chưa chạy weights |
| Backbone load + generate FP16 distributed | 00 model-check backbone | Chưa chạy weights |
| Smoke đủ 7 stages, 4 methods, real local judge | 00–06 | Chưa chạy GPU |
| Hook đúng site/token, delta !=0, cleanup | 05 audit + 06 verify | Chỉ qua tensor CPU |
| Interrupt/resume generation thật và calibration thật | 01 resume test, 04 state/history | Chỉ qua state/tensor CPU |
| OOM batch reduction, max VRAM, runtime/throughput | extract/events, pilot telemetry | Chưa đo T4 |
| NF4/int8 + Qwen3.5 hybrid architecture | smoke riêng precision profile | Chưa xác nhận |
| Full 3 backbones, 25 combos, final scores/CI | full 00–06 từng model, combine_reports | Chưa có kết quả thực nghiệm |

Các rủi ro thực thi còn lại có căn cứ: loader/device_map và hybrid-cache giữa
hai GPU; tốc độ fallback PyTorch của Qwen3.5 trên T4; phân phối Qwen scores có thể
làm bộ lọc thiếu cặp; full có thể cần rất nhiều phiên/quota GPU. Pipeline dừng
với lỗi và giữ shard đã hoàn thành, không đổi model/method/threshold/precision
hoặc giảm số câu hỏi/token ngầm để che các lỗi này.
