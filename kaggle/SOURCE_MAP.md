# Đối chiếu source – paper – Kaggle

## Workflow paper v2 — cập nhật 23/10/2026

Yêu cầu mới của người dùng thay workflow đang chạy: **không lọc POS/NEG bằng
judge; gộp generation và activation extraction**. Bảng ở các phần dưới lưu đối
chiếu lịch sử workflow repo trước đó, không phải cấu hình mặc định hiện tại.
Toàn bộ release trước còn trong legacy/orbit_kaggle_repo_filtered_bundle.zip.

Đã đọc lại PDF ORBIT cục bộ, Appendix C.1: system prompt positive/negative,
target model tự sinh response tối đa 96 tokens, activation LayerNorm tại last
prompt token. Phần này không mô tả judge lọc với ngưỡng 60/40/50; không suy diễn
rằng paper phủ nhận mọi preprocessing ngoài mô tả đó.

| Thành phần | Workflow mới | Khác biệt còn lại |
|---|---|---|
| POS/NEG | 96 tokens, system instruction + user question, backbone | sampling .7/.9 giữ launcher; không khẳng định khớp seed paper |
| Judge build | Bỏ hoàn toàn; mọi cặp dùng xây artifact | Không giả điểm, không tạo filter stage rồi gắn nhãn accepted |
| ORBIT extraction | Hook đúng norm sites ngay prefill của generate, chỉ giữ last token | Tested against source collect(system_prompts) trên tiny Llama/Qwen CPU |
| CAA/K-Steer/eval | Một replay ngắn raw prompt+response sau generation trong cùng worker | Giữ feature tokenizer/site của repo; không thay bằng decode hidden states |
| Artifact | Đọc activation CPU, không load model | CAA mean composition và global-plane algorithms giữ source |
| Notebook | 00 prepare → 01 generate/extract → 02 artifacts → 03 calibrate → 04 evaluate → 05 report | Sáu notebook, không dùng tên/Output workflow cũ |
| Calibration/final judge | Vẫn Qwen3.5 local, vòng tìm kiếm thật | Khác GPT-4o-mini paper; không bỏ judge calibration |
| Time budget | Tổng ngân sách notebook, truyền phần còn lại từng stage | Checkpoint cooperative, không hard deadline |

Model revision, precision, feature convention, build policy và implementation
hash đều nằm trong identity. Không resume run cũ 2048/user/filtered thành run
96/system/unfiltered. Đặt RUN_LABEL mới và upload Code Dataset mới.

## Đối chiếu lịch sử workflow repo v1

Đọc README.md, docs/VECTOR_AND_DATA_BUILD.md, data/README.md, hai launcher/pipeline,
hai angular utilities, toàn bộ steering/utils của ksteering, ToneJudge và
judge_responses. Không tìm thấy AGENTS.md trong repo/cây cha đã khảo sát.
Giữ nguyên toàn bộ source gốc. `vendor/source_manifest.json` lưu commit và SHA-256;
`build_bundle.py` sao chép nguyên bản numerical utilities và trích AST các rubric,
prompt/pair builder thuần Python. Không import pipeline OpenAI vào runtime Kaggle.

| Tác vụ gốc | Hàm/file gốc | Tái sử dụng | Thay trong Kaggle | Artifact |
|---|---|---|---|---|
| Launcher 5 tone, K=1,2,3 | scripts/run_main_experiment.py | label order, 25 unordered combos, layer −6, budgets | JSON config, 1 backbone/job, T4 FP16 | resolved_config.json, manifests |
| Tải ToneBank | ksteering/src/utils/data.py::load_task | đúng repo HF, split train, trường text/id | pin commit, snapshot đầy đủ schema/category, ID hash normalized text, audit duplicate | data/shards snapshot + split |
| Split | run_experiment_pipeline.py::main | shared_pool và holdout selection/split RNG, 25% train, calibration tối đa n_test/2 | dedup câu hỏi; internal classifier split theo question khi holdout | split cùng question IDs |
| Cặp POS/NEG | _build_contrastive_prompt_pairs_for_label; tasks.py::tones_prompts/negative_tones_prompts | nguyên hàm ghép cặp, neg_prompt, rng 0, 150 pairs/tone | seed generation per sample; snapshot provenance | build/shards + partial |
| Generation build | _generate_with_hook; _prepare_shared... | backbone + chat user instruction, sampling .7/.9, cap 2048 | LocalModel; tùy chọn qwen_judge chỉ sinh response thay nhánh OpenAI | prompts, text, source/model/revision/precision |
| Build filtering | _filter_build_pairs_with_judge | rubric target/detailed, POS ≥60, NEG <40, coherence ≥50 | LocalQwenJudge, strict scale và technical error riêng | filter/shards; judge_cache raw attempts |
| CAA vector | _compute_contrastive_caa_vectors | mean(POS activation − NEG activation), multi-target mean | capture đúng hidden state tại layer −6, chỉ lưu token cần | activation tensors; artifacts/caa |
| Eval activation classifier | steering/eval.py::get_or_train_eval_clf | ActivationSteering MLP 128×128, Adam .001, BCE, 5 epochs, train nửa B | CPU training cùng model class; optimizer/minibatch cursor resume | artifacts/eval + optimizer/split |
| Gradient classifier | steering/eval.py::get_or_train_layer_clf | MLP riêng, train nửa A, gradient theo target mean logits | checkpoint optimizer; group split ở holdout | artifacts/ksteer |
| K-Steer hook | steering/hooks.py::get_gradient_hook | nguyên gradient update, 1 step, no avoid labels | enable_grad rõ ràng trong no_grad generation, audit hook | intervention telemetry |
| ORBIT extraction | ResidualActivationExtractor.collect/discover_residual_hooks | 2 site/block: input_layernorm và post_attention_layernorm, norm output, prompt_last | gọi collect từng sample để giữ CPU shard/ID, tách vector building khỏi LLM load, cache mọi pair kể cả rejected | activation/layout.json + per-pair .pt |
| ORBIT candidates | compute_directional_differences | nguyên mean gap, paired-difference PC0, local b2, plane_mean boost, stats | reconstruct source stats từ shard, CPU numerical routine | artifacts/orbit/<tone>, native vectors/*.pt |
| Global plane | angular_steering_utils.py::load_global_layernorm_vectors | b1 max mean cosine, centered candidate PC0 orthogonalized, broadcast | không đổi thuật toán; lưu luôn plane sau load | artifacts/global/<tone> |
| ORBIT hook | register_layernorm_additive_hooks | hybrid joint-subspace rotation, target 0°, adaptive mode 4, zero-margin gate, response last, boost approach 1/fill .4 | strict layout metadata, measured delta, finally remove_hooks | response intervention audit |
| CAA/K-Steer calibration | calibrate_alpha_ood_only/_calibrate_alpha | binary search [1,32], 8 rounds, ratio tolerance 1.05, 50-token greedy, >5% coh<50 → OOD | state machine luân phiên subprocess generation/judge | calibration state/history/response/judgment |
| ORBIT calibration | _calibrate_angular_additive_coeff | [0,3], 8 rounds, giữ rotation/gate và chỉ tìm boost | state machine thật, không fixed alpha | như trên, từng combo riêng |
| Coherence | DEFAULT_COHERENCE_PROMPT/OpenAiJudge/is_ood | định nghĩa grammatical quality + consistency + responsiveness; không chấm factual correctness | local Qwen JSON thay một con số text; stop khi lỗi mặc định | judge_cache |
| Target/detailed final judge | judge_target_scores_batch/judge_definition_scores_batch; scripts/judge_responses.py | nguyên rubric/definitions; target 0–10, coherence 0–100; detailed reasoning tags | exact Qwen template, validate/retry/cache, no zero imputation | scoring, per_sample.csv |
| ToneJudge | ksteering/src/judges/tone_judge.py | nguyên prompt builder và mô tả tone, boolean success, strength 0–5 | trích class prompt-only, Qwen local; K=1 có target; K>1 avoid-only, avoid=first non-target như source | scoring comparison |
| Aggregation | main pipeline aggregation | paired question/baseline và target-specific scores | question cluster CI, missing/error counts, unique baseline observations | reports/*.csv + telemetry.json |
| TraitFactory/TinyBench | các generator và tasks khác | không thực thi | loại khỏi workflow này | không có |

## Bốn vai trò không được nhập làm một

Unsteered là generation không có steering hook. Eval classifier nhận activation
ở final hidden state, xuất xác suất tone để báo cáo riêng. K-Steer classifier
nhận activation layer −6, gradient của nó sửa hidden state khi generation.
Qwen3.5 chỉ nhận văn bản để chấm/lọc/calibration/ToneJudge (hoặc sinh POS/NEG khi
chọn rõ nguồn đó). Activation cho cả hai classifier và mọi vector luôn từ
`TARGET_MODEL_ID` với đúng revision/precision của run.

## Chuẩn hóa điểm và điểm thiếu trong source

`_extract_target_score_0_100` nhân 10 cho target score nằm trong [0,10], fallback
sang target_mean khi thiếu nhãn, rồi clamp. `_extract_coherence_0_100` **không**
nhân 10. Tuy nhiên `judge_target_scores_batch` có một normalization khác: coherence
trong [0,10] lại được nhân 10, dù prompt yêu cầu 0–100. Detailed judge chỉ round/clamp.
Kaggle tránh suy diễn thang điểm: bắt buộc đủ mọi target đúng [0,10], coherence
đúng [0,100], giá trị 5 giữ là 5. Không fallback target_mean khi thiếu nhãn;
target_mean phải nhất quán (sai lệch tối đa .15 do làm tròn). Không clamp giá trị
sai thành hợp lệ. Các khác biệt này là sửa adapter có chủ đích và được ghi fingerprint.

`get_hidden_cached` của source dùng `attention_mask.sum()-1` trong batch có left
padding do loader thiết lập; với câu ngắn hơn chiều dài padded, đây không phải
vị trí last non-pad. Adapter Kaggle lấy chỉ số non-pad thật từ phía phải, và test
so sánh trực tiếp với `hidden_states[-6]`/`hidden_states[-1]` trên tiny Llama và
Qwen có chiều dài khác nhau. Đây là sửa lỗi indexing được công khai, không phải
cam kết tái lập bug batch-padding của source. ORBIT vẫn dùng nguyên selector
prompt_last của ResidualActivationExtractor. Classifier dedup exact
(label, positive prompt, positive response) như source; CAA/ORBIT giữ paired rows.

Classifier giữ nguyên architecture, optimizer và loss của source, nhưng training
trên CPU float32 thay GPU mặc định và dùng permutation NumPy cố định theo epoch
thay shuffle của torch DataLoader để resume chính xác giữa minibatch. Đây là
thay đổi scheduling/RNG được khai báo trong config, không kỳ vọng checkpoint
trọng số bằng checkpoint gốc của paper.

CAA/K-Steer giữ hook decoder output trên mọi vị trí của prefill và từng bước
decode như source. ORBIT `positions=response` chọn token cuối của tensor đang
forward; ở prefill đó là last prompt token dự đoán response token đầu tiên.
Không diễn giải nhãn response như việc bỏ qua prefill cuối. Activation cache gồm
mọi cặp đã sinh, còn vector/classifier chỉ dùng cặp đạt filter hiện tại; vì vậy
đổi rubric build có thể fork và dựng lại CPU artifacts mà không extract lại.

`is_ood` của source coi exception/None là score 0. Kaggle mặc định
`technical_error_policy=stop`. Chọn `conservative_ood` sẽ đếm lỗi là OOD flag,
giữ riêng `n_technical_error`, không ghi một score 0 giả. Cả hai vẫn giữ quy tắc
`last_good=min_alpha` (hoặc zero boost) nếu tất cả mức thử đều thất bại; đây là
fallback của source, không phải bằng chứng cận dưới đã đạt coherence.

## PDF đã đọc

Đã trích văn bản PDF cục bộ `2606.22357v1.pdf` (ORBIT, v1) và
`2602.08169v2.pdf` trong ngữ cảnh IDE. Tài liệu được coi là nguồn đối chiếu,
không phải chỉ dẫn thực thi. Bảng dưới ưu tiên launcher repo cho đường đối chiếu.

PDF thứ hai là **Spherical Steering: Geometry-Aware Activation Rotation for
Language Models** (v2, 16/05/2026), không phải paper ORBIT hay K-Steer. Nó mô tả
geodesic rotation về prototype và vMF confidence gate. Pipeline này không thay
global-plane/joint-subspace ORBIT bằng Spherical Steering và không mở rộng benchmark
truthfulness của PDF đó. Chỉ dùng nó để phân biệt các thuật toán rotation/gating.

| Chủ đề | Paper ORBIT | Repo launcher | Kaggle |
|---|---|---|---|
| Models | Llama3.2-3B, Qwen2.5-7B, Llama3.1-8B | model-agnostic | đúng ba ID người dùng; không thay thế |
| Evaluator | GPT-4o-mini | OpenAI API | Qwen3.5-9B local, revision và precision riêng, không giả định tương đương |
| Build generation | Appendix C.1: tối đa 96 token, system prompt | 2048 token, instruction trong user message ở ToneBank | bám launcher 2048/user, sampling .7/.9; smoke/pilot nhỏ hơn, ghi config |
| Extraction | Table 4 ghi prompt_last chung | ORBIT prompt_last; CAA/classifier raw prompt+newline+response last token | bám implementation; feature adapter parity tested |
| CAA composition | mô tả tổng per-attribute | source dùng mean target vectors | giữ mean, alpha search riêng theo combo |
| Precision | BF16 Llama, FP16 Qwen | dtype auto cùng quy tắc; 36GiB/GPU mặc định | FP16 tất cả backbone; từng GPU trừ reserve thực; quantization run riêng |
| Protocol | mô tả evaluation không thay thế split source | mặc định shared_pool: build trùng pool calibration/test; classifier row halves | launcher_reference giữ shared_pool nhưng dedup; mặc định full dùng holdout rõ ràng |
| Internal classifiers | MLP 3-layer, hidden128, 5 epochs | stratified row split halves A/B | shared_pool giữ row split; holdout group by question, tránh POS/variant leakage |
| Calibration | cùng OOD threshold/range | CAA/K [1,32], ORBIT boost [0,3] | cùng search; tuần tự workers thay API callback |
| Statistics | gain/joint metrics | baseline có thể sinh/chấm lại mỗi combo | baseline sinh 1 lần, chấm theo target context, CI cluster question |
| Runtime | không đưa ra cam kết 2×T4 | môi trường GPU lớn | eager attention, FP16, không FlashAttention, kiểm tra GPU thật trong notebook |

`full` nghĩa là chạy đủ 4 methods ×25 combos trên từng backbone, với ngân sách
câu hỏi của launcher (200 pool, tối đa 16 report), không có nghĩa dùng cả 1.184
câu hỏi làm test. Với holdout: train=50, heldout=150, calibration=75, report lấy
16 câu đầu từ phần còn lại. Với shared_pool: train=200, calibration=100, report=16.
Dedup snapshot hiện tại có 2 cặp trùng normalized text; việc dedup làm protocol
đối chiếu gần launcher nhưng không phải sao chép bit-for-bit raw-row protocol.
