#!/usr/bin/env python3
"""Generate valid nbformat-4 notebooks; code cells contain plain Python (no magics)."""
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent

PARAMETERS='''# Giữ các tham số giống nhau trong cả sáu notebook của workflow paper mới.
MODE = "smoke"  # smoke -> pilot -> full; POS/NEG 96 tokens, không judge lọc
TARGET_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
SPLIT_MODE = "question_holdout"  # hoặc shared_pool; không đổi giữa các stage
PRECISION_PROFILE = "fp16"  # fp16 | judge_nf4 | low_memory_nf4; run riêng cho mỗi profile
POS_NEG_SOURCE = "backbone"  # workflow inline chỉ dùng backbone để thu activation đúng model
RUN_LABEL = "tonebank-paper-smoke-v2"  # run mới; không dùng checkpoint workflow cũ
BUNDLE_ZIP = None  # Tự tìm ZIP hoặc bundle đã giải nén; cũng có thể đặt đường dẫn thư mục
PREVIOUS_RUN_INPUT = __PREVIOUS__
TIME_BUDGET_SECONDS = 30000  # tổng budget notebook, gồm setup; chừa thời gian Save Version
'''

BOOTSTRAP=(ROOT / "bootstrap_bundle.py").read_text()+"\n"+(ROOT / "runtime_env.py").read_text()+"\n"+(ROOT / "checkpoint_tools.py").read_text()+'''
import os, sys, json, hashlib, shutil, zipfile, subprocess, time
from pathlib import Path
NOTEBOOK_START = time.monotonic()
assert TIME_BUDGET_SECONDS > 0
assert sys.version_info >= (3, 11), "Notebook bootstrap cần Python >=3.11; worker sẽ dùng Python 3.12 riêng"
CODE_ROOT = prepare_bundle(BUNDLE_ZIP)
assert (CODE_ROOT / "orbit_kaggle/fused_build.py").is_file(), "Upload Code Dataset bằng bundle paper mới; bundle cũ không có fused build"
sys.path.insert(0, str(CODE_ROOT))
os.environ["PYTHONPATH"] = str(CODE_ROOT)
os.environ["HF_HOME"] = "/tmp/orbit-hf"  # weight cache không bị Save Output vô tình
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from orbit_kaggle.session import restore_run, run_stage
PYTHON = install_env(CODE_ROOT)
# HF_TOKEN chỉ ở environment; không in token, không ghi token vào artifact.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as exc:
    if TARGET_MODEL_ID.startswith("meta-llama/"):
        raise RuntimeError("Bật secret HF_TOKEN có quyền đọc model Llama gated") from exc
config = json.loads((CODE_ROOT / "configs" / f"{MODE}.json").read_text())
config["target_model_id"] = TARGET_MODEL_ID
config["split"]["mode"] = SPLIT_MODE
profile = json.loads((CODE_ROOT / "configs/precision_profiles.json").read_text())[PRECISION_PROFILE]
config["precision"] = profile["precision"]
config["judge"]["precision"] = profile["judge_precision"]
config["build"]["response_source"] = POS_NEG_SOURCE
assert config["build"].get("extraction") == "generation_prefill", "Cần config workflow paper mới"
RUN_ROOT = Path("/kaggle/working/orbit_runs") / RUN_LABEL / TARGET_MODEL_ID.replace("/", "--")
RUN_ROOT.mkdir(parents=True, exist_ok=True)
if PREVIOUS_RUN_INPUT:
    restore_run(PREVIOUS_RUN_INPUT, RUN_ROOT)
if (RUN_ROOT / "checkpoint_index.json").exists():
    restore_run(RUN_ROOT, RUN_ROOT)  # re-open own compacted checkpoint in this same session
(RUN_ROOT / "runtime_environment.json").write_text((Path(PYTHON).parents[1] / "orbit-install.json").read_text())
CONFIG_PATH = RUN_ROOT / "resolved_config.json"
if CONFIG_PATH.exists():
    assert json.loads(CONFIG_PATH.read_text()) == config, "Config khác checkpoint; dùng RUN_LABEL mới hoặc giữ đúng config cũ"
else:
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\\n")
CHECKPOINTED = False
def stage(name, **kwargs):
    global CHECKPOINTED
    if CHECKPOINTED:
        print("SKIPPED until resume:", name)
        return 75
    intentional = kwargs.pop("allow_checkpoint", False)
    remaining = int(TIME_BUDGET_SECONDS - (time.monotonic() - NOTEBOOK_START))
    if remaining <= 0:
        CHECKPOINTED = True
        print("Notebook time budget reached. Snapshot current run, save output and resume.")
        return 75
    code = run_stage(PYTHON, CONFIG_PATH, RUN_ROOT, name, remaining, allow_checkpoint=True, **kwargs)
    if code == 75 and not intentional:
        CHECKPOINTED = True
        print("CHECKPOINT SAVED ON SESSION DISK. Save this version's output and resume this notebook; upstream stage incomplete.")
    return code
print("Run:", RUN_ROOT, "model:", TARGET_MODEL_ID, "mode:", MODE, "split:", SPLIT_MODE, "precision:", profile)
'''

STAGES=[
("00_prepare_data", "Snapshot ToneBank, kiểm tra quyền truy cập và hardware", "Không cần output notebook trước. Input: Code Dataset (zip + sha256). Bật Internet, GPU T4 x2 và HF_TOKEN trong Add-ons → Secrets. Metadata/access kiểm tra cả bốn ID trước tải weight; hai lệnh model-check load từng model lần lượt bằng subprocess. Không có model thay thế.", [
'''# Contract tests dùng tensor/tiny random models CPU; không tạo kết quả ToneBank giả.
subprocess.run([str(PYTHON), "-m", "unittest", "discover", "-s", str(CODE_ROOT / "tests"), "-v"], check=True)
(RUN_ROOT / "cpu_contract_checks.json").write_text(json.dumps({"executed": True, "scope": "CPU contracts only", "returncode": 0}))''',
'''stage("preflight")
stage("prepare")
stage("estimate")
if not CHECKPOINTED:
    print((RUN_ROOT / "preflight.json").read_text())
    print((RUN_ROOT / "estimate.json").read_text())''',
'''# Kiểm tra judge thật trên chính GPU phiên hiện tại, rồi worker thoát và giải phóng VRAM.
stage("model-check", extra=["--role", "judge"])
stage("model-check", extra=["--role", "backbone"])''']),
("01_generate_and_extract", "Sinh POS/NEG và thu activation cùng worker backbone", "Input: RUN_ROOT từ 00 hoặc checkpoint mới nhất của 01. System prompt POS/NEG, sampling 0.7/0.9, tối đa 96 token như Appendix C.1. ORBIT thu prompt_last ngay trong generation prefill; không forward prompt riêng. CAA/K-Steer/eval giữ feature raw prompt+response của repo, cần một replay ngắn cho hai response. Không judge lọc; lưu mọi cặp. Resume từng phía POS/NEG không sinh lại phía đã lưu.", [
'''if MODE == "smoke" and not (RUN_ROOT / "resume_test.json").exists():
    code = stage("build-extract", extra=["--stop-after", "1"], allow_checkpoint=True)
    assert code == 75, "Thiếu lần ngắt có chủ ý"
    from orbit_kaggle.storage import read_json
    manifest = read_json(RUN_ROOT / "build/manifest.json")
    intentional_checkpoint = manifest.get("reason", "").startswith("Intentional interruption")
    shards = list((RUN_ROOT / "build/shards").glob("*.json"))
    shards += list((RUN_ROOT / "activation/shards").glob("*.json"))
    before = {str(p.relative_to(RUN_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in shards}
    if intentional_checkpoint:
        assert before
        (RUN_ROOT / "resume_before.json").write_text(json.dumps(before))
    else:
        CHECKPOINTED = True
        print("Time budget reached before intentional resume check. Save checkpoint and resume notebook 01.")
stage("build-extract")
if MODE == "smoke" and not CHECKPOINTED and not (RUN_ROOT / "resume_test.json").exists():
    before = json.loads((RUN_ROOT / "resume_before.json").read_text())
    assert all(hashlib.sha256((RUN_ROOT / p).read_bytes()).hexdigest() == value for p, value in before.items())
    from orbit_kaggle.storage import open_existing
    build = open_existing(RUN_ROOT, "build")
    ids = [key for key, row in build.rows()]
    assert len(ids) == len(set(ids))
    (RUN_ROOT / "resume_test.json").write_text(json.dumps({"intentional_interrupt": True, "completed_shards_preserved": True, "unique_pairs": len(ids)}))''']),
("02_build_artifacts", "CPU dựng CAA, classifier K-Steer/eval và ORBIT global plane", "Input: RUN_ROOT hoàn thành 01, có cả build và activation. Không load backbone, không load judge, không lọc cặp. Đọc CPU shards đã lưu để dựng artifact. Training classifier có optimizer và minibatch cursor để resume. Không extract activation lần nữa.", ['stage("artifacts")']),
("03_calibrate", "Calibration tuần tự theo vòng, có judge thật", "Input: RUN_ROOT hoàn thành 02. Mỗi vòng: worker backbone sinh các yêu cầu độc lập → worker judge chấm coherence → cập nhật binary search. Full tối đa 8 vòng, lưu checkpoint từng response và state. Vẫn cần judge cho calibration; chỉ bỏ judge lọc build. Không chọn alpha tùy tiện.", ['stage("calibrate")']),
("04_generate_evaluation", "Sinh Unsteered, CAA, K-Steer, ORBIT", "Input: RUN_ROOT hoàn thành 03. Baseline sinh một lần question/seed. Dùng cùng câu hỏi, seed và token budget cho các phương pháp. Kiểm hook can thiệp và cleanup; lưu response, hệ số, thời gian và VRAM.", ['stage("evaluate")']),
("05_judge_and_report", "Qwen chấm target/detailed, tổng hợp paired delta và CI", "Input: RUN_ROOT hoàn thành 04. Judge Qwen local vẫn chấm kết quả cuối; không thay classifier. Báo tone/coherence, paired delta, CI theo question ID, lỗi/thiếu và telemetry. Build filter phải được ghi none/not applied, không có tỷ lệ lọc giả.", ['stage("score")','stage("verify")',
'''if not CHECKPOINTED:
    latest = json.loads((RUN_ROOT / "latest_report.json").read_text())
    report_dir = RUN_ROOT / latest["relative_path"]
    print((report_dir / "REPORT.md").read_text())
    import pandas as pd
    display(pd.read_csv(report_dir / "summary_model_method_K.csv"))
    print("Tải artifact báo cáo tại:", report_dir)'''])]


def cell(kind,source,index):
    r={"cell_type":kind,"metadata":{},"id":hashlib.sha256((str(index)+source).encode()).hexdigest()[:12],"source":source.splitlines(keepends=True)}
    if kind=="code":r.update(execution_count=None,outputs=[])
    return r


def main():
    output=ROOT/"notebooks";output.mkdir(exist_ok=True)
    # The entire previous release is preserved in legacy/orbit_kaggle_repo_filtered_bundle.zip.
    # Remove obsolete generated names to prevent accidentally mixing two workflows.
    wanted={slug+".ipynb" for slug,*_ in STAGES}
    for path in output.glob("*.ipynb"):
        if path.name not in wanted:path.unlink()
    for i,(slug,title,intro,codes) in enumerate(STAGES):
        previous="None" if i==0 else '"/kaggle/input/OUTPUT-NOTEBOOK-TRUOC/orbit_runs/tonebank-paper-smoke-v2/meta-llama--Llama-3.2-3B-Instruct"'
        sources=[("markdown",f"# {slug}: {title}\n\n{intro}\n\nMỗi notebook chạy từ kernel mới. Thay PREVIOUS_RUN_INPUT bằng đường dẫn Add Input thật; stage 00 đặt None. Khi tiếp tục cùng notebook, dùng output checkpoint mới nhất của chính notebook đó.\n\nChưa có kết quả T4 được chạy sẵn trong notebook này."),
                 ("code",PARAMETERS.replace("__PREVIOUS__",previous)),("code",BOOTSTRAP)]
        sources.extend(("code",code) for code in codes)
        sources.append(("code",'''# Đóng gói checkpoint trước Save Version để tránh hàng chục nghìn output files.
# Archive theo phần <=1GiB uncompressed; bản unpacked còn giữ trong /tmp của phiên.
from orbit_kaggle.session import snapshot_run
snapshot = snapshot_run(RUN_ROOT)
validate_checkpoint(RUN_ROOT)  # cũng kiểm được bundle cũ; không chỉ tin log hoặc index
print(json.dumps(snapshot, indent=2))
checkpoint_files = [RUN_ROOT / "checkpoint_index.json"] + [RUN_ROOT / part["name"] for part in snapshot["archives"]]
for path in checkpoint_files:
    assert path.is_file(), f"Thiếu checkpoint trên đĩa phiên: {path}"
    print("SESSION FILE:", path, "bytes:", path.stat().st_size)
download_zip = export_checkpoint_zip(RUN_ROOT, Path("/kaggle/working") / f"orbit-checkpoint-{RUN_ROOT.parent.name}-{RUN_ROOT.name}.zip")
if download_zip is not None:
    print("DOWNLOAD ZIP (ngay trong /kaggle/working):", download_zip, "bytes:", download_zip.stat().st_size)
else:
    print("Checkpoint >64MiB: không tạo ZIP bản sao; giữ index và archive parts trong RUN_ROOT.")
print("Các file trên mới tồn tại trong phiên. Chỉ tiếp tục khi saved version có Output files thực tế; log cell không đủ.")'''))
        sources.append(("markdown","## Giữ output và tiếp tục\n\nNếu stage báo checkpoint/time budget: không chạy stage tiếp theo. Dùng **Save Version** ở góc trên phải; chọn **Quick Save → Advanced Settings → Save output files** nếu giao diện có lựa chọn đó. Chờ version lưu xong, mở đúng version → **Output** và xác nhận có `orbit_runs/.../checkpoint_index.json` cùng các `checkpoint-part-*.tar.gz`.\n\n**Nếu Output ghi 0 files:** version đó chưa có checkpoint để notebook sau đọc, dù vẫn hiện log cell cũ. Bấm Edit, chạy lại riêng cell snapshot khi phiên vẫn còn RUN_ROOT và kiểm tra các dòng SESSION FILE. Lưu version có output files. Nếu phiên đã mất file, chạy lại notebook từ các input đã lưu. Có thể chọn **Save & Run All**, nhưng thao tác này chạy một phiên sạch từ đầu, không chuyển file tương tác sang; khi resume phải khai báo PREVIOUS_RUN_INPUT đã được lưu thật. README có cách tải checkpoint dự phòng nếu phiên còn file.\n\nỞ notebook tiếp theo chọn **Add Input → Notebook Outputs**, tìm notebook/version đã có file, Add. Copy đường dẫn RUN_ROOT dưới `/kaggle/input/...` vào PREVIOUS_RUN_INPUT. Chỉ đính kèm một snapshot cumulative mới nhất của run. Không coi file còn trong `/kaggle/working` là đã bền vững. Xem README_KAGGLE.md cho cách giới hạn output và lặp qua ba backbone."))
        nb={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
             "language_info":{"name":"python","version":"3.11"},"accelerator":"GPU","kaggle":{"isInternetEnabled":True,"isGpuEnabled":True}},
             "cells":[cell(kind,source,j) for j,(kind,source) in enumerate(sources)]}
        (output/(slug+".ipynb")).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n")


if __name__=="__main__":main()
