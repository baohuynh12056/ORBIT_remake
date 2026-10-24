from __future__ import annotations
import itertools
import hashlib
from pathlib import Path
from .storage import read_json, digest

MODELS = ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]
JUDGE = "Qwen/Qwen3.5-9B"
DATASET = "withmartian/tone_agnostic_questions"
LABELS = ["expert", "cautious", "empathetic", "casual", "concise"]
METHODS = ["Unsteered", "CAA", "K-Steer", "ORBIT"]


def implementation(*names):
    root=Path(__file__).resolve().parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def load_config(path):
    c = read_json(path)
    if c["mode"] not in ("smoke","pilot","full"):
        raise ValueError("Unknown run mode")
    if c["target_model_id"] not in MODELS or c["judge"]["model_id"] != JUDGE:
        raise ValueError("Model substitution is forbidden; use one of the three exact backbone IDs and Qwen3.5-9B judge.")
    if c["dataset"]["id"] != DATASET or c["labels"] != LABELS or c["methods"] != METHODS:
        raise ValueError("Only ToneBank, the five launcher tones, and the four requested methods are supported.")
    if c["split"]["mode"] not in ("shared_pool", "question_holdout"):
        raise ValueError("Explicit split mode required")
    if c["build"]["response_source"] not in ("backbone", "qwen_judge"):
        raise ValueError("One POS/NEG response source per run is required")
    if c["build"].get("filter_policy","judge") not in ("none","judge"):
        raise ValueError("Unknown build filter policy")
    if c["build"].get("extraction") == "generation_prefill":
        if (c["build"].get("filter_policy"),c["build"].get("prompt_role"),c["build"]["response_source"]) != ("none","system","backbone"):
            raise ValueError("Paper inline build requires system prompts, no filtering and backbone responses")
        if c["build"].get("feature_replay_batch_size") not in (1,2):
            raise ValueError("Declare feature replay batch size 1 or 2; generation stays per sample")
    if c["mode"] == "full" and c["k_values"] != [1, 2, 3]:
        raise ValueError("Full requires all 25 combinations")
    if c["precision"] not in ("fp16", "int8", "nf4") or c["judge"]["precision"] not in ("fp16", "int8", "nf4"):
        raise ValueError("Unsupported explicit precision")
    for model in MODELS + [JUDGE]:
        revision = c["revisions"][model]
        if len(revision) != 40 or any(x not in "0123456789abcdef" for x in revision):
            raise ValueError(f"Pin a resolved commit for {model}")
    if c["generation"]["batch_size"] < 1 or c["generation"]["repeats"] < 1:
        raise ValueError("Positive batch size/repeats required")
    if not 1<=c["judge"]["max_attempts"]<=10 or not isinstance(c["judge"]["thinking"],bool):
        raise ValueError("Judge needs boolean thinking and bounded 1..10 attempts")
    if min(c["build"]["max_pairs"],c["build"]["max_new_tokens"],c["generation"]["max_new_tokens"],
           c["generation"]["n_questions"],c["calibration"]["iterations"],c["judge"]["max_new_tokens"],
           c["report"]["bootstrap_draws"])<1:
        raise ValueError("Sample/token/iteration budgets must be positive")
    expected_steering={"layer":-6,"eval_layer":-1,"orbit_plane":"global","orbit_token":"prompt_last",
                       "orbit_positions":"response","orbit_gate":"zero_margin","orbit_rotation":"full_alignment",
                       "orbit_boost_fill_fraction":0.4}
    if c["steering"]!=expected_steering:
        raise ValueError("This release preserves the fixed launcher steering configuration; unsupported overrides would be misleading")
    if c["generation"]["do_sample"] or c["judge"]["do_sample"]:
        raise ValueError("This release implements greedy evaluation/judging and sampled build responses, matching launcher")
    if not set(c["report"]["judge_types"]).issubset({"target","detailed"}):
        raise ValueError("Unknown judge rubric")
    if c["calibration"]["technical_error_policy"] not in ("stop","conservative_ood"):
        raise ValueError("Explicit calibration technical-error policy required")
    if (c["classifier"].get("device"),c["classifier"].get("dtype"),c["classifier"].get("shuffle"))!=("cpu","float32","stateless_numpy_epoch"):
        raise ValueError("Classifier training uses declared CPU float32 and resumable stateless epoch shuffling")
    return c


def combos(c):
    return [list(x) for k in c["k_values"] for x in itertools.combinations(c["labels"], k)]


def model_identity(c):
    import importlib.metadata
    manifest = read_json(Path(__file__).resolve().parents[1] / "vendor" / "source_manifest.json")
    # Rubric/docs/commit-only changes must not invalidate backbone activation caches.
    # Full source provenance remains in each stage manifest outside the result fingerprint.
    numerical={name:checksum for name,checksum in manifest["sha256"].items()
               if name.startswith(("angular_","ksteering/src/steering/","ksteering/src/utils/"))}
    packages={name:importlib.metadata.version(name) for name in ("torch","transformers","numpy","accelerate","scikit-learn")}
    try:packages["bitsandbytes"]=importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError:packages["bitsandbytes"]=None
    return {"model": c["target_model_id"], "revision": c["revisions"][c["target_model_id"]],
            "precision": c["precision"], "attention": "eager", "source_numerics": numerical,
            "layer": c["steering"]["layer"], "activation_storage_dtype": "float32", "packages":packages,
            "implementation":implementation("models.py","artifacts.py","steering.py","config.py","storage.py","fused_build.py"),
            "build_protocol":{k:c["build"].get(k) for k in ("filter_policy","prompt_role","extraction","max_new_tokens")}}


def stable_seed(c, question_id, repeat=0, namespace="evaluation"):
    # No method/combo/shard/order in evaluation seed: paired common random numbers.
    return int(digest([c["seed"], question_id, repeat, namespace])[:8], 16) % (2**31)


def evaluation_protocol(c):
    return {"backbone":model_identity(c),"dataset":c["dataset"],"split":c["split"],"generation":c["generation"],
            "classifier":c["classifier"],"combos":combos(c),"seed":c["seed"]}


def check_evaluation_protocol(c,identity):
    for key,value in evaluation_protocol(c).items():
        if identity.get(key)!=value:
            raise ValueError(f"Saved evaluation protocol mismatch ({key}); scoring/report cannot relabel existing generations")
