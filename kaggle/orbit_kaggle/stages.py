from __future__ import annotations
import time
from pathlib import Path
from .storage import Store, open_existing, digest, write_json
from .config import model_identity, stable_seed, combos, METHODS, evaluation_protocol, check_evaluation_protocol


def generate_build(c,root,budget):
    if c["build"].get("extraction")=="generation_prefill":
        raise ValueError("Paper build requires generate_extract; legacy generation would drop activation and system-role provenance")
    from .data import build_pairs
    from .models import LocalModel
    data=open_existing(root,"data")
    pairs=build_pairs(c,data.get("split"))
    identity={"data":data.fp,"model":model_identity(c),"build":c["build"],"seed":c["seed"],"pair_prompts":digest(pairs)}
    if c["build"]["response_source"]=="qwen_judge":
        identity["response_evaluator"]={"revision":c["revisions"][c["judge"]["model_id"]],"config":c["judge"]}
    store=Store(root,"build",identity)
    with store.transaction():
        runtime=None
        store.log("generation_policy",requested_batch=c["generation"]["batch_size"],effective_batch=1,
                  reason="per_sample_seed_independent_of_shard_and_resume")
        for pair in pairs:
            key=pair["pair_id"]
            if store.get(key) is not None:
                continue
            budget.check()
            if runtime is None:
                runtime=LocalModel(c,judge=c["build"]["response_source"]=="qwen_judge")
            row={**pair,"response_source":c["build"]["response_source"]}
            # Side records survive interruption even before a complete pair is committed.
            for side in ("pos","neg"):
                side_path=store.root / "partial" / f"{key}-{side}.json"
                if side_path.exists():
                    from .storage import read_json
                    partial=read_json(side_path)
                    if partial["fingerprint"]!=store.fp or digest(partial["payload"])!=partial["checksum"]:
                        raise ValueError("Partial response checksum/config mismatch")
                    row[side]=partial["payload"]
                else:
                    budget.check()
                    result=runtime.generate(pair[side+"_prompt"], stable_seed(c,pair["question_id"],namespace=f"build/{pair['label']}/{side}/{key}"),
                                           c["build"]["max_new_tokens"],sampling=True)
                    write_json(side_path,{"fingerprint":store.fp,"payload":result,"checksum":digest(result)})
                    row[side]=result
            store.put(key,row)
            store.log("build_response",model=c["target_model_id"],sample_id=key,combo=[pair["label"]],question_id=pair["question_id"],
                      elapsed_s=sum(row[s]["elapsed_s"] for s in ("pos","neg")))
            budget.committed()
    return store


def filter_build(c,root,budget):
    if c["build"].get("filter_policy")=="none":
        raise ValueError("Judge filtering is disabled by this run's paper protocol")
    from .judge import LocalQwenJudge,filter_decision
    build=open_existing(root,"build")
    judge=LocalQwenJudge(c,root)
    store=Store(root,"filter",{"build":build.fp,"content":digest(list(build.rows())),"judge":judge.identity,
              "thresholds":{k:c["build"][k] for k in ("pos_threshold","neg_threshold","coherence_threshold","filter_judge")}})
    with store.transaction():
        for key,pair in build.rows():
            if store.get(key) is not None:
                continue
            budget.check()
            judgments=judge.filter_pair(pair)
            decision=filter_decision(pair,judgments,c)
            store.put(key,{"pair_id":key,"judgments":judgments,"decision":decision,"label":pair["label"]})
            store.log("filter",model=c["target_model_id"],sample_id=key,combo=[pair["label"]],decision=decision)
            budget.committed()
    return store


def generate_evaluation(c,root,budget):
    from .models import LocalModel
    from .steering import Steering,hook_signature
    from .calibration import calibration_store,states,state_key
    from .artifacts import extract_features
    from src.steering.models import ActivationSteering
    import torch
    data=open_existing(root,"data"); artifacts=open_existing(root,"artifacts")
    calibration=calibration_store(c,root)
    current=dict(states(c,calibration))
    if any(s["phase"]!="completed" or s["selected"] is None for s in current.values()):
        raise RuntimeError("Calibration search is incomplete; resume the calibration notebook")
    selected={key:s["selected"] for key,s in current.items()}
    store=Store(root,"evaluation",{**evaluation_protocol(c),"artifacts":artifacts.fp,"data":data.fp,"strengths":selected})
    with store.transaction():
        runtime=None;steering=None
        checkpoint=artifacts.load_tensor("eval")
        clf=ActivationSteering(checkpoint["input_dim"],len(c["labels"]),hidden_dim=c["classifier"]["hidden_dim"],device="cpu")
        clf.classifier.load_state_dict(checkpoint["state_dict"]);clf.classifier.eval()
        # One physical baseline per question/repeat. Projection into combinations occurs only in reporting.
        jobs=[("Unsteered",[],q,r) for q in data.get("split")["test"] for r in range(c["generation"]["repeats"])]
        jobs += [(method,targets,q,r) for targets in combos(c) for method in METHODS[1:] for q in data.get("split")["test"]
                 for r in range(c["generation"]["repeats"])]
        for method,targets,q,repeat in jobs:
            key=digest([method,targets,q["question_id"],repeat])
            if store.get(key) is not None:
                continue
            budget.check()
            if runtime is None:
                runtime=LocalModel(c);steering=Steering(c,root,runtime)
            strength=0.0 if method=="Unsteered" else selected[state_key(method,targets)]
            result=steering.generate(q["text"],stable_seed(c,q["question_id"],repeat),method,targets,strength,c["generation"]["max_new_tokens"])
            # Evaluate final hidden state of the saved output as in source; no generation hook remains.
            features=extract_features(runtime,[q["text"]+"\n"+result["text"]],c["steering"]["layer"])
            probabilities=clf.predict_proba(features["eval"].numpy())[0].tolist()
            store.put(key,{"sample_id":key,"question_id":q["question_id"],"question":q["text"],"method":method,"targets":targets,
                           "repeat":repeat,"generation":result,"eval_classifier_probabilities":dict(zip(c["labels"],probabilities)),
                           "strength":strength})
            store.log("evaluation_response",model=c["target_model_id"],method=method,combo=targets,sample_id=key,
                      question_id=q["question_id"],elapsed_s=result["elapsed_s"])
            budget.committed()
    return store


def judge_evaluation(c,root,budget):
    from .judge import LocalQwenJudge
    evaluation=open_existing(root,"evaluation")
    check_evaluation_protocol(c,evaluation.identity)
    judge=LocalQwenJudge(c,root)
    identity={"evaluation":evaluation.fp,"content":digest(list(evaluation.rows())),"judge":judge.identity,
              "judge_types":c["report"]["judge_types"],"tone_comparison":c["report"]["tone_comparison"],
              "comparison_protocol":"source_target_if_k1_else_avoid_only_v1"}
    store=Store(root,"scoring/"+digest(identity),identity)
    rows=dict(evaluation.rows())
    baselines={(row["question_id"],row["repeat"]):row for row in rows.values() if row["method"]=="Unsteered"}
    with store.transaction():
        for key,row in rows.items():
            if store.get(key) is not None:
                continue
            budget.check()
            # Score identical target sets/rubrics on each paired baseline and steered response.
            # Baseline generation is reused, but its 25 scoring contexts remain references,
            # never 25 independent response observations.
            contexts=combos(c) if row["method"]=="Unsteered" else [row["targets"]]
            judgments={",".join(targets):{kind:judge.score(row["question"],row["generation"]["text"],targets,kind)
                        for kind in c["report"]["judge_types"]} for targets in contexts}
            comparison={}
            if c["report"]["tone_comparison"] and row["method"]!="Unsteered":
                baseline=baselines[(row["question_id"],row["repeat"])]
                avoid=next(t for t in c["labels"] if t not in row["targets"])
                # Match launcher: target_style for K=1; avoid-only for K>1.
                target=row["targets"][0] if len(row["targets"])==1 else None
                comparison[target or "avoid_only"]=judge.score(row["question"],row["generation"]["text"],[target] if target else [],"comparison",
                                                             baseline=baseline["generation"]["text"],avoid_style=avoid)
            store.put(key,{"sample_id":key,"judgments":judgments,"comparison":comparison})
            budget.committed()
    return store


def estimate(c,root):
    from transformers import AutoConfig
    from .config import MODELS
    import shutil
    from .storage import read_json
    preflight_path=Path(root)/"preflight.json"
    access=read_json(preflight_path).get("access",{}) if preflight_path.exists() else {}
    result={"scope":"upper bounds before filtering; no model weights loaded","models":{}}
    for mid in MODELS:
        conf=AutoConfig.from_pretrained(mid,revision=c["revisions"][mid],trust_remote_code=False)
        n,d=conf.num_hidden_layers,conf.hidden_size
        pairs=len(c["labels"])*c["build"]["max_pairs"]
        activation_bytes=pairs*2*(2*n+2)*d*4
        evals=c["generation"]["n_questions"]*c["generation"]["repeats"]*(1+3*len(combos(c)))
        cal=len(combos(c))*3*c["calibration"]["iterations"]*c["split"]["calibration_samples"]
        # Conservative text/raw-judge budget (bytes/token varies by language), plus 3 judge attempts.
        text_bytes=(pairs*2*c["build"]["max_new_tokens"]+evals*c["generation"]["max_new_tokens"]+cal*c["calibration"]["max_new_tokens"])*16
        filter_calls=0 if c["build"].get("filter_policy")=="none" else pairs*2
        judge_bytes=(filter_calls+evals*len(c["report"]["judge_types"])+cal)*c["judge"]["max_attempts"]*(c["judge"]["max_new_tokens"]*16+32768)
        result["models"][mid]={"layers":n,"hidden_dim":d,"activation_float32_bytes":activation_bytes,
             "physical_eval_responses":evals,"calibration_upper_responses":cal,"estimated_output_bytes":activation_bytes+text_bytes+judge_bytes,
             "one_pair_shard_bytes":2*(2*n+2)*d*4,"checkpoint_weights_bytes_from_hub":access.get(mid,{}).get("weights_bytes"),
             "output_partition":"one_backbone_cumulative_run; sample shards; <=1GiB uncompressed archive parts"}
    result["free_disk_bytes"]=shutil.disk_usage(root).free
    result["output_soft_budget_bytes"]=18*2**30
    selected=result["models"][c["target_model_id"]]["estimated_output_bytes"]
    result["selected_model_fits_estimated_output_budget"]=selected<min(result["free_disk_bytes"],result["output_soft_budget_bytes"])
    write_json(Path(root)/"estimate.json",result)
    if not result["selected_model_fits_estimated_output_budget"]:
        raise RuntimeError("Conservative estimated output exceeds disk/18GiB budget. Split runs/versions before full; do not silently shrink the experiment. See estimate.json.")
    return result
