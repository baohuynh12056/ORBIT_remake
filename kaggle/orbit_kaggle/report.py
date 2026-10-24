from __future__ import annotations
import io
from pathlib import Path
from .config import combos, METHODS, check_evaluation_protocol
from .storage import open_existing, Store, digest, write_json, atomic_bytes, read_json


def csv_bytes(rows):
    import pandas as pd
    return pd.DataFrame(rows).to_csv(index=False).encode()


def cluster_ci(frame,column,seed=42,draws=2000):
    import numpy as np
    values=frame.groupby("question_id")[column].mean().dropna().to_numpy()
    if len(values)<2:
        return [None,None]
    rng=np.random.default_rng(seed)
    samples=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(draws)])
    return np.quantile(samples,[0.025,0.975]).tolist()


def report(c,root,scored):
    import numpy as np
    import pandas as pd
    evaluation=open_existing(root,"evaluation")
    check_evaluation_protocol(c,evaluation.identity)
    identity={"evaluation":evaluation.fp,"scoring":scored.fp,"report":c["report"],"schema":1}
    out=Store(root,"reports/"+digest(identity),identity)
    generations=dict(evaluation.rows()); judgments=dict(scored.rows())
    baselines={(r["question_id"],r["repeat"]):r for r in generations.values() if r["method"]=="Unsteered"}
    long,errors=[],[]
    with out.transaction():
        for sid,row in generations.items():
            baseline=baselines[(row["question_id"],row["repeat"])]
            contexts=combos(c) if row["method"]=="Unsteered" else [row["targets"]]
            for targets in contexts:
                combo=",".join(targets)
                for kind in c["report"]["judge_types"]:
                    j=judgments[sid]["judgments"][combo][kind]
                    bj=judgments[baseline["sample_id"]]["judgments"][combo][kind]
                    record={"model":c["target_model_id"],"model_revision":c["revisions"][c["target_model_id"]],
                          "precision":c["precision"],"evaluator":c["judge"]["model_id"],
                          "evaluator_revision":c["revisions"][c["judge"]["model_id"]],"evaluator_precision":c["judge"]["precision"],
                          "evaluator_fingerprint":digest(j["evaluator"]),"split_mode":c["split"]["mode"],
                          "build_filter_policy":c["build"].get("filter_policy","judge"),
                          "build_prompt_role":c["build"].get("prompt_role","user"),
                          "build_max_new_tokens":c["build"]["max_new_tokens"],
                          "method":row["method"],"K":len(targets),"combo":combo,"question_id":row["question_id"],
                          "sample_id":sid,"baseline_sample_id":baseline["sample_id"],"baseline_reference":row["method"]=="Unsteered",
                          "repeat":row["repeat"],"seed":row["generation"]["seed"],"judge_type":kind,"status":j["status"],
                          "baseline_status":bj["status"],"retry_count":j["retry_count"],
                          "judge_cache_key":j["cache_key"],"judge_elapsed_s":sum(a.get("elapsed_s") or 0 for a in j["attempts"]),
                          "judge_truncated":any(a["truncated"] for a in j["attempts"]),
                          "response_truncated":row["generation"]["truncated"],"strength":row["strength"],
                          "generation_elapsed_s":row["generation"]["elapsed_s"],"target_mean":None,"coherence":None,
                          "generated_tokens":row["generation"]["generated_tokens"],
                          "delta_tone":None,"delta_coherence":None,"all_targets_improve":None,
                          "fraction_targets_improve":None,"minimum_delta":None,"geometric_gain":None,"steering_success":None,
                          "activation_target_probability":sum(row["eval_classifier_probabilities"][t] for t in targets)/len(targets)}
                    peaks={v["device"]:v["peak"] for v in row["generation"]["resources"].get("vram",[])}
                    for device in (0,1):record[f"peak_vram_gpu{device}_bytes"]=peaks.get(device)
                    if j["status"]=="ok":
                        s=j["scores"]
                        record.update(target_mean=sum(s["target_scores"][t] for t in targets)/len(targets),coherence=s["coherence"])
                        for target in targets:
                            record["tone_"+target]=s["target_scores"][target]
                    if j["status"]==bj["status"]=="ok":
                        deltas=np.array([j["scores"]["target_scores"][t]-bj["scores"]["target_scores"][t] for t in targets])
                        record.update(delta_tone=float(deltas.mean()),delta_coherence=j["scores"]["coherence"]-bj["scores"]["coherence"],
                                      all_targets_improve=float((deltas>0).all()),fraction_targets_improve=float((deltas>0).mean()),
                                      minimum_delta=float(deltas.min()),geometric_gain=float(np.exp(np.log(deltas).mean())) if (deltas>0).all() else 0.0,
                                      steering_success=float((deltas>0).all() and j["scores"]["coherence"]>=c["report"]["success_coherence_threshold"]))
                        for target,delta in zip(targets,deltas):
                            record["delta_"+target]=float(delta)
                    if j["status"]!="ok" or bj["status"]!="ok":
                        errors.append({**record,"judge_cache_key":j["cache_key"],"baseline_judge_cache_key":bj["cache_key"]})
                    comp=judgments[sid]["comparison"]
                    if comp:
                        valid=[v for v in comp.values() if v["status"]=="ok"]
                        record["tonejudge_n_valid"]=len(valid)
                        record["tonejudge_success"]=sum(v["scores"]["steering_successful"] for v in valid)/len(valid) if valid else None
                    long.append(record)
        frame=pd.DataFrame(long)
        summary,detail=[],[]
        metrics=["target_mean","coherence","delta_tone","delta_coherence","all_targets_improve","fraction_targets_improve",
                 "minimum_delta","geometric_gain","steering_success","activation_target_probability"]
        metrics += [x for x in frame.columns if x.startswith(("tone_","delta_")) and x not in metrics and x!="tone_comparison"]
        for groups,destination in [(["model","method","K","judge_type"],summary),(["model","method","K","combo","judge_type"],detail)]:
            for key,group in frame.groupby(groups,dropna=False):
                r=dict(zip(groups,key))
                r.update(n_questions=group.question_id.nunique(),n_physical_responses=group.sample_id.nunique(),
                         n_scoring_contexts=len(group),n_valid_contexts=int((group.status=="ok").sum()),
                         n_error_contexts=int((group.status!="ok").sum()),n_paired_valid=int(group.delta_tone.notna().sum()),
                         n_missing_score_contexts=int(group.target_mean.isna().sum()),
                         n_failed_baseline_contexts=int((group.baseline_status!="ok").sum()),
                         evaluator=c["judge"]["model_id"],evaluator_revision=c["revisions"][c["judge"]["model_id"]],
                         evaluator_precision=c["judge"]["precision"],evaluator_fingerprint=group.evaluator_fingerprint.iloc[0],
                         model_revision=c["revisions"][c["target_model_id"]],precision=c["precision"],split_mode=c["split"]["mode"])
                r.update(build_filter_policy=c["build"].get("filter_policy","judge"),
                         build_prompt_role=c["build"].get("prompt_role","user"),build_max_new_tokens=c["build"]["max_new_tokens"])
                r["generation_elapsed_s_unique"]=float(group.drop_duplicates("sample_id").generation_elapsed_s.sum())
                r["generated_tokens_unique"]=int(group.drop_duplicates("sample_id").generated_tokens.sum())
                r["generation_tokens_per_second"]=r["generated_tokens_unique"]/max(r["generation_elapsed_s_unique"],1e-9)
                r["judge_elapsed_s_unique_cache"]=float(group.drop_duplicates("judge_cache_key").judge_elapsed_s.sum())
                for device in (0,1):r[f"peak_vram_gpu{device}_bytes"]=group[f"peak_vram_gpu{device}_bytes"].max()
                for metric in metrics:
                    if metric not in group or not pd.api.types.is_numeric_dtype(group[metric]):
                        continue
                    # Equal weight to question clusters; all contexts/repeats within a cluster are averaged first.
                    r[metric]=group.groupby("question_id")[metric].mean().mean()
                    ci=cluster_ci(group,metric,c["seed"],c["report"]["bootstrap_draws"])
                    r[metric+"_ci_low"],r[metric+"_ci_high"]=ci
                destination.append(r)
        atomic_bytes(out.root/"per_sample.csv",csv_bytes(long))
        atomic_bytes(out.root/"summary_model_method_K.csv",csv_bytes(summary))
        atomic_bytes(out.root/"detail_combinations.csv",csv_bytes(detail))
        atomic_bytes(out.root/"errors.csv",csv_bytes(errors))
        # Unique cache requests, not aliases of the same reused baseline judgment.
        unique={}
        for row in judgments.values():
            for context in row["judgments"].values():
                for j in context.values(): unique[j["cache_key"]]=j
            for j in row["comparison"].values(): unique[j["cache_key"]]=j
        js=list(unique.values())
        filter_policy=c["build"].get("filter_policy","judge")
        if filter_policy == "none":
            filter_rows=[]
        else:
            filtering=open_existing(root,"filter")
            filter_rows=[r for k,r in filtering.rows()]
        gen=list(generations.values())
        n=len(js)
        telemetry={"config":c,"evaluator":js[0]["evaluator"] if js else None,
           "scoring_requests":n,"parse_success_rate":sum(j["status"]=="ok" for j in js)/n if n else None,
           "parse_failure_count":sum(j["status"]!="ok" for j in js),
           "judge_truncation_request_rate":sum(any(a["truncated"] for a in j["attempts"]) for j in js)/n if n else None,
           "unique_response_count":len(gen),"baseline_physical_responses":len(baselines),
           "baseline_reference_rows_are_not_independent":True,
           "evaluation_generation_s":sum(r["generation"]["elapsed_s"] for r in gen),
           "evaluation_generated_tokens":sum(r["generation"]["generated_tokens"] for r in gen),
           "peak_vram_bytes_by_device":{},"filter_pairs":len(filter_rows),
           "filter_accepted":sum(r["decision"]["keep"] for r in filter_rows),
           "filter_technical_unscored":sum(r["decision"]["status"]=="unscored" for r in filter_rows),
           "stage_manifests":{},"output_bytes":sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file()),
           "bootstrap_unit":"question_id; contexts and repeats averaged within question; percentile CI",
           "success_definition":"all target scores strictly exceed paired baseline AND coherence >= configured threshold",
           "paper_joint_fraction":"fraction of targets strictly exceeding paired baseline",
           "sensitivity_note":"Threshold sweep is descriptive only; never used to tune from test data."}
        for row in gen:
            for v in row["generation"]["resources"].get("vram",[]):
                key=str(v["device"]);telemetry["peak_vram_bytes_by_device"][key]=max(telemetry["peak_vram_bytes_by_device"].get(key,0),v["peak"])
        for p in Path(root).rglob("manifest.json"):
            if "reports" not in p.parts:
                telemetry["stage_manifests"][str(p.relative_to(root))]=read_json(p)
        telemetry["tokens_per_generation_second"] = telemetry["evaluation_generated_tokens"]/max(telemetry["evaluation_generation_s"],1e-9)
        def quality(requests):
            requests=list({j["cache_key"]:j for j in requests}.values())
            count=len(requests)
            return {"unique_requests":count,"valid":sum(j["status"]=="ok" for j in requests),
                    "technical_errors":sum(j["status"]!="ok" for j in requests),
                    "parse_success_rate":sum(j["status"]=="ok" for j in requests)/count if count else None,
                    "truncated_requests":sum(any(a["truncated"] for a in j["attempts"]) for j in requests),
                    "attempts":sum(len(j["attempts"]) for j in requests)}
        build_judges=[j for row in filter_rows if "judgments" in row for j in row["judgments"].values()]
        calibration_judges=[]
        if (Path(root)/"calibration/manifest.json").exists():
            calibration=open_existing(root,"calibration")
            calibration_judges=[v for key,v in calibration.rows() if key.startswith("judgment/")]
        telemetry["judge_quality_by_stage"]={"build_filter":quality(build_judges),"calibration":quality(calibration_judges),"final":quality(js)}
        telemetry["build_filter_policy"]=filter_policy
        telemetry["judge_quality_by_stage"]["build_filter"]["applied"]=filter_policy!="none"
        if filter_policy == "none":
            telemetry.update(filter_pairs=None,filter_accepted=None,filter_technical_unscored=None)
            build_rows=list(open_existing(root,"build").rows()) if (Path(root)/"build/manifest.json").exists() else []
            telemetry["build_pairs_by_label"]={label:sum(r["label"]==label for key,r in build_rows) for label in c["labels"]}
            telemetry["build_response_truncations"]={side:sum(r[side]["truncated"] for key,r in build_rows) for side in ("pos","neg")}
        evaluator_peaks={}
        for judgment in build_judges+calibration_judges+js:
            for attempt in judgment["attempts"]:
                for device in attempt.get("resources",{}).get("vram",[]):
                    key=str(device["device"]);evaluator_peaks[key]=max(evaluator_peaks.get(key,0),device["peak"])
        telemetry["evaluator_peak_vram_bytes_by_device"]=evaluator_peaks
        telemetry["filter_by_label"]={label:{"total":sum(r.get("label")==label for r in filter_rows),
                "accepted":sum(r.get("label")==label and r["decision"]["keep"] for r in filter_rows),
                "technical_unscored":sum(r.get("label")==label and r["decision"]["status"]=="unscored" for r in filter_rows)} for label in c["labels"]}
        if filter_policy == "none":
            telemetry["filter_by_label"]=None
        write_json(out.root/"telemetry.json",telemetry)
        # Distribution and threshold sensitivity, kept separate from the fixed primary protocol.
        distributions=frame.groupby(["method","judge_type"])[["target_mean","coherence"]].describe()
        atomic_bytes(out.root/"score_distributions.csv",distributions.to_csv().encode())
        sensitivity=[]
        for threshold in (40,50,60,70):
            for (method,kind),group in frame[frame.method!="Unsteered"].groupby(["method","judge_type"]):
                valid=group[group.all_targets_improve.notna()].copy()
                valid["sensitivity_success"]=(valid.all_targets_improve.astype(bool)&(valid.coherence>=threshold)).astype(float)
                sensitivity.append({"method":method,"judge_type":kind,"coherence_threshold":threshold,
                    "success":valid.groupby("question_id").sensitivity_success.mean().mean(),"n_questions":valid.question_id.nunique()})
        atomic_bytes(out.root/"threshold_sensitivity.csv",csv_bytes(sensitivity))
        build_sensitivity=[]
        for threshold in (40,50,60,70,80):
            for label in c["labels"]:
                eligible=[r for r in filter_rows if r.get("label")==label and all(j["status"]=="ok" for j in r["judgments"].values())]
                kept=sum(r["judgments"]["pos"]["scores"]["target_scores"][label]*10>=threshold and
                         r["judgments"]["neg"]["scores"]["target_scores"][label]*10<100-threshold and
                         min(r["judgments"][side]["scores"]["coherence"] for side in ("pos","neg"))>=50 for r in eligible)
                build_sensitivity.append({"label":label,"pos_threshold":threshold,"neg_threshold":100-threshold,
                                          "n_scored":len(eligible),"kept":kept,"applied_to_training":False})
        if filter_policy != "none":
            atomic_bytes(out.root/"build_filter_threshold_sensitivity.csv",csv_bytes(build_sensitivity))
        write_json(out.root/"build_protocol.json",{"filter_policy":filter_policy,"build":c["build"],
                   "note":"All generated pairs used; no judge selection" if filter_policy=="none" else "Judge-filtered repo protocol"})
        # A portable Markdown entry point; CSV keeps full numeric precision.
        text=("# Báo cáo ToneBank\n\nEvaluator: "+c["judge"]["model_id"]+" @ "+c["revisions"][c["judge"]["model_id"]]+"\n\n"
              "Đây là đánh giá lại với Qwen local, không phải tái lập evaluator GPT-4o-mini.\n\n"
              f"Split: {c['split']['mode']}; backbone precision: {c['precision']}; judge: {c['judge']['precision']}.\n\n"
              f"Build POS/NEG filter: {filter_policy}; build tokens: {c['build']['max_new_tokens']}; prompt role: {c['build'].get('prompt_role','user')}.\n\n"
              "Xem summary_model_method_K.csv, detail_combinations.csv, per_sample.csv, errors.csv và telemetry.json. "
              "Baseline chỉ sinh một lần mỗi question/seed; các dòng baseline theo tổ hợp là tham chiếu. "
              "CI bootstrap theo question_id, không dùng số dòng tham chiếu làm cỡ mẫu độc lập.\n")
        atomic_bytes(out.root/"REPORT.md",text.encode())
    write_json(Path(root)/"latest_report.json",{"path":str(out.root),"relative_path":str(out.root.relative_to(root)),"fingerprint":out.fp})
    return out


def combine_reports(roots,destination):
    """Combine the three real completed backbone runs; no synthetic/missing-model rows."""
    import pandas as pd
    from .config import MODELS
    paths=[Path(root)/read_json(Path(root)/"latest_report.json")["relative_path"] for root in roots]
    frames=[pd.read_csv(path/"summary_model_method_K.csv") for path in paths]
    result=pd.concat(frames,ignore_index=True)
    if set(result.model)!=set(MODELS):
        raise ValueError("Full combined report requires the exact three evaluated backbones")
    for column in ("evaluator_revision","evaluator_precision","evaluator_fingerprint","precision","split_mode"):
        if result[column].nunique()!=1:
            raise ValueError(f"Incompatible runs in combined report: {column}")
    configs=[read_json(path/"telemetry.json")["config"] for path in paths]
    for section in ("dataset","split","generation","build","calibration","report","k_values"):
        if len({digest(config[section]) for config in configs})!=1:
            raise ValueError(f"Incompatible protocol in combined report: {section}")
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    atomic_bytes(destination/"all_models_method_K.csv",result.to_csv(index=False).encode())
    atomic_bytes(destination/"all_combinations.csv",pd.concat([pd.read_csv(p/"detail_combinations.csv") for p in paths]).to_csv(index=False).encode())
