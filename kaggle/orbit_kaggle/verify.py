from pathlib import Path
from .storage import open_existing, read_json, write_json
from .config import METHODS,combos


def verify_run(c,root):
    from .calibration import calibration_store,states
    build=open_existing(root,"build")
    act=open_existing(root,"activation")
    artifacts=open_existing(root,"artifacts")
    evaluation=open_existing(root,"evaluation")
    split=open_existing(root,"data").get("split")
    rows=[r for k,r in evaluation.rows()]
    assert set(r["method"] for r in rows)==set(METHODS)
    expected=len(split["test"])*c["generation"]["repeats"]*(1+3*len(combos(c)))
    assert len(rows)==expected,(len(rows),expected)
    ids=[r["sample_id"] for r in rows];assert len(ids)==len(set(ids))
    for method in METHODS[1:]:
        selected=[r for r in rows if r["method"]==method]
        assert all(r["generation"]["intervention"]["hook_calls"]>0 for r in selected),method
        assert sum(r["generation"]["intervention"]["changed_tokens"] for r in selected)>0,f"No measured intervention: {method}"
        assert {tuple(r["targets"]) for r in selected}=={tuple(x) for x in combos(c)}
    assert all(r["generation"]["intervention"]["hook_calls"]==0 for r in rows if r["method"]=="Unsteered")
    for key,row in act.rows():act.load_tensor(key)
    if c["build"].get("extraction")=="generation_prefill":
        assert {key for key,row in build.rows()}=={key for key,row in act.rows()}
        assert all(row["metadata"]["norm_capture"]=="generation_prefill_last" for key,row in act.rows())
        assert not (Path(root)/"filter/manifest.json").exists(),"Paper run must not contain a build judge/filter stage"
    for key,row in artifacts.rows():artifacts.load_tensor(key)
    calibration=calibration_store(c,root)
    for key,state in states(c,calibration):
        assert state["phase"]=="completed" and state["history"]
        assert all(h["judge_cache_keys"] for h in state["history"])
    report_dir=Path(read_json(Path(root)/"latest_report.json")["path"])
    telemetry=read_json(report_dir/"telemetry.json")
    assert telemetry["scoring_requests"]>0
    assert telemetry["parse_success_rate"]>0,"No real parsed Qwen judgments"
    checks={"gpu_pipeline_checks_executed":True,"methods":METHODS,"n_physical_responses":expected,
            "intervention_verified":True,"activation_checksums_shapes_finite":True,
            "calibration_judge_dependency":True,"source_model_ids":c["target_model_id"],
            "judge_revision":c["revisions"][c["judge"]["model_id"]],
            "build_filter_policy":c["build"].get("filter_policy","judge"),
            "build_extraction":c["build"].get("extraction","separate"),
            "resume_test":read_json(Path(root)/"resume_test.json") if (Path(root)/"resume_test.json").exists() else "not_run"}
    if c["mode"]=="smoke" and checks["resume_test"]=="not_run":
        raise AssertionError("Smoke requires notebook 01 intentional interrupted build/resume check")
    write_json(Path(root)/"gpu_verification.json",checks)
    print(checks)
