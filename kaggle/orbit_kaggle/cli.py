from __future__ import annotations
import argparse
import sys
from pathlib import Path
from .config import load_config
from .storage import Budget, BudgetExpired, write_json


def main():
    parser=argparse.ArgumentParser(description="Local-only staged ToneBank benchmark")
    parser.add_argument("stage",choices=["prepare","preflight","model-check","estimate","build-extract","build","filter","extract","artifacts",
                        "calibrate","calibration-generate","calibration-judge","evaluate","score","report","verify"])
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--time-budget",type=int,default=0)
    parser.add_argument("--stop-after",type=int,default=0,help="Intentional interruption after N committed shards; exit 75")
    parser.add_argument("--role",choices=["backbone","judge"],default="judge")
    args=parser.parse_args()
    c=load_config(args.config)
    args.root.mkdir(parents=True,exist_ok=True)
    # Resolved configuration is archived per invocation, never contains HF_TOKEN.
    from .storage import digest
    write_json(args.root/"configs"/(digest(c)+".json"),c)
    budget=Budget(args.time_budget,args.stop_after)
    stage=args.stage
    try:
        if stage=="prepare":
            from .data import prepare
            prepare(c,args.root,budget)
        elif stage=="preflight":
            from .models import preflight
            from .storage import Store
            with Store(args.root,"preflight_stage",{"hardware":c["hardware"],"revisions":c["revisions"]}).transaction():
                preflight(c,args.root)
        elif stage=="model-check":
            from .models import LocalModel
            from .storage import Store
            from .config import model_identity
            store=Store(args.root,"model_check/"+args.role,{"backbone":model_identity(c),"judge":c["judge"],"revisions":c["revisions"]})
            with store.transaction():
                runtime=LocalModel(c,judge=args.role=="judge")
                if args.role=="judge":
                    from .judge import LocalQwenJudge
                    judge=LocalQwenJudge(c,args.root,runtime)
                    result=judge.score("What is 2 + 2?","2 + 2 = 4.",["concise"])
                    store.put("result",result)
                    if result["status"]!="ok":
                        raise RuntimeError("Real Qwen3.5 judge smoke failed. Inspect saved raw output/schema/truncation.")
                else:
                    result=runtime.generate("What is 2 + 2?",42,32)
                    store.put("result",result)
                    if not result["text"].strip():raise RuntimeError("Empty backbone smoke output")
                write_json(args.root/("model_check_"+args.role+".json"),result)
        elif stage=="estimate":
            from .stages import estimate
            from .storage import Store
            with Store(args.root,"estimate_stage",{"config":c}).transaction():
                print(estimate(c,args.root))
        elif stage=="build-extract":
            from .fused_build import generate_extract
            generate_extract(c,args.root,budget)
        elif stage=="build":
            if c["build"].get("extraction") == "generation_prefill":
                raise ValueError("Use build-extract for paper system prompts and inline activation capture")
            from .stages import generate_build
            generate_build(c,args.root,budget)
        elif stage=="filter":
            if c["build"].get("filter_policy") == "none":
                raise ValueError("Paper workflow has no build judge/filter stage; run build-extract then artifacts")
            from .stages import filter_build
            filter_build(c,args.root,budget)
        elif stage=="extract":
            if c["build"].get("extraction") == "generation_prefill":
                raise ValueError("Paper activations are captured in build-extract; no separate extract stage")
            from .artifacts import extract
            extract(c,args.root,budget)
        elif stage=="artifacts":
            from .artifacts import build_artifacts
            build_artifacts(c,args.root,budget)
        elif stage=="calibrate":
            from .calibration import orchestrate
            orchestrate(args.config,c,args.root,budget)
        elif stage=="calibration-generate":
            from .calibration import calibration_generate
            calibration_generate(c,args.root,budget)
        elif stage=="calibration-judge":
            from .calibration import calibration_judge
            calibration_judge(c,args.root,budget)
        elif stage=="evaluate":
            from .stages import generate_evaluation
            generate_evaluation(c,args.root,budget)
        elif stage in ("score","report"):
            from .report import report
            if stage=="score":
                from .stages import judge_evaluation
                store=judge_evaluation(c,args.root,budget)
                report(c,args.root,store)
            else:
                from .judge import LocalQwenJudge
                from .storage import Store,open_existing,digest
                evaluation=open_existing(args.root,"evaluation");judge=LocalQwenJudge(c,args.root)
                identity={"evaluation":evaluation.fp,"content":digest(list(evaluation.rows())),"judge":judge.identity,
                          "judge_types":c["report"]["judge_types"],"tone_comparison":c["report"]["tone_comparison"],
                          "comparison_protocol":"source_target_if_k1_else_avoid_only_v1"}
                store=open_existing(args.root,"scoring/"+digest(identity))
                report(c,args.root,store)
        elif stage=="verify":
            from .verify import verify_run
            verify_run(c,args.root)
    except BudgetExpired as e:
        print(str(e),file=sys.stderr)
        return 75
    return 0


if __name__=="__main__":raise SystemExit(main())
