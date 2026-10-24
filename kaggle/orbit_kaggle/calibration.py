from __future__ import annotations
import subprocess
import sys
import time
from .config import combos, model_identity, stable_seed
from .storage import Store, open_existing, digest


def calibration_store(c,root):
    from .judge import LocalQwenJudge
    artifacts=open_existing(root,"artifacts")
    data=open_existing(root,"data")
    if artifacts.identity.get("algorithm")!=model_identity(c) or artifacts.identity.get("classifier")!=c["classifier"]:
        raise ValueError("Steering artifacts belong to a different backbone/revision/precision/implementation")
    if data.identity["dataset"]!=c["dataset"] or data.identity["split"]!=c["split"]:
        raise ValueError("Saved question split does not match calibration config")
    evaluator=LocalQwenJudge(c,root)
    return Store(root,"calibration",{"backbone":model_identity(c),"artifacts":artifacts.fp,
                 "data":data.fp,"search":c["calibration"],"judge":evaluator.identity,
                 "combos":combos(c),"seed":c["seed"]})


def initial_state(c,method,targets):
    lo,hi=c["calibration"]["boost_range" if method=="ORBIT" else "alpha_range"]
    return {"method":method,"targets":targets,"lo":lo,"hi":hi,"last_good":lo,
            "iteration":0,"phase":"pending_generation","history":[],"selected":None}


def state_key(method,targets):
    return method+"/"+",".join(targets)


def states(c,store):
    result=[]
    for targets in combos(c):
        for method in ("CAA","K-Steer","ORBIT"):
            key=state_key(method,targets)
            state=store.get(key)
            if state is None:
                state=initial_state(c,method,targets);store.put(key,state)
            result.append((key,state))
    return result


def update_search(state,judgments,c):
    errors=sum(j["status"]!="ok" for j in judgments)
    if errors and c["calibration"]["technical_error_policy"]=="stop":
        raise RuntimeError("Calibration has technical judge errors. Raw outputs are saved; fix judge in a NEW calibration directory, or explicitly select conservative_ood policy. No missing score became quality zero.")
    if not judgments:
        raise ValueError("Empty calibration judgments")
    # Explicit compatibility policy: missing scores count as OOD flags, not quality measurements.
    bad=sum(j["status"]!="ok" or j["scores"]["coherence"]<c["calibration"]["coherence_threshold"] for j in judgments)
    is_ood=100*bad/len(judgments)>c["calibration"]["max_bad_percent"]
    mid=(state["lo"]+state["hi"])/2
    state["history"].append({"round":state["iteration"],"strength":mid,"ood":is_ood,
                            "n":len(judgments),"n_technical_error":errors,"bad_count":bad,
                            "judge_cache_keys":[j["cache_key"] for j in judgments]})
    if is_ood:
        state["hi"]=mid
    else:
        state["lo"]=mid;state["last_good"]=mid
    state["iteration"]+=1
    done=state["iteration"]>=c["calibration"]["iterations"]
    if state["method"]!="ORBIT":
        done=done or state["hi"]/state["lo"]<=1.05
    state["phase"]="completed" if done else "pending_generation"
    if done:
        state["selected"]=state["last_good"]
    return state


def calibration_generate(c,root,budget):
    from .models import LocalModel
    from .steering import Steering
    store=calibration_store(c,root)
    split=open_existing(root,"data").get("split")
    requests=[(k,s) for k,s in states(c,store) if s["phase"]=="pending_generation"]
    if not requests:
        return
    with store.transaction():
        budget.check()
        runtime=LocalModel(c); steering=Steering(c,root,runtime)
        for key,state in requests:
            mid=(state["lo"]+state["hi"])/2
            for q in split["calibration"]:
                budget.check()
                gkey=f"generation/{key}/{state['iteration']}/{q['question_id']}"
                if store.get(gkey) is not None:
                    continue
                result=steering.generate(q["text"],stable_seed(c,q["question_id"],namespace="calibration"),state["method"],
                    state["targets"],mid,c["calibration"]["max_new_tokens"])
                store.put(gkey,{"question":q,"generation":result,"method":state["method"],"targets":state["targets"],"strength":mid})
                store.log("calibration_generation",model=c["target_model_id"],method=state["method"],combo=state["targets"],
                          sample_id=q["question_id"],round=state["iteration"],elapsed_s=result["elapsed_s"])
            state["phase"]="pending_judge";store.put(key,state)


def calibration_judge(c,root,budget):
    from .judge import LocalQwenJudge
    store=calibration_store(c,root)
    split=open_existing(root,"data").get("split")
    judge=LocalQwenJudge(c,root)
    with store.transaction():
        for key,state in states(c,store):
            if state["phase"]!="pending_judge":
                continue
            judgments=[]
            for q in split["calibration"]:
                budget.check()
                gkey=f"generation/{key}/{state['iteration']}/{q['question_id']}"
                jkey=gkey.replace("generation/","judgment/",1)
                result=store.get(jkey)
                if result is None:
                    generation=store.get(gkey)
                    result=judge.score(q["text"],generation["generation"]["text"],kind="coherence")
                    store.put(jkey,result)
                judgments.append(result)
            store.put(key,update_search(state,judgments,c))


def orchestrate(config_path,c,root,budget):
    """Workers exit between loads. State advances only after actual judge results."""
    store=calibration_store(c,root)
    while any(s["phase"]!="completed" for k,s in states(c,store)):
        for phase in ("calibration-generate","calibration-judge"):
            budget.check()
            # Remaining budget is shared with worker; no fixed alpha or fabricated judge.
            remaining=max(1,int(budget.deadline-time.monotonic())) if budget.deadline!=float("inf") else 0
            cmd=[sys.executable,"-m","orbit_kaggle.cli",phase,"--config",str(config_path),"--root",str(root),"--time-budget",str(remaining)]
            result=subprocess.run(cmd)
            if result.returncode==75:
                from .storage import BudgetExpired
                raise BudgetExpired("Calibration worker checkpointed; resume the same search state next session")
            result.check_returncode()
    # Worker transaction completion is not equivalent to the entire search being complete.
    with store.transaction():
        if any(s["selected"] is None for k,s in states(c,store)):
            raise RuntimeError("Incomplete calibration")
