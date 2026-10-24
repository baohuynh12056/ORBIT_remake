"""CPU contract tests. Random tiny models/fixtures are tests, NEVER ToneBank results."""
from __future__ import annotations
import copy
import importlib.abc
import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from orbit_kaggle.storage import Store,Budget,BudgetExpired,read_json,digest
from orbit_kaggle.config import load_config,combos,stable_seed
from orbit_kaggle.judge import validate,LocalQwenJudge,filter_decision
from orbit_kaggle.calibration import initial_state,update_search

ROOT=Path(__file__).resolve().parents[1]


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.c=load_config(ROOT/"configs/smoke.json")

    def test_all_full_combinations(self):
        c=load_config(ROOT/"configs/full.json")
        self.assertEqual(len(combos(c)),25)
        self.assertEqual([sum(len(x)==k for x in combos(c)) for k in [1,2,3]],[5,10,10])

    def test_json_and_tensor_integrity_and_mismatch(self):
        s=Store(self.root,"a",{"x":1})
        with s.transaction():s.save_tensor("q",{"h":torch.ones(2,4)},{"sample_id":"q"})
        self.assertEqual(s.load_tensor("q")["h"].shape,(2,4))
        with self.assertRaises(ValueError):Store(self.root,"a",{"x":2})
        p=s.root/"tensors"/s.get("q")["tensor_file"]
        p.write_bytes(p.read_bytes()+b"bad")
        with self.assertRaises(ValueError):s.load_tensor("q")

    def test_interrupted_shard_resume(self):
        s=Store(self.root,"a",{"x":1});b=Budget(stop_after=1)
        with self.assertRaises(BudgetExpired):
            with s.transaction():
                s.put("first",{"id":"first","seed":stable_seed(self.c,"q")});b.committed()
        self.assertEqual(read_json(s.root/"manifest.json")["status"],"pending")
        with s.transaction():
            if s.get("first") is None:self.fail("Completed sample lost")
            s.put("second",{"id":"second"})
        self.assertEqual(len(list(s.rows())),2)
        self.assertEqual(s.get("first")["seed"],stable_seed(self.c,"q"))

    def test_archive_parts_restore_and_checksum(self):
        from orbit_kaggle.session import snapshot_run,restore_run
        from orbit_kaggle.storage import write_json
        run=self.root/"run";run.mkdir()
        write_json(run/"resolved_config.json",self.c)
        store=Store(run,"stage",{})
        with store.transaction():
            store.save_tensor("sample",torch.ones(3,4),{"sample_id":"sample"})
        info=snapshot_run(run,part_bytes=2048)
        self.assertTrue((run/"checkpoint_index.json").exists())
        self.assertFalse((run/"stage/manifest.json").exists())
        destination=self.root/"restored"
        restore_run(run,destination)
        loaded=Store(destination,"stage",{}).load_tensor("sample")
        self.assertTrue(torch.equal(loaded,torch.ones(3,4)))
        archive=run/info["archives"][0]["name"];archive.write_bytes(archive.read_bytes()+b"corruption")
        with self.assertRaises(ValueError):restore_run(run,self.root/"corrupt-restore")

    def test_json_scale_missing_and_tags(self):
        s=validate('<think>ignore {}</think><final_json>{"target_scores":{"expert":6},"target_mean":6,"coherence":5}</final_json>',["expert"],"detailed")
        self.assertEqual(s["coherence"],5) # source target-normalizer's accidental x10 is deliberately fixed.
        for score in (None,True,-1,11,float("nan")):
            with self.assertRaises(ValueError):
                validate(json.dumps({"target_scores":{"expert":score},"target_mean":6,"coherence":50}),["expert"],"target")
        with self.assertRaises(ValueError):validate('{"target_scores":{},"coherence":50}',["expert"],"target")

    def test_judge_retry_cache_and_failures(self):
        class Runtime:
            calls=0
            def generate(self,prompt,**kwargs):
                self.calls+=1
                return {"text":"not JSON" if self.calls==1 else '{"target_scores":{"expert":7},"target_mean":7,"coherence":80}',
                        "truncated":False}
        runtime=Runtime();judge=LocalQwenJudge(self.c,self.root,runtime)
        j=judge.score("question","answer",["expert"])
        self.assertEqual(j["status"],"ok");self.assertEqual(j["retry_count"],1)
        judge.score("question","answer",["expert"]);self.assertEqual(runtime.calls,2)
        class Bad:
            def generate(self,*args,**kwargs):return {"text":"{}","truncated":True}
        judge=LocalQwenJudge(self.c,self.root,Bad());j=judge.score("q2","a2",["expert"])
        self.assertEqual(j["status"],"technical_error");self.assertIsNone(j["scores"])
        self.assertEqual(len(j["attempts"]),3)

    def test_thresholds_and_technical_failures(self):
        # Legacy adapter remains tested, but the paper workflow never calls it.
        self.c["build"].update(pos_threshold=60,neg_threshold=40,coherence_threshold=50)
        def j(t,coh):return {"status":"ok","scores":{"target_scores":{"expert":t},"coherence":coh}}
        pair={"label":"expert"}
        self.assertTrue(filter_decision(pair,{"pos":j(6,50),"neg":j(3.9,50)},self.c)["keep"])
        self.assertFalse(filter_decision(pair,{"pos":j(6,50),"neg":j(4,50)},self.c)["keep"])
        self.assertEqual(filter_decision(pair,{"pos":{"status":"technical_error"},"neg":j(1,60)},self.c)["status"],"unscored")

    def test_calibration_state_depends_on_judge_and_resume(self):
        c=copy.deepcopy(self.c);s=initial_state(c,"CAA",["expert"])
        good=[{"status":"ok","scores":{"coherence":90},"cache_key":"real-worker-cache-id"}]
        first=(s["lo"]+s["hi"])/2
        s=update_search(s,good,c);self.assertEqual(s["lo"],first)
        saved=json.loads(json.dumps(s))
        bad=[{"status":"ok","scores":{"coherence":10},"cache_key":"bad-cache-id"}]
        s=update_search(saved,bad,c);self.assertEqual(s["selected"],first)
        with self.assertRaises(RuntimeError):update_search(initial_state(c,"ORBIT",["expert"]),[{"status":"technical_error","scores":None,"cache_key":"err"}],c)

    def test_question_split_and_duplicate_detection(self):
        from orbit_kaggle.data import snapshot_rows,split_rows,build_pairs
        rows=[{"id":i,"text":f"Fixture question {i}?"} for i in range(40)]
        rows.append({"id":"duplicate","text":"  fixture   QUESTION 0? "})
        snap,dups=snapshot_rows(rows);self.assertEqual(len(dups),1)
        split=split_rows(snap,self.c)
        shared=copy.deepcopy(self.c);shared["split"]["mode"]="shared_pool"
        ordered=split_rows(snap,shared)
        self.assertEqual([r["source_index"] for r in ordered["train"]],list(range(40)))
        a,b,d=[{q["question_id"] for q in split[x]} for x in ("train","calibration","test")]
        self.assertFalse(a&b or a&d or b&d)
        pairs=build_pairs(self.c,split);self.assertEqual(len(pairs),80)
        self.assertTrue(all(p["question_id"] in a for p in pairs))
        self.assertEqual(pairs,build_pairs(self.c,split))

    def test_classifier_optimizer_resumes_exactly(self):
        from orbit_kaggle.artifacts import train_classifier
        c=copy.deepcopy(self.c);c["split"]["mode"]="shared_pool";c["classifier"]["epochs"]=2;c["classifier"]["batch_size"]=5
        rng=np.random.default_rng(8);x=rng.normal(size=(40,8)).astype("float32");y=np.repeat(np.arange(5),8)
        a=Store(self.root,"train_a",{});b=Store(self.root,"train_b",{})
        class InterruptBudget:
            n=0
            def check(self):
                self.n+=1
                if self.n==3:raise BudgetExpired("test minibatch interrupt")
        with self.assertRaises(BudgetExpired):train_classifier(c,a,"ksteer",x,y,list(map(str,range(40))),InterruptBudget())
        self.assertGreater(a.load_tensor("ksteer")["next_step"],0)
        resumed=train_classifier(c,a,"ksteer",x,y,list(map(str,range(40))),Budget())
        reference=train_classifier(c,b,"ksteer",x,y,list(map(str,range(40))),Budget())
        for k,v in reference.classifier.state_dict().items():self.assertTrue(torch.equal(v,resumed.classifier.state_dict()[k]),k)

    def test_real_source_math_hooks_token_and_cleanup(self):
        from types import SimpleNamespace
        from torch import nn
        from src.steering.models import ActivationSteering
        from orbit_kaggle.steering import Steering,hook_signature
        from angular_residual_utils import ORBIT_EXTRACTION_LAYOUT_VERSION
        class Block(nn.Module):
            def __init__(self):
                super().__init__();self.input_layernorm=nn.Identity();self.post_attention_layernorm=nn.Identity()
            def forward(self,x):return self.post_attention_layernorm(self.input_layernorm(x))
        class Model(nn.Module):
            def __init__(self):
                super().__init__();self.model=nn.Module();self.model.layers=nn.ModuleList([Block() for _ in range(6)])
            def forward(self,x):
                for block in self.model.layers:x=block(x)
                return x
        model=Model();steer=Steering.__new__(Steering);steer.c=self.c;steer.runtime=SimpleNamespace(model=model)
        steer.caa=torch.ones(5,8);steer.clf=ActivationSteering(8,5,hidden_dim=8,device="cpu")
        entry={"first_direction":torch.tensor([1.,0,0,0,0,0,0,0]),"second_direction":torch.tensor([0.,1.,0,0,0,0,0,0]),
               "delta_proj_vector":torch.ones(8),"hook_type":"output","extraction_layout_version":ORBIT_EXTRACTION_LAYOUT_VERSION}
        modules={f"model.layers.{i}.{n}":entry for i in range(6) for n in ["input_layernorm","post_attention_layernorm"]}
        steer.global_vectors={"expert":{"module_vectors":modules,"r_typical":{n:1.0 for n in modules}}}
        x=torch.ones(1,3,8);x[:,:,0]=-2
        original=model(x).clone();before=hook_signature(model)
        for method in METHODS_FOR_TEST:
            with torch.no_grad(),steer.apply(method,["expert"],1.0) as audit:
                out=model(x.clone())
            self.assertTrue(torch.isfinite(out).all());self.assertGreater(audit["changed_tokens"],0,method)
            if method=="ORBIT":self.assertTrue(torch.equal(out[:,:-1],original[:,:-1]))
            self.assertEqual(hook_signature(model),before)
            self.assertTrue(torch.equal(model(x),original))
        with steer.apply("Unsteered",[],0) as audit:model(x)
        self.assertEqual(audit["hook_calls"],0)
        with steer.apply("ORBIT",["expert"],0) as audit:
            rotated=model(x.clone())
        self.assertGreater(audit["changed_tokens"],0)
        self.assertTrue(torch.allclose(rotated.norm(dim=-1),original.norm(dim=-1),atol=1e-5))

    def test_feature_adapter_matches_source_hidden_index(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast,LlamaConfig,LlamaForCausalLM,Qwen2Config,Qwen2ForCausalLM
        from types import SimpleNamespace
        from orbit_kaggle.artifacts import extract_features
        backend=Tokenizer(WordLevel({"[UNK]":0,"[PAD]":1,"a":2,"b":3,"c":4},unk_token="[UNK]"));backend.pre_tokenizer=Whitespace()
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token="[UNK]",pad_token="[PAD]")
        tokenizer.padding_side="left"
        for cls,conf in [(LlamaForCausalLM,LlamaConfig),(Qwen2ForCausalLM,Qwen2Config)]:
            model=cls(conf(vocab_size=5,hidden_size=16,intermediate_size=32,num_hidden_layers=6,num_attention_heads=2,num_key_value_heads=2)).eval()
            model.requires_grad_(False)
            runtime=SimpleNamespace(model=model,tokenizer=tokenizer,device="cpu")
            texts=["a b c","a c"]
            ours=extract_features(runtime,texts,-6)
            inputs=tokenizer(texts,return_tensors="pt",padding=True)
            with torch.no_grad():ref=model.model(**inputs,output_hidden_states=True,use_cache=False)
            self.assertTrue(torch.allclose(ours["ksteer"],ref.hidden_states[-6][:,-1].float(),atol=1e-6))
            self.assertTrue(torch.allclose(ours["eval"],ref.hidden_states[-1][:,-1].float(),atol=1e-6))
            from angular_residual_utils import ResidualActivationExtractor
            extractor=ResidualActivationExtractor(model,tokenizer,batch_size=1)
            stats=extractor.collect(texts,["", ""],token_mode="prompt_last")
            self.assertEqual(len(stats),12)
            self.assertEqual(len({h.module_name for h in extractor.hooks}),12)
            for key,entry in stats.items():
                self.assertEqual(len(entry["prompt_last_samples"]),2)
                self.assertEqual(entry["prompt_last_samples"][0].shape,(16,))
                self.assertTrue(torch.isfinite(entry["prompt_last_samples"][0]).all())

    def test_question_cluster_bootstrap_not_baseline_alias_count(self):
        import pandas as pd
        from orbit_kaggle.report import cluster_ci
        f=pd.DataFrame({"question_id":["a","b"],"score":[2.,8.]})
        repeated=pd.concat([f]*25,ignore_index=True)
        self.assertEqual(cluster_ci(f,"score",draws=100),cluster_ci(repeated,"score",draws=100))

    def test_source_candidates_global_plane_and_artifacts_from_saved_shards(self):
        self.c["build"]["filter_policy"]="judge"  # explicit legacy filtered fixture
        from orbit_kaggle.artifacts import build_artifacts
        from orbit_kaggle.storage import write_json
        act=Store(self.root,"activation",{});build=Store(self.root,"build",{});filtering=Store(self.root,"filter",{})
        layout=[{"key":[i,side],"layer":i,"module":f"model.layers.{i}.{name}","hook_type":"output"}
                for i in range(6) for side,name in [("mid","post_attention_layernorm"),("post","input_layernorm")]]
        write_json(act.root/"layout.json",layout)
        rng=torch.Generator().manual_seed(12)
        with act.transaction(),build.transaction(),filtering.transaction():
            for li,label in enumerate(self.c["labels"]):
                for i in range(4):
                    key=f"{label}-{i}"
                    norms={side:{str(tuple(h["key"])):torch.randn(8,generator=rng)+(1 if side=="pos" else -1) for h in layout} for side in ["pos","neg"]}
                    act.save_tensor(key,{"norm":norms,"features":{"ksteer":torch.randn(2,8,generator=rng),"eval":torch.randn(2,8,generator=rng)}},
                                    {"label":label,"question_id":str(i),"layout":layout})
                    build.put(key,{"question_id":str(i),"pair_index":i,"pos_prompt":"fixture "+str(i),"pos":{"text":"fixture response"}})
                    filtering.put(key,{"decision":{"keep":not(li==0 and i==3)}})
        out=build_artifacts(self.c,self.root,Budget())
        self.assertEqual(out.load_tensor("caa").shape,(5,8))
        expected=torch.stack([act.load_tensor("expert-"+str(i))["features"]["ksteer"][0]-act.load_tensor("expert-"+str(i))["features"]["ksteer"][1] for i in range(3)]).mean(0)
        self.assertTrue(torch.allclose(out.load_tensor("caa")[0],expected))
        for label in self.c["labels"]:
            global_=out.load_tensor("global/"+label)
            self.assertEqual(len(global_["module_vectors"]),12)
            for module,v in global_["module_vectors"].items():
                self.assertTrue(torch.isfinite(v["first_direction"]).all())
                self.assertAlmostEqual(float(v["first_direction"].norm()),1.,places=5)
                if v["second_direction"] is not None:self.assertLess(abs(float(v["first_direction"]@v["second_direction"])),1e-5)
        self.assertTrue(out.get("eval")["metadata"]["training_complete"])
        # Re-running from disk requires no LLM and retains train cursor and weights.
        saved=out.load_tensor("ksteer")["state_dict"]
        again=build_artifacts(self.c,self.root,Budget())
        self.assertTrue(all(torch.equal(v,again.load_tensor("ksteer")["state_dict"][k]) for k,v in saved.items()))

    def test_report_valid_errors_and_unique_baseline(self):
        from orbit_kaggle.report import report
        from orbit_kaggle.storage import write_json
        from orbit_kaggle.config import METHODS
        c=copy.deepcopy(self.c);c["report"]["judge_types"]=["target"];c["report"]["bootstrap_draws"]=10
        from orbit_kaggle.config import evaluation_protocol
        ev=Store(self.root,"evaluation",evaluation_protocol(c));sc=Store(self.root,"scoring-fixture",{});fi=Store(self.root,"filter",{})
        evaluator={"id":"unit-test-fixture-not-experiment","revision":"fixture"}
        def j(targets,value=5,ok=True):
            return {"status":"ok" if ok else "technical_error","scores":{"target_scores":{t:value for t in targets},"coherence":80} if ok else None,
                    "retry_count":0,"attempts":[{"truncated":False}],"evaluator":evaluator,"cache_key":digest([targets,value,ok])}
        with ev.transaction(),sc.transaction(),fi.transaction():
            fi.put("pair",{"decision":{"keep":True,"status":"accepted"}})
            for qid in ["q0","q1"]:
                for method in METHODS:
                    contexts=combos(c) if method=="Unsteered" else combos(c)
                    for targets in ([[]] if method=="Unsteered" else contexts):
                        key=digest([method,qid,targets])
                        row={"sample_id":key,"question_id":qid,"repeat":0,"method":method,"targets":targets,"strength":1.,
                            "generation":{"seed":42,"truncated":False,"elapsed_s":1.,"generated_tokens":10,"resources":{"vram":[]}},
                            "eval_classifier_probabilities":{t:0.5 for t in c["labels"]}}
                        ev.put(key,row)
                        context_scores={",".join(ts):{"target":j(ts,5 if method=="Unsteered" else 7,not(method=="CAA" and qid=="q1"))} for ts in contexts if method=="Unsteered" or ts==targets}
                        sc.put(key,{"judgments":context_scores,"comparison":{}})
        out=report(c,self.root,sc)
        import pandas as pd
        summary=pd.read_csv(out.root/"summary_model_method_K.csv")
        baseline=summary[summary.method=="Unsteered"].iloc[0]
        self.assertEqual(baseline.n_physical_responses,2)
        self.assertEqual(baseline.n_questions,2)
        self.assertEqual(baseline.n_scoring_contexts,10)
        self.assertEqual(summary[summary.method=="CAA"].iloc[0].n_error_contexts,5)
        self.assertAlmostEqual(summary[summary.method=="ORBIT"].iloc[0].delta_tone,2.)
        self.assertTrue((out.root/"telemetry.json").exists())
        telemetry=json.loads((out.root/"telemetry.json").read_text())
        self.assertEqual(telemetry["build_filter_policy"],"none")
        self.assertIsNone(telemetry["filter_accepted"])
        self.assertFalse(telemetry["judge_quality_by_stage"]["build_filter"]["applied"])
        self.assertTrue((summary.build_filter_policy=="none").all())
        self.assertFalse((out.root/"build_filter_threshold_sensitivity.csv").exists())

    def test_notebooks_schema_and_syntax(self):
        import ast
        import nbformat
        paths=list((ROOT/"notebooks").glob("*.ipynb"))
        self.assertEqual(len(paths),6)
        for path in paths:
            notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
            self.assertTrue(any(c.cell_type=="markdown" for c in notebook.cells))
            for cell in notebook.cells:
                if cell.cell_type=="code":ast.parse(cell.source,filename=str(path))

    def test_no_openai_imports_needed(self):
        import sys
        class BlockOpenAI(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,*args):
                if fullname.split(".")[0]=="openai":raise AssertionError("OpenAI SDK import attempted")
        blocker=BlockOpenAI();sys.meta_path.insert(0,blocker)
        try:
            import orbit_kaggle.cli,orbit_kaggle.stages,orbit_kaggle.models,orbit_kaggle.artifacts
            from source_prompts import TonePrompt
            self.assertIn("JSON",TonePrompt()._create_evaluation_prompt("a","b","expert","casual"))
            self.assertNotIn("openai",sys.modules)
        finally:sys.meta_path.remove(blocker)

    def test_cross_backbone_precision_and_protocol_reuse_rejected(self):
        from types import SimpleNamespace
        from orbit_kaggle.config import model_identity,evaluation_protocol,check_evaluation_protocol
        from orbit_kaggle.steering import Steering
        identity={"algorithm":model_identity(self.c),"classifier":self.c["classifier"]}
        art=Store(self.root,"artifacts",identity)
        with art.transaction():pass
        for key,value in [("target_model_id","Qwen/Qwen2.5-7B-Instruct"),("precision","nf4")]:
            wrong=copy.deepcopy(self.c);wrong[key]=value
            with self.assertRaises(ValueError):Steering(wrong,self.root,SimpleNamespace(model=torch.nn.Identity()))
        frozen=evaluation_protocol(self.c)
        wrong=copy.deepcopy(self.c);wrong["split"]["mode"]="shared_pool"
        with self.assertRaises(ValueError):check_evaluation_protocol(wrong,frozen)
        # Pure evaluator/report changes leave backbone activations/generations reusable.
        rescore=copy.deepcopy(self.c);rescore["judge"]["max_new_tokens"]+=256
        check_evaluation_protocol(rescore,frozen)


METHODS_FOR_TEST=["CAA","K-Steer","ORBIT"]
if __name__=="__main__":unittest.main()
