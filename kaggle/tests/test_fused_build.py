"""Tiny real CPU models exercise inline capture, token sites and resume; no HF weights."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from orbit_kaggle.config import load_config
from orbit_kaggle.storage import Store,Budget,BudgetExpired,open_existing
from orbit_kaggle.fused_build import capture_prefill,generate_extract

ROOT=Path(__file__).resolve().parents[1]


def tiny_runtime(c, family="llama"):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast,LlamaConfig,LlamaForCausalLM,Qwen2Config,Qwen2ForCausalLM
    from orbit_kaggle.models import LocalModel
    backend=Tokenizer(WordLevel({"[UNK]":0,"[PAD]":1,"[EOS]":2,"user":3,"system":4,"assistant":5,
                                "a":6,"b":7,"c":8,"d":9,"brief":10,"verbose":11},unk_token="[UNK]"))
    backend.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token="[UNK]",pad_token="[PAD]",eos_token="[EOS]")
    tokenizer.chat_template="{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
    tokenizer.padding_side="left"
    cls,config=(LlamaForCausalLM,LlamaConfig) if family=="llama" else (Qwen2ForCausalLM,Qwen2Config)
    torch.manual_seed(8)
    model=cls(config(vocab_size=12,hidden_size=16,intermediate_size=24,num_hidden_layers=6,
                     num_attention_heads=2,num_key_value_heads=2,pad_token_id=1,eos_token_id=2)).eval().requires_grad_(False)
    model.generation_config.temperature=None;model.generation_config.top_p=None;model.generation_config.top_k=None
    runtime=LocalModel.__new__(LocalModel)
    runtime.model=model;runtime.tokenizer=tokenizer;runtime.device="cpu";runtime.is_judge=False
    runtime.c=c;runtime.model_id=c["target_model_id"];runtime.revision=c["revisions"][c["target_model_id"]]
    runtime.precision="fp16-config-with-float32-test-fixture"
    return runtime


class FusedBuild(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.c=copy.deepcopy(load_config(ROOT/"configs/smoke.json"))
        self.c["build"]["max_pairs"]=4
        self.c["build"]["max_new_tokens"]=2  # test fixture only; no experimental output
        data=Store(self.root,"data",{"fixture":True})
        with data.transaction():
            data.put("split",{"train":[{"question_id":str(i),"text":q} for i,q in enumerate(["a b","b c","c d","a d"])]})

    def test_prefill_matches_source_system_prompt_extraction_llama_qwen(self):
        from angular_residual_utils import ResidualActivationExtractor
        for family in ("llama","qwen"):
            runtime=tiny_runtime(self.c,family)
            extractor=ResidualActivationExtractor(runtime.model,runtime.tokenizer,batch_size=1)
            with capture_prefill(extractor.hooks) as captured:
                runtime.generate("a b",42,3,sampling=True,system="brief")
            reference=extractor.collect(["a b"],["unused response"],token_mode="prompt_last",system_prompts=["brief"])
            self.assertEqual(len(captured),12)
            for site in extractor.hooks:
                self.assertTrue(torch.allclose(captured[str(site.key)],reference[site.key]["prompt_last_samples"][0],atol=2e-6))
            self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in runtime.model.modules()))

    def test_hook_cleanup_when_generation_fails(self):
        from angular_residual_utils import ResidualActivationExtractor
        runtime=tiny_runtime(self.c);extractor=ResidualActivationExtractor(runtime.model,runtime.tokenizer)
        with self.assertRaisesRegex(RuntimeError,"interrupted"):
            with capture_prefill(extractor.hooks):
                raise RuntimeError("interrupted generation")
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in runtime.model.modules()))

    def test_pair_checkpoint_and_system_prompt_provenance(self):
        runtime=tiny_runtime(self.c)
        with patch("orbit_kaggle.models.LocalModel",return_value=runtime),patch.object(runtime,"generate",wraps=runtime.generate) as generate:
            with self.assertRaises(BudgetExpired):
                generate_extract(self.c,self.root,Budget(stop_after=1))
            self.assertEqual(generate.call_count,2)
            call=generate.call_args
            self.assertIn(call.args[0],["a b","b c","c d","a d"])
            self.assertTrue(call.kwargs["system"])
            build_manifest=__import__('json').loads((self.root/"build/manifest.json").read_text())
            self.assertEqual(build_manifest["status"],"pending")
            changed=copy.deepcopy(self.c);changed["build"]["max_new_tokens"]+=1
            with self.assertRaisesRegex(ValueError,"Cache config mismatch"):
                generate_extract(changed,self.root,Budget())
            self.assertEqual(generate.call_count,2)

    def test_mid_pair_resume_never_regenerates_completed_side_and_no_filter(self):
        runtime=tiny_runtime(self.c)
        first_result={}
        class InterruptAfterSide(Budget):
            def check(inner):
                if list((self.root/"build_activation_sides/shards").glob("*.json")):
                    raise BudgetExpired("intentional mid-pair interruption")
        with patch("orbit_kaggle.models.LocalModel",return_value=runtime), \
             patch.object(runtime,"generate",wraps=runtime.generate) as generate, \
             patch("orbit_kaggle.judge.LocalQwenJudge",side_effect=AssertionError("Build must never load judge")):
            with self.assertRaises(BudgetExpired):
                generate_extract(self.c,self.root,InterruptAfterSide())
            self.assertEqual(generate.call_count,1)
            pending=list((self.root/"build_activation_sides/shards").glob("*.json"))
            self.assertEqual(len(pending),1)
            generate_extract(self.c,self.root,Budget())
            self.assertEqual(generate.call_count,40) # 5 tones * 4 pairs * 2 sides
            generate_extract(self.c,self.root,Budget())
            self.assertEqual(generate.call_count,40)
        self.assertFalse((self.root/"filter").exists())
        self.assertEqual(len(list(open_existing(self.root,"build").rows())),20)
        activation=open_existing(self.root,"activation")
        self.assertEqual(len(list(activation.rows())),20)
        self.assertEqual(len(list(open_existing(self.root,"build_activation_sides").rows())),0)
        from orbit_kaggle.artifacts import extract_features,build_artifacts
        for key,row in open_existing(self.root,"build").rows():
            value=activation.load_tensor(key)
            reference=extract_features(runtime,[row[s+"_prompt"]+"\n"+row[s]["text"] for s in ("pos","neg")],-6)
            for name in reference:self.assertTrue(torch.equal(value["features"][name],reference[name]))
        with patch("orbit_kaggle.models.LocalModel",side_effect=AssertionError("CPU artifact build must not load model")), \
             patch("orbit_kaggle.judge.LocalQwenJudge",side_effect=AssertionError("No build judge")):
            out=build_artifacts(self.c,self.root,Budget())
        self.assertEqual(out.load_tensor("caa").shape,(5,16))
        self.assertTrue(out.get("ksteer")["metadata"]["training_complete"])
        self.assertTrue(out.get("eval")["metadata"]["training_complete"])
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in runtime.model.modules()))

    def test_notebook_budget_is_shared_across_stages(self):
        import ast,json
        from unittest.mock import Mock
        nb=json.loads((ROOT/"notebooks/00_prepare_data.ipynb").read_text())
        tree=ast.parse(''.join(nb["cells"][2]["source"]))
        definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="stage")
        run=Mock(return_value=0)
        namespace={"CHECKPOINTED":False,"TIME_BUDGET_SECONDS":100,"NOTEBOOK_START":0,
                   "time":SimpleNamespace(monotonic=Mock(side_effect=[10,40,101])),"run_stage":run,
                   "PYTHON":"python","CONFIG_PATH":"config","RUN_ROOT":"run"}
        exec(compile(ast.Module(body=[definition],type_ignores=[]),"notebook-budget","exec"),namespace)
        self.assertEqual(namespace["stage"]("first"),0)
        self.assertEqual(run.call_args.args[4],90)
        self.assertEqual(namespace["stage"]("second"),0)
        self.assertEqual(run.call_args.args[4],60)
        self.assertEqual(namespace["stage"]("third"),75)
        self.assertTrue(namespace["CHECKPOINTED"])
        self.assertEqual(run.call_count,2)


if __name__=="__main__":unittest.main()
