from __future__ import annotations
import contextlib
from .storage import open_existing
from .config import model_identity


def hook_signature(model):
    return {name: (tuple(module._forward_hooks),tuple(module._forward_pre_hooks)) for name,module in model.named_modules()
            if module._forward_hooks or module._forward_pre_hooks}


class Steering:
    def __init__(self, c, root, runtime):
        from src.steering.models import ActivationSteering
        self.c,self.runtime = c,runtime
        self.clean_signature=hook_signature(runtime.model)
        self.store = open_existing(root,"artifacts")
        if self.store.identity.get("algorithm")!=model_identity(c) or self.store.identity.get("classifier")!=c["classifier"]:
            raise ValueError("Backbone/revision/precision/classifier config does not match steering artifacts. Cross-backbone reuse is forbidden.")
        self.caa = self.store.load_tensor("caa")
        checkpoint = self.store.load_tensor("ksteer")
        if not self.store.get("ksteer")["metadata"]["training_complete"]:
            raise RuntimeError("K-Steer classifier training incomplete")
        self.clf = ActivationSteering(checkpoint["input_dim"],len(c["labels"]),hidden_dim=c["classifier"]["hidden_dim"],device="cpu")
        self.clf.classifier.load_state_dict(checkpoint["state_dict"])
        self.clf.classifier.eval()
        self.global_vectors = {label:self.store.load_tensor("global/"+label) for label in c["labels"]}

    @contextlib.contextmanager
    def apply(self, method, targets, strength):
        import torch
        from src.steering.hooks import get_additive_hook, get_gradient_hook
        from angular_steering_utils import register_layernorm_additive_hooks, remove_hooks
        model = self.runtime.model
        before = hook_signature(model)
        if hasattr(self,"clean_signature") and before!=self.clean_signature:
            raise RuntimeError("Unexpected hooks exist before generation; baseline/method isolation cannot be guaranteed")
        handles, observers, captured = [], [], {}
        audit = {"method":method,"hook_calls":0,"changed_tokens":0,"max_abs_delta":0.0,"modules":{},"strength":strength}
        if method == "Unsteered":
            try:
                yield audit
            finally:
                if hook_signature(model) != before:
                    raise RuntimeError("Baseline hook state changed")
            return
        if method not in ("CAA","K-Steer","ORBIT"):
            raise ValueError(method)
        idx=[self.c["labels"].index(t) for t in targets]
        if method == "ORBIT":
            modules={n: model.get_submodule(n) for n in self.global_vectors[targets[0]]["module_vectors"]}
        else:
            index=len(model.model.layers)+self.c["steering"]["layer"]
            modules={f"model.layers.{index}":model.model.layers[self.c["steering"]["layer"]]}
        def capture(name):
            def f(module,args,out):
                h=out[0] if isinstance(out,tuple) else out
                # ORBIT response mode changes the last causal position, including prefill's last prompt token.
                captured[name]=(h[:,-1:,:] if method=="ORBIT" else h).detach().clone()
            return f
        def measure(name):
            def f(module,args,out):
                h=out[0] if isinstance(out,tuple) else out
                h=h[:,-1:,:] if method=="ORBIT" else h
                delta=(h.detach().float()-captured.pop(name).to(h.device,dtype=torch.float32)).abs()
                if not torch.isfinite(h).all():
                    raise ValueError(f"Nonfinite steered hidden state at {name}")
                n=int((delta.amax(-1)>0).sum())
                audit["hook_calls"]+=1; audit["changed_tokens"]+=n
                audit["max_abs_delta"]=max(audit["max_abs_delta"],float(delta.max()))
                audit["modules"][name]=audit["modules"].get(name,0)+n
            return f
        try:
            for name,module in modules.items():
                observers.append(module.register_forward_hook(capture(name)))
            if method == "ORBIT":
                handles=register_layernorm_additive_hooks(model, None, coeff=1.0,positions="response",
                    add_module_vectors_list=[self.global_vectors[t]["module_vectors"] for t in targets],
                    add_r_typical_list=[self.global_vectors[t]["r_typical"] for t in targets],
                    adaptive_mode=4,adaptive_threshold=0.0,parallel_method="sum",sequential_mode="full",
                    target_angle=0.0,hybrid_multitrait=True,hybrid_gate_style="margin",hybrid_soft_temperature=0.5,
                    hybrid_min_gate=0.0,hybrid_gate_reference="zero",hybrid_rotation_gate_mode="full",
                    additive_boost=True,additive_coeff=strength,boost_gate_mode="adaptive",boost_abs_tau=0.0,
                    boost_min_gate=0.0,boost_clamp_ratio=0.0,boost_delta_clamp_mode="off",boost_approach=1,
                    boost_fill_fraction=0.4,intervention_layout="norm",require_layout_metadata=True)
                if len(handles)!=len(modules):
                    raise RuntimeError("ORBIT failed to attach at every configured LayerNorm site")
            else:
                fn = (get_additive_hook(self.caa[idx].mean(0),alpha=strength) if method=="CAA" else
                      get_gradient_hook(steer_module=self.clf,target_labels=idx,avoid_labels=[],alpha=strength,steps=1))
                # Explicit autograd enable for K-Steer inside outer no_grad generation.
                if method=="K-Steer":
                    original=fn
                    def fn(*args):
                        with torch.inference_mode(False), torch.enable_grad():
                            return original(*args)
                handles=[next(iter(modules.values())).register_forward_hook(fn)]
            for name,module in modules.items():
                observers.append(module.register_forward_hook(measure(name)))
            yield audit
        finally:
            remove_hooks(handles+observers)
            if hook_signature(model)!=before:
                raise RuntimeError(f"Steering hook leaked after {method}")

    def generate(self, question, seed, method, targets, strength, max_new_tokens):
        with self.apply(method,targets,strength) as audit:
            result=self.runtime.generate(question,seed,max_new_tokens)
        result["intervention"]=audit
        return result
