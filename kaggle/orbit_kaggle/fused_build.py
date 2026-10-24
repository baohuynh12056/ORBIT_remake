"""Paper-style unfiltered build, capturing ORBIT during generation prefill.

CAA/K-Steer/eval retain the repo's raw-prompt/response last-token features.
Those require one short replay; they are not silently replaced by decode states.
"""
from contextlib import contextmanager
import time
from .storage import Store, open_existing, digest, write_json
from .config import model_identity, stable_seed


@contextmanager
def capture_prefill(hooks):
    """Read only the first visit to each extraction site, never retain all tokens."""
    import torch
    values, handles = {}, []
    def capture(key, output):
        if key in values:
            return
        tensor = output[0] if isinstance(output, tuple) else output
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError("Inline prefill capture requires generation batch=1")
        value = tensor[0, -1].detach().float().cpu().clone()
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite inline activation: {key}")
        values[key] = value
    try:
        for site in hooks:
            key = str(site.key)
            if site.hook_type == "pre":
                handles.append(site.module.register_forward_pre_hook(
                    lambda module, args, key=key: capture(key, args[0])))
            else:
                handles.append(site.module.register_forward_hook(
                    lambda module, args, output, key=key: capture(key, output)))
        yield values
        if set(values) != {str(site.key) for site in hooks}:
            raise RuntimeError("Not all ORBIT prefill hooks fired")
    finally:
        for handle in handles:
            handle.remove()


def generate_extract(c, root, budget):
    import torch
    from .data import build_pairs
    from .models import LocalModel, retry_oom_batches
    from .artifacts import extract_features
    from angular_residual_utils import ResidualActivationExtractor
    if c["build"].get("filter_policy") != "none" or c["build"]["response_source"] != "backbone":
        raise ValueError("Fused paper build requires unfiltered backbone responses")
    data = open_existing(root, "data")
    pairs = build_pairs(c, data.get("split"))
    identity = {"data":data.fp,"model":model_identity(c),"build":c["build"],
                "seed":c["seed"],"pair_prompts":digest(pairs),"workflow":"paper_inline_v1"}
    build = Store(root, "build", identity)
    act_identity = {"backbone":model_identity(c),"build":build.fp,"pair_prompts":digest(pairs),
                    "selection":"all_unfiltered_pairs","norm_token":"system_chat_prefill_last",
                    "feature_token":"raw_prompt_response_last","layout":"norm"}
    activation = Store(root, "activation", act_identity)
    sides = Store(root, "build_activation_sides", act_identity)
    runtime = extractor = None
    def compact_sides(key):
        # Full build JSON + activation shard are now committed. Retain only the
        # one authoritative norm copy, instead of doubling the full activation cache.
        for side in ("pos","neg"):
            side_key=key+"/"+side
            saved=sides.get(side_key)
            if saved is not None:
                sides.path(side_key).unlink()
                (sides.root/"tensors"/saved["tensor_file"]).unlink(missing_ok=True)
    with build.transaction(), activation.transaction(), sides.transaction():
        for pair in pairs:
            key = pair["pair_id"]
            if build.get(key) is not None and activation.get(key) is not None:
                activation.load_tensor(key)  # checksum/shape/finiteness before cache reuse
                compact_sides(key)
                continue
            budget.check()
            if runtime is None:
                runtime = LocalModel(c)
                extractor = ResidualActivationExtractor(runtime.model, runtime.tokenizer, batch_size=1)
                layout = [{"key":list(h.key),"module":h.module_name,"layer":h.layer_idx,
                           "hook_type":h.hook_type,"token":"prompt_last"} for h in extractor.hooks]
                write_json(activation.root / "layout.json", layout)
            row = {**pair,"response_source":"backbone","filter_policy":"none"}
            norms = {}
            for side in ("pos", "neg"):
                budget.check()
                side_key = key + "/" + side
                saved = sides.load_tensor(side_key)
                if saved is None:
                    with capture_prefill(extractor.hooks) as captured:
                        result = runtime.generate(pair[side+"_user"],
                            stable_seed(c,pair["question_id"],namespace=f"build/{pair['label']}/{side}/{key}"),
                            c["build"]["max_new_tokens"],sampling=True,system=pair[side+"_system"])
                    saved = {"generation":result,"norm":captured}
                    sides.save_tensor(side_key, saved, {"pair_id":key,"question_id":pair["question_id"],
                        "label":pair["label"],"polarity":side,"capture":"generation_prefill_last",
                        "layout":layout,"compute_precision":c["precision"],"storage_dtype":"float32"})
                    build.log("build_side_committed",model=c["target_model_id"],sample_id=key,
                              combo=[pair["label"]],polarity=side)
                row[side], norms[side] = saved["generation"], saved["norm"]
            # Commit text before feature replay. An interruption resumes with saved sides,
            # so no completed POS/NEG generation is repeated, even mid-pair.
            build.put(key, row)
            budget.check()
            texts = [pair[side+"_prompt"] + "\n" + row[side]["text"] for side in ("pos","neg")]
            replay_start=time.monotonic()
            def collect(part):
                budget.check()
                result = extract_features(runtime, part, c["steering"]["layer"])
                return [{name:value[i] for name,value in result.items()} for i in range(len(part))]
            features = []
            for part, result in retry_oom_batches(texts, c["build"]["feature_replay_batch_size"], collect, activation):
                features.extend(result)
            replay_s=time.monotonic()-replay_start
            activation.save_tensor(key, {"norm":norms,"features":{
                name:torch.stack([item[name] for item in features]) for name in ("ksteer","eval")}},
                {"pair_id":key,"question_id":pair["question_id"],"label":pair["label"],
                 "polarity_order":["pos","neg"],"layout":layout,"compute_precision":c["precision"],
                 "storage_dtype":"float32","norm_capture":"generation_prefill_last",
                 "feature_replay_elapsed_s":replay_s,
                 "feature_modules":{"ksteer":f"model.layers.{len(runtime.model.model.layers)+c['steering']['layer']}",
                                    "eval":"model.norm"},"feature_token":"last_nonpad_raw_prompt_response"})
            build.log("build_response",model=c["target_model_id"],sample_id=key,combo=[pair["label"]],
                      question_id=pair["question_id"],elapsed_s=sum(row[s]["elapsed_s"] for s in ("pos","neg")))
            activation.log("inline_activation",model=c["target_model_id"],sample_id=key,
                           combo=[pair["label"]],question_id=pair["question_id"],feature_replay_elapsed_s=replay_s)
            compact_sides(key)
            budget.committed()
    return build
