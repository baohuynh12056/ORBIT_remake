from __future__ import annotations
from pathlib import Path
from .storage import Store, open_existing, digest, write_json, atomic_bytes
from .config import model_identity


def extract_features(runtime, texts, layer):
    """Same two hidden-state indices as launcher, capture only selected last tokens.

    With N blocks hidden_states[-6] == output of blocks[-6]; final hidden state
    is after model.norm. Raw prompt + '\n' + response is intentional source behavior.
    """
    import torch
    model, tokenizer = runtime.model, runtime.tokenizer
    modules = {"ksteer": model.model.layers[layer], "eval": model.model.norm}
    values, handles = {}, []
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False).to(runtime.device)
    positions = inputs["attention_mask"].shape[1] - 1 - inputs["attention_mask"].flip(1).argmax(1)
    def hook(name):
        def capture(module, args, output):
            h = output[0] if isinstance(output,tuple) else output
            values[name] = h[torch.arange(h.shape[0],device=h.device), positions.to(h.device)].detach().float().cpu()
        return capture
    try:
        for name,module in modules.items():
            handles.append(module.register_forward_hook(hook(name)))
        with torch.no_grad():
            model.model(**inputs, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()
    if set(values) != set(modules):
        raise RuntimeError("Feature hooks did not fire")
    if any(not torch.isfinite(v).all() for v in values.values()):
        raise ValueError("Nonfinite activation")
    return values


def extract(c, root, budget):
    import torch
    from angular_residual_utils import ResidualActivationExtractor
    from .models import LocalModel, retry_oom_batches
    build = open_existing(root, "build")
    filtering = open_existing(root, "filter")
    if build.identity["model"]!=model_identity(c):
        raise ValueError("Build responses belong to a different target backbone/revision/precision configuration")
    accepted = [key for key,row in filtering.rows() if row["decision"]["keep"]]
    pairs = dict(build.rows())
    if any(sum(pairs[key]["label"]==label for key in accepted) < 2 for label in c["labels"]):
        raise RuntimeError("Fewer than two accepted POS/NEG pairs for a tone. Inspect filtering/technical failures; no synthetic fallback or automatic threshold change.")
    # Cache all generated pairs, including currently rejected pairs. A later rubric
    # can change filtering without requiring the backbone to extract again.
    identity = {"backbone": model_identity(c), "pairs": digest(pairs),"selection":"all_generated_pairs",
                "norm_token": "prompt_last", "feature_token": "raw_prompt_response_last", "layout": "norm"}
    store = Store(root, "activation", identity)
    with store.transaction():
        pending = [(k, p) for k,p in pairs.items() if store.get(k) is None]
        if not pending:
            return store
        runtime = LocalModel(c)
        extractor = ResidualActivationExtractor(runtime.model, runtime.tokenizer, batch_size=1)
        layout = [{"key": list(h.key), "module": h.module_name, "layer": h.layer_idx,
                   "hook_type": h.hook_type, "token": "prompt_last"} for h in extractor.hooks]
        write_json(store.root / "layout.json", layout)
        def collect_batch(part):
            budget.check()
            # Features may batch; ORBIT's source accumulator is called per sample,
            # so all returned activation samples keep a stable question/polarity ID.
            texts = [p[side+"_prompt"] + "\n" + p[side]["text"] for k,p in part for side in ("pos","neg")]
            features = extract_features(runtime, texts, c["steering"]["layer"])
            outputs = []
            for j,(key,pair) in enumerate(part):
                norms = {}
                for side in ("pos","neg"):
                    budget.check()
                    # collect(prompt_last) intentionally does not condition on the generated response.
                    stats = extractor.collect([pair[side+"_prompt"]], [pair[side]["text"]], token_mode="prompt_last")
                    norms[side] = {str(h.key): stats[h.key]["prompt_last_samples"][0].float().cpu() for h in extractor.hooks}
                outputs.append({"norm": norms, "features": {name:value[j*2:j*2+2] for name,value in features.items()}})
            return outputs
        for part, results in retry_oom_batches(pending, c["hardware"]["extraction_batch_size"], collect_batch, store):
            for (key,pair), result in zip(part, results):
                store.save_tensor(key, result, {"pair_id": key, "question_id": pair["question_id"], "label": pair["label"],
                    "polarity_order": ["pos","neg"], "layout": layout, "compute_precision": c["precision"],
                    "storage_dtype": "float32", "feature_modules": {"ksteer": f"model.layers.{len(runtime.model.model.layers)+c['steering']['layer']}", "eval":"model.norm"},
                    "feature_token": "last_nonpad_raw_prompt_response"})
                store.log("activation", model=c["target_model_id"], sample_id=key, combo=[pair["label"]])
    return store


def train_classifier(c, store, name, X, y, question_ids, budget):
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from src.steering.models import ActivationSteering, one_hot
    if c["split"]["mode"] == "shared_pool":
        a,b = train_test_split(np.arange(len(y)), test_size=0.5, random_state=42, stratify=y)
    else:
        # Both classifiers use the build partition only, and internal halves are disjoint by question.
        ids = sorted(set(question_ids), key=lambda q:digest([42,q]))
        first = set(ids[::2]); second = set(ids[1::2])
        a,b = np.array([i for i,q in enumerate(question_ids) if q in first]), np.array([i for i,q in enumerate(question_ids) if q in second])
        if not len(a) or not len(b):
            raise ValueError("Need at least two distinct accepted build questions for question-level classifier split")
    train_idx, val_idx = (a,b) if name == "ksteer" else (b,a)
    if set(y[train_idx].tolist()) != set(range(len(c["labels"]))):
        raise ValueError(f"{name} training half misses a tone; increase build sample count, inspect filter")
    torch.manual_seed(c["seed"])
    clf = ActivationSteering(X.shape[1],len(c["labels"]),hidden_dim=c["classifier"]["hidden_dim"],
                            lr=c["classifier"]["lr"],device="cpu")
    checkpoint = store.load_tensor(name)
    cursor = 0
    if checkpoint:
        clf.classifier.load_state_dict(checkpoint["state_dict"])
        clf.optimizer.load_state_dict(checkpoint["optimizer"])
        cursor = checkpoint["next_step"]
    batch = c["classifier"]["batch_size"]
    batches = (len(train_idx) + batch - 1) // batch
    X_t = torch.as_tensor(X, dtype=torch.float32)
    Y_t = torch.as_tensor(one_hot(y,len(c["labels"])))
    for epoch in range(c["classifier"]["epochs"]):
        # Stateless epoch shuffle; optimizer + next minibatch fully resumes training.
        order = np.random.default_rng(c["seed"]+epoch).permutation(train_idx)
        for bidx,start in enumerate(range(0,len(order),batch)):
            step = epoch*batches+bidx
            if step < cursor:
                continue
            budget.check()
            clf.classifier.train()
            idx = order[start:start+batch]
            clf.optimizer.zero_grad()
            loss = clf.loss_fn(clf.classifier(X_t[idx]), Y_t[idx])
            loss.backward(); clf.optimizer.step()
            store.save_tensor(name, {"state_dict": clf.classifier.state_dict(), "optimizer": clf.optimizer.state_dict(),
                    "next_step": step+1, "input_dim": X.shape[1], "train_idx": torch.tensor(train_idx),
                    "validation_idx": torch.tensor(val_idx), "labels": c["labels"],"question_ids":question_ids,
                    "y":torch.as_tensor(y)}, {"training_complete": False})
            store.log("classifier_training",model=c["target_model_id"],method=name,epoch=epoch,minibatch=bidx,
                      completed_step=step+1,total_steps=batches*c["classifier"]["epochs"],loss=float(loss.detach()))
    clf.classifier.eval()
    with torch.no_grad():
        accuracy = float((clf.classifier(X_t[val_idx]).argmax(-1).numpy()==y[val_idx]).mean())
    value = store.load_tensor(name)
    value["validation_accuracy"] = accuracy
    store.save_tensor(name,value,{"training_complete":True,"internal_split":"row_stratified" if c["split"]["mode"]=="shared_pool" else "question_disjoint"})
    return clf


def build_artifacts(c, root, budget):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from angular_residual_utils import ResidualActivationExtractor, ResidualHook, ORBIT_EXTRACTION_LAYOUT_VERSION
    from angular_steering_utils import load_global_layernorm_vectors
    np.random.seed(c["seed"])
    torch.manual_seed(c["seed"])
    act = open_existing(root, "activation")
    pairs = open_existing(root,"build")
    if c["build"].get("filter_policy") == "none":
        if (pairs.identity.get("model")!=model_identity(c) or pairs.identity.get("build")!=c["build"]
                or act.identity.get("backbone")!=model_identity(c) or act.identity.get("build")!=pairs.fp):
            raise ValueError("Paper build/activation protocol mismatch; do not mix legacy response or activation caches")
        accepted={key for key,row in pairs.rows()}
    else:
        filtering=open_existing(root,"filter")
        accepted={key for key,row in filtering.rows() if row["decision"]["keep"]}
    store = Store(root, "artifacts", {"activation": act.fp, "activation_content": digest(list(act.rows())),
                  "accepted_pairs":sorted(accepted),"filter_policy":c["build"].get("filter_policy","judge"),
                  "classifier": c["classifier"], "split_mode": c["split"]["mode"], "seed":c["seed"],
                  "algorithm": model_identity(c)})
    with store.transaction():
        layout = __import__('json').loads((act.root / "layout.json").read_text())
        for key,row in act.rows():
            if row["metadata"]["layout"]!=layout:
                raise ValueError(f"Extraction layout checksum/schema mismatch against shard metadata: {key}")
        # Only source numerical routines are needed now; no LLM load for vector construction.
        ext = ResidualActivationExtractor.__new__(ResidualActivationExtractor)
        ext.extraction_layout = "norm"
        ext.extraction_layout_version = ORBIT_EXTRACTION_LAYOUT_VERSION
        ext.max_pca_samples = 2048
        hooks = [ResidualHook(x["layer"],x["key"][1],x["module"],None,x["hook_type"]) for x in layout]
        ext._hook_map = {h.key:h for h in hooks}
        all_features, ys, qids, caa = {"ksteer":[],"eval":[]}, [], [], []
        classifier_seen=set()
        for label in c["labels"]:
            budget.check()
            entries = [(key,act.load_tensor(key)) for key,row in act.rows() if key in accepted and row["metadata"]["label"]==label]
            entries.sort(key=lambda kv:pairs.get(kv[0])["pair_index"])
            if len(entries)<2:
                raise ValueError(f"Insufficient activations: {label}")
            stats = {}
            for side in ("pos","neg"):
                stats[side] = {}
                for hook in hooks:
                    samples = [v["norm"][side][str(hook.key)] for key,v in entries]
                    stats[side][hook.key] = {"prompt_last_samples": samples, "prompt_last_sum":torch.stack(samples).sum(0),
                                             "prompt_last_count":torch.tensor(len(samples))}
            vectors = ext.compute_directional_differences(stats["pos"],stats["neg"],delta_proj_mode="plane_mean")
            metadata = {"label":label,"backbone":model_identity(c),"token_mode":"prompt_last","delta_proj_mode":"plane_mean",
                        "pair_ids":[key for key,v in entries],"pairing_version":"question_aligned_v3",
                        "activation_fingerprint":act.fp,"build_fingerprint":pairs.fp,
                        "response_source":c["build"]["response_source"],
                        "response_model_id":c["target_model_id"] if c["build"]["response_source"]=="backbone" else c["judge"]["model_id"]}
            store.save_tensor("orbit/"+label,{"vectors":vectors,"metadata":metadata},metadata)
            # The source global loader reads its native file format. Export exact native payload atomically.
            import io
            buff=io.BytesIO(); torch.save({"vectors":vectors,"metadata":metadata},buff)
            path=store.root / "vectors" / f"{label}_angular_instruct.pt"
            atomic_bytes(path,buff.getvalue())
            mv,rt=load_global_layernorm_vectors(path,r_typical_method="pos_mean")
            store.save_tensor("global/"+label,{"module_vectors":mv,"r_typical":rt},metadata)
            values = torch.stack([v["features"]["ksteer"][0]-v["features"]["ksteer"][1] for key,v in entries]).mean(0)
            caa.append(values)
            for key,v in entries:
                pair=pairs.get(key)
                positive_key=(label,pair["pos_prompt"],pair["pos"]["text"])
                if positive_key in classifier_seen:
                    continue  # match source _build_rows_from_shared_build_data dedup
                classifier_seen.add(positive_key)
                for name in all_features:
                    all_features[name].append(v["features"][name][0].numpy())
                ys.append(c["labels"].index(label)); qids.append(pair["question_id"])
        store.save_tensor("caa",torch.stack(caa),{"labels":c["labels"],"composition":"mean_of_target_vectors","layer":c["steering"]["layer"]})
        for name in ("ksteer","eval"):
            train_classifier(c,store,name,np.stack(all_features[name]),np.array(ys),qids,budget)
    return store
