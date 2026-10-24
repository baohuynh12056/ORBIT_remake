from __future__ import annotations
from collections import defaultdict
from .storage import Store, digest
from .config import DATASET, stable_seed, implementation


def normalize(text):
    return " ".join(text.split()).casefold()


def snapshot_rows(rows):
    snapshot, seen, duplicates = [], {}, []
    for index, row in enumerate(rows):
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"ToneBank schema: nonempty text required at row {index}")
        qid = "tb-" + digest(normalize(row["text"]))[:24]
        item = {"question_id": qid, "source_index": index, "source_id": str(row.get("id", index)),
                "text": row["text"], "original": row}
        snapshot.append(item)
        if qid in seen:
            if normalize(seen[qid]["text"]) != normalize(row["text"]):
                raise ValueError("Question ID collision")
            duplicates.append({"question_id": qid, "indices": [seen[qid]["source_index"], index]})
        else:
            seen[qid] = item
    return snapshot, duplicates


def split_rows(snapshot, c):
    import numpy as np
    # Keep source order. Duplicate texts remain in raw snapshot, but count once for splitting/reporting.
    unique,seen=[],set()
    for row in snapshot:
        if row["question_id"] not in seen:
            seen.add(row["question_id"]);unique.append(row)
    s = c["split"]
    if s["mode"] == "shared_pool":
        selected = unique[:s["max_questions"]]
        train, test = list(selected), list(selected)
    else:
        rng = np.random.default_rng(s["seed"])
        selected = [unique[int(i)] for i in rng.permutation(len(unique))[:s["max_questions"]]]
        perm = np.random.default_rng(s["seed"] + 1).permutation(len(selected))
        n = max(1, min(len(selected) - 1, round(len(selected) * s["train_fraction"])))
        idx = set(perm[:n].tolist())
        train, test = [r for i,r in enumerate(selected) if i in idx], [r for i,r in enumerate(selected) if i not in idx]
    ncal = min(s["calibration_samples"], len(test) // 2)
    calibration, evaluation = test[:ncal], test[ncal:]
    evaluation = evaluation[:c["generation"]["n_questions"]]
    if not train or not calibration or not evaluation:
        raise ValueError("Empty train/calibration/test split; increase max_questions")
    if s["mode"] == "question_holdout":
        sets = [{r["question_id"] for r in part} for part in [train, calibration, evaluation]]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise AssertionError("Question leakage")
    return {"train": train, "calibration": calibration, "test": evaluation,
            "selected": selected, "split_config": s, "dedup_policy": "normalized_text_first_occurrence"}


def prepare(c, root, budget):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(DATASET, revision=c["dataset"]["revision"])
    if info.sha != c["dataset"]["revision"]:
        raise ValueError("Dataset revision did not resolve to configured commit")
    store = Store(root, "data", {"dataset": c["dataset"], "split": c["split"],
                                 "n_eval": c["generation"]["n_questions"],"implementation":implementation("data.py")})
    with store.transaction():
        if store.get("snapshot") is None:
            budget.check()
            ds = load_dataset(DATASET, revision=info.sha, split="train")
            rows, duplicates = snapshot_rows([dict(r) for r in ds])
            store.put("snapshot", {"rows": rows, "duplicates": duplicates, "schema": str(ds.features),
                                   "hf_fingerprint": ds._fingerprint, "revision": info.sha,
                                   "content_fingerprint": digest(rows)})
        snap = store.get("snapshot")
        store.put("split", split_rows(snap["rows"], c))
        store.log("snapshot", rows=len(snap["rows"]), duplicates=len(snap["duplicates"]))
    return store


def build_pairs(c, split):
    import numpy as np
    from source_prompts import _build_contrastive_prompt_pairs_for_label
    from src.utils.tasks import tones_prompts, negative_tones_prompts
    pos = {k: [v] for k, v in tones_prompts()}
    neg = {k: [v] for k, v in negative_tones_prompts()}
    questions = [r["text"] for r in split["train"]]
    id_by_text = {r["text"]: r["question_id"] for r in split["train"]}
    rng = np.random.default_rng(c["build"]["negative_seed"])
    result = []
    for label in c["labels"]:
        pp, np_, pq, nq = _build_contrastive_prompt_pairs_for_label(
            label=label, labels=c["labels"], questions=questions, pos_prompts=pos, neg_prompts=neg,
            negatives_mode="neg_prompt", rng=rng, max_pairs=c["build"]["max_pairs"],
            repeat_each_question_across_variants=False)
        for i, (p,n,q,nq_) in enumerate(zip(pp,np_,pq,nq)):
            assert q == nq_
            result.append({"pair_id": digest([label,i,id_by_text[q],p,n])[:32], "pair_index":i,"label": label,
                           "question_id": id_by_text[q], "question": q, "pos_prompt": p, "neg_prompt": n})
            if c["build"].get("prompt_role") == "system":
                for side, text in (("pos",p),("neg",n)):
                    suffix = "\n" + q
                    if not text.endswith(suffix):
                        raise ValueError("Cannot separate source instruction from question")
                    result[-1][side+"_system"] = text[:-len(suffix)]
                    result[-1][side+"_user"] = q
    return result
