"""Atomic per-sample shards with checksums; interrupted writes never become cache hits."""
from __future__ import annotations
import contextlib
import hashlib
import io
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, value):
    atomic_bytes(path, (canonical(value) + "\n").encode())


def read_json(path):
    return json.loads(Path(path).read_text())


class BudgetExpired(RuntimeError):
    pass


class Budget:
    def __init__(self, seconds=0, stop_after=0):
        self.deadline = time.monotonic() + seconds if seconds else float("inf")
        self.stop_after = stop_after
        self.commits = 0

    def check(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExpired("Time budget reached; completed shards are safe on this session's disk.")

    def committed(self):
        self.commits += 1
        if self.stop_after and self.commits >= self.stop_after:
            raise BudgetExpired("Intentional interruption after committed shard (resume test)")


def resources():
    import shutil
    result = {"disk_free_bytes": shutil.disk_usage(Path.cwd()).free}
    try:
        import torch
        result["vram"] = [{"device": i, "allocated": torch.cuda.memory_allocated(i),
                            "peak": torch.cuda.max_memory_allocated(i)} for i in range(torch.cuda.device_count())]
    except ImportError:
        result["vram"] = []
    return result


class Store:
    def __init__(self, root, stage, identity):
        self.root = Path(root) / stage
        self.stage = stage
        self.identity = identity
        self.fp = digest(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "manifest.json"
        if path.exists():
            old = read_json(path)
            if old["fingerprint"] != self.fp:
                raise ValueError(f"Cache config mismatch: {path}. Use a new run/report directory; do not overwrite old artifacts.")
            if old["identity"] != identity:
                raise ValueError(f"Manifest identity mismatch: {path}")
        else:
            provenance_path=Path(__file__).resolve().parents[1]/"vendor/source_manifest.json"
            provenance=read_json(provenance_path) if provenance_path.exists() else None
            write_json(path, {"schema": 1, "stage": stage, "identity": identity,
                              "fingerprint": self.fp, "status": "pending","source_provenance":provenance})

    def path(self, key):
        return self.root / "shards" / f"{digest(key)}.json"

    def get(self, key):
        path = self.path(key)
        if not path.exists():
            return None
        row = read_json(path)
        if row["fingerprint"] != self.fp or row["key"] != key or row["schema"] != 1:
            raise ValueError(f"Invalid shard identity: {path}")
        if digest(row["payload"]) != row["checksum"]:
            raise ValueError(f"Corrupt JSON shard: {path}")
        return row["payload"]

    def put(self, key, payload):
        write_json(self.path(key), {"schema": 1, "key": key, "fingerprint": self.fp,
                                   "checksum": digest(payload), "payload": payload})

    def rows(self):
        for path in sorted((self.root / "shards").glob("*.json")):
            row = read_json(path)
            yield row["key"], self.get(row["key"])

    def save_tensor(self, key, value, metadata):
        import torch
        tensors = {}
        def inspect(obj, prefix=""):
            if torch.is_tensor(obj):
                if obj.device.type != "cpu" or not torch.isfinite(obj).all():
                    raise ValueError(f"Tensor must be finite CPU tensor: {prefix}")
                tensors[prefix] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    inspect(v, prefix + "/" + str(k))
            elif isinstance(obj, (list, tuple)):
                for k, v in enumerate(obj):
                    inspect(v, prefix + "/" + str(k))
        inspect(value)
        out = io.BytesIO()
        torch.save(value, out)
        data = out.getvalue()
        name = f"{digest(key)}.pt"
        atomic_bytes(self.root / "tensors" / name, data)
        self.put(key, {"tensor_file": name, "sha256": hashlib.sha256(data).hexdigest(),
                       "bytes": len(data), "tensors": tensors, "metadata": metadata})

    def load_tensor(self, key):
        import torch
        row = self.get(key)
        if row is None:
            return None
        data = (self.root / "tensors" / row["tensor_file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"Tensor checksum mismatch: {key}")
        value = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        for path, meta in row["tensors"].items():
            obj = value
            for part in path.split("/")[1:]:
                if isinstance(obj, (tuple,list)):
                    obj=obj[int(part)]
                elif part in obj:
                    obj=obj[part]
                else:
                    obj=obj[int(part)]  # torch optimizer.state uses integer parameter IDs
            if list(obj.shape) != meta["shape"] or str(obj.dtype) != meta["dtype"] or not torch.isfinite(obj).all():
                raise ValueError(f"Tensor shape/dtype/finiteness mismatch: {key} {path}")
        return value

    def log(self, event, **fields):
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), "stage": self.stage,
               "event": event, **resources(), **fields}
        line = canonical(row)
        with open(self.root / "events.jsonl", "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    @contextlib.contextmanager
    def transaction(self):
        start = time.monotonic()
        def status(s, **extra):
            m = read_json(self.root / "manifest.json")
            m.update(status=s, **extra)
            write_json(self.root / "manifest.json", m)
        status("running")
        try:
            yield self
        except BudgetExpired as e:
            status("pending", reason=str(e))
            self.log("checkpoint", elapsed_s=time.monotonic() - start)
            raise
        except BaseException as e:
            status("failed", error=f"{type(e).__name__}: {e}")
            self.log("failed", error=str(e), elapsed_s=time.monotonic() - start)
            raise
        else:
            status("completed", elapsed_s=time.monotonic() - start)
            self.log("completed", elapsed_s=time.monotonic() - start)


def open_existing(root, stage):
    path = Path(root) / stage / "manifest.json"
    manifest = read_json(path)
    if manifest["status"] != "completed":
        raise ValueError(f"Upstream stage not completed: {path}")
    return Store(root, stage, manifest["identity"])


def content_identity(store):
    return digest([(key, row) for key, row in store.rows()])
