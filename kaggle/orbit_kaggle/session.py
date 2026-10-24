"""Notebook-side helpers: standard library only, models live in isolated subprocesses."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
import tarfile
import uuid
from pathlib import Path
from checkpoint_tools import validate_checkpoint


def restore_run(source,destination):
    source,destination=Path(source),Path(destination)
    if not source.exists():raise FileNotFoundError(source)
    destination.mkdir(parents=True,exist_ok=True)
    index=source/"checkpoint_index.json"
    if index.exists():
        info=json.loads(index.read_text())
        if info.get("schema")!=1:raise ValueError("Unknown checkpoint archive schema")
        for archive in info["archives"]:
            path=source/archive["name"]
            if path.name!=archive["name"] or _file_hash(path)!=archive["sha256"]:
                raise ValueError(f"Checkpoint archive checksum mismatch: {path}")
            with tarfile.open(path,"r:gz") as tar:
                for member in tar:
                    dst=destination/member.name
                    if not dst.resolve().is_relative_to(destination.resolve()) or not member.isfile():
                        raise ValueError(f"Unsafe checkpoint member {member.name}")
                    incoming=tar.extractfile(member)
                    if dst.exists():
                        h=hashlib.sha256()
                        for data in iter(lambda:incoming.read(2**20),b""):h.update(data)
                        if h.hexdigest()!=_file_hash(dst):raise ValueError(f"Restore conflict: {dst}")
                    else:
                        dst.parent.mkdir(parents=True,exist_ok=True)
                        tmp=dst.with_name(dst.name+".restore.tmp")
                        with tmp.open("wb") as f:shutil.copyfileobj(incoming,f,2**20)
                        tmp.replace(dst)
        return
    for src in source.rglob("*"):
        if not src.is_file():continue
        dst=destination/src.relative_to(source)
        if dst.exists():
            if hashlib.sha256(src.read_bytes()).digest()!=hashlib.sha256(dst.read_bytes()).digest():
                raise ValueError(f"Restore would overwrite a different checkpoint: {dst}. Use one newest complete cumulative input.")
        else:
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)


def _file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda:f.read(2**20),b""):h.update(data)
    return h.hexdigest()


def fork_build_judge_run(source,destination,config):
    """Fork only immutable inputs/activations; rebuild filter/artifacts/calibration explicitly.

    Source must be unpacked. All generated pairs are cached in activation stage.
    Final-only re-scoring does not need a fork.
    """
    source,destination=Path(source),Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Fork destination must be empty")
    destination.mkdir(parents=True,exist_ok=True)
    for stage in ("data","build","activation"):
        manifest=json.loads((source/stage/"manifest.json").read_text())
        if manifest["status"]!="completed":raise ValueError(f"Cannot fork incomplete {stage}")
        shutil.copytree(source/stage,destination/stage)
    (destination/"resolved_config.json").write_text(json.dumps(config,indent=2)+"\n")
    return destination


def snapshot_run(root,part_bytes=1024**3):
    """Bound saved-output file count; keep unpacked working copy in /tmp after verified archive.

    This is a session checkpoint, not publication/upload. Kaggle must still save its version.
    Archive parts cap uncompressed bytes (except an individually larger file); gzip level 1.
    """
    root=Path(root)
    if (root/"checkpoint_index.json").exists() and not (root/"resolved_config.json").exists():
        return validate_checkpoint(root)
    files=sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("checkpoint-part-")
                 and p.name!="checkpoint_index.json" and not p.name.endswith(".tmp"))
    if not files:raise ValueError("No checkpoint files to archive")
    temporary=root.with_name(root.name+".packing-"+uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    parts=[];group=[];size=0
    def save_group(group,number):
        path=temporary/f"checkpoint-part-{number:04d}.tar.gz"
        with tarfile.open(path,"w:gz",compresslevel=1) as tar:
            for file in group:tar.add(file,arcname=str(file.relative_to(root)),recursive=False)
        parts.append({"name":path.name,"sha256":_file_hash(path),"bytes":path.stat().st_size,"files":len(group)})
    for path in files:
        n=path.stat().st_size
        if group and size+n>part_bytes:
            save_group(group,len(parts));group=[];size=0
        group.append(path);size+=n
    if group:save_group(group,len(parts))
    info={"schema":1,"archives":parts,"uncompressed_bytes":sum(p.stat().st_size for p in files),
          "unpacked_files":len(files),"durable_on_kaggle":False,
          "note":"Save Version with output files, then verify saved Output before ending session"}
    (temporary/"checkpoint_index.json").write_text(json.dumps(info,indent=2)+"\n")
    validate_checkpoint(temporary)
    # Retain the original unpacked files locally, never delete the only copy.
    live=Path("/tmp")/("orbit-unpacked-"+uuid.uuid4().hex)
    shutil.move(str(root),str(live))
    temporary.replace(root)
    validate_checkpoint(root)
    print("Saved-output snapshot prepared:",root,"live unpacked copy:",live,flush=True)
    return info


def run_stage(python,config,root,stage,time_budget=30000,extra=(),allow_checkpoint=False):
    cmd=[str(python),"-m","orbit_kaggle.cli",stage,"--config",str(config),"--root",str(root),"--time-budget",str(time_budget),*extra]
    result=subprocess.run(cmd)
    if result.returncode==75 and allow_checkpoint:
        return result.returncode
    if result.returncode:
        if result.returncode==75:
            raise RuntimeError("Stage checkpointed before completion. Save Version with current output, attach it next session, rerun this stage.")
        raise subprocess.CalledProcessError(result.returncode,cmd)
    return result.returncode


def install_env(code_root,env_root="/tmp/orbit-runtime",cpu=False):
    import importlib.util
    path=Path(code_root)/"runtime_env.py"
    spec=importlib.util.spec_from_file_location("orbit_runtime_environment",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.install_env(code_root,env_root,cpu)
