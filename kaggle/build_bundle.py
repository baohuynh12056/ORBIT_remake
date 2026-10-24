#!/usr/bin/env python3
"""Freeze unchanged numerical code + selected pure prompt/pairing definitions; produce self-contained zip."""
from __future__ import annotations
import ast
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def main():
    vendor=HERE/"vendor";vendor.mkdir(exist_ok=True)
    files=["angular_residual_utils.py","angular_steering_utils.py"]
    files += [str(p.relative_to(ROOT)) for p in (ROOT/"ksteering/src/steering").glob("*.py")]
    files += [str(p.relative_to(ROOT)) for p in (ROOT/"ksteering/src/utils").glob("*.py")]
    files += ["ksteering/src/__init__.py","ksteering/LICENSE"]
    for rel in files:
        dst=vendor/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dst)
    source=ROOT/"scripts/run_experiment_pipeline.py"
    tree=ast.parse(source.read_text())
    names={"DEFAULT_COHERENCE_PROMPT","TARGET_ATTRIBUTE_PROMPT","DEFINITION_JUDGE_PROMPT","TONE_DEFINITIONS",
           "_allocate_stratified_counts","_build_contrastive_prompt_pairs_for_label","_format_prompts_for_model"}
    selected=[]
    for node in tree.body:
        name=node.name if isinstance(node,(ast.FunctionDef,ast.ClassDef)) else None
        if isinstance(node,ast.Assign):
            name=node.targets[0].id if isinstance(node.targets[0],ast.Name) else None
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):name=node.target.id
        if name in names:selected.append(node)
    header="from __future__ import annotations\nfrom typing import *\nimport numpy as np\nPromptVariantsMap = Dict[str, List[str]]\n"
    judge_tree=ast.parse((ROOT/"ksteering/src/judges/tone_judge.py").read_text())
    tone=next(n for n in judge_tree.body if isinstance(n,ast.ClassDef))
    init=next(n for n in tone.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
    descriptions=next(n for n in init.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Attribute) and n.targets[0].attr=="tone_descriptions")
    init_new=ast.parse("def __init__(self):\n    pass").body[0]
    init_new.body=[descriptions]
    methods=[n for n in tone.body if isinstance(n,ast.FunctionDef) and n.name in ("_style_description","_create_evaluation_prompt")]
    prompt_cls=ast.ClassDef(name="TonePrompt",bases=[],keywords=[],body=[init_new]+methods,decorator_list=[])
    module=ast.Module(body=selected+[prompt_cls],type_ignores=[])
    ast.fix_missing_locations(module)
    (vendor/"source_prompts.py").write_text(header+ast.unparse(module)+"\n")
    references=["scripts/run_main_experiment.py","scripts/run_experiment_pipeline.py","scripts/judge_responses.py",
                "ksteering/src/judges/tone_judge.py","README.md","docs/VECTOR_AND_DATA_BUILD.md","data/README.md"]
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files+references}
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    manifest={"commit":commit,"sha256":hashes,"prompt_extraction":"AST selected pure definitions; TonePrompt excludes network client",
              "generated_prompt_sha256":hashlib.sha256((vendor/"source_prompts.py").read_bytes()).hexdigest()}
    (vendor/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    dist=HERE/"dist";dist.mkdir(exist_ok=True)
    output=dist/"orbit_kaggle_bundle.zip"
    paths=[]
    # Keep legacy ZIP separately in the repo: Kaggle recursively expands nested ZIPs,
    # making legacy code look like a second bundle and invalidating its file manifest.
    for folder in ("orbit_kaggle","vendor","configs","requirements","tests","notebooks","checks"):
        paths.extend(path for path in sorted((HERE/folder).rglob("*"))
                     if path.is_file() and "__pycache__" not in path.parts)
    paths.extend(HERE/name for name in ("bootstrap_bundle.py","runtime_env.py","checkpoint_tools.py","README_KAGGLE.md","SOURCE_MAP.md","TEST_REPORT.md") if (HERE/name).exists())
    integrity={"schema":1,"files":{str(path.relative_to(HERE)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for path in paths:z.write(path,str(path.relative_to(HERE)))
        z.writestr("bundle_files.sha256.json",json.dumps(integrity,sort_keys=True,indent=2)+"\n")
    (dist/"orbit_kaggle_bundle.sha256").write_text(hashlib.sha256(output.read_bytes()).hexdigest()+"  "+output.name+"\n")
    print(output)


if __name__=="__main__":main()
