"""Pinned worker Python independent of the notebook's pip/venv/ensurepip.

Embedded in notebooks as well as shipped in the bundle, so the updated notebook
can also launch the first release's already attached code Dataset.
"""
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

WORKER_PYTHON = "3.12.11"
UV_VERSION = "0.8.22"
PIP_VERSION = "25.2"
UV_WHEEL_URL = "https://files.pythonhosted.org/packages/3a/a9/a83cee9b8cf63e57ce64ba27c77777cc66410e144fd178368f55af1fa18d/uv-0.8.22-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
UV_WHEEL_SHA256 = "8efec4ef5acddc35f0867998c44e0b15fc4dace1e4c26d01443871a2fbb04bf6"


def _runtime_write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    tmp.replace(path)


def _runtime_run(command, env):
    print("Runtime setup:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), env=env, check=True)


def ensure_uv(tools_root):
    """Extract a checksum-pinned native wheel; requires neither host pip nor root."""
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("This Kaggle runtime bootstrap supports Linux x86_64")
    directory = Path(tools_root) / ("uv-" + UV_VERSION)
    directory.mkdir(parents=True, exist_ok=True)
    wheel = directory / "uv.whl"
    if not wheel.exists():
        print("Downloading checksum-pinned uv", UV_VERSION, flush=True)
        try:
            with urllib.request.urlopen(UV_WHEEL_URL, timeout=120) as response:
                data = response.read()
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError("Không tải được uv bootstrap. Bật Internet trên Kaggle và thử lại; không cần apt/sudo/ensurepip.") from exc
        if hashlib.sha256(data).hexdigest() != UV_WHEEL_SHA256:
            raise ValueError("uv wheel checksum mismatch")
        temporary = wheel.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(wheel)
    data = wheel.read_bytes()
    if hashlib.sha256(data).hexdigest() != UV_WHEEL_SHA256:
        raise ValueError("Cached uv wheel checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = [name for name in archive.namelist() if Path(name).name == "uv"]
        if len(members) != 1:
            raise ValueError("Unexpected uv wheel executable layout")
        executable = archive.read(members[0])
    uv = directory / "uv"
    if uv.exists() and uv.read_bytes() != executable:
        raise ValueError("Cached uv executable checksum mismatch")
    if not uv.exists():
        temporary = uv.with_suffix(".tmp")
        temporary.write_bytes(executable)
        temporary.chmod(0o755)
        temporary.replace(uv)
    return uv


def runtime_environment(tools_root):
    """Keep managed Python, cache and executable links inside the task directory."""
    root = Path(tools_root)
    env = dict(os.environ)
    env.update(UV_PYTHON_INSTALL_DIR=str(root / "python"), UV_PYTHON_BIN_DIR=str(root / "bin"),
               UV_CACHE_DIR=str(root / "cache"), UV_NATIVE_TLS="true")
    # Worker environments must not inherit another interpreter's site-packages.
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(name, None)
    return env


def create_worker_environment(env_root, tools_root):
    env_root, tools_root = Path(env_root), Path(tools_root)
    uv = ensure_uv(tools_root)
    env = runtime_environment(tools_root)
    python = env_root / "bin/python"
    if not python.exists():
        _runtime_run([uv, "--no-config", "python", "install", WORKER_PYTHON], env)
        _runtime_run([uv, "--no-config", "venv", "--managed-python", "--python", WORKER_PYTHON, env_root], env)
    probe = json.loads(subprocess.check_output(
        [str(python), "-I", "-c", "import sys,json; print(json.dumps({'version':list(sys.version_info[:3]),'executable':sys.executable}))"],
        env=env, text=True))
    if probe["version"] != [int(x) for x in WORKER_PYTHON.split(".")]:
        raise RuntimeError(f"Unexpected worker Python {probe}; expected {WORKER_PYTHON}. Use a fresh environment path.")
    return python, uv, env, probe


def install_env(code_root, env_root="/tmp/orbit-runtime", cpu=False, tools_root=None):
    lock = Path(code_root) / "requirements/runtime.lock.txt"
    identity = {"schema": 2, "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(), "cpu": bool(cpu),
                "worker_python": WORKER_PYTHON, "uv": UV_VERSION, "uv_wheel_sha256": UV_WHEEL_SHA256,
                "pip": PIP_VERSION, "torch": "2.6.0", "torchvision": "0.21.0", "bitsandbytes": None if cpu else "0.48.2"}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    base = Path(env_root)
    # A failed legacy /tmp/orbit-runtime created by Python 3.13 is left untouched.
    actual = base.with_name(base.name + "-py312-" + fingerprint[:16])
    tools_root = Path(tools_root) if tools_root else base.parent / "orbit-tools"
    marker = actual / "orbit-install.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get("fingerprint") != fingerprint:
            raise RuntimeError("Environment fingerprint mismatch")
        python = actual / "bin/python"
        if not python.exists():
            raise RuntimeError("Installed environment lost its Python executable")
        return python
    actual.parent.mkdir(parents=True, exist_ok=True)
    python, uv, env, probe = create_worker_environment(actual, tools_root)
    state = actual / "orbit-install-state.json"
    _runtime_write_json(state, {"identity": identity, "status": "running", "worker": probe})
    try:
        # uv seeds pip without invoking the notebook interpreter's ensurepip.
        _runtime_run([uv, "--no-config", "pip", "install", "--python", python, f"pip=={PIP_VERSION}"], env)
        index = "https://download.pytorch.org/whl/cpu" if cpu else "https://download.pytorch.org/whl/cu124"
        _runtime_run([python, "-m", "pip", "install", "--only-binary=:all:", "torch==2.6.0", "torchvision==0.21.0", "--index-url", index], env)
        _runtime_run([python, "-m", "pip", "install", "--only-binary=:all:", "-r", lock], env)
        if not cpu:
            _runtime_run([python, "-m", "pip", "install", "--only-binary=:all:", "bitsandbytes==0.48.2"], env)
        _runtime_run([python, "-m", "pip", "check"], env)
    except BaseException as exc:
        _runtime_write_json(state, {"identity": identity, "status": "failed", "worker": probe,
                                    "error": f"{type(exc).__name__}: {exc}"})
        raise
    result = {"identity": identity, "fingerprint": fingerprint, "worker": probe,
              "host_python": sys.version, "status": "completed"}
    _runtime_write_json(marker, result)
    _runtime_write_json(state, result)
    print("Worker ready:", python, "Python", WORKER_PYTHON, flush=True)
    return python
