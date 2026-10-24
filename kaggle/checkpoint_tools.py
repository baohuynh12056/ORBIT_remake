"""Standard-library checkpoint validation and small download exports.

Embedded in the notebooks so it also works with an attached legacy code bundle.
"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def validate_checkpoint(root):
    root = Path(root)
    index = root / "checkpoint_index.json"
    info = json.loads(index.read_text())
    if info.get("schema") != 1 or not isinstance(info.get("archives"), list) or not info["archives"]:
        raise ValueError(f"Invalid checkpoint index: {index}")
    seen = set()
    for part in info["archives"]:
        name = part["name"]
        if not isinstance(name, str) or Path(name).name != name or name in seen:
            raise ValueError(f"Unsafe or duplicate checkpoint part: {name}")
        seen.add(name)
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint part in current session: {path}")
        if path.stat().st_size != part["bytes"]:
            raise ValueError(f"Checkpoint size mismatch: {path}")
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(2**20), b""):
                checksum.update(block)
        if checksum.hexdigest() != part["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return info


def export_checkpoint_zip(root, destination, max_bytes=64 * 1024**2):
    """Export a verified small checkpoint. Return None above the explicit size cap.

    ZIP contains the index and original gzip parts directly at its root, so a
    Kaggle auto-expanded Dataset can be passed straight to restore_run().
    This only creates a local file; it does not save a Kaggle version.
    """
    root, destination = Path(root), Path(destination)
    info = validate_checkpoint(root)
    files = [root / "checkpoint_index.json"] + [root / part["name"] for part in info["archives"]]
    total = sum(path.stat().st_size for path in files)
    if total > max_bytes:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < total + 1024**2:
        raise OSError(f"Not enough free disk space for checkpoint download ZIP: {destination}")
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in files:
                archive.write(path, path.name)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("Checkpoint download ZIP failed CRC validation")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
