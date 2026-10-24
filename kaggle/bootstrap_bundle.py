"""Standard-library bootstrap, embedded in every notebook before package import."""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path


def _bundle_file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _is_bundle_root(path):
    return all((path / name).is_file() for name in (
        "orbit_kaggle/__init__.py", "orbit_kaggle/session.py", "configs/smoke.json",
        "requirements/runtime.lock.txt", "vendor/source_manifest.json"))


def _bundle_roots(path):
    if _is_bundle_root(path):
        return [path]
    candidates = {p.parent.parent for p in path.rglob("source_manifest.json")
                  if p.parent.name == "vendor" and _is_bundle_root(p.parent.parent)}
    # Kaggle can recursively expand archived releases inside the outer bundle.
    return sorted(p for p in candidates if not any(parent in candidates for parent in p.parents))


def locate_bundle(bundle_path=None, input_root="/kaggle/input"):
    """BUNDLE_ZIP remains compatible; its value can now also be a directory."""
    root = Path(bundle_path) if bundle_path is not None else Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"Bundle input không tồn tại: {root}")
    if root.is_file():
        if not zipfile.is_zipfile(root):
            raise ValueError(f"Bundle file không phải ZIP: {root}")
        return root
    if bundle_path is not None and _is_bundle_root(root):
        return root
    zips = sorted(root.rglob("orbit_kaggle_bundle.zip"))
    if len(zips) == 1:
        return zips[0]
    roots = _bundle_roots(root)
    if not zips and len(roots) == 1:
        return roots[0]
    raise ValueError(
        "Cần đúng một bundle. BUNDLE_ZIP có thể là đường dẫn ZIP HOẶC thư mục "
        "chứa orbit_kaggle/, configs/, vendor/, requirements/. "
        f"ZIP tìm thấy: {[str(p) for p in zips]}; thư mục bundle: {[str(p) for p in roots]}")


def _verify_bundle_tree(root):
    manifest_path = root / "bundle_files.sha256.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != 1 or not isinstance(manifest.get("files"), dict):
            raise ValueError("Sai schema bundle_files.sha256.json")
        for name, checksum in manifest["files"].items():
            path = root / name
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Unsafe manifest path: {name}")
            if not path.is_file() or _bundle_file_hash(path) != checksum:
                raise ValueError(f"Bundle checksum mismatch: {name}")
        # Every executable/config file must be covered, including after auto-extraction.
        actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
                  and p.name not in ("bundle_files.sha256.json", "orbit_kaggle_bundle.sha256")
                  and "__pycache__" not in p.parts}
        if actual != set(manifest["files"]):
            raise ValueError("Bundle file list differs from checksum manifest")
        return manifest["files"]
    # Compatibility with the first released ZIP when Kaggle has removed the ZIP.
    # Its original outer ZIP hash cannot verify the expanded adapter files.
    source = json.loads((root / "vendor/source_manifest.json").read_text())
    for name, checksum in source["sha256"].items():
        if name.startswith(("angular_", "ksteering/src/steering/", "ksteering/src/utils/")) or name in ("ksteering/src/__init__.py", "ksteering/LICENSE"):
            path = root / "vendor" / name
            if not path.is_file() or _bundle_file_hash(path) != checksum:
                raise ValueError(f"Legacy vendor checksum mismatch: {name}")
    if _bundle_file_hash(root / "vendor/source_prompts.py") != source["generated_prompt_sha256"]:
        raise ValueError("Legacy prompt checksum mismatch")
    print("Bundle cũ đã giải nén: đã kiểm checksum vendor; chưa có checksum đầy đủ "
          "cho adapter/config. Bundle mới có bundle_files.sha256.json để kiểm toàn bộ file.")
    return {str(p.relative_to(root)): _bundle_file_hash(p) for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}


def prepare_bundle(bundle_path=None, input_root="/kaggle/input", destination_base="/tmp/orbit-code"):
    import tempfile
    source = locate_bundle(bundle_path, input_root)
    with tempfile.TemporaryDirectory(prefix="orbit-bootstrap-") as tmp:
        if source.is_file():
            checksum_file = source.with_suffix(".sha256")
            if not checksum_file.is_file():
                raise FileNotFoundError("Đính kèm orbit_kaggle_bundle.sha256 bên cạnh ZIP")
            if _bundle_file_hash(source) != checksum_file.read_text().split()[0]:
                raise ValueError("Sai checksum bundle ZIP")
            tree = Path(tmp)
            with zipfile.ZipFile(source) as archive:
                for member in archive.infolist():
                    if not (tree / member.filename).resolve().is_relative_to(tree.resolve()):
                        raise ValueError("Unsafe ZIP path")
                archive.extractall(tree)
        else:
            tree = source
        if not _is_bundle_root(tree):
            raise ValueError(f"Thiếu module/config trong bundle: {tree}")
        files = _verify_bundle_tree(tree)
        fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        destination = Path(destination_base) / fingerprint
        if not destination.exists():
            destination.mkdir(parents=True)
            for name in files:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tree / name, target)
            if (tree / "bundle_files.sha256.json").exists():
                shutil.copy2(tree / "bundle_files.sha256.json", destination / "bundle_files.sha256.json")
        for name, checksum in files.items():
            if not (destination / name).is_file() or _bundle_file_hash(destination / name) != checksum:
                raise ValueError(f"Local code cache checksum mismatch: {name}")
        loaded = sys.modules.get("orbit_kaggle")
        if loaded is not None and Path(loaded.__file__).resolve().parent != destination / "orbit_kaggle":
            raise RuntimeError("Bundle đã đổi trong kernel đang dùng. Restart kernel rồi chạy lại.")
    print("Code input:", source, "→ CODE_ROOT:", destination)
    return destination
