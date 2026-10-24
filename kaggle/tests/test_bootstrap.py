"""Bundle bootstrapping uses only stdlib: no GPU, Torch or network required."""
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("orbit_bundle_bootstrap_test", ROOT / "bootstrap_bundle.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class BundleBootstrap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / "input"
        self.inputs.mkdir()
        self.destination = self.root / "code"
        self.modules = patch.dict(sys.modules, {"orbit_kaggle": None})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def fixture(self, name="datasets/user/code/1", legacy=False):
        path = self.inputs / name
        files = {"orbit_kaggle/__init__.py": "# fixture\n", "orbit_kaggle/session.py": "# fixture\n",
                 "configs/smoke.json": "{}\n", "requirements/runtime.lock.txt": "# fixture\n",
                 "vendor/source_prompts.py": "# fixture prompts\n", "vendor/angular_residual_utils.py": "# fixture numerical code\n"}
        source = {"sha256": {"angular_residual_utils.py": hashlib.sha256(files["vendor/angular_residual_utils.py"].encode()).hexdigest(),
                              "ksteering/src/judges/tone_judge.py": "reference-only-not-bundled"},
                  "generated_prompt_sha256": hashlib.sha256(files["vendor/source_prompts.py"].encode()).hexdigest()}
        files["vendor/source_manifest.json"] = json.dumps(source)
        for name, content in files.items():
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        if not legacy:
            checksums = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()}
            (path / "bundle_files.sha256.json").write_text(json.dumps({"schema": 1, "files": checksums}))
        return path

    def prepare(self, explicit=None):
        return bootstrap.prepare_bundle(explicit, self.inputs, self.destination)

    def test_auto_extracted_folder_and_explicit_parent(self):
        source = self.fixture()
        # Kaggle can leave the separately uploaded outer checksum beside expanded files.
        (source / "orbit_kaggle_bundle.sha256").write_text("original-zip-checksum")
        self.assertEqual(bootstrap.locate_bundle(input_root=self.inputs), source)
        result = self.prepare()
        self.assertTrue((result / "orbit_kaggle/session.py").is_file())
        self.assertEqual(self.prepare(self.inputs / "datasets"), result)
        self.assertEqual(self.prepare(source), result)

    def test_original_zip_with_outer_checksum(self):
        source = self.fixture()
        archive = self.inputs / "orbit_kaggle_bundle.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for path in source.rglob("*"):
                if path.is_file(): z.write(path, str(path.relative_to(source)))
        archive.with_suffix(".sha256").write_text(hashlib.sha256(archive.read_bytes()).hexdigest())
        self.assertEqual(bootstrap.locate_bundle(input_root=self.inputs), archive)
        self.assertEqual(self.prepare(archive), self.prepare(source))

    def test_nested_expanded_legacy_is_not_another_input_bundle(self):
        source = self.fixture("paper")
        nested = self.fixture("paper/legacy/old_bundle")
        self.assertEqual(bootstrap.locate_bundle(input_root=self.inputs), source)
        self.assertEqual(bootstrap.locate_bundle(nested), nested)

    def test_missing_and_ambiguous_inputs(self):
        with self.assertRaisesRegex(ValueError, "thư mục bundle"):
            self.prepare()
        first = self.fixture("one")
        self.fixture("two")
        with self.assertRaisesRegex(ValueError, "BUNDLE_ZIP"):
            self.prepare()
        self.assertEqual(bootstrap.locate_bundle(first), first)

    def test_expanded_checksum_and_local_cache_corruption(self):
        source = self.fixture()
        destination = self.prepare()
        (destination / "configs/smoke.json").write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "Local code cache checksum"):
            self.prepare()
        (source / "configs/smoke.json").write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "Bundle checksum"):
            self.prepare()

    def test_legacy_expanded_bundle_vendor_checks(self):
        source = self.fixture(legacy=True)
        self.assertTrue((self.prepare() / "configs/smoke.json").exists())
        (source / "vendor/source_prompts.py").write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "Legacy prompt checksum"):
            self.prepare()

    def test_unsafe_zip_path_rejected(self):
        archive = self.inputs / "orbit_kaggle_bundle.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("../escaped.py", "unsafe")
        archive.with_suffix(".sha256").write_text(hashlib.sha256(archive.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, "Unsafe ZIP"):
            self.prepare()
        self.assertFalse((self.root / "escaped.py").exists())


if __name__ == "__main__":
    unittest.main()
