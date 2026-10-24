"""Real filesystem checks for snapshots and small downloadable ZIPs; no Torch."""
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("checkpoint_tools", ROOT / "checkpoint_tools.py")
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)
spec = importlib.util.spec_from_file_location("checkpoint_session_test", ROOT / "orbit_kaggle/session.py")
session = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"checkpoint_tools": checkpoint}):
    spec.loader.exec_module(session)


class CheckpointDownloads(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        (self.run / "resolved_config.json").write_text('{"fixture":true}\n')
        (self.run / "data").mkdir()
        (self.run / "data/snapshot.json").write_text('{"questions":[]}\n')
        with patch.object(session.shutil, "move", wraps=shutil.move) as move:
            self.info = session.snapshot_run(self.run)
            retained = Path(move.call_args.args[1])
        self.retained = retained
        self.addCleanup(shutil.rmtree, retained)

    def test_exported_zip_expands_to_restorable_dataset(self):
        output = checkpoint.export_checkpoint_zip(self.run, self.root / "download.zip")
        with zipfile.ZipFile(output) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(set(archive.namelist()), {"checkpoint_index.json", "checkpoint-part-0000.tar.gz"})
            archive.extractall(self.root / "input")
        session.restore_run(self.root / "input", self.root / "restored")
        self.assertEqual(json.loads((self.root / "restored/resolved_config.json").read_text()), {"fixture": True})
        self.assertEqual(json.loads((self.root / "restored/data/snapshot.json").read_text()), {"questions": []})

    def test_existing_snapshot_revalidated_instead_of_trusting_stale_log(self):
        self.assertEqual(session.snapshot_run(self.run), self.info)
        part = self.run / self.info["archives"][0]["name"]
        original = part.read_bytes()
        part.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        with self.assertRaisesRegex(ValueError, "checksum"):
            session.snapshot_run(self.run)
        with self.assertRaisesRegex(ValueError, "checksum"):
            checkpoint.export_checkpoint_zip(self.run, self.root / "download.zip")
        part.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing checkpoint part"):
            session.snapshot_run(self.run)

    def test_raw_backup_download_restores_when_working_checkpoint_missing(self):
        shutil.rmtree(self.run)
        backup = Path(shutil.make_archive(str(self.root / "recovery"), "zip", root_dir=self.retained))
        with zipfile.ZipFile(backup) as archive:
            archive.extractall(self.root / "recovery-input")
        session.restore_run(self.root / "recovery-input", self.root / "restored")
        self.assertEqual((self.root / "restored/resolved_config.json").read_bytes(),
                         (self.retained / "resolved_config.json").read_bytes())
        self.assertEqual((self.root / "restored/data/snapshot.json").read_bytes(),
                         (self.retained / "data/snapshot.json").read_bytes())

    def test_size_cap_does_not_duplicate_large_checkpoint(self):
        self.assertIsNone(checkpoint.export_checkpoint_zip(self.run, self.root / "download.zip", max_bytes=1))
        self.assertFalse((self.root / "download.zip").exists())
        self.assertEqual(checkpoint.validate_checkpoint(self.run), self.info)

    def test_failed_export_keeps_previous_zip_and_cleans_temporary(self):
        output = self.root / "download.zip"
        checkpoint.export_checkpoint_zip(self.run, output)
        original = output.read_bytes()
        with patch.object(checkpoint.zipfile, "ZipFile", side_effect=OSError("interrupted export")):
            with self.assertRaisesRegex(OSError, "interrupted export"):
                checkpoint.export_checkpoint_zip(self.run, output)
        self.assertEqual(output.read_bytes(), original)
        self.assertFalse(output.with_name(output.name + ".tmp").exists())

    def test_duplicate_or_traversal_part_rejected(self):
        index = self.run / "checkpoint_index.json"
        self.info["archives"] *= 2
        index.write_text(json.dumps(self.info))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            checkpoint.validate_checkpoint(self.run)
        self.info["archives"] = [{"name": "../outside.tar.gz"}]
        index.write_text(json.dumps(self.info))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            checkpoint.validate_checkpoint(self.run)


if __name__ == "__main__":
    unittest.main()
