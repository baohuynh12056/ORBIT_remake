"""Installer contracts without downloading dependencies or requiring Torch."""
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("runtime_env_test", ROOT / "runtime_env.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimeBootstrap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.code = self.root / "code"
        (self.code / "requirements").mkdir(parents=True)
        (self.code / "requirements/runtime.lock.txt").write_text("numpy==1.26.4\n")
        self.base = self.root / "orbit-runtime"
        self.base.mkdir()
        (self.base / "legacy-failed-venv").write_text("preserve")
        self.calls = []

    def fake_create(self, actual, tools):
        python = actual / "bin/python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        return python, self.root / "uv", {}, {"version": [3, 12, 11], "executable": str(python)}

    def fake_run(self, command, env):
        self.calls.append(list(map(str, command)))

    def install(self, cpu=True):
        return runtime.install_env(self.code, self.base, cpu=cpu)

    def test_isolated_worker_and_completed_environment_reused(self):
        with patch.object(runtime, "create_worker_environment", side_effect=self.fake_create) as create, \
                patch.object(runtime, "_runtime_run", side_effect=self.fake_run), \
                patch.object(runtime.sys, "version", "3.13.0 (notebook kernel)"):
            python = self.install()
            marker = json.loads((python.parents[1] / "orbit-install.json").read_text())
            self.assertEqual(marker["identity"]["worker_python"], "3.12.11")
            self.assertIn("3.13", marker["host_python"])
            self.assertIn("-py312-", str(python))
            self.assertEqual((self.base / "legacy-failed-venv").read_text(), "preserve")
            self.assertEqual(self.install(), python)
            self.assertEqual(create.call_count, 1)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(self.calls[0][-1], "pip==25.2")
        self.assertIn("https://download.pytorch.org/whl/cpu", self.calls[1])
        self.assertEqual(self.calls[-1][-3:], ["-m", "pip", "check"])
        self.assertFalse(any("ensurepip" in command or "venv" in command for command in self.calls))

    def test_gpu_pins_and_lock_change_gets_new_environment(self):
        with patch.object(runtime, "create_worker_environment", side_effect=self.fake_create), \
                patch.object(runtime, "_runtime_run", side_effect=self.fake_run):
            gpu = self.install(cpu=False)
            self.assertIn("https://download.pytorch.org/whl/cu124", self.calls[1])
            self.assertIn("torch==2.6.0", self.calls[1])
            self.assertIn("bitsandbytes==0.48.2", self.calls[-2])
            cpu = self.install()
            self.assertNotEqual(cpu, gpu)
            (self.code / "requirements/runtime.lock.txt").write_text("numpy==1.26.4\n# revised\n")
            self.assertNotEqual(self.install(), cpu)

    def test_failed_install_has_no_completed_marker_and_can_resume(self):
        with patch.object(runtime, "create_worker_environment", side_effect=self.fake_create), \
                patch.object(runtime, "_runtime_run", side_effect=subprocess.CalledProcessError(1, ["pip"])):
            with self.assertRaises(subprocess.CalledProcessError):
                self.install()
        actual, = self.root.glob("orbit-runtime-py312-*")
        self.assertFalse((actual / "orbit-install.json").exists())
        self.assertEqual(json.loads((actual / "orbit-install-state.json").read_text())["status"], "failed")
        with patch.object(runtime, "create_worker_environment", side_effect=self.fake_create), \
                patch.object(runtime, "_runtime_run", side_effect=self.fake_run):
            self.assertEqual(self.install().parents[1], actual)
        self.assertEqual(json.loads((actual / "orbit-install.json").read_text())["status"], "completed")

    def test_create_uses_managed_python_and_checks_actual_version(self):
        commands = []
        with patch.object(runtime, "ensure_uv", return_value=self.root / "uv"), \
                patch.object(runtime, "_runtime_run", side_effect=lambda command, env: commands.append(list(map(str, command)))), \
                patch.object(runtime.subprocess, "check_output", return_value=json.dumps({"version": [3, 12, 11]})):
            runtime.create_worker_environment(self.root / "worker", self.root / "tools")
        self.assertEqual(commands[0][-3:], ["python", "install", "3.12.11"])
        self.assertIn("--managed-python", commands[1])
        with patch.object(runtime, "ensure_uv", return_value=self.root / "uv"), \
                patch.object(runtime, "_runtime_run"), \
                patch.object(runtime.subprocess, "check_output", return_value=json.dumps({"version": [3, 13, 0]})):
            with self.assertRaisesRegex(RuntimeError, "Unexpected worker Python"):
                runtime.create_worker_environment(self.root / "worker", self.root / "tools")

    def test_runtime_environment_removes_foreign_python_paths(self):
        with patch.dict(os.environ, {"PYTHONPATH": "/host/packages", "PYTHONHOME": "/host/python", "VIRTUAL_ENV": "/old/venv"}):
            env = runtime.runtime_environment(self.root)
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("PYTHONHOME", env)
            self.assertNotIn("VIRTUAL_ENV", env)
            self.assertEqual(os.environ["PYTHONPATH"], "/host/packages")
            self.assertEqual(env["UV_PYTHON_INSTALL_DIR"], str(self.root / "python"))

    def test_uv_checksum_and_executable_integrity_without_host_pip(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("uv.data/scripts/uv", "fixture native executable")
        data = buffer.getvalue()
        directory = self.root / ("uv-" + runtime.UV_VERSION)
        directory.mkdir()
        wheel = directory / "uv.whl"
        wheel.write_bytes(data)
        with patch.object(runtime, "UV_WHEEL_SHA256", hashlib.sha256(data).hexdigest()):
            executable = runtime.ensure_uv(self.root)
            self.assertEqual(executable.read_text(), "fixture native executable")
            self.assertTrue(executable.stat().st_mode & 0o100)
            executable.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "executable checksum"):
                runtime.ensure_uv(self.root)
        wheel.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "wheel checksum"):
            runtime.ensure_uv(self.root)


if __name__ == "__main__":
    unittest.main()
