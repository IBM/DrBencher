"""
Enroot-sandboxed stateful Python execution environment.

Adapted from jit_gf/react/python_tool.py. Spawns a persistent Python REPL inside
an enroot container (a ``.sqsh`` image), so a question's variables/imports/defs
persist across calls. Requires the ``enroot`` CLI on PATH and a ``.sqsh`` image.

DrBencher uses this to replace the fragile pip-built ``.python_tool_env`` venv
(which needs outbound pip access that cluster nodes often block). The image is
loaded from the repo (default ``<repo>/assets/python_tool.sqsh``, overridable
with the ``DRBENCHER_ENROOT_SQSH`` env var).
"""
from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

# Repo root = parent of tools/. Default image lives under the repo for portability.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SQSH = str(_REPO_ROOT / "assets" / "python_tool.sqsh")


def enroot_image_path() -> str:
    """Resolve the enroot image path (env override wins over the repo default)."""
    return os.environ.get("DRBENCHER_ENROOT_SQSH") or DEFAULT_SQSH


def enroot_available() -> bool:
    """True if enroot is usable: CLI on PATH, image present, and not disabled."""
    if os.environ.get("DRBENCHER_NO_ENROOT") == "1":
        return False
    return shutil.which("enroot") is not None and os.path.exists(enroot_image_path())


# REPL script that runs inside the enroot container — maintains state across
# executions via a persistent ``local_env`` dict. JSON lines over stdin/stdout.
_ENROOT_REPL_SCRIPT = r'''
import sys
import io
import json
import contextlib
import warnings
import builtins
import ast as _ast

def _auto_capture_last_expr(code):
    """Auto-capture last bare expression (Jupyter-style)."""
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code, None
    if not tree.body:
        return code, None
    last_node = tree.body[-1]
    if not isinstance(last_node, _ast.Expr):
        return code, None
    if len(tree.body) == 1:
        setup_code = ""
    else:
        last_lineno = last_node.lineno
        lines = code.splitlines(True)
        setup_code = "".join(lines[:last_lineno - 1])
    lines = code.splitlines(True)
    last_expr_source = "".join(lines[last_node.lineno - 1:]).strip().rstrip()
    return setup_code, last_expr_source

local_env = {"__builtins__": builtins, "sys": sys, "io": io}

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)

while True:
    try:
        line = sys.stdin.readline()
        if not line:
            break
        task = json.loads(line.strip())
        if task.get('reset'):
            local_env.clear()
            local_env.update({"__builtins__": builtins, "sys": sys, "io": io})
            print(json.dumps({"output": "State reset."}), flush=True)
            continue
        code = task.get('code', '')
        f = io.StringIO()
        result_data = {}
        with contextlib.redirect_stdout(f):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    warnings.filterwarnings('ignore', category=FutureWarning)
                    setup_code, last_expr = _auto_capture_last_expr(code)
                    if last_expr is not None:
                        if setup_code.strip():
                            exec(setup_code, local_env, local_env)
                        expr_result = eval(last_expr, local_env, local_env)
                        if expr_result is not None:
                            local_env['_result'] = expr_result
                    else:
                        exec(code, local_env, local_env)
                result = local_env.get("_result", None)
                printed_output = f.getvalue().strip()
                if printed_output and result is not None:
                    result_data['output'] = f"{printed_output}\n---\n{repr(result)}"
                elif printed_output:
                    result_data['output'] = printed_output
                elif result is not None:
                    result_data['output'] = repr(result)
                else:
                    result_data['output'] = "Code executed successfully with no output."
                local_env.pop('_result', None)
            except SystemExit:
                result_data['output'] = "Python code called exit() or quit(), ignored."
            except KeyboardInterrupt:
                result_data['output'] = "Execution interrupted."
            except Exception as e:
                import traceback
                result_data['error'] = f"Python execution error: {e}\n\n{traceback.format_exc()}"
        print(json.dumps(result_data), flush=True)
    except json.JSONDecodeError as e:
        print(json.dumps({'error': f'Invalid JSON: {e}'}), flush=True)
    except Exception as e:
        print(json.dumps({'error': f'REPL error: {e}'}), flush=True)
'''


class EnrootEnvironment:
    """Persistent Python REPL inside an enroot container. State persists across
    ``execute()`` calls. Communication is JSON lines over the container's
    stdin/stdout."""

    def __init__(self, enroot_image: str | None = None, timeout: int = 60):
        self._image = enroot_image or enroot_image_path()
        self._timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._container_name = f"drbench-py-{os.getpid()}-{id(self)}"

        # Redirect all enroot paths to writable temp dirs (cluster defaults are
        # often read-only), matching jit_gf's approach.
        self._enroot_tmpdir = os.path.join(
            tempfile.gettempdir(), f"enroot_{os.getpid()}_{id(self)}"
        )
        self._enroot_env = os.environ.copy()
        for subdir, env_key in {
            "data": "ENROOT_DATA_PATH", "cache": "ENROOT_CACHE_PATH",
            "tmp": "ENROOT_TEMP_PATH", "run": "ENROOT_RUNTIME_PATH",
        }.items():
            path = os.path.join(self._enroot_tmpdir, subdir)
            os.makedirs(path, exist_ok=True)
            self._enroot_env[env_key] = path

        self._repl_script_path = os.path.join(self._enroot_tmpdir, "enroot_repl.py")
        with open(self._repl_script_path, "w") as f:
            f.write(_ENROOT_REPL_SCRIPT)

        config_dir = os.path.join(self._enroot_tmpdir, "config")
        os.makedirs(config_dir, exist_ok=True)
        with open(os.path.join(config_dir, "enroot.conf"), "w") as cf:
            cf.write(
                f"ENROOT_DATA_PATH {os.path.join(self._enroot_tmpdir, 'data')}\n"
                f"ENROOT_CACHE_PATH {os.path.join(self._enroot_tmpdir, 'cache')}\n"
                f"ENROOT_TEMP_PATH {os.path.join(self._enroot_tmpdir, 'tmp')}\n"
                f"ENROOT_RUNTIME_PATH {os.path.join(self._enroot_tmpdir, 'run')}\n"
                f"ENROOT_MOUNT_HOME n\nENROOT_RESTRICT_DEV y\n"
            )
        self._enroot_env["ENROOT_CONFIG_PATH"] = config_dir
        self._enroot_env["ENROOT_MOUNT_HOME"] = "n"
        self._enroot_env["ENROOT_RESTRICT_DEV"] = "y"

        self._create_container()
        self._start_process()

    def _create_container(self, _max_retries: int = 3) -> None:
        for attempt in range(1, _max_retries + 1):
            result = subprocess.run(
                ["enroot", "create", "--name", self._container_name, self._image],
                capture_output=True, timeout=300, env=self._enroot_env,
            )
            if result.returncode == 0:
                break
            stderr = result.stderr.decode(errors="replace").strip()
            if attempt < _max_retries:
                subprocess.run(["enroot", "remove", "-f", self._container_name],
                               capture_output=True, env=self._enroot_env)
                print(f"[EnrootEnv] create attempt {attempt} failed: {stderr}; retrying...", flush=True)
                time.sleep(attempt)
            else:
                raise subprocess.CalledProcessError(result.returncode, result.args,
                                                     output=result.stdout, stderr=result.stderr)
        print(f"[EnrootEnv] Created container {self._container_name!r} from {self._image}", flush=True)

    def _start_process(self) -> None:
        cmd = [
            "enroot", "start", "--rw",
            "-m", f"{self._repl_script_path}:/work/repl.py:none:bind,ro,x-create=file",
            "-e", "HOME=/tmp", "-e", "PYTHONUNBUFFERED=1",
            self._container_name, "python", "-u", "/work/repl.py",
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=self._enroot_env, bufsize=0,
        )
        print(f"[EnrootEnv] Started REPL process (PID: {self._proc.pid})", flush=True)

    def _restart(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()
        subprocess.run(["enroot", "remove", "-f", self._container_name],
                       capture_output=True, env=self._enroot_env)
        self._create_container()
        self._start_process()

    def execute(self, code: str) -> str:
        """Execute *code* in the sandboxed REPL; return combined output/error text."""
        if self._proc is None or self._proc.poll() is not None:
            self._restart()
        request = json.dumps({"code": code}) + "\n"
        try:
            self._proc.stdin.write(request.encode())
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._restart()
            self._proc.stdin.write(request.encode())
            self._proc.stdin.flush()

        ready, _, _ = select.select([self._proc.stdout], [], [], self._timeout)
        if not ready:
            self._restart()
            return f"Execution timed out after {self._timeout} seconds"
        result_line = self._proc.stdout.readline().decode(errors="replace").strip()
        if not result_line:
            return "Empty response from sandbox"
        try:
            data = json.loads(result_line)
        except json.JSONDecodeError:
            return result_line
        return data.get("error") or data.get("output", "")

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        subprocess.run(["enroot", "remove", "-f", self._container_name],
                       capture_output=True, env=self._enroot_env)
        try:
            if self._repl_script_path and os.path.exists(self._repl_script_path):
                os.remove(self._repl_script_path)
        except OSError:
            pass
        print(f"[EnrootEnv] Closed container {self._container_name!r}", flush=True)
