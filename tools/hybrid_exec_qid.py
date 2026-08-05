# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

import builtins
import ast
import traceback
import asyncio
import subprocess
import os
import tempfile
import venv
from gpt_oss.tools.python_docker.docker_tool import PythonTool as BasePythonTool
from openai_harmony import Message, Role, Author, TextContent
import sys
import contextlib
import io
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import pickle
import signal
import queue as queue_module

def _auto_capture_last_expr(code):
    """
    Rewrite code so that a trailing bare expression is captured into _result.

    Mimics Jupyter/IPython behaviour: if the last statement in the code block
    is an expression (not an assignment, import, etc.), we compile everything
    before it normally and then eval() the last expression so its value is
    returned.

    Returns (setup_code, last_expr) where *last_expr* is None when the last
    statement is not a bare expression.
    """
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code, None

    if not tree.body:
        return code, None

    last_node = tree.body[-1]

    # Only rewrite if the last statement is a bare Expr node
    if not isinstance(last_node, _ast.Expr):
        return code, None

    # Split: everything before the last expression runs via exec,
    # the last expression is eval'd separately.
    if len(tree.body) == 1:
        setup_code = ""
    else:
        # Reconstruct source for all statements except the last.
        # Use line numbers to slice the original source so we preserve
        # formatting, comments, etc.
        last_lineno = last_node.lineno  # 1-indexed
        lines = code.splitlines(True)
        setup_code = "".join(lines[:last_lineno - 1])

    # The expression source: from last_node start to end of code
    lines = code.splitlines(True)
    last_expr_source = "".join(lines[last_node.lineno - 1:]).strip()
    # Remove any trailing newline/whitespace
    last_expr_source = last_expr_source.rstrip()

    return setup_code, last_expr_source


def _persistent_worker(conn):
    """
    Persistent worker process that maintains state between executions.
    Receives code to execute via pipe connection, sends results back.

    Behaves like a Jupyter cell: if the last statement is a bare expression
    its value is automatically captured and returned.
    """
    import builtins
    import sys
    import io
    import contextlib
    import traceback
    import warnings
    import json

    # Initialize environment for this worker
    local_env = {
        "__builtins__": builtins,
        "sys": sys,
        "io": io,
    }

    while True:
        try:
            # Wait for code to execute
            if not conn.poll(timeout=3600):  # 1 hour timeout for idle workers
                break

            task_str = conn.recv()

            if task_str is None:  # Shutdown signal
                break

            task = json.loads(task_str)
            code = task['code']

            # Execute code
            f = io.StringIO()
            result_data = {}

            with contextlib.redirect_stdout(f):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning)
                        warnings.filterwarnings('ignore', category=FutureWarning)

                        # Auto-capture last expression (Jupyter-style)
                        setup_code, last_expr = _auto_capture_last_expr(code)

                        if last_expr is not None:
                            # Run setup statements
                            if setup_code.strip():
                                exec(setup_code, local_env, local_env)
                            # Eval the last expression
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

                    # Clear _result for next execution
                    local_env.pop('_result', None)

                except SystemExit:
                    result_data['output'] = "Python code called exit() or quit(), ignored."
                except KeyboardInterrupt:
                    result_data['output'] = "Execution interrupted."
                except BaseException as e:
                    tb = traceback.format_exc()
                    result_data['error'] = f"Python execution error: {e}\n\n{tb}"

            # Send result back as JSON string
            conn.send(json.dumps(result_data))

        except Exception as e:
            # Unexpected error in worker
            try:
                conn.send(json.dumps({'error': f"Worker error: {e}"}))
            except:
                pass
            break

class HybridPythonTool(BasePythonTool):
    """
    Hybrid Python execution:
    - Safe code runs in current environment (fast)
    - Dangerous code runs in isolated venv (safe)
    """
    def __init__(self, timeout=60, venv_path=None):  # REDUCED from 120s to 60s for faster hang detection
        super().__init__()
        self.timeout = timeout
        self._current_qid = None  # Track current question for natural cleanup boundaries
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._qid_workers = {}    # qid -> (process, conn)  [venv-worker fallback]
        self._qid_enroot = {}     # qid -> EnrootEnvironment [enroot mode]

        # Prefer an enroot sandbox (a .sqsh image loaded from the repo) when
        # available: it avoids the fragile pip-built venv, which needs outbound
        # pip access that cluster nodes block. Fall back to the venv worker.
        from tools.enroot_env import enroot_available, enroot_image_path
        self._use_enroot = enroot_available()

        if self._use_enroot:
            self._enroot_image = enroot_image_path()
            # venv paths unused in enroot mode, but keep attrs defined.
            self.venv_path = venv_path or os.path.join(os.getcwd(), ".python_tool_env")
            self.python_bin = os.path.join(self.venv_path, "bin", "python")
            print(f"[HybridPythonTool] Initialized in ENROOT mode "
                  f"(image={self._enroot_image}, timeout={self.timeout}s)", flush=True)
            return

        # ---- fallback: venv worker (original behavior) ----
        if venv_path is None:
            venv_path = os.path.join(os.getcwd(), ".python_tool_env")
        self.venv_path = venv_path
        self.python_bin = os.path.join(venv_path, "bin", "python")
        self._ensure_tool_environment()
        self._setup_tool_environment_path()
        print(f"[HybridPythonTool] Initialized with tool env at {self.venv_path}, timeout={self.timeout}s", flush=True)
        print(f"[HybridPythonTool] ⚠️ RESTRICTED MODE: Only pre-installed packages available, dynamic installation blocked", flush=True)
    
    def _ensure_tool_environment(self):
        """
        Create special tool environment with pre-installed packages.

        This environment provides additional packages beyond the main environment.
        All packages are installed UPFRONT - no dynamic installation allowed.
        """
        if os.path.exists(self.python_bin):
            print(f"[HybridPythonTool] Using existing tool env at {self.venv_path}", flush=True)
            self._list_available_packages()
            return

        print(f"[HybridPythonTool] Creating tool environment at {self.venv_path}...", flush=True)

        try:
            # Create venv with access to main environment site-packages
            venv.create(self.venv_path, with_pip=True, clear=True, system_site_packages=True)

            # Install additional safe packages not in main environment
            pip_bin = os.path.join(self.venv_path, "bin", "pip")
            packages = [
                # Core scientific computing (numpy/scipy/pandas from main env via system_site_packages)
                "matplotlib",
                "scikit-learn",

                # Symbolic math & arbitrary precision
                "sympy",
                "mpmath",

                # Graph algorithms
                "networkx",

                # Statistical modeling
                "statsmodels",

                # HTTP/Web
                "requests",
                "beautifulsoup4",
                "lxml",

                # Data formats & databases
                "pyyaml",
                "pillow",
                "duckdb",

                # Utilities
                "python-dateutil",
                "pytz",
                "tqdm",
                "sortedcontainers",
            ]

            subprocess.run(
                [pip_bin, "install", "--no-cache-dir"] + packages,
                check=True,
                capture_output=True,
                timeout=300  # 5 minute timeout for installation
            )

            print(f"[HybridPythonTool] ✅ Tool environment created successfully", flush=True)
            self._list_available_packages()

        except Exception as e:
            print(f"[HybridPythonTool] ⚠️ Warning: Could not create tool environment: {e}", flush=True)
            print(f"[HybridPythonTool] Will use main environment only with blocked imports", flush=True)

    def _list_available_packages(self):
        """List available packages in tool environment"""
        if not os.path.exists(self.python_bin):
            print(f"[HybridPythonTool] Available packages: main environment only", flush=True)
            return

        try:
            result = subprocess.run(
                [self.python_bin, "-m", "pip", "list", "--format=freeze"],
                capture_output=True,
                text=True,
                timeout=10
            )
            packages = [line.split('==')[0] for line in result.stdout.strip().split('\n') if '==' in line]
            print(f"[HybridPythonTool] Pre-installed packages ({len(packages)} total): {', '.join(sorted(packages)[:10])}...", flush=True)
        except Exception as e:
            print(f"[HybridPythonTool] Could not list packages: {e}", flush=True)

    def _setup_tool_environment_path(self):
        """
        Add tool environment site-packages to sys.path.

        This allows worker processes (spawned via fork) to import packages
        from both the main environment and the tool environment.
        """
        if not os.path.exists(self.venv_path):
            print(f"[HybridPythonTool] Tool environment not available, using main environment only", flush=True)
            return

        # Find site-packages directory in tool environment
        import site
        import glob

        # Typical paths: .python_tool_env/lib/python3.X/site-packages
        site_packages_pattern = os.path.join(self.venv_path, "lib", "python*", "site-packages")
        site_packages_dirs = glob.glob(site_packages_pattern)

        if site_packages_dirs:
            for site_packages_dir in site_packages_dirs:
                if site_packages_dir not in sys.path:
                    sys.path.insert(0, site_packages_dir)
                    print(f"[HybridPythonTool] Added to sys.path: {site_packages_dir}", flush=True)
        else:
            print(f"[HybridPythonTool] ⚠️ Could not find site-packages in tool environment", flush=True)

        # Also ensure main site-packages are accessible (should already be there)
        main_site_packages = site.getsitepackages()
        for sp_dir in main_site_packages:
            if sp_dir not in sys.path:
                sys.path.append(sp_dir)

    def set_qid(self, qid):
        """
        Set the current question ID for natural cleanup boundaries.

        When qid changes, worker is terminated to prevent state leakage
        between questions while preserving multi-step reasoning within a question.

        Args:
            qid: Question identifier (typically from dataset)
        """
        if qid is not None and qid != self._current_qid:
            if self._current_qid is not None:
                # New question detected - terminate old worker
                print(f"[HybridPython] New question (qid {self._current_qid} → {qid}), terminating worker", flush=True)
                self._terminate_worker(self._current_qid)
            self._current_qid = qid

    def _terminate_worker(self, qid):
        """Terminate the per-QID execution context (enroot container or venv worker)."""
        # Enroot mode: close the container for this question.
        if qid in self._qid_enroot:
            try:
                self._qid_enroot[qid].close()
            except Exception:
                pass
            del self._qid_enroot[qid]
        if qid in self._qid_workers:
            proc, conn = self._qid_workers[qid]
            try:
                conn.close()
                proc.terminate()
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                print(f"[HybridPython] Terminated worker for qid={qid}", flush=True)
            except:
                pass
            del self._qid_workers[qid]

    def _get_or_create_worker(self, qid):
        """Get or create persistent worker process for given QID"""
        if qid not in self._qid_workers:
            # Set start method to 'fork' to avoid pickle issues
            try:
                multiprocessing.set_start_method('fork', force=True)
            except RuntimeError:
                pass  # Already set

            # Create new worker with pipe communication
            parent_conn, child_conn = multiprocessing.Pipe()
            proc = multiprocessing.Process(
                target=_persistent_worker,
                args=(child_conn,),
                daemon=True
            )
            proc.start()
            child_conn.close()  # Close child end in parent process
            self._qid_workers[qid] = (proc, parent_conn)
            print(f"[HybridPython] Created persistent worker for qid={qid}", flush=True)

        return self._qid_workers[qid]

    def _get_or_create_enroot(self, qid):
        """Get or create the per-QID enroot Python sandbox (stateful REPL)."""
        env = self._qid_enroot.get(qid)
        if env is None:
            from tools.enroot_env import EnrootEnvironment
            env = EnrootEnvironment(self._enroot_image, timeout=self.timeout)
            self._qid_enroot[qid] = env
            print(f"[HybridPython] Created enroot sandbox for qid={qid}", flush=True)
        return env

    async def _run_in_enroot(self, qid, code_str):
        """Run code in the per-QID enroot container (blocking execute off-loop)."""
        loop = asyncio.get_event_loop()
        try:
            env = self._get_or_create_enroot(qid)
            return await loop.run_in_executor(self._executor, env.execute, code_str)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Enroot execution error: {e}"

    async def _run_in_current_env(self, code_input):
        """Execute code in the per-QID context (enroot container, or venv worker)."""
        import json

        print(f"[HybridPython] _run_in_current_env called", flush=True)

        qid = self._current_qid or "default"

        # Enroot mode: route to the container REPL and return.
        if getattr(self, "_use_enroot", False):
            code_str = code_input if isinstance(code_input, str) else str(code_input)
            return await self._run_in_enroot(qid, code_str)

        # Get or create worker for this QID
        print(f"[HybridPython] Getting worker for qid={qid}", flush=True)
        try:
            proc, conn = self._get_or_create_worker(qid)
            print(f"[HybridPython] Got worker, proc.is_alive()={proc.is_alive()}", flush=True)
        except Exception as e:
            print(f"[HybridPython] ❌ Failed to get worker: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return f"Failed to create worker process: {e}"

        # Check if worker is still alive
        if not proc.is_alive():
            print(f"[HybridPython] Worker died, restarting for qid={qid}", flush=True)
            self._terminate_worker(qid)
            proc, conn = self._get_or_create_worker(qid)

        # Convert to string if it's compiled code, otherwise use as-is
        if isinstance(code_input, str):
            code_str = code_input
        else:
            # It's compiled code - decompile it
            import dis
            import types
            # We can't easily decompile, so this is an error
            return "Error: Cannot execute compiled code in worker process. This is a bug."

        # Send code to worker as JSON
        print(f"[HybridPython] Sending code to worker ({len(code_str)} chars):\n{code_str[:1000]}", flush=True)
        conn.send(json.dumps({'code': code_str}))

        # Wait for result with timeout
        def get_result():
            try:
                if conn.poll(timeout=self.timeout):
                    result_str = conn.recv()
                    return json.loads(result_str)
                else:
                    # Timeout - kill worker
                    print(f"[HybridPython] Worker timeout, terminating for qid={qid}", flush=True)
                    self._terminate_worker(qid)
                    return {'error': f"Execution timed out after {self.timeout} seconds"}
            except Exception as e:
                return {'error': f"Error getting result: {e}"}

        try:
            result_data = await asyncio.get_event_loop().run_in_executor(None, get_result)
            print(f"[HybridPython] Worker result_data keys: {list(result_data.keys())}", flush=True)

            if 'error' in result_data:
                output = result_data['error']
                print(f"[HybridPython] Worker ERROR: {output[:500]}", flush=True)
            else:
                output = result_data.get('output', '')
                print(f"[HybridPython] Worker OUTPUT ({len(output)} chars): {output[:500]}", flush=True)

        except Exception as e:
            output = f"Unexpected error during execution: {e}"
            print(f"[HybridPython] Unexpected error: {output}", flush=True)

        return output

    async def process(self, message: Message):
        """Async interface for GPT-OSS to interact with this tool"""
        async for result in self._process(message):
            yield result
    
    async def _process(self, message: Message):
        """
        Main processing logic - all code runs in worker process with restricted environment.

        The worker process has access to:
        - Main environment packages (via system_site_packages=True)
        - Pre-installed packages in .python_tool_env
        - Runtime protections block dangerous imports and subprocess calls
        """
        code_text = message.content[0].text if message.content else ""
        print(f"[HybridPythonTool] _process called, message recipient={getattr(message, 'recipient', None)}", flush=True)
        print(f"[HybridPythonTool] Code to execute ({len(code_text)} chars):", flush=True)
        print(f"--- CODE START ---\n{code_text}\n--- CODE END ---", flush=True)
        if not code_text.strip():
            print(f"[HybridPythonTool] Empty code, returning error", flush=True)
            yield Message(
                author=Author.new(Role.TOOL, "python"),
                content=[TextContent(text="No Python code provided.")],
            ).with_recipient("assistant")
            return

        # All code runs in worker process with pre-installed packages
        # Runtime protections (import/subprocess) handle dangerous operations
        print(f"[HybridPythonTool] Executing in worker process with restricted environment", flush=True)
        output = await self._run_in_current_env(code_text)

        print(f"[HybridPythonTool] Execution output ({len(output)} chars):", flush=True)
        print(f"--- OUTPUT START ---\n{output[:2000]}\n--- OUTPUT END ---", flush=True)

        yield Message(
            author=Author.new(Role.TOOL, "python"),
            content=[TextContent(text=output)],
        ).with_recipient("assistant")
        
