import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from constants import CONTAINER_NAME, DEFAULT_NIM_MODEL, DEFAULT_NIM_PORT

try:
    from dotenv import load_dotenv
    from dotenv import dotenv_values
except ImportError:
    load_dotenv = None
    dotenv_values = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _error_and_exit(message: str, code: int = 1) -> None:
    logging.error("ERROR: This is not supported/compatible - %s", message)
    sys.exit(code)


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        _error_and_exit(f"missing required environment variable `{name}`")
    if name == "PRODCONFIG_NIM__API_KEY" and ("nvapi-your" in value or "your-key" in value):
        _error_and_exit("PRODCONFIG_NIM__API_KEY is a placeholder; set a real NVIDIA key")
    return value


def _resolve_shell_command(script_path: Path, env_exports: str) -> list[str]:
    system = platform.system().lower()
    script_posix = script_path.as_posix()

    if system == "windows":
        bash_path = shutil.which("bash")
        if bash_path:
            return [bash_path, "-lc", f"{env_exports} '{script_posix}'"]

        if shutil.which("wsl") is None:
            _error_and_exit("Windows runtime requires bash (Git Bash/WSL) for nim_llm/start_llama_container.sh")

        try:
            wsl_script = subprocess.check_output(
                ["wsl", "wslpath", "-a", str(script_path)],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except subprocess.CalledProcessError as exc:
            _error_and_exit(f"failed to resolve WSL path for launcher script: {exc.output.strip()}")

        return ["wsl", "bash", "-lc", f"{env_exports} '{wsl_script}'"]

    bash_path = shutil.which("bash") or "/bin/bash"
    if not Path(bash_path).exists() and shutil.which("bash") is None:
        _error_and_exit("bash is required to run nim_llm/start_llama_container.sh")

    return [bash_path, "-lc", f"{env_exports} '{script_posix}'"]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    env_file = script_dir.parent / ".env"
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_file, override=False)

    if dotenv_values is not None:
        env_map = dotenv_values(env_file)
        file_key = (env_map.get("PRODCONFIG_NIM__API_KEY") or "").strip()
        proc_key = os.getenv("PRODCONFIG_NIM__API_KEY", "").strip()
        proc_placeholder = "nvapi-your" in proc_key or "your-key" in proc_key
        if file_key and (not proc_key or proc_placeholder):
            os.environ["PRODCONFIG_NIM__API_KEY"] = file_key

    script_to_run = script_dir / "start_llama_container.sh"

    if not script_to_run.exists():
        _error_and_exit(f"launcher script not found: {script_to_run}")

    try:
        script_to_run.chmod(0o755)
    except Exception:
        # Non-fatal on Windows filesystems.
        pass

    api_key = _require_env("PRODCONFIG_NIM__API_KEY")
    nim_port = os.getenv("PRODCONFIG_NIM__PORT", DEFAULT_NIM_PORT).strip() or DEFAULT_NIM_PORT
    nim_model = os.getenv("PRODCONFIG_NIM__MODEL", DEFAULT_NIM_MODEL).strip() or DEFAULT_NIM_MODEL

    env_exports = (
        f"export PRODCONFIG_NIM__API_KEY='{api_key}'; "
        f"export PRODCONFIG_NIM__PORT='{nim_port}'; "
        f"export PRODCONFIG_NIM__MODEL='{nim_model}'; "
        f"export CONTAINER_NAME='{CONTAINER_NAME}';"
    )

    cmd = _resolve_shell_command(script_to_run, env_exports)
    log_file_path = script_dir / "llama_container.log"

    logging.info("Launching local NIM container using port %s and model %s", nim_port, nim_model)
    logging.info("Logging to %s", log_file_path)

    with open(log_file_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        process.wait()
        if process.returncode != 0:
            _error_and_exit(f"container launcher exited with code {process.returncode}", process.returncode)

    logging.info("NIM launcher exited successfully")


if __name__ == "__main__":
    main()
