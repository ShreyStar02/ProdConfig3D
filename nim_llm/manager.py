import subprocess
import time

from constants import CONTAINER_NAME


def _run_docker(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def is_container_running() -> bool:
    """Check if the configured NIM container is currently running."""
    try:
        result = _run_docker(["ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"], timeout=10)
        if result.returncode != 0:
            return False
        return any(line.strip() == CONTAINER_NAME for line in result.stdout.splitlines())
    except Exception:
        return False


def stop_container() -> bool:
    """Stop and remove the running NIM container if present."""
    try:
        if not is_container_running():
            print(f"Container {CONTAINER_NAME} is not running")
            return True

        print(f"Stopping container {CONTAINER_NAME}...")
        result = _run_docker(["stop", CONTAINER_NAME], timeout=30)
        if result.stdout.strip():
            print(result.stdout.strip())

        if result.returncode != 0:
            print(
                f"ERROR: This is not supported/compatible - failed to stop container {CONTAINER_NAME}. "
                f"docker output: {result.stdout.strip()}"
            )
            return False

        start_time = time.time()
        while time.time() - start_time < 10:
            if not is_container_running():
                print(f"Container {CONTAINER_NAME} stopped successfully")
                _run_docker(["rm", "-f", CONTAINER_NAME], timeout=30)
                return True
            time.sleep(1)

        print(f"ERROR: This is not supported/compatible - timeout while stopping container {CONTAINER_NAME}")
        return False
    except Exception as exc:
        print(f"ERROR: This is not supported/compatible - failed to manage container lifecycle: {exc}")
        return False
