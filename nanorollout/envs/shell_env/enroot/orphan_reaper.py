"""
Detached helper: when the parent worker exits, tear down enroot start/exec PIDs and remove the container.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

# Same marker written into commands by EnrootEnvironment.execute (cwd extraction).
EXEC_LINE_MARKER = "__NANOROLLOUT_PWD__"

_REAPER_SCAN_INTERVAL_S = 2.0
_SIGTERM_GRACE_S = 1.0


def _parse_args(argv: list[str]) -> tuple[int, int, int, str, str]:
    if len(argv) != 6:
        print(
            "usage: ... parent_pid start_pid container_pid container_name enroot_exe",
            file=sys.stderr,
        )
        sys.exit(2)
    return (
        int(argv[1]),
        int(argv[2]),
        int(argv[3]),
        argv[4],
        argv[5],
    )


def _parent_alive(parent_pid: int) -> bool:
    try:
        os.kill(parent_pid, 0)
    except OSError:
        return False
    return True


def _read_cmdline(pid: int) -> str:
    path = os.path.join("/proc", str(pid), "cmdline")
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError:
        return ""
    if not raw:
        return ""
    return raw.decode("utf-8", "ignore").replace("\x00", " ")


def _looks_like_our_start_process(start_process_pid: int, container_name: str) -> bool:
    cmdline = _read_cmdline(start_process_pid)
    if not cmdline:
        return False
    return (
        "enroot" in cmdline
        and " start " in f" {cmdline} "
        and container_name in cmdline
    )


def _kill_start_process_group(
    start_process_pid: int, container_name: str, sig: int
) -> None:
    if not _looks_like_our_start_process(start_process_pid, container_name):
        return
    try:
        os.killpg(start_process_pid, sig)
    except OSError:
        pass


def _kill_matching_exec(
    container_pid: int, container_name: str, sig: int
) -> None:
    needle_pid = f"enroot exec {container_pid} "
    needle_name = f"enroot exec {container_name} "
    proc_root = "/proc"
    if not os.path.isdir(proc_root):
        return
    self_pid = os.getpid()
    for name in os.listdir(proc_root):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == self_pid:
            continue
        cmdline_path = os.path.join(proc_root, name, "cmdline")
        try:
            with open(cmdline_path, "rb") as handle:
                raw = handle.read()
        except OSError:
            continue
        if not raw:
            continue
        cmdline = raw.decode("utf-8", "ignore").replace("\x00", " ")
        if needle_pid not in cmdline and needle_name not in cmdline:
            continue
        if EXEC_LINE_MARKER not in cmdline:
            continue
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def run(
    parent_pid: int,
    start_process_pid: int,
    container_pid: int,
    container_name: str,
    executable: str,
) -> None:
    while _parent_alive(parent_pid):
        time.sleep(_REAPER_SCAN_INTERVAL_S)

    _kill_start_process_group(start_process_pid, container_name, signal.SIGTERM)
    _kill_matching_exec(container_pid, container_name, signal.SIGTERM)
    time.sleep(_SIGTERM_GRACE_S)
    _kill_start_process_group(start_process_pid, container_name, signal.SIGKILL)
    _kill_matching_exec(container_pid, container_name, signal.SIGKILL)

    try:
        subprocess.run(
            [executable, "remove", "-f", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except Exception:
        pass


def main() -> None:
    run(*_parse_args(sys.argv))


if __name__ == "__main__":
    main()
