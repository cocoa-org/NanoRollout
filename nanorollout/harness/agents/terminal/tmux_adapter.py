"""
Adapted from Harbor's TmuxSession (terminus_2/tmux_session.py):
"""

import logging
import re
import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nanorollout.envs.shell_env.base import ShellEnvironment as BaseEnvironment

logger = logging.getLogger(__name__)

_SESSION_NAME = "main"
_ENTER_KEYS = {"Enter", "C-m", "KPEnter", "C-j", "^M", "^J"}
_ENDS_WITH_NEWLINE_PATTERN = r"[\r\n]$"
_NEWLINE_CHARS = "\r\n"
_TMUX_COMPLETION_COMMAND = "; tmux wait -S done"
_REMOTE_PANE_LOG = "/tmp/terminus2.pane"
_REMOTE_CAST_PATH = "/tmp/recording.cast"
_INSTALL_TIMEOUT_SEC = 900
_BUILD_TIMEOUT_SEC = 1800
_VERIFY_TIMEOUT_SEC = 30


class TmuxAdapter:
    """Drives a tmux session inside a container via BaseEnvironment.execute()."""

    def __init__(
        self,
        environment: "BaseEnvironment",
        session_name: str = _SESSION_NAME,
        enable_asciinema: bool = False,
        local_cast_path: Optional[Path] = None,
        pane_width: int = 160,
        pane_height: int = 40,
    ) -> None:
        self._env = environment
        self._session_name = session_name
        self._enable_asciinema = enable_asciinema
        self._local_cast_path = local_cast_path
        self._pane_width = pane_width
        self._pane_height = pane_height
        self._previous_buffer: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Create and start a tmux session inside the container."""
        self._attempt_tmux_installation()

        cmd = (
            f"export TERM=xterm-256color && "
            f"export SHELL=/bin/bash && "
            f'script -qc "'
            f"tmux new-session -x {self._pane_width} -y {self._pane_height} -d -s {self._session_name} 'bash --login' \\; "
            f"pipe-pane -t {self._session_name} "
            f"'cat > {_REMOTE_PANE_LOG}'"
            f'" /dev/null'
        )
        result = self._env.execute(cmd)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to start tmux session: {result.output}"
            )

        self._env.execute("tmux set-option -g history-limit 10000000")

        if self._enable_asciinema:
            self.send_keys(
                keys=[f"asciinema rec --stdin {_REMOTE_CAST_PATH}", "Enter"],
                min_timeout_sec=1.0,
            )
            self.send_keys(keys=["clear", "Enter"])

    def stop(self) -> None:
        """Kill the tmux session, optionally downloading asciinema recording."""
        if self._enable_asciinema:
            logger.debug("Stopping asciinema recording.")
            self.send_keys(keys=["C-d"], min_timeout_sec=0.1)
            time.sleep(0.5)

            if self._local_cast_path is not None:
                self._local_cast_path.parent.mkdir(parents=True, exist_ok=True)
                result = self._env.read_file(str(_REMOTE_CAST_PATH))
                if result.exit_code == 0 and result.output:
                    self._local_cast_path.write_text(result.output, encoding="utf-8")
                else:
                    logger.warning(
                        "Failed to download asciinema recording: %s", result.output
                    )

        self._env.execute(f"tmux kill-session -t {self._session_name}")

    def is_session_alive(self) -> bool:
        result = self._env.execute(f"tmux has-session -t {self._session_name}")
        return result.exit_code == 0

    # ------------------------------------------------------------------
    # Key preparation helpers (ported from Harbor's TmuxSession)
    # ------------------------------------------------------------------

    def _is_enter_key(self, key: str) -> bool:
        return key in _ENTER_KEYS

    def _ends_with_newline(self, key: str) -> bool:
        return re.search(_ENDS_WITH_NEWLINE_PATTERN, key) is not None

    def _is_executing_command(self, key: str) -> bool:
        return self._is_enter_key(key) or self._ends_with_newline(key)

    def _prevent_execution(self, keys: list[str]) -> list[str]:
        keys = keys.copy()
        while keys and self._is_executing_command(keys[-1]):
            if self._is_enter_key(keys[-1]):
                keys.pop()
            else:
                stripped = keys[-1].rstrip(_NEWLINE_CHARS)
                if stripped:
                    keys[-1] = stripped
                else:
                    keys.pop()
        return keys

    def _prepare_keys(
        self,
        keys: "str | list[str]",
        block: bool,
    ) -> "tuple[list[str], bool]":
        if isinstance(keys, str):
            keys = [keys]

        if not block or not keys or not self._is_executing_command(keys[-1]):
            return keys, False

        keys = self._prevent_execution(keys)
        keys.extend([_TMUX_COMPLETION_COMMAND, "Enter"])
        return keys, True

    def _tmux_send_keys(self, keys: list[str]) -> str:
        escaped = [shlex.quote(key) for key in keys]
        return " ".join(
            ["tmux", "send-keys", "-t", shlex.quote(self._session_name), *escaped]
        )

    # ------------------------------------------------------------------
    # Sending keystrokes
    # ------------------------------------------------------------------

    def send_keys(
        self,
        keys: "str | list[str]",
        *,
        block: bool = False,
        min_timeout_sec: float = 0.0,
        max_timeout_sec: float = 180.0,
    ) -> None:
        """Send keystrokes to the tmux session.

        Args:
            keys: String or list of key names / text to send.
            block: Wait for command to complete using tmux wait mechanism.
            min_timeout_sec: Minimum seconds to wait after non-blocking send.
            max_timeout_sec: Maximum seconds to wait for blocking commands.
        """
        prepared, is_blocking = self._prepare_keys(keys=keys, block=block)
        if is_blocking:
            self._send_blocking_keys(prepared, timeout=max_timeout_sec)
        else:
            self._send_non_blocking_keys(prepared, min_timeout_sec=min_timeout_sec)

    def _send_blocking_keys(self, keys: list[str], timeout: float) -> None:
        self._env.execute(self._tmux_send_keys(keys))
        result = self._env.execute(
            f"timeout {timeout}s tmux wait done",
            timeout=int(timeout) + 10,
        )
        if result.exit_code != 0:
            raise TimeoutError(f"Command timed out after {timeout} seconds")

    def _send_non_blocking_keys(self, keys: list[str], min_timeout_sec: float) -> None:
        started = time.time()
        self._env.execute(self._tmux_send_keys(keys))
        elapsed = time.time() - started
        remaining = min_timeout_sec - elapsed
        if remaining > 0:
            time.sleep(remaining)

    # ------------------------------------------------------------------
    # Capturing output
    # ------------------------------------------------------------------

    def capture_pane(self, capture_entire: bool = False) -> str:
        if capture_entire:
            result = self._env.execute(
                f"tmux capture-pane -p -S - -t {self._session_name}"
            )
        else:
            result = self._env.execute(
                f"tmux capture-pane -p -t {self._session_name}"
            )
        return result.output

    def get_incremental_output(self) -> str:
        """Get new terminal output since last call, or current screen."""
        current_buffer = self.capture_pane(capture_entire=True)

        if self._previous_buffer is None:
            self._previous_buffer = current_buffer
            return f"Current Terminal Screen:\n{self.capture_pane(capture_entire=False)}"

        new_content = self._find_new_content(current_buffer)
        self._previous_buffer = current_buffer

        if new_content is not None and new_content.strip():
            return f"New Terminal Output:\n{new_content}"

        return f"Current Terminal Screen:\n{self.capture_pane(capture_entire=False)}"

    def _find_new_content(self, current_buffer: str) -> Optional[str]:
        if self._previous_buffer is None:
            return None
        pb = self._previous_buffer.strip()
        if pb in current_buffer:
            idx = current_buffer.index(pb)
            if "\n" in pb:
                idx = pb.rfind("\n")
            return current_buffer[idx:]
        return None

    # ------------------------------------------------------------------
    # Installation helpers (ported from Harbor's TmuxSession)
    # ------------------------------------------------------------------

    def _attempt_tmux_installation(self) -> None:
        """Install tmux and (optionally) asciinema if not present."""
        tmux_installed = self._env.execute("tmux -V").exit_code == 0

        if self._enable_asciinema:
            asciinema_installed = self._env.execute("asciinema --version").exit_code == 0
        else:
            asciinema_installed = True

        if tmux_installed and asciinema_installed:
            logger.debug("tmux and asciinema are already installed")
            return

        tools_needed = []
        if not tmux_installed:
            tools_needed.append("tmux")
        if self._enable_asciinema and not asciinema_installed:
            tools_needed.append("asciinema")

        logger.debug("Installing: %s", ", ".join(tools_needed))
        pkg_manager = self._detect_package_manager()

        if pkg_manager:
            install_cmd = self._get_install_command(pkg_manager, tools_needed)
            if install_cmd:
                result = self._env.execute(install_cmd, timeout=_INSTALL_TIMEOUT_SEC)
                if result.exit_code == 0:
                    if not tmux_installed:
                        if self._env.execute("tmux -V", timeout=_VERIFY_TIMEOUT_SEC).exit_code != 0:
                            logger.warning("tmux verification failed, building from source")
                            self._build_tmux_from_source()
                    if self._enable_asciinema and not asciinema_installed:
                        if self._env.execute(
                            "asciinema --version", timeout=_VERIFY_TIMEOUT_SEC
                        ).exit_code != 0:
                            logger.warning("asciinema verification failed, trying pip")
                            self._install_asciinema_with_pip()
                    return
                logger.warning(
                    "Package manager install for %s failed with exit code %s: %s",
                    ", ".join(tools_needed),
                    result.exit_code,
                    result.output,
                )

        # Fallback
        if not tmux_installed:
            self._build_tmux_from_source()
        if self._enable_asciinema and not asciinema_installed:
            self._install_asciinema_with_pip()

    def _detect_package_manager(self) -> Optional[str]:
        for pm in ("apt-get", "dnf", "yum", "apk", "pacman", "brew", "pkg", "zypper"):
            if self._env.execute(f"which {pm} >/dev/null 2>&1").exit_code == 0:
                return pm
        return None

    def _get_install_command(self, pkg_manager: str, tools: list[str]) -> str:
        packages = " ".join(tools)
        commands = {
            "apt-get": f"DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y {packages}",
            "dnf": f"dnf install -y {packages}",
            "yum": f"yum install -y {packages}",
            "apk": f"apk add --no-cache {packages}",
            "pacman": f"pacman -S --noconfirm {packages}",
            "brew": f"brew install {packages}",
            "pkg": f"ASSUME_ALWAYS_YES=yes pkg install -y {packages}",
            "zypper": f"zypper install -y -n {packages}",
        }
        return commands.get(pkg_manager, "")

    def _build_tmux_from_source(self) -> None:
        dep_commands = [
            "DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential libevent-dev libncurses5-dev curl",
            "yum groupinstall -y 'Development Tools' && yum install -y libevent-devel ncurses-devel curl",
            "dnf groupinstall -y 'Development Tools' && dnf install -y libevent-devel ncurses-devel curl",
            "apk add --no-cache build-base libevent-dev ncurses-dev curl",
        ]
        for cmd in dep_commands:
            if self._env.execute(cmd, timeout=_INSTALL_TIMEOUT_SEC).exit_code == 0:
                break
        build_cmd = (
            "cd /tmp && "
            "curl -L https://github.com/tmux/tmux/releases/download/3.4/tmux-3.4.tar.gz -o tmux.tar.gz && "
            "tar -xzf tmux.tar.gz && "
            "cd tmux-3.4 && "
            "./configure --prefix=/usr/local && "
            "make && make install"
        )
        result = self._env.execute(build_cmd, timeout=_BUILD_TIMEOUT_SEC)
        if self._env.execute(
            "tmux -V || /usr/local/bin/tmux -V", timeout=_VERIFY_TIMEOUT_SEC
        ).exit_code == 0:
            logger.debug("tmux built from source successfully")
        else:
            logger.error("Failed to build tmux from source: %s", result.output)

    def _install_asciinema_with_pip(self) -> None:
        pip_dep_commands = [
            "DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip",
            "yum install -y python3-pip",
            "dnf install -y python3-pip",
            "apk add --no-cache python3 py3-pip",
        ]
        for cmd in pip_dep_commands:
            if self._env.execute(cmd, timeout=_INSTALL_TIMEOUT_SEC).exit_code == 0:
                break
        for pip_cmd in ("pip3 install asciinema", "pip install asciinema"):
            if self._env.execute(pip_cmd, timeout=_INSTALL_TIMEOUT_SEC).exit_code == 0:
                if self._env.execute(
                    "asciinema --version", timeout=_VERIFY_TIMEOUT_SEC
                ).exit_code == 0:
                    logger.debug("asciinema installed via pip")
                    return
        logger.error("Failed to install asciinema via pip")
