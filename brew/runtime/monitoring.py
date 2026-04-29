import concurrent.futures
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psutil
import ray

logger = logging.getLogger(__name__)


@ray.remote(max_restarts=1, num_cpus=0)
class NodeResourceMonitorActor:
    def __init__(self, node_id: str, node_ip: str) -> None:
        self.node_id = node_id
        self.node_ip = node_ip
        self.hostname = os.uname().nodename
        self._last_cpu_at = time.time()
        self._last_disk = psutil.disk_io_counters()
        self._last_net = psutil.net_io_counters()

    def sample(self) -> Dict[str, Any]:
        now = time.time()
        elapsed = max(1e-6, now - self._last_cpu_at)
        self._last_cpu_at = now

        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk_usage = psutil.disk_usage("/")
        disk_io = psutil.disk_io_counters()
        net_io = psutil.net_io_counters()
        load_avg = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)

        io_read_bps = 0.0
        io_write_bps = 0.0
        if self._last_disk and disk_io:
            io_read_bps = float(disk_io.read_bytes - self._last_disk.read_bytes) / elapsed
            io_write_bps = float(disk_io.write_bytes - self._last_disk.write_bytes) / elapsed

        net_recv_bps = 0.0
        net_sent_bps = 0.0
        if self._last_net and net_io:
            net_recv_bps = float(net_io.bytes_recv - self._last_net.bytes_recv) / elapsed
            net_sent_bps = float(net_io.bytes_sent - self._last_net.bytes_sent) / elapsed

        self._last_disk = disk_io
        self._last_net = net_io

        return {
            "timestamp": now,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "hostname": self.hostname,
            "cpu_percent": float(psutil.cpu_percent(interval=None)),
            "cpu_count_logical": int(psutil.cpu_count(logical=True) or 0),
            "cpu_count_physical": int(psutil.cpu_count(logical=False) or 0),
            "load_1m": float(load_avg[0]),
            "load_5m": float(load_avg[1]),
            "load_15m": float(load_avg[2]),
            "mem_total_bytes": int(vm.total),
            "mem_available_bytes": int(vm.available),
            "mem_used_bytes": int(vm.used),
            "mem_percent": float(vm.percent),
            "swap_total_bytes": int(swap.total),
            "swap_used_bytes": int(swap.used),
            "swap_percent": float(swap.percent),
            "disk_total_bytes": int(disk_usage.total),
            "disk_used_bytes": int(disk_usage.used),
            "disk_free_bytes": int(disk_usage.free),
            "disk_percent": float(disk_usage.percent),
            "io_read_bytes": int(disk_io.read_bytes if disk_io else 0),
            "io_write_bytes": int(disk_io.write_bytes if disk_io else 0),
            "io_read_bps": io_read_bps,
            "io_write_bps": io_write_bps,
            "net_recv_bytes": int(net_io.bytes_recv if net_io else 0),
            "net_sent_bytes": int(net_io.bytes_sent if net_io else 0),
            "net_recv_bps": net_recv_bps,
            "net_sent_bps": net_sent_bps,
        }


def _sanitize_node_name(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", raw)


def _extract_alive_nodes(timeout: float = 10.0) -> Optional[List[Dict[str, str]]]:
    """Extract alive nodes from Ray cluster.

    Returns a list of node dicts on success, or ``None`` if ``ray.nodes()``
    timed out or failed (so callers can distinguish "zero nodes" from "GCS
    unreachable").
    """
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            raw_nodes = pool.submit(ray.nodes).result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logger.warning(
            "ray.nodes() timed out after %.1fs — GCS may be congested by log backpressure.",
            timeout,
        )
        return None
    except Exception:
        logger.exception("ray.nodes() call failed.")
        return None

    nodes = []
    for node in raw_nodes:
        if not node.get("Alive"):
            continue
        node_id = str(node.get("NodeID"))
        node_ip = str(node.get("NodeManagerAddress") or node.get("NodeName") or "")
        if not node_id or not node_ip:
            continue
        nodes.append({"node_id": node_id, "node_ip": node_ip})
    return nodes


@dataclass
class ClusterResourceMonitor:
    enabled: bool = False
    interval_s: float = 30.0
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_job_type: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_config: Dict[str, Any] = field(default_factory=dict)
    print_summary: bool = True

    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _actors: Dict[str, Any] = field(default_factory=dict, init=False)
    _wandb_run: Any = field(default=None, init=False)
    _consecutive_gcs_failures: int = field(default=0, init=False)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._init_wandb()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="tinyflow-cluster-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Cluster monitoring started (interval=%ss, wandb=%s)",
            self.interval_s,
            self.wandb_enabled,
        )

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(2.0, self.interval_s + 2.0))
        self._thread = None
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                logger.exception("Failed to finish wandb run for monitoring.")
            self._wandb_run = None

    def _init_wandb(self) -> None:
        if not self.wandb_enabled:
            return
        try:
            import wandb  # lazy import

            self._wandb_run = wandb.init(
                project=self.wandb_project or "tinyflow-monitor",
                entity=self.wandb_entity,
                name=self.wandb_run_name,
                group=self.wandb_group,
                job_type=self.wandb_job_type or "cluster-monitor",
                tags=self.wandb_tags or None,
                config=self.wandb_config or None,
                reinit=True,
            )
            logger.info("W&B monitoring run initialized.")
        except Exception:
            logger.exception("Failed to initialize wandb. Monitoring continues without wandb.")
            self._wandb_run = None

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.time()
            try:
                snapshots = self._collect_snapshots()
                if snapshots:
                    self._log_snapshots(snapshots)
            except Exception:
                logger.exception("Cluster monitoring loop error.")
            elapsed = time.time() - started
            sleep_s = max(0.0, self.interval_s - elapsed)
            if self._stop_event.wait(timeout=sleep_s):
                break

    def _collect_snapshots(self) -> List[Dict[str, Any]]:
        gcs_timeout = min(10.0, self.interval_s * 0.3)
        alive_nodes = _extract_alive_nodes(timeout=gcs_timeout)
        if alive_nodes is not None:
            self._consecutive_gcs_failures = 0
            self._sync_actors(alive_nodes)
        else:
            self._consecutive_gcs_failures += 1
            logger.warning(
                "Skipping actor sync (GCS unreachable, %d consecutive failures). "
                "Sampling %d cached actors.",
                self._consecutive_gcs_failures,
                len(self._actors),
            )
        if not self._actors:
            return []
        refs = [actor.sample.remote() for actor in self._actors.values()]
        timeout = max(1.0, self.interval_s * 0.8)
        done, not_done = ray.wait(refs, num_returns=len(refs), timeout=timeout)
        snapshots: List[Dict[str, Any]] = []
        if done:
            snapshots.extend(ray.get(done))
        if not_done:
            logger.warning("Monitoring timeout: %d node samples not returned.", len(not_done))
        return snapshots

    def _sync_actors(self, alive_nodes: List[Dict[str, str]]) -> None:
        wanted = {node["node_id"]: node for node in alive_nodes}
        stale_node_ids = [node_id for node_id in self._actors if node_id not in wanted]
        for node_id in stale_node_ids:
            self._actors.pop(node_id, None)

        for node_id, node in wanted.items():
            if node_id in self._actors:
                continue
            node_ip = node["node_ip"]
            node_resource_key = f"node:{node_ip}"
            opts = {"resources": {node_resource_key: 0.001}, "name": f"tinyflow-monitor-{node_id[:8]}"}
            self._actors[node_id] = NodeResourceMonitorActor.options(**opts).remote(node_id, node_ip)

    def _log_snapshots(self, snapshots: List[Dict[str, Any]]) -> None:
        now = time.time()
        payload: Dict[str, Any] = {
            "cluster/node_count": len(snapshots),
            "cluster/cpu_percent_avg": _avg(snapshots, "cpu_percent"),
            "cluster/mem_percent_avg": _avg(snapshots, "mem_percent"),
            "cluster/disk_percent_avg": _avg(snapshots, "disk_percent"),
            "cluster/io_read_bps_total": _sum(snapshots, "io_read_bps"),
            "cluster/io_write_bps_total": _sum(snapshots, "io_write_bps"),
            "cluster/net_recv_bps_total": _sum(snapshots, "net_recv_bps"),
            "cluster/net_sent_bps_total": _sum(snapshots, "net_sent_bps"),
            "cluster/timestamp": now,
        }
        for snap in snapshots:
            node_name = _sanitize_node_name(snap.get("hostname") or snap.get("node_ip") or snap["node_id"][:8])
            prefix = f"nodes/{node_name}"
            payload[f"{prefix}/cpu_percent"] = snap["cpu_percent"]
            payload[f"{prefix}/mem_percent"] = snap["mem_percent"]
            payload[f"{prefix}/disk_percent"] = snap["disk_percent"]
            payload[f"{prefix}/io_read_bps"] = snap["io_read_bps"]
            payload[f"{prefix}/io_write_bps"] = snap["io_write_bps"]
            payload[f"{prefix}/net_recv_bps"] = snap["net_recv_bps"]
            payload[f"{prefix}/net_sent_bps"] = snap["net_sent_bps"]

        if self.print_summary:
            logger.info(
                "[monitor] nodes=%d cpu_avg=%.1f mem_avg=%.1f disk_avg=%.1f io_r=%.2fMB/s io_w=%.2fMB/s",
                int(payload["cluster/node_count"]),
                float(payload["cluster/cpu_percent_avg"]),
                float(payload["cluster/mem_percent_avg"]),
                float(payload["cluster/disk_percent_avg"]),
                float(payload["cluster/io_read_bps_total"]) / (1024 * 1024),
                float(payload["cluster/io_write_bps_total"]) / (1024 * 1024),
            )
        if self._wandb_run is not None:
            try:
                self._wandb_run.log(payload)
            except Exception:
                logger.exception("Failed to log monitoring payload to wandb.")


def _avg(items: List[Dict[str, Any]], key: str) -> float:
    if not items:
        return 0.0
    return float(sum(float(item.get(key, 0.0)) for item in items) / len(items))


def _sum(items: List[Dict[str, Any]], key: str) -> float:
    return float(sum(float(item.get(key, 0.0)) for item in items))
