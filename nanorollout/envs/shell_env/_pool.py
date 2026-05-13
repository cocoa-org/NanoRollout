"""VM pool coordinator for the GCE shell environment.

PoolCoordinator owns a per-MIG view of worker VMs and hands them out
to GCEEnvironment via claim() / release(). Two construction paths:
production (mig_name+project+zone) and test bypass (explicit_ip).

Uses threading.Lock (not asyncio.Lock) so one coordinator can be shared
across GCEEnvironment instances with their own event loops. claim() is
declared async for caller compatibility but does not actually await
(GCP API + HealthCheck use sync clients)."""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

import grpc

from .gce_worker.proto import worker_pb2, worker_pb2_grpc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VMInfo:
    name: str           # GCE instance name, or "explicit" for the test bypass.
    internal_ip: str    # 10.x.y.z reachable on worker_port from the caller's VPC.


class PoolExhausted(RuntimeError):
    """Raised by ``claim()`` when the MIG has zero available healthy VMs.

    Callers should treat this as "increase MIG size or wait" --- the pool
    does NOT autoscale on demand (autoscaler is server-side, CPU-driven).
    """


class PoolCoordinator:
    """Per-MIG VM pool. See module docstring for concurrency model."""

    def __init__(
        self,
        mig_name: str,
        project: str,
        zone: str,
        worker_port: int = 50051,
    ) -> None:
        self._mig_name = mig_name
        self._project = project
        self._zone = zone
        self._worker_port = worker_port
        self._explicit_ip: Optional[str] = None

        self._free_vms: deque[VMInfo] = deque()
        self._active: dict[str, VMInfo] = {}

        # Two locks so refresh (slow, network) doesn't block fast pop/push.
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()

        # Lazily imported so ``explicit_ip`` callers (B.4) don't pay the
        # cost of importing google-cloud-compute.
        self._compute_module = None

    @classmethod
    def explicit_ip(cls, ip: str, worker_port: int = 50051) -> "PoolCoordinator":
        """Test bypass: pool that always returns the same VM, no MIG involved.
        release() is a no-op in this mode (caller manages VM lifecycle)."""
        self = cls.__new__(cls)
        self._mig_name = None  # type: ignore[assignment]
        self._project = None  # type: ignore[assignment]
        self._zone = None  # type: ignore[assignment]
        self._worker_port = worker_port
        self._explicit_ip = ip
        self._free_vms = deque()
        self._active = {}
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._compute_module = None
        return self

    @property
    def worker_port(self) -> int:
        return self._worker_port

    # ----- claim / release -----

    async def claim(self) -> VMInfo:
        """Return an idle worker VM.

        IMPORTANT: ``async def`` only for caller compatibility. The body
        never awaits, which means concurrent ``asyncio.gather(claim(),
        claim(), ...)`` calls on the **same** event loop will run
        sequentially (not interleaved). The B.5 smoke parallelizes via
        ``ThreadPoolExecutor`` so each thread has its own loop --- that
        works because state is guarded by ``threading.Lock``.

        Raises ``PoolExhausted`` if the MIG has no available healthy VMs
        after a fresh refresh.
        """
        if self._explicit_ip is not None:
            vm = VMInfo(name="explicit", internal_ip=self._explicit_ip)
            with self._lock:
                self._active[vm.name] = vm
            return vm

        # Fast path: pop from free queue.
        with self._lock:
            if self._free_vms:
                vm = self._free_vms.popleft()
                self._active[vm.name] = vm
                return vm

        # Slow path: refresh from GCP. Use a separate lock so concurrent
        # claimers serialize on the GCP call but don't block fast pops.
        with self._refresh_lock:
            # Double-check: a sibling refresh may have populated the queue
            # while we were waiting.
            with self._lock:
                if self._free_vms:
                    vm = self._free_vms.popleft()
                    self._active[vm.name] = vm
                    return vm

            self._refresh_from_gcp()

            with self._lock:
                if not self._free_vms:
                    raise PoolExhausted(
                        f"MIG {self._mig_name} has no available healthy VMs; "
                        f"active={len(self._active)}"
                    )
                vm = self._free_vms.popleft()
                self._active[vm.name] = vm
                return vm

    async def release(self, vm: VMInfo) -> None:
        """Return a VM to the idle pool.

        For ``explicit_ip`` mode this is a no-op (the VM is caller-managed).

        For MIG mode: pops from active, pushes to free queue. NOTE: does
        not re-HealthCheck the VM --- if the worker daemon died during
        the claim, this re-enqueues a dead IP. Future B.6+ improvement:
        either probe at release-time or rely on a faster background
        refresh to prune stale entries.
        """
        if self._explicit_ip is not None:
            return

        with self._lock:
            self._active.pop(vm.name, None)
            self._free_vms.append(vm)

    # ----- diagnostics -----

    def active_vms(self) -> list[VMInfo]:
        """Return a snapshot of currently claimed VMs.

        Used by the B.5 smoke to verify 4 concurrent envs land on 4
        distinct workers. Snapshot is consistent at lock-acquire time;
        callers should not assume it stays valid.
        """
        with self._lock:
            return list(self._active.values())

    def free_vms(self) -> list[VMInfo]:
        """Return a snapshot of currently idle VMs."""
        with self._lock:
            return list(self._free_vms)

    # ----- internals -----

    def _ensure_compute(self):
        if self._compute_module is None:
            from google.cloud import compute_v1  # type: ignore[import-not-found]
            self._compute_module = compute_v1
        return self._compute_module

    def _refresh_from_gcp(self) -> None:
        """Repopulate ``_free_vms`` with healthy MIG members.

        Called under ``_refresh_lock``. Gathers RUNNING instances, fetches
        each one's internal IP, runs HealthCheck, and enqueues the OK
        ones (skipping any already in ``_active``).
        """
        compute_v1 = self._ensure_compute()

        mig_client = compute_v1.InstanceGroupManagersClient()
        instances_client = compute_v1.InstancesClient()

        listed = mig_client.list_managed_instances(
            project=self._project,
            zone=self._zone,
            instance_group_manager=self._mig_name,
        )

        # mig_client returns ManagedInstance objects; the URL has the
        # short instance name as last path component.
        candidates: list[str] = []
        for mi in listed:
            if mi.instance_status != "RUNNING":
                logger.debug("skipping MIG member %s (status=%s)", mi.instance, mi.instance_status)
                continue
            short_name = mi.instance.rsplit("/", 1)[-1]
            candidates.append(short_name)

        with self._lock:
            already_active = set(self._active.keys())
            already_free = {vm.name for vm in self._free_vms}

        added = 0
        for name in candidates:
            if name in already_active or name in already_free:
                continue
            try:
                inst = instances_client.get(
                    project=self._project,
                    zone=self._zone,
                    instance=name,
                )
                ip = inst.network_interfaces[0].network_i_p
            except Exception as exc:
                logger.warning("refresh: get(%s) failed: %s", name, exc)
                continue

            if not self._health_check(ip):
                logger.warning("refresh: HealthCheck failed for %s (%s); skipping", name, ip)
                continue

            with self._lock:
                # Re-check under lock: another refresh may have added it.
                if name in self._active or any(v.name == name for v in self._free_vms):
                    continue
                self._free_vms.append(VMInfo(name=name, internal_ip=ip))
                added += 1

        logger.info(
            "refresh from MIG %s: %d candidates, %d added (active=%d, free=%d)",
            self._mig_name, len(candidates), added,
            len(self._active), len(self._free_vms),
        )

    def _health_check(self, ip: str) -> bool:
        """Sync gRPC HealthCheck with a 3s deadline. Returns True if healthy."""
        target = f"{ip}:{self._worker_port}"
        channel = grpc.insecure_channel(target)
        try:
            stub = worker_pb2_grpc.WorkerServiceStub(channel)
            response = stub.HealthCheck(worker_pb2.Empty(), timeout=3)
            return bool(response.healthy)
        except Exception as exc:
            logger.debug("health_check(%s) failed: %s", target, exc)
            return False
        finally:
            channel.close()
