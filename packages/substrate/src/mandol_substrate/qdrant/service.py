"""Owned single-node Qdrant service for an independent persistent run root.

Supports the same pinned binary natively and in Docker for measured deployment
comparison. Never discovers/kills arbitrary processes, reuses a stale PID or
removes a storage directory. Service identity survives the Python owner dying.
"""

import ctypes
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx

from ..state.models import StateUnavailable, canonical_json


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def process_identity(pid):
    try:
        # comm can contain spaces/parentheses; fields after its final ')' start
        # at state (field3); starttime is field22. Epoch clocks are irrelevant.
        text = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if text[0] == "Z":
            return None
        return {"boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "start_ticks": int(text[19])}
    except FileNotFoundError:
        return None


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def owns_loopback_listener(pid, port):
    """Match listening socket inode to the exact process, not just its netns."""
    try:
        sockets = set()
        for path in Path(f"/proc/{pid}/fd").iterdir():
            try:
                target = os.readlink(path)
            except FileNotFoundError:
                # Listing our own fd directory creates a short-lived directory
                # fd; unrelated application descriptors can close concurrently.
                continue
            if target.startswith("socket:["):
                sockets.add(target[8:-1])
        address = f"0100007F:{port:04X}"
        return any(parts[1] == address and parts[3] == "0A" and parts[9] in sockets
                   for line in Path(f"/proc/{pid}/net/tcp").read_text().splitlines()[1:]
                   if (parts := line.split()))
    except (FileNotFoundError, ProcessLookupError):
        return False


def signal_owned_process(pid, identity, sig):
    """Linux pidfd syscall wrappers, also on Python builds lacking os.pidfd_open."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
    libc.pidfd_open.restype = ctypes.c_int
    libc.pidfd_send_signal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    libc.pidfd_send_signal.restype = ctypes.c_int
    fd = libc.pidfd_open(pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "cannot open owned process pidfd")
    try:
        if process_identity(pid) != identity:
            raise StateUnavailable("process identity changed before stop")
        if libc.pidfd_send_signal(fd, sig, None, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot signal owned process pidfd")
    finally:
        os.close(fd)


class QdrantServiceManager:
    def __init__(self, root, storage_id, *, mode, binary=None, binary_sha256=None,
                 image_id=None, version="1.19.1", port=None, allow_scratch=False):
        original = Path(root).absolute()
        if any(p.is_symlink() for p in (original, *original.parents)):
            raise ValueError("symlink run storage is not permitted")
        self.root = original.resolve()
        if self.root == Path("/") or (str(self.root).startswith(("/mnt/", "/tmp/")) and not allow_scratch):
            raise ValueError("formal run storage requires a persistent Linux filesystem")
        if not self.root.is_dir() or not storage_id or mode not in {"native", "docker"}:
            raise ValueError("explicit existing run root/storage identity/mode required")
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest.get("storage_id") != storage_id:
            raise ValueError("service storage identity differs from run manifest")
        self.storage_id, self.mode, self.version = storage_id, mode, version
        self.binary = str(Path(binary).resolve()) if binary else None
        self.binary_sha256, self.image_id = binary_sha256, image_id
        if mode == "native":
            if not self.binary or hashlib.sha256(Path(self.binary).read_bytes()).hexdigest() != binary_sha256:
                raise ValueError("pinned native Qdrant binary required")
        elif not image_id or not image_id.startswith("sha256:"):
            raise ValueError("immutable Docker image ID required")
        self.qroot = self.root / "qdrant"
        for path in [self.qroot, self.qroot / "storage", self.qroot / "snapshots", self.qroot / "scratch", self.root / "logs"]:
            if path.is_symlink():
                raise ValueError("unsafe service directory")
            path.mkdir(parents=True, exist_ok=True)
        self.owner_path = self.qroot / "service.json"
        self._lock = (self.qroot / "service.lock").open("a+b")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise StateUnavailable("another Qdrant lifecycle owner holds this run") from None
        except BaseException:
            self._lock.close()
            raise
        self.port = port
        self.owner = None
        self.process = None
        self._log = None

    @property
    def uri(self):
        return f"http://127.0.0.1:{self.port}"

    def _alive(self, owner):
        if owner["mode"] == "native":
            return process_identity(owner["pid"]) == owner["identity"]
        result = subprocess.run(["docker", "inspect", owner["container_id"]], capture_output=True, text=True, check=False)
        if result.returncode:
            if "No such object" not in result.stderr:
                raise StateUnavailable("cannot prove Docker service stopped")
            return False
        data = json.loads(result.stdout)[0]
        if (data["Config"]["Labels"].get("mandol.storage_id") != self.storage_id
                or data["Image"] != owner["image_id"]):
            raise StateUnavailable("Docker ownership differs; refusing lifecycle action")
        mounts = {m["Destination"]: m["Source"] for m in data["Mounts"]}
        if mounts.get("/run/qdrant") != str(Path(owner["root"]) / "qdrant"):
            raise StateUnavailable("Docker storage mount differs")
        bindings = data["HostConfig"]["PortBindings"].get("6333/tcp")
        if bindings != [{"HostIp": "127.0.0.1", "HostPort": str(owner["port"])}]:
            raise StateUnavailable("Docker loopback port ownership differs")
        return data["State"]["Running"]

    def start(self, *, timeout=60):
        started = time.monotonic()
        if self.owner_path.exists():
            owner = json.loads(self.owner_path.read_text())
            if owner["storage_id"] != self.storage_id:
                raise StateUnavailable("service receipt belongs to another storage identity")
            alive = self._alive(owner)
            if owner["root"] != str(self.root):
                if alive:
                    raise StateUnavailable("cannot relocate a live Qdrant owner")
                # A stopped run is portable. Preserve its old observation, then
                # regenerate runtime absolute paths; never reuse the old PID.
                identity = hashlib.sha256(self.owner_path.read_bytes()).hexdigest()
                saved = self.root / "artifacts" / ("relocated-service-" + identity + ".json")
                if not saved.exists():
                    atomic_json(saved, owner)
            if alive:
                if (owner["mode"] != self.mode or owner.get("image_id") != self.image_id
                        or owner.get("binary_sha256") != self.binary_sha256):
                    raise StateUnavailable("running Qdrant deployment profile changed")
                self.owner, self.port = owner, owner["port"]
                self.health()
                return {"reused": True, "startup_seconds": time.monotonic() - started, "uri": self.uri}
        self.port = self.port or free_port()
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", self.port))
        prefix = self.qroot if self.mode == "native" else Path("/run/qdrant")
        config = {"log_level": "WARN", "telemetry_disabled": True,
            "storage": {"storage_path": str(prefix / "storage"), "snapshots_path": str(prefix / "snapshots"),
                "temp_path": str(prefix / "scratch"), "performance": {"max_search_threads": 4},
                "optimizers": {"max_optimization_threads": 2}},
            "service": {"host": "127.0.0.1" if self.mode == "native" else "0.0.0.0",
                "http_port": self.port if self.mode == "native" else 6333,
                "grpc_port": None, "max_workers": 4, "enable_cors": False},
            "cluster": {"enabled": False}}
        # The pinned server config reader enables YAML, not JSON file discovery.
        # JSON content is a valid YAML subset; the extension must be .yaml.
        config_path = self.qroot / "config.yaml"
        atomic_json(config_path, config)
        owner = {"storage_id": self.storage_id, "root": str(self.root), "mode": self.mode,
                 "port": self.port, "image_id": self.image_id, "binary_sha256": self.binary_sha256,
                 "version": self.version, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
        if self.mode == "native":
            command = [self.binary, "--config-path", str(config_path)]
            self._log = (self.root / "logs/qdrant.log").open("ab")
            # Do not inherit ambient QDRANT config overrides or credentials.
            env = {k: v for k, v in os.environ.items() if not k.startswith("QDRANT__")}
            env["RUN_MODE"] = "production"
            # Even a future config-loader failure must not open public listeners.
            env["QDRANT__SERVICE__HOST"] = "127.0.0.1"
            env["QDRANT__TELEMETRY_DISABLED"] = "true"
            self.process = subprocess.Popen(command, cwd=self.qroot, env=env,
                stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True)
            identity = process_identity(self.process.pid)
            if identity is None:
                raise StateUnavailable("Qdrant exited during launch")
            owner.update(pid=self.process.pid, identity=identity, argv=command)
        else:
            command = ["docker", "run", "-d", "--label", f"mandol.storage_id={self.storage_id}",
                "--publish", f"127.0.0.1:{self.port}:6333", "--mount",
                f"type=bind,src={self.qroot},dst=/run/qdrant", "--entrypoint", "/qdrant/qdrant",
                self.image_id, "--config-path", "/run/qdrant/config.yaml"]
            owner["container_id"] = subprocess.check_output(command, text=True).strip()
        self.owner = owner
        atomic_json(self.owner_path, owner)
        try:
            while time.monotonic() - started < timeout:
                if not self._alive(owner):
                    raise StateUnavailable("owned Qdrant exited before ready")
                try:
                    self.health()
                    return {"reused": False, "startup_seconds": time.monotonic() - started, "uri": self.uri}
                except (httpx.HTTPError, StateUnavailable):
                    time.sleep(.05)
            raise StateUnavailable("Qdrant startup deadline exceeded")
        except BaseException:
            self.stop()
            raise

    def health(self):
        if not self.owner or not self._alive(self.owner):
            raise StateUnavailable("owned Qdrant process is not running")
        if self.mode == "native" and not owns_loopback_listener(self.owner["pid"], self.port):
            raise StateUnavailable("Qdrant listener is not owned by the recorded process")
        with httpx.Client(timeout=2, trust_env=False) as client:
            response = client.get(self.uri + "/readyz")
            response.raise_for_status()
            info = client.get(self.uri + "/")
            info.raise_for_status()
            if info.json().get("version") != self.version:
                raise StateUnavailable("owned Qdrant reports unexpected version")
        return True

    def stop(self, *, kill=False, timeout=30):
        started = time.monotonic()
        owner = self.owner
        if not owner:
            return {"already_stopped": True}
        if self._alive(owner):
            if self.mode == "native":
                # pidfd identifies the process even if its numerical PID is reused.
                signal_owned_process(owner["pid"], owner["identity"],
                                     signal.SIGKILL if kill else signal.SIGTERM)
            else:
                subprocess.run(["docker", "kill" if kill else "stop", owner["container_id"]],
                               check=True, capture_output=True, timeout=timeout)
            while self._alive(owner):
                if time.monotonic() - started >= timeout:
                    raise StateUnavailable("owned Qdrant did not stop; storage retained")
                time.sleep(.05)
        if self.process:
            self.process.wait(timeout=1)
        if self.mode == "docker":
            # Capture logs and remove only this exact-owned stopped container.
            logs = subprocess.run(["docker", "logs", owner["container_id"]], capture_output=True, check=False)
            with (self.root / "logs/qdrant.log").open("ab") as handle:
                handle.write(logs.stdout + logs.stderr)
            subprocess.run(["docker", "rm", owner["container_id"]], check=True, capture_output=True)
        if self._log:
            self._log.close()
            self._log = None
        self.owner = None
        return {"stopped": True, "sigkill": kill, "seconds": time.monotonic() - started}

    def close(self):
        try:
            self.stop()
        finally:
            if self._log:
                self._log.close()
            self._lock.close()
