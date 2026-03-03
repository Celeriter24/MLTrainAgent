"""
Docker Sandbox — safely runs ML code in an isolated container.
"""

import os
import platform
import tarfile
import io
import logging
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass

import docker
from docker.errors import ContainerError, ImageNotFound, DockerException

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    output: str          # combined
    files: dict[str, bytes]   # filename → bytes for /results/ artifacts
    duration: float


class DockerSandbox:
    def __init__(self, config: dict):
        self.cfg = config["docker"]
        self.mem_limit = self.cfg.get("memory_limit", "8g")
        self.cpu_limit = float(self.cfg.get("cpu_limit", 4.0))
        self.timeout = int(self.cfg.get("timeout", 600))
        self.network = self.cfg.get("network", "none")
        self.auto_cleanup = self.cfg.get("auto_cleanup", True)

        gpu_requested = self.cfg.get("gpu", False)
        system = platform.system()

        if system == "Darwin":
            if gpu_requested:
                logger.warning("macOS detected — GPU flag ignored (Docker on Mac has no GPU support)")
            self.gpu = False
            self.dockerfile = "sandbox/Dockerfile.sandbox"
            self.image = self.cfg.get("image_cpu", "ml-sandbox:cpu")
        elif gpu_requested:
            self.gpu = True
            self.dockerfile = "sandbox/Dockerfile.sandbox-gpu"
            self.image = self.cfg.get("image_gpu", "ml-sandbox:gpu")
        else:
            self.gpu = False
            self.dockerfile = "sandbox/Dockerfile.sandbox"
            self.image = self.cfg.get("image_cpu", "ml-sandbox:cpu")

        logger.info(f"Platform: {system} | GPU: {self.gpu} | Image: {self.image}")

        try:
            self.client = docker.from_env()
        except DockerException as e:
            raise RuntimeError(f"Cannot connect to Docker daemon: {e}") from e

    def run_code(self, code: str, extra_files: dict[str, str] = None) -> ExecutionResult:
        """
        Execute Python code string inside the ML sandbox.
        extra_files: {filename: content} to copy into /workspace/
        Returns ExecutionResult with logs and any /results/ artifacts.
        """
        start = time.time()
        container = None

        try:
            # Build device_requests for GPU
            device_requests = []
            if self.gpu:
                device_requests = [
                    docker.types.DeviceRequest(
                        device_ids=[self.cfg.get("gpu_device", "0")],
                        capabilities=[["gpu"]],
                    )
                ]

            backend = "gpu" if self.gpu else "cpu"
            container_name = f"mlresearch-{backend}-{time.strftime('%Y%m%d-%H%M%S')}"

            container = self.client.containers.create(
                image=self.image,
                name=container_name,
                command=["python", "/workspace/experiment.py"],
                mem_limit=self.mem_limit,
                nano_cpus=int(self.cpu_limit * 1e9),
                network_mode=self.network,
                device_requests=device_requests,
                working_dir="/workspace",
                detach=True,
            )

            # Copy code into container
            self._copy_to_container(container, "experiment.py", code.encode())

            # Copy any extra files
            if extra_files:
                for fname, content in extra_files.items():
                    self._copy_to_container(container, fname, content.encode())

            # Start and wait with timeout
            container.start()
            logger.info(f"Container {container.short_id} started")

            try:
                result = container.wait(timeout=self.timeout)
                exit_code = result["StatusCode"]
            except Exception:
                container.kill()
                exit_code = -1
                logger.warning("Container timed out — killed")

            # Collect logs
            logs = container.logs(stdout=True, stderr=True).decode(errors="replace")
            stdout = container.logs(stdout=True, stderr=False).decode(errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode(errors="replace")

            # Collect /results/ artifacts
            files = self._collect_results(container)

            duration = time.time() - start
            logger.info(f"Container finished in {duration:.1f}s — exit code {exit_code}")

            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                output=logs,
                files=files,
                duration=duration,
            )

        finally:
            if container and self.auto_cleanup:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    def _copy_to_container(self, container, filename: str, data: bytes):
        """Copy a single file into /workspace/ of a stopped/created container."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=filename)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        buf.seek(0)
        container.put_archive("/workspace", buf)

    def _collect_results(self, container) -> dict[str, bytes]:
        """Pull all files from /results/ inside the container."""
        files = {}
        try:
            bits, _ = container.get_archive("/results")
            buf = io.BytesIO(b"".join(bits))
            with tarfile.open(fileobj=buf) as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        f = tar.extractfile(member)
                        if f:
                            fname = Path(member.name).name
                            files[fname] = f.read()
            logger.info(f"Collected {len(files)} result files from container")
        except Exception as e:
            logger.debug(f"No /results/ to collect (or empty): {e}")
        return files

    def image_exists(self) -> bool:
        try:
            self.client.images.get(self.image)
            return True
        except ImageNotFound:
            return False

    def build_image(self):
        logger.info(f"Building {self.image} from {self.dockerfile} ...")
        self.client.images.build(
            path=str(Path.cwd()),
            dockerfile=self.dockerfile,
            tag=self.image,
            rm=True,
        )
        logger.info("Image built successfully")
