"""
Base repository profile class.

This module defines the abstract base class for repository profiles that specify
installation and testing configurations for different repositories.
"""

import docker
import os
import platform
import shutil
import subprocess
import hashlib
import random
import string

from abc import ABC, abstractmethod, ABCMeta
from collections import UserDict
from dataclasses import dataclass, field
from docker.models.containers import Container
from dotenv import load_dotenv
from enum import Enum
from functools import cached_property
from ghapi.all import GhApi
from multiprocessing import Lock
from pathlib import Path
from typing import Any
from unidiff import PatchSet

# Note: swesmith.bug_gen.adapters is imported lazily in extract_entities() to avoid
# loading tree-sitter dependencies when only using Registry/get_valid_report

from nanorollout.eval.swebench.constants import (
    DOCKER_USER,
    DOCKER_WORKDIR,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
)

ENV_NAME = "testbed"
KEY_PATCH = "patch"
LOG_DIR_ENV = Path("logs/build_images/env")
ORG_NAME_DH = "swebench"
ORG_NAME_GH = "swesmith"
INSTANCE_REF = "instance_ref"


class CodeProperty(Enum):
    # Core entity types
    IS_FUNCTION = "is_function"
    IS_CLASS = "is_class"

    # Control flow
    HAS_EXCEPTION = "has_exception"
    HAS_IF = "has_if"
    HAS_IF_ELSE = "has_if_else"
    HAS_LOOP = "has_loop"
    HAS_SWITCH = "has_switch"  # Added for switch statements

    # Operations
    HAS_ARITHMETIC = "has_arithmetic"
    HAS_ASSIGNMENT = "has_assignment"
    HAS_DECORATOR = "has_decorator"
    HAS_FUNCTION_CALL = "has_function_call"
    HAS_IMPORT = "has_import"
    HAS_LAMBDA = "has_lambda"
    HAS_LIST_COMPREHENSION = "has_list_comprehension"
    HAS_LIST_INDEXING = "has_list_indexing"
    HAS_OFF_BY_ONE = "has_off_by_one"
    HAS_PARENT = "has_parent"
    HAS_RETURN = "has_return"
    HAS_WRAPPER = "has_wrapper"

    # Operations by type
    HAS_BINARY_OP = "has_binary_op"
    HAS_BOOL_OP = "has_bool_op"
    HAS_TERNARY = "has_ternary"
    HAS_UNARY_OP = "has_unary_op"


class CodeEntityMeta(type):
    def __new__(mcs, name, bases, namespace):
        # Create properties for all enum values
        for prop in CodeProperty:
            namespace[prop.value] = property(lambda self, p=prop: p in self._tags)
        return super().__new__(mcs, name, bases, namespace)


@dataclass
class CodeEntity(metaclass=CodeEntityMeta):
    """Data class to hold information about a code entity (e.g. function, class)."""

    file_path: str
    indent_level: int
    indent_size: int
    line_end: int
    line_start: int
    node: Any
    src_code: Any

    def __post_init__(self):
        self._tags: set[CodeProperty] = set()
        self._analyze_properties()

    def _analyze_properties(self):
        """To be implemented by language-specific classes"""
        pass

    @property
    def complexity(self) -> int:
        """Get the complexity of the code entity."""
        return -1  # Default value = no notion of complexity implemented

    @property
    def ext(self) -> str:
        if isinstance(self.file_path, Path):
            self.file_path = str(self.file_path)
        return self.file_path.rsplit(".", 1)[-1].lower()

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the code entity."""
        pass

    @property
    @abstractmethod
    def signature(self) -> str:
        """Get the signature of the code entity."""
        pass

    @property
    @abstractmethod
    def stub(self) -> str:
        """Get stub (code with implementation removed) for the code entity."""
        pass


class BugRewrite:
    cost: float = 0
    explanation: str = ""
    output: str
    rewrite: str
    strategy: str

    def __init__(
        self,
        rewrite: str,
        explanation: str,
        strategy: str,
        cost: float = 0,
        output: str = "",
    ):
        self.rewrite = rewrite
        self.explanation = explanation
        self.cost = cost
        self.strategy = strategy
        self.output = output

    def get_hash(self) -> str:
        """Generates a hash for the bug rewrite."""
        return generate_hash(self.rewrite)

    def to_dict(self) -> dict[str, Any]:
        """Converts the bug rewrite to a dictionary."""
        return {
            "cost": self.cost,
            "explanation": self.explanation,
            "output": self.output,
            "rewrite": self.rewrite,
            "strategy": self.strategy,
        }


def generate_hash(s):
    rng = random.Random(int(hashlib.sha256(s.encode()).hexdigest(), 16))
    return "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(8))


load_dotenv()


class SingletonMeta(ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class RepoProfile(ABC, metaclass=SingletonMeta):
    """
    Base class for repository profiles that define installation and testing specifications.

    This class provides a language-agnostic interface for repository configuration,
    allowing different languages (Python, Go, Rust, etc.) to have their own
    installation and testing patterns while maintaining a consistent API.
    """

    org_dh: str = ORG_NAME_DH
    org_gh: str = ORG_NAME_GH
    arch: str = "x86_64" if platform.machine() not in {"aarch64", "arm64"} else "arm64"

    @property
    def pltf(self) -> str:
        if self.arch == "x86_64":
            return "linux/x86_64"
        elif self.arch == "arm64":
            return "linux/arm64/v8"
        else:
            raise ValueError(
                f"Architecture {self.arch} not supported. Must be one of ['x86_64', 'arm64']"
            )

    exts: list[str] = field(default_factory=list)  # Must be set by subclass
    eval_sets: set[str] = field(default_factory=set)

    # Install + Test specifications
    timeout: int = 90  # timeout (sec) for running test suite for a single instance
    timeout_ref: int = 900  # timeout for running entire test suite

    # `min_testing`: If set, then subset of tests (not all) are run for post-bug validation
    # Affects get_test_cmd, get_valid_report
    min_testing: bool = False

    # `min_pregold`: If set, then for pre-bug validation, individual runs are
    # performed instead of running the entire test suite
    # Affects valid.py
    min_pregold: bool = False

    # The lock is to prevent concurrent clones of the same repository.
    # In this repo, all subclasses of RepoProfile are meant to be Singletons (only one instance
    # of the class will ever be created). If this changes for some reason in the future,
    # this design may have to be updated.
    _lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    # GitHub API instance (lazily initialized)
    _api: GhApi | None = field(default=None, init=False, repr=False, compare=False)

    # Class-level caches
    _cache_test_paths = None
    _cache_branches = None
    _cache_mirror_exists = None

    ### START: Properties, Methods that *do not* require (re-)implementation ###

    @property
    def api(self) -> GhApi:
        """Get GitHub API instance with lazy initialization."""
        if self._api is None:
            token = os.getenv("GITHUB_TOKEN")
            self._api = GhApi(token=token)
        return self._api

    @property
    def image_name(self) -> str:
        return f"{self.org_dh}/swesmith.{self.arch}.{self.owner}_1776_{self.repo}.{self.commit[:8]}".lower()

    @cached_property
    def _cache_image_exists(self) -> bool:
        """Check if Docker image exists locally."""
        try:
            subprocess.run(
                f"docker image inspect {self.image_name}",
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    @property
    def mirror_name(self):
        return f"{self.org_gh}/{self.repo_name}"

    @property
    def repo_name(self):
        return f"{self.owner}__{self.repo}.{self.commit[:8]}"

    @property
    def branches(self):
        """Get task instance branches corresponding to this repo"""
        if self._cache_branches is None:
            self._cache_branches = []
            increase, page = 0, 1
            while page == 1 or increase > 0:
                prev = len(self._cache_branches)
                self._cache_branches.extend(
                    self.api.repos.list_branches(
                        owner=self.org_gh,
                        repo=self.repo_name,
                        prefix=self.repo_name,
                        per_page=100,
                        page=page,
                    )
                )
                increase = len(self._cache_branches) - prev
                page += 1
            self._cache_branches = [
                b.name
                for b in self._cache_branches
                if b.name.startswith(self.repo_name)
            ]
        return self._cache_branches

    def _get_cached_test_paths(self) -> list[Path]:
        """Clone the repo, get all testing file paths relative to the repo directory, then clean up."""
        if self._cache_test_paths is None:
            with self._lock:  # Only one process enters this block at a time
                dir_path, cloned = self.clone()
                self._cache_test_paths = [
                    Path(os.path.relpath(os.path.join(root, file), self.repo_name))
                    for root, _, files in os.walk(Path(self.repo_name).resolve())
                    for file in files
                    if self._is_test_path(root, file)
                ]
                if cloned:
                    shutil.rmtree(dir_path)

        return self._cache_test_paths

    def _mirror_exists(self):
        """Check if mirror repository exists under organization"""
        if self._cache_mirror_exists is not True:
            try:
                self.api.repos.get(owner=self.org_gh, repo=self.repo_name)
                self._cache_mirror_exists = True
            except:
                self._cache_mirror_exists = False
        return self._cache_mirror_exists

    def build_image(self):
        """Build a Docker image (execution environment) for this repository profile."""
        env_dir = LOG_DIR_ENV / self.repo_name
        env_dir.mkdir(parents=True, exist_ok=True)
        dockerfile_path = env_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(self.dockerfile)
        with open(env_dir / "build_image.log", "w") as log_file:
            subprocess.run(
                f"docker build -f {dockerfile_path} --platform {self.pltf} --no-cache -t {self.image_name} .",
                check=True,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

    def create_mirror(self):
        """Create a mirror of this repository at the specified commit."""
        if self._mirror_exists():
            return
        if self.repo_name in os.listdir():
            shutil.rmtree(self.repo_name)
        self.api.repos.create_in_org(self.org_gh, self.repo_name)

        # Clone the repository
        subprocess.run(
            f"git clone git@github.com:{self.owner}/{self.repo}.git {self.repo_name}",
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Build the git commands
        git_cmds = [
            f"cd {self.repo_name}",
            f"git checkout {self.commit}",
        ]

        # Add submodule update if submodules exist
        if os.path.exists(os.path.join(self.repo_name, ".gitmodules")):
            git_cmds.append("git submodule update --init --recursive")

        # Add the rest of the commands
        git_cmds.extend(
            [
                "rm -rf .git",
                "git init",
                'git config user.name "swesmith"',
                'git config user.email "swesmith@anon.com"',
                "rm -rf .github/workflows",
                "rm -rf .github/dependabot.y*",
                "git add .",
                "git commit --no-gpg-sign -m 'Initial commit'",
                "git branch -M main",
                f"git remote add origin git@github.com:{self.mirror_name}.git",
                "git push -u origin main",
            ]
        )

        # Execute the commands
        subprocess.run(
            "; ".join(git_cmds),
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Clean up
        subprocess.run(
            f"rm -rf {self.repo_name}",
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def clone(self, dest: str | None = None) -> tuple[str, bool]:
        """Clone repository locally"""
        if not self._mirror_exists():
            raise ValueError(
                "Mirror clone repo must be created first (call .create_mirror)"
            )
        dest = self.repo_name if not dest else dest
        if not os.path.exists(dest):
            token = os.getenv("GITHUB_TOKEN")
            if token:
                base_url = (
                    f"https://x-access-token:{token}@github.com/{self.mirror_name}.git"
                )
            else:
                base_url = f"git@github.com:{self.mirror_name}.git"

            clone_cmd = (
                f"git clone {base_url}"
                if dest is None
                else f"git clone {base_url} {dest}"
            )
            subprocess.run(
                clone_cmd,
                check=True,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return dest, True
        else:
            return dest, False

    def extract_entities(
        self,
        dirs_exclude: list[str] = [],
        dirs_include: list[str] = [],
        exclude_tests: bool = True,
        max_entities: int = -1,
    ) -> list[CodeEntity]:
        """
        Extracts entities (functions, classes, etc.) from files in a directory.
        Args:
            directory_path (str): Path to the directory to scan.
            exclude_tests (bool): Whether to exclude test files and directories.
        Returns:
            List[CodeEntity]: List of CodeEntity objects containing entity information.
        """
        raise NotImplementedError

    def get_container(self, instance: dict) -> Container:
        """Return a docker container with the task instance initialized"""
        raise NotImplementedError("Subclasses must implement this method")

    def pull_image(self):
        """Pull the Docker image for this repository profile."""
        raise NotImplementedError

    def push_image(self, rebuild_image: bool = False):
        raise NotImplementedError

    def set_github_token(self, token: str):
        """Set a custom GitHub token and reset the API instance."""
        raise NotImplementedError

    ### END: Properties, Methods that *do not* require (re-)implementation ###

    ### START: Properties, Methods that *may* require (re-)implementation ###

    def _is_test_path(self, root: str, file: str) -> bool:
        """Check whether the file path corresponds to a testing related file"""
        if len(self.exts) > 1 and not any([file.endswith(ext) for ext in self.exts]):
            return False
        if file.lower().startswith("test") or file.rsplit(".", 1)[0].endswith("test"):
            return True
        dirs = root.split("/")
        if any([x in dirs for x in ["tests", "test", "specs"]]):
            return True
        return False

    def get_test_files(self, instance: dict) -> tuple[list[str], list[str]]:
        """Given an instance, return files corresponding to F2P, P2P test files"""
        return [], []

    def get_test_cmd(
        self, instance: dict, f2p_only: bool = False
    ) -> tuple[str, list[Path]]:
        assert instance[KEY_INSTANCE_ID].rsplit(".", 1)[0] == self.repo_name, (
            f"WARNING: {instance[KEY_INSTANCE_ID]} not from {self.repo_name}"
        )
        test_command = self.test_cmd

        if f2p_only:
            f2p_files, _ = self.get_test_files(instance)
            test_command += f" {' '.join(f2p_files)}"
            return test_command, f2p_files

        if self.min_testing and FAIL_TO_PASS in instance:
            f2p_files, p2p_files = self.get_test_files(instance)
            if len(f2p_files + p2p_files) > 0:
                test_command += f" {' '.join(f2p_files + p2p_files)}"
            return test_command, f2p_files + p2p_files

        if not self.min_testing or KEY_PATCH not in instance:
            # If min testing is not enabled or there's no patch
            # return test command as is (usually = run whole test suite)
            return test_command, []

        # Get all testing related file paths in the repo
        test_paths = self._get_cached_test_paths()

        # For PR Mirroring (SWE-bench style) instances
        if (
            INSTANCE_REF in instance
            and len(instance[INSTANCE_REF]["test_patch"].strip()) > 0
        ):
            # if test patch is available, use that information
            test_patch = instance[INSTANCE_REF]["test_patch"]
            rv = []
            for x in PatchSet(test_patch):
                for test_path in test_paths:
                    if str(test_path).endswith(x.path) or str(test_path).endswith(
                        Path(x.path).name
                    ):
                        rv.append(test_path)
            if len(rv) > 0:
                test_command += f" {' '.join([str(v) for v in rv])}"
                return test_command, rv

        # Identify relevant test files based on the patch
        patch_paths = [Path(f.path) for f in PatchSet(instance[KEY_PATCH])]
        rv = []
        for pp in patch_paths:
            for test_path in test_paths:
                # Check for common test file naming conventions first
                # If found, add to list and break
                common_test_names = [
                    f"test_{pp.stem}{pp.suffix}",
                    f"test{pp.stem}{pp.suffix}",
                    f"{pp.stem}_test{pp.suffix}",
                    f"{pp.stem}test{pp.suffix}",
                ]
                if any([str(test_path).endswith(name) for name in common_test_names]):
                    rv.append(test_path)
                    break
            else:
                for test_path in test_paths:
                    if pp.parent.name == test_path.parent.name:
                        # If similar testing folder found, add to list and break
                        rv.append(test_path.parent)
                        break
                    elif any(
                        [
                            test_path.stem
                            in {
                                f"test_{pp.parent.name}",
                                f"test{pp.parent.name}",
                                f"{pp.parent.name}_test",
                                f"{pp.parent.name}test",
                            }
                        ]
                    ):
                        rv.append(test_path)

        if len(rv) > 0:
            # Remove duplicates
            test_files = [x for x in rv if x.is_file()]
            final = [x for x in rv if not x.is_file()]
            for test_file in test_files:
                if os.path.dirname(test_file) not in final:
                    final.append(test_file)
            test_command += f" {' '.join(sorted([str(v) for v in set((final))]))}"

        return test_command, rv

    ### END: Properties, Methods that *may* require (re-)implementation ###

    ### START: Properties, Methods that require implementation ###

    owner: str = ""
    repo: str = ""
    commit: str = ""
    test_cmd: str = ""

    @abstractmethod
    def log_parser(self, log: str) -> dict[str, str]:
        """Parse test output logs and extract relevant information."""
        pass

    @property
    def dockerfile(self) -> str:
        """Return the Dockerfile path for this repository profile."""
        pass

    ### END: Properties, Methods that require implementation ###


### MARK: Profile Registry ###


class Registry(UserDict):
    """A registry mapping repo/mirror names to RepoProfile subclasses."""

    def __init__(self, github_token: str | None = None):
        super().__init__()
        self.github_token = github_token

    def register_profile(self, profile_class: type):
        """Register a RepoProfile subclass (except base types)."""
        # Skip base types
        if profile_class.__name__ in {
            "RepoProfile",
            "PythonProfile",
            "GoProfile",
            "RustProfile",
        }:
            # TODO: Update for new languages
            return
        # Create temporary instance to get properties
        p = profile_class()
        self.data[p.repo_name] = profile_class
        self.data[p.mirror_name] = profile_class

    def get(self, key: str) -> RepoProfile:
        """Get a profile class by mirror name or repo name."""
        cls = self.data.get(key)
        if cls is None:
            raise KeyError(f"No profile registered for key: {key}")
        profile = cls()
        if self.github_token:
            profile.set_github_token(self.github_token)
        return profile

    def get_from_inst(self, instance: dict) -> RepoProfile:
        """Get a profile class by a SWE-smith instance dict."""
        key = instance.get("repo", instance[KEY_INSTANCE_ID].rsplit(".", 1)[0])
        return self.get(key)

    def keys(self):
        return self.data.keys()

    def values(self):
        profiles = []
        for cls in set(self.data.values()):
            profile = cls()
            if self.github_token:
                profile.set_github_token(self.github_token)
            profiles.append(profile)
        return profiles

    def set_github_token(self, token: str):
        """Set GitHub token for all profiles retrieved from this registry."""
        self.github_token = token


# Global registry instance that can be shared across modules
registry = Registry()
