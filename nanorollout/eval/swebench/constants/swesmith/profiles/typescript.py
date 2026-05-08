from dataclasses import dataclass, field

from nanorollout.eval.swebench.constants import ENV_NAME, TestStatus
from .base import RepoProfile, registry
from .javascript import parse_log_jest, parse_log_vitest


@dataclass
class TypeScriptProfile(RepoProfile):
    """
    Profile for TypeScript repositories.
    """

    exts: list[str] = field(default_factory=lambda: [".ts", ".tsx"])

    def extract_entities(
        self,
        dirs_exclude: list[str] | None = None,
        dirs_include: list[str] = [],
        exclude_tests: bool = True,
        max_entities: int = -1,
    ) -> list:
        """
        Override to exclude TypeScript/JavaScript build artifacts by default.
        """
        if dirs_exclude is None:
            dirs_exclude = [
                "dist",
                "build",
                "node_modules",
                "coverage",
                ".next",
                "out",
                "examples",
                "docs",
                "bin",
                "lib",  # Common TS output directory
            ]

        return super().extract_entities(
            dirs_exclude=dirs_exclude,
            dirs_include=dirs_include,
            exclude_tests=exclude_tests,
            max_entities=max_entities,
        )


def default_npm_install_dockerfile(mirror_name: str, node_version: str = "20") -> str:
    """Default Dockerfile for TypeScript projects using npm."""
    return f"""FROM node:{node_version}-bullseye
RUN apt update && apt install -y git
RUN git clone https://github.com/{mirror_name} /{ENV_NAME}
WORKDIR /{ENV_NAME}
RUN npm install
"""


def default_pnpm_install_dockerfile(mirror_name: str, node_version: str = "20") -> str:
    """Default Dockerfile for TypeScript projects using pnpm."""
    return f"""FROM node:{node_version}-bullseye
RUN apt update && apt install -y git
RUN npm install -g pnpm
RUN git clone https://github.com/{mirror_name} /{ENV_NAME}
WORKDIR /{ENV_NAME}
RUN pnpm install
"""


@dataclass
class CrossEnv9951937a(TypeScriptProfile):
    owner: str = "kentcdodds"
    repo: str = "cross-env"
    commit: str = "9951937a7d3d4a1ea7bd2ce3133bcfb687125813"
    test_cmd: str = "npm test"

    @property
    def dockerfile(self) -> str:
        return f"""FROM node:18-slim
RUN apt-get update && apt-get install -y git procps && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/{self.mirror_name} /{ENV_NAME}
WORKDIR /{ENV_NAME}
RUN npm install
"""

    def log_parser(self, log: str) -> dict[str, str]:
        return parse_log_vitest(log)


@dataclass
class Trpc2f40ba93(TypeScriptProfile):
    owner: str = "trpc"
    repo: str = "trpc"
    commit: str = "2f40ba935ad7f7d29eec3f9c45d353450b43e852"
    test_cmd: str = "pnpm test"

    @property
    def dockerfile(self) -> str:
        return f"""FROM node:22
RUN apt-get update && apt-get install -y git procps && rm -rf /var/lib/apt/lists/*
RUN npm install -g pnpm
RUN git clone https://github.com/{self.mirror_name} /{ENV_NAME}
WORKDIR /{ENV_NAME}
RUN pnpm install
"""

    def log_parser(self, log: str) -> dict[str, str]:
        return parse_log_vitest(log)


@dataclass
class ClassValidator977d2c70(TypeScriptProfile):
    owner: str = "typestack"
    repo: str = "class-validator"
    commit: str = "977d2c707930db602b6450d0c03ee85c70756f1f"
    test_cmd: str = "npm test"

    @property
    def dockerfile(self) -> str:
        return f"""FROM node:18-slim
RUN apt-get update && apt-get install -y git procps && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/{self.mirror_name} /{ENV_NAME}
WORKDIR /{ENV_NAME}
RUN npm install
"""

    def log_parser(self, log: str) -> dict[str, str]:
        return parse_log_jest(log)


# Register all TypeScript profiles with the global registry
for name, obj in list(globals().items()):
    if (
        isinstance(obj, type)
        and issubclass(obj, TypeScriptProfile)
        and obj.__name__ != "TypeScriptProfile"
    ):
        registry.register_profile(obj)
