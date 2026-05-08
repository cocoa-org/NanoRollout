import re

from dataclasses import dataclass, field
from nanorollout.eval.swebench.constants import TestStatus
from .base import RepoProfile, registry, ENV_NAME


DEFAULT_CPP_BUG_GEN_DIRS_EXCLUDE = [
    # Docs / metadata.
    "/doc",
    "/docs",
    # Examples / benchmarks are typically not covered by ctest.
    "/bench",
    "/benchmark",
    "/example",
    "/examples",
    # Build / tooling.
    "/cmake",
    "/scripts",
    "/tools",
]


@dataclass
class CppProfile(RepoProfile):
    """
    Profile for C++ repositories.
    """

    exts: list[str] = field(
        default_factory=lambda: [".cpp", ".cc", ".cxx", ".h", ".hpp"]
    )
    # Exclude directories that are typically not built/executed by unit tests.
    bug_gen_dirs_exclude: list[str] = field(
        default_factory=lambda: list(DEFAULT_CPP_BUG_GEN_DIRS_EXCLUDE)
    )

    def extract_entities(
        self,
        dirs_exclude: list[str] | None = None,
        dirs_include: list[str] = [],
        exclude_tests: bool = True,
        max_entities: int = -1,
    ) -> list:
        if dirs_exclude is None:
            dirs_exclude = []
        merged_excludes = [*dirs_exclude, *self.bug_gen_dirs_exclude]
        return super().extract_entities(
            dirs_exclude=merged_excludes,
            dirs_include=dirs_include,
            exclude_tests=exclude_tests,
            max_entities=max_entities,
        )


@dataclass
class Catch29b3f508a(CppProfile):
    owner: str = "catchorg"
    repo: str = "Catch2"
    commit: str = "9b3f508a1b1579f5366cf83d19822cb395f23528"
    test_cmd: str = (
        "cd build && cmake --build . -j$(nproc) && ctest --output-on-failure --verbose"
    )
    timeout: int = 300  # 5 minutes - allows time for incremental rebuild + 71 tests
    # Exclude directories not used in cmake build or not covered by tests
    bug_gen_dirs_exclude: list[str] = field(
        default_factory=lambda: [
            *DEFAULT_CPP_BUG_GEN_DIRS_EXCLUDE,
            "/extras",  # Amalgamated single-file version (not used in cmake build)
            "/third_party",  # Bundled third-party libraries (not tested)
            "/fuzzing",  # Fuzzing harness code (not library code)
        ]
    )

    @property
    def dockerfile(self):
        return f"""FROM gcc:12
RUN apt-get update && apt-get install -y \
    libbrotli-dev libcurl4-openssl-dev \
    clang build-essential cmake \
    python3 python3-dev python3-pip

RUN git clone https://github.com/{self.mirror_name} /{ENV_NAME}
WORKDIR /{ENV_NAME}
RUN mkdir build && cd build \
    && cmake .. -DCATCH_DEVELOPMENT_BUILD=ON \
    && make all \
    && ctest"""

    def log_parser(self, log: str) -> dict[str, str]:
        test_status_map = {}
        re_passes = [
            re.compile(r"^-- Performing Test (.+) - Success$", re.IGNORECASE),
            re.compile(
                r"^\d+/\d+ Test\s+#\d+: (.+) \.+\s+ Passed\s+.+$", re.IGNORECASE
            ),
        ]
        re_fails = [
            re.compile(r"^-- Performing Test (.+) - Failed$", re.IGNORECASE),
            re.compile(
                r"^\d+/\d+ Test\s+#\d+: (.+) \.+\*\*\*Failed\s+.+$", re.IGNORECASE
            ),
        ]
        re_skips = [
            re.compile(r"^-- Performing Test (.+) - skipped$", re.IGNORECASE),
        ]

        for line in log.splitlines():
            line = line.strip().lower()
            if not line:
                continue

            for re_pass in re_passes:
                pass_match = re_pass.match(line)
                if pass_match:
                    test = pass_match.group(1)
                    test_status_map[test] = TestStatus.PASSED.value

            for re_fail in re_fails:
                fail_match = re_fail.match(line)
                if fail_match:
                    test = fail_match.group(1)
                    test_status_map[test] = TestStatus.FAILED.value

            for re_skip in re_skips:
                skip_match = re_skip.match(line)
                if skip_match:
                    test = skip_match.group(1)
                    test_status_map[test] = TestStatus.SKIPPED.value

        return test_status_map


registry.register_profile(Catch29b3f508a)
