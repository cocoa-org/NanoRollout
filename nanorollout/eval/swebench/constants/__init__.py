from enum import Enum


class TestStatus(Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    XFAIL = "XFAIL"


class EvalType(Enum):
    PASS_AND_FAIL = "PASS_AND_FAIL"
    FAIL_ONLY = "FAIL_ONLY"


class ResolvedStatus(Enum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    NO = "NO"


FAIL_TO_PASS = "FAIL_TO_PASS"
PASS_TO_PASS = "PASS_TO_PASS"
FAIL_TO_FAIL = "FAIL_TO_FAIL"
PASS_TO_FAIL = "PASS_TO_FAIL"

KEY_INSTANCE_ID = "instance_id"
KEY_MODEL = "model_name_or_path"
KEY_PREDICTION = "model_patch"
KEY_PATCH = "patch"

DOCKER_PATCH = "/tmp/patch.diff"
DOCKER_USER = "root"
DOCKER_WORKDIR = "/testbed"
ENV_NAME = "testbed"

LOG_REPORT = "report.json"
LOG_INSTANCE = "run_instance.log"
LOG_TEST_OUTPUT = "test_output.txt"
UTF8 = "utf-8"

START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"

NON_TEST_EXTS = [
    ".json",
    ".png",
    "csv",
    ".txt",
    ".md",
    ".jpg",
    ".jpeg",
    ".pkl",
    ".yml",
    ".yaml",
    ".toml",
]

FAIL_ONLY_REPOS = {
    "chartjs/Chart.js",
    "markedjs/marked",
    "processing/p5.js",
}

# ---- Derived maps (imported after all base constants are defined) ----

from .python import MAP_REPO_VERSION_TO_SPECS_PY
from .swesmith import (
    MAP_REPO_TO_EXT_SWESMITH,
    MAP_REPO_VERSION_TO_SPECS_SWESMITH,
    REPOS_SWESMITH,
)

REPOS_PY = [
    "astropy/astropy",
    "bokeh/bokeh",
    "conan-io/conan",
    "dask/dask",
    "dbt-labs/dbt-core",
    "django/django",
    "facebookresearch/hydra",
    "getmoto/moto",
    "hypothesisworks/hypothesis",
    "iterative/dvc",
    "marshmallow-code/marshmallow",
    "matplotlib/matplotlib",
    "modin-project/modin",
    "mwaskom/seaborn",
    "pandas-dev/pandas",
    "pallets/flask",
    "project-monai/monai",
    "psf/requests",
    "pydantic/pydantic",
    "pvlib/pvlib-python",
    "pydata/xarray",
    "pydicom/pydicom",
    "pylint-dev/astroid",
    "pylint-dev/pylint",
    "python/mypy",
    "pytest-dev/pytest",
    "pyvista/pyvista",
    "scikit-learn/scikit-learn",
    "spyder-ide/spyder",
    "sphinx-doc/sphinx",
    "sqlfluff/sqlfluff",
    "swe-bench/humaneval",
    "sympy/sympy",
]

REPOS = sorted(set(REPOS_PY) | set(REPOS_SWESMITH))

MAP_REPO_TO_EXT = {
    **{k: "py" for k in REPOS_PY},
    **{k.lower(): v for k, v in MAP_REPO_TO_EXT_SWESMITH.items()},
    **MAP_REPO_TO_EXT_SWESMITH,
}

MAP_REPO_VERSION_TO_SPECS = {
    **MAP_REPO_VERSION_TO_SPECS_PY,
    **MAP_REPO_VERSION_TO_SPECS_SWESMITH,
}
