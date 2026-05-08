from . import python as py_parsers
from .python import MAP_REPO_TO_PARSER_PY
from ..constants.swesmith import (
    MAP_REPO_TO_PARSER_SWESMITH,
    PARSER_NAME_MAP_SWESMITH,
)


def _normalize_parser_output(fn):
    def _wrapped(log: str, test_spec=None) -> dict[str, str]:
        rv = fn(log, test_spec) if fn.__code__.co_argcount >= 2 else fn(log)
        if not isinstance(rv, dict):
            return {}
        return {str(k): str(v) for k, v in rv.items()}

    _wrapped.__name__ = fn.__name__
    return _wrapped


PARSER_NAME_MAP = {
    name: _normalize_parser_output(obj)
    for name, obj in vars(py_parsers).items()
    if name.startswith("parse_log_") and callable(obj)
}
PARSER_NAME_MAP.update(PARSER_NAME_MAP_SWESMITH)

MAP_REPO_TO_PARSER = {**MAP_REPO_TO_PARSER_PY, **MAP_REPO_TO_PARSER_SWESMITH}


def get_parser_by_repo(repo: str):
    """
    Get the appropriate parser for the repository.
    """
    repo_key = repo.lower()
    parser = MAP_REPO_TO_PARSER.get(repo_key) or MAP_REPO_TO_PARSER.get(repo)
    if parser:
        return parser
    return MAP_REPO_TO_PARSER_PY["default"]


def get_parser_by_fn_name(fn_name: str):
    """
    Get the appropriate parser for the function name.
    """
    if not fn_name:
        raise ValueError("fn_name is required")
    return PARSER_NAME_MAP.get(fn_name)
