"""Build SWE-smith constant maps from the profile registry."""

import sys

from .profiles import registry


def _normalize_parser(fn):
    """Adapt a log-parser to a uniform ``(log, test_spec=None) -> dict[str, str]`` signature.

    Profile parsers have inconsistent signatures: standalone functions like
    ``parse_log_jest(log)`` accept one arg, while some accept two.  Callers
    (e.g. ``grading.get_eval_report``) always pass two positional args.

    This wrapper bridges the gap by trying ``fn(log, test_spec)`` first and
    falling back to ``fn(log)`` on TypeError, and coerces the return value
    to ``dict[str, str]`` for safety.
    """
    def _wrapped(log, test_spec=None):
        try:
            rv = fn(log, test_spec)
        except TypeError:
            # Parser only accepts (log,) — retry without test_spec.
            rv = fn(log)
        if not isinstance(rv, dict):
            return {}
        return {str(k): str(v) for k, v in rv.items()}
    _wrapped.__name__ = getattr(fn, "__name__", "parser")
    return _wrapped


def _build_maps():
    repo_version_specs = {}
    repo_to_ext = {}
    repo_to_parser = {}
    parser_name_map = {}

    seen = set()
    for profile_cls in registry.data.values():
        if id(profile_cls) in seen:
            continue
        seen.add(id(profile_cls))

        profile = profile_cls()
        specs = {"test_cmd": profile.test_cmd}
        parser = _normalize_parser(profile.log_parser)
        ext = profile.exts[0].lstrip(".") if profile.exts else ""

        for key in (profile.repo_name, profile.mirror_name):
            repo_version_specs[key] = {"default": specs}
            if ext:
                repo_to_ext[key] = ext
            repo_to_parser[key] = parser

    # Collect standalone parse_log_* functions from already-loaded profile modules.
    pkg_prefix = __name__.rsplit(".", 1)[0] + ".profiles."
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith(pkg_prefix) or mod is None:
            continue
        for attr in dir(mod):
            if attr.startswith("parse_log_") and callable(getattr(mod, attr)):
                parser_name_map[attr] = _normalize_parser(getattr(mod, attr))

    return repo_version_specs, repo_to_ext, repo_to_parser, parser_name_map


(
    MAP_REPO_VERSION_TO_SPECS_SWESMITH,
    MAP_REPO_TO_EXT_SWESMITH,
    MAP_REPO_TO_PARSER_SWESMITH,
    PARSER_NAME_MAP_SWESMITH,
) = _build_maps()

REPOS_SWESMITH = sorted(MAP_REPO_VERSION_TO_SPECS_SWESMITH.keys())
