import json

from .constants import (
    END_TEST_OUTPUT,
    FAIL_ONLY_REPOS,
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    START_TEST_OUTPUT,
    EvalType,
    ResolvedStatus,
    TestStatus,
)
from .log_parsers import get_parser_by_fn_name, get_parser_by_repo
from .repo_specs import get_repo_specs
from .test_spec import TestSpec


def _coerce_test_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
    return [str(value)]


def _test_passed(case: str, status_map: dict[str, str]) -> bool:
    return case in status_map and status_map[case] in [
        TestStatus.PASSED.value,
        TestStatus.XFAIL.value,
    ]


def _test_failed(case: str, status_map: dict[str, str]) -> bool:
    return case not in status_map or status_map[case] in [
        TestStatus.FAILED.value,
        TestStatus.ERROR.value,
    ]


def _get_eval_tests_report(
    eval_status_map: dict[str, str],
    gold_results: dict[str, list[str]],
    eval_type: EvalType,
    *,
    calculate_to_fail: bool = False,
) -> dict[str, dict[str, list[str]]]:
    def check_pass_and_fail(test_case, success, failed):
        if _test_passed(test_case, eval_status_map):
            success.append(test_case)
        elif _test_failed(test_case, eval_status_map):
            failed.append(test_case)

    def check_fail_only(test_case, success, failed):
        if (
            test_case in eval_status_map
            and eval_status_map[test_case] == TestStatus.FAILED.value
        ):
            failed.append(test_case)
        else:
            success.append(test_case)

    check_test_case = (
        check_pass_and_fail if eval_type == EvalType.PASS_AND_FAIL else check_fail_only
    )

    f2p_success: list[str] = []
    f2p_failure: list[str] = []
    for test_case in gold_results.get(FAIL_TO_PASS, []):
        check_test_case(test_case, f2p_success, f2p_failure)

    p2p_success: list[str] = []
    p2p_failure: list[str] = []
    for test_case in gold_results.get(PASS_TO_PASS, []):
        check_test_case(test_case, p2p_success, p2p_failure)

    results = {
        FAIL_TO_PASS: {
            "success": f2p_success,
            "failure": f2p_failure,
        },
        PASS_TO_PASS: {
            "success": p2p_success,
            "failure": p2p_failure,
        },
    }

    f2f_success: list[str] = []
    f2f_failure: list[str] = []
    p2f_success: list[str] = []
    p2f_failure: list[str] = []
    if calculate_to_fail:
        for test_case in gold_results.get(FAIL_TO_FAIL, []):
            check_test_case(test_case, f2f_success, f2f_failure)
        for test_case in gold_results.get(PASS_TO_FAIL, []):
            check_test_case(test_case, p2f_success, p2f_failure)

    results.update(
        {
            FAIL_TO_FAIL: {
                "success": f2f_success,
                "failure": f2f_failure,
            },
            PASS_TO_FAIL: {
                "success": p2f_success,
                "failure": p2f_failure,
            },
        }
    )
    return results


def _compute_rate(report: dict[str, list[str]]) -> float:
    total = len(report.get("success", [])) + len(report.get("failure", []))
    if total == 0:
        return 1.0
    return len(report.get("success", [])) / total


def get_eval_report(
    instance: dict,
    log_content: str,
    *,
    test_spec: TestSpec | None = None,
) -> dict:
    """
    Generate a report of evaluation results.

    Args:
        instance: The instance dict containing 'repo', 'FAIL_TO_PASS', 'PASS_TO_PASS'
        log_content: The content of the evaluation log (stdout from docker)
    """
    repo = test_spec.repo if test_spec else instance["repo"]
    parser = None
    parser_name = None

    # 1) Instance-level explicit parser (SWE-rebench install_config).
    install_config = instance.get("install_config")
    if isinstance(install_config, dict):
        parser_name = install_config.get("log_parser")
        if parser_name:
            parser = get_parser_by_fn_name(parser_name)

    # 2) Repo spec parser from constants map.
    if not parser:
        repo_specs = get_repo_specs(repo, instance.get("version"), None)
        if isinstance(repo_specs, dict):
            parser_name = repo_specs.get("log_parser")
            if parser_name:
                parser = get_parser_by_fn_name(str(parser_name))

    # 3) Repo-based fallback parser.
    if not parser:
        parser = get_parser_by_repo(repo)

    if not parser:
        raise ValueError(
            f"No parser found for repo={repo}, version={instance.get('version')}, parser_name={parser_name}"
        )

    has_test_output = START_TEST_OUTPUT in log_content and END_TEST_OUTPUT in log_content
    # Extract test output part from the log
    try:
        if has_test_output:
            test_content = (
                log_content.split(START_TEST_OUTPUT)[1]
                .split(END_TEST_OUTPUT)[0]
            )
            status_map = parser(test_content, test_spec)
        else:
            # Test output missing (possibly patch failed or tests did not run)
            print(f"No test output found in log: {log_content}")
            status_map = {}
    except Exception as e:
        print(f"Error parsing log: {e}")
        status_map = {}

    # Calculate metrics
    # 1. FAIL_TO_PASS: Tests that failed before but should pass now (Resolution)
    if test_spec:
        fail_to_pass = _coerce_test_list(test_spec.FAIL_TO_PASS)
        pass_to_pass = _coerce_test_list(test_spec.PASS_TO_PASS)
        fail_to_fail = _coerce_test_list(getattr(test_spec, FAIL_TO_FAIL, []))
        pass_to_fail = _coerce_test_list(getattr(test_spec, PASS_TO_FAIL, []))
    else:
        fail_to_pass = _coerce_test_list(instance.get(FAIL_TO_PASS))
        pass_to_pass = _coerce_test_list(instance.get(PASS_TO_PASS))
        fail_to_fail = _coerce_test_list(instance.get(FAIL_TO_FAIL))
        pass_to_fail = _coerce_test_list(instance.get(PASS_TO_FAIL))

    eval_type = (
        EvalType.FAIL_ONLY
        if repo in FAIL_ONLY_REPOS
        else EvalType.PASS_AND_FAIL
    )
    eval_ref = {
        FAIL_TO_PASS: fail_to_pass,
        PASS_TO_PASS: pass_to_pass,
        FAIL_TO_FAIL: fail_to_fail,
        PASS_TO_FAIL: pass_to_fail,
    }

    report = _get_eval_tests_report(
        status_map,
        eval_ref,
        eval_type,
        calculate_to_fail=bool(fail_to_fail or pass_to_fail),
    )

    # Determine resolution status
    # Full resolution means all F2P passed AND all P2P passed
    f2p_rate = _compute_rate(report[FAIL_TO_PASS])
    p2p_rate = _compute_rate(report[PASS_TO_PASS])

    resolved_status = ResolvedStatus.NO
    if f2p_rate == 1.0 and p2p_rate == 1.0:
        resolved_status = ResolvedStatus.FULL
    elif f2p_rate > 0 and p2p_rate == 1.0:
        resolved_status = ResolvedStatus.PARTIAL

    metrics = {
        FAIL_TO_PASS: {**report[FAIL_TO_PASS], "rate": f2p_rate},
        PASS_TO_PASS: {**report[PASS_TO_PASS], "rate": p2p_rate},
        FAIL_TO_FAIL: {
            **report[FAIL_TO_FAIL],
            "rate": _compute_rate(report[FAIL_TO_FAIL]),
        },
        PASS_TO_FAIL: {
            **report[PASS_TO_FAIL],
            "rate": _compute_rate(report[PASS_TO_FAIL]),
        },
    }

    return {
        "instance_id": (
            test_spec.instance_id if test_spec else instance.get("instance_id")
        ),
        "resolved_status": resolved_status.value,
        "patch_successfully_applied": has_test_output,
        "metrics": metrics,
        "test_status_map": status_map,
    }
