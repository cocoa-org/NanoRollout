from .constants import (
    MAP_REPO_TO_EXT,
    START_TEST_OUTPUT,
    END_TEST_OUTPUT,
    FAIL_TO_PASS,
    PASS_TO_PASS,
)
from .repo_specs import get_repo_eval_commands
from .utils import get_modified_files


def test_files_from_instance_swe_smith(instance: dict) -> list[str]:
    """Extract test file paths from FAIL_TO_PASS / PASS_TO_PASS lists.

    SWE-smith instances don't carry a ``test_patch`` (the tests are already
    baked into the repo image).  We still need the file paths so the eval
    script can ``git checkout HEAD~1 <files>`` to reset them between
    the agent's edits and the test run.  The paths are derived by taking the
    portion before the first ``::`` in each pytest node-id, e.g.
    ``"tests/test_utils.py::TestFoo::test_bar"`` -> ``"tests/test_utils.py"``.
    """
    test_files: set[str] = set()
    for key in (FAIL_TO_PASS, PASS_TO_PASS):
        for test_case in instance.get(key):
            candidate = test_case.split("::", 1)[0].strip()
            if candidate:
                test_files.add(candidate)
    return sorted(test_files)


def _resolve_test_files(instance: dict, test_patch: str) -> list[str]:
    """Return the list of test files that the eval script should reset.

    For SWE-bench instances ``test_patch`` is a git diff that adds acceptance
    tests, so we parse modified paths from it.  For SWE-smith instances
    ``test_patch`` is empty -- fall back to extracting paths from the
    FAIL_TO_PASS / PASS_TO_PASS lists stored in the instance dict.
    """
    test_files = get_modified_files(test_patch)
    if test_files:
        return test_files
    return test_files_from_instance_swe_smith(instance)


def _resolve_reset_ref(instance: dict, base_commit: str, test_patch: str) -> str:
    """Return the git ref used to reset test files before/after running tests.

    For SWE-smith (no test_patch) the container history at eval time is:
      HEAD   = agent changes (committed by get_git_diff)
      HEAD~1 = instance branch tip (tests removed so agent can't see them)
      HEAD~2 = initial commit with full source + all test files
    We need HEAD~2 to restore the complete test files.
    """
    if not test_patch.strip():
        return "HEAD~2"
    return base_commit


def _make_apply_test_patch_command(test_patch: str, *, common: bool) -> str:
    if not test_patch.strip():
        return "echo 'No test patch'"

    HEREDOC_DELIMITER = "EOF_114329324912"
    if common:
        return (
            "git apply --verbose --reject - "
            f"<<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
        )
    return (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    )


def make_eval_script(
    instance: dict,
    repo_directory: str,
    base_commit: str,
    env_name: str,
    test_patch: str,
    test_cmd: str,
    build_cmds: list[str] = None,
) -> str:
    """
    Generate the evaluation script for a given instance.

    Args:
        instance: The instance dictionary (must contain 'repo' and 'version')
        repo_directory: Path to the repository in the container
        base_commit: The commit hash to reset to
        env_name: Name of the environment (e.g. conda env name)
        test_patch: The acceptance test patch content
        test_cmd: The command to run tests (e.g. "pytest")
        build_cmds: Optional list of build commands (for C/Java)
    """
    repo = instance["repo"]
    repo_key = repo.lower()
    ext = MAP_REPO_TO_EXT.get(repo) or MAP_REPO_TO_EXT.get(repo_key, "py")

    if ext == "py":
        return make_eval_script_py(
            instance, repo_directory, base_commit, env_name, test_patch, test_cmd
        )
    else:
        return make_eval_script_common(
            instance,
            repo_directory,
            base_commit,
            env_name,
            test_patch,
            test_cmd,
            build_cmds,
        )


def make_eval_script_py(
    instance, repo_directory, base_commit, env_name, test_patch, test_cmd
) -> str:
    test_files = _resolve_test_files(instance, test_patch)
    reset_ref = _resolve_reset_ref(instance, base_commit, test_patch)

    reset_tests_command = (
        f"git checkout {reset_ref} {' '.join(test_files)}"
        if test_files
        else "echo 'No test files to reset'"
    )

    apply_test_patch_command = _make_apply_test_patch_command(test_patch, common=False)

    eval_commands = get_repo_eval_commands(
        instance.get("repo"), instance.get("version"), instance
    )

    eval_steps = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
        *eval_commands,
        f"git config --global --add safe.directory {repo_directory}",
        reset_tests_command,
        apply_test_patch_command,
        f"echo '{START_TEST_OUTPUT}'",
        test_cmd,
        f"echo '{END_TEST_OUTPUT}'",
        reset_tests_command,  # Revert tests
    ]

    return "\n".join(eval_steps)


def make_eval_script_common(
    instance,
    repo_directory,
    base_commit,
    env_name,
    test_patch,
    test_cmd,
    build_cmds=None,
) -> str:
    test_files = _resolve_test_files(instance, test_patch)
    reset_ref = _resolve_reset_ref(instance, base_commit, test_patch)

    reset_tests_command = (
        f"git checkout {reset_ref} {' '.join(test_files)}"
        if test_files
        else "echo 'No test files to reset'"
    )

    apply_test_patch_command = _make_apply_test_patch_command(test_patch, common=True)

    eval_commands = [
        f"cd {repo_directory}",
        f"git config --global --add safe.directory {repo_directory}",
        reset_tests_command,
        apply_test_patch_command,
    ]

    if build_cmds:
        eval_commands.extend(build_cmds)

    eval_commands.extend(
        [
            f"echo '{START_TEST_OUTPUT}'",
            test_cmd,
            f"echo '{END_TEST_OUTPUT}'",
            reset_tests_command,
        ]
    )

    return "\n".join(eval_commands)
