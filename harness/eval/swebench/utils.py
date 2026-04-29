import re

from unidiff import PatchSet


def get_modified_files(patch: str) -> list[str]:
    """
    Get the list of modified files in a patch.
    """
    source_files = []
    for file in PatchSet(patch):
        if file.source_file != "/dev/null":
            source_files.append(file.source_file)
    # remove 'a/' prefix
    source_files = [x[2:] for x in source_files if x.startswith("a/")]
    return source_files


def ansi_escape(text: str) -> str:
    """
    Remove ANSI escape sequences from text.
    """
    return re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub("", text)
