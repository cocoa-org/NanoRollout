import re
from typing import Any, Dict
import shlex


# XML Action parsing  (stolen from r2egym/agenthub/action/action.py)
_FN_PATTERN = re.compile(r"<function\s*=\s*([^>]+)>")
_PARAM_PATTERN = re.compile(
    r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>", re.DOTALL
)
_ACTION_BLOCK_PATTERN = re.compile(r"(?s)(<function=.*?</function>)")
_THINK_PATTERN = re.compile(r"(?s)(<think>.*?</think>)")


def _parse_xml_action(action_str: str) -> tuple[str, dict[str, str]]:
    """
    Parses a string of the form:

        <function=FUNCTION_NAME>
            <parameter=KEY>VALUE</parameter>
            ...
        </function>

    For example:
        <function=file_editor>
            <parameter=command>view</parameter>
            <parameter=path>./sympy/tensor/array/dense_ndim_array.py</parameter>
            <parameter=concise>True</parameter>
        </function>

    yields an Action with:
        function_name = "file_editor"
        parameters = {
            "command":  "view",
            "path":     "./sympy/tensor/array/dense_ndim_array.py",
            "concise":  "True"
        }
    """

    # Extract the function name: <function=...>
    fn_match = re.search(r"<function\s*=\s*([^>]+)>", action_str)
    function_name = fn_match.group(1).strip() if fn_match else ""

    # Extract parameters of the form: <parameter=KEY>VALUE</parameter>
    # DOTALL allows the captured VALUE to span multiple lines
    pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
    param_matches = re.findall(pattern, action_str, flags=re.DOTALL)
    params = {}
    for param_key, param_value in param_matches:
        param_key = param_key.strip()
        param_value = param_value.strip()
        params[param_key] = param_value

    return function_name, params


def _action_to_xml(function_name: str, parameters: dict[str, str]) -> str:
    """
    Returns an XML-like string representation of this action.

    Example:
        <function=file_editor>
            <parameter=command>view</parameter>
            <parameter=path>./sympy/tensor/array/dense_ndim_array.py</parameter>
            <parameter=concise>True</parameter>
        </function>
    """
    # Start with the function name
    xml_str = f"<function={function_name}>\n"

    # Add each parameter as <parameter=KEY>VALUE</parameter>
    for param_key, param_value in parameters.items():
        xml_str += f"  <parameter={param_key}>{param_value}</parameter>\n"

    xml_str += "</function>"
    return xml_str


def _action_to_bashcmd(function_name: str, parameters: dict[str, Any]) -> str:
    """
    Converts this action into a Bash command string.

    Examples:
        If function_name == "execute_bash" and parameters = {
            "command": "search_dir",
            "search_term": "foo"
        }
        then this returns:
        execute_bash search_dir --search_term 'foo'

        If function_name == "file_editor" and parameters = {
            "command": "view",
            "path": "./some/path.py",
            "concise": "True"
        }
        then this returns:
        file_editor view --path './some/path.py' --concise 'True'
    """
    if not function_name:
        return ""
    elif function_name == "finish" or function_name == "submit":
        return "echo '<<<Finished>>>'"

    # Start building the command
    cmd_parts = [shlex.quote(function_name)]

    # If there's a 'command' parameter, put that next
    base_command = parameters.get("command")
    if base_command is not None:
        cmd_parts.append(shlex.quote(base_command))

    # Append all other parameters
    for param_key, param_value in parameters.items():
        if param_key == "command":
            continue

        # Safely quote the param_value
        param_value_quoted = shlex.quote(str(param_value))
        cmd_parts.append(f"--{param_key}")
        cmd_parts.append(param_value_quoted)

    return " ".join(cmd_parts)


if __name__ == "__main__":
    # Sample usage

    # Example 1
    xml_1 = """
    <function=file_editor>
      <parameter=command>view</parameter>
      <parameter=path>./sympy/tensor/array/dense_ndim_array.py</parameter>
      <parameter=concise>True</parameter>
    </function>
    """
    action1 = _parse_xml_action(xml_1)
    print("[Example 1] Action as dict:", action1)
    print("[Example 1] Action as bashcmd:", _action_to_bashcmd(action1[0], action1[1]), "\n")

    # Example 2
    xml_2 = """
    <function=execute_bash>
      <parameter=command>search_dir</parameter>
      <parameter=search_term>class ImmutableDenseNDimArray</parameter>
    </function>
    """
    action2 = _parse_xml_action(xml_2)
    print("[Example 2] Action as dict:", action2)
    print("[Example 2] Action as bashcmd:", _action_to_bashcmd(action2[0], action2[1]))
