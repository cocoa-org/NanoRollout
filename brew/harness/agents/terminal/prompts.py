"""Prompt templates for Terminal Bench agents."""

SYSTEM_TEMPLATE = """\
You are a helpful assistant that can interact multiple times with a computer shell to solve tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
THOUGHT: Your reasoning and analysis here

```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected.
"""

INSTANCE_TEMPLATE = """\
<task_description>
{{task}}
</task_description>

<instructions>
# Task Instructions

## Overview
You are working in a Linux terminal environment. Your goal is to complete the task described above by issuing shell commands.

IMPORTANT: This is an interactive process where you will think and issue ONE command, see its result, then think and issue your next command.

For each response:
1. Include a THOUGHT section explaining your reasoning
2. Provide exactly ONE bash command to execute

## Recommended Workflow
1. Read the task carefully and explore the environment
2. Plan your approach
3. Execute commands step by step
4. Verify your solution is correct

## Command Execution Rules
You are operating in an environment where
1. You write a single command
2. The system executes that command in a subshell
3. You see the result
4. You write your next command

Each response should include:
1. A **THOUGHT** section where you explain your reasoning and plan
2. A single bash code block with your command

Format your responses like this:

<format_example>
THOUGHT: Here I explain my reasoning process, analysis of the current situation,
and what I'm trying to accomplish with the command below.

```bash
your_command_here
```
</format_example>

Commands must be specified in a single bash code block:

```bash
your_command_here
```

**CRITICAL REQUIREMENTS:**
- Your response SHOULD include a THOUGHT section explaining your reasoning
- Your response MUST include EXACTLY ONE bash code block
- This bash block MUST contain EXACTLY ONE command (or a set of commands connected with && or ||)
- If you include zero or multiple bash blocks, or no command at all, YOUR RESPONSE WILL FAIL
- Do NOT try to run multiple independent commands in separate blocks in one response
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

## Environment Details
- You have a full Linux shell environment
- Always use non-interactive flags (-y, -f) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- If a command isn't available, you can install it

## Completion
When you have completed the task and verified the result, run:
```bash
echo "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
```
</instructions>\
"""

JSON_PROMPT_TEMPLATE = """\
You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {{
      "keystrokes": "ls -la\\n",
      "duration": 0.1
    }}
  ],
  "task_complete": true
}}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete (defaults to false if not present)

Command object structure:
- "keystrokes": String containing the exact keystrokes to send to the terminal (required)
- "duration": Number of seconds to wait for the command to complete before the next command will be executed (defaults to 1.0 if not present)

IMPORTANT: The text inside "keystrokes" will be used completely verbatim as keystrokes. Write commands exactly as you want them sent to the terminal:
- For regular shell commands you must end every command with a newline (\\n) or it will not execute.
- Special key sequences (Ctrl+C, Tab, arrow keys, etc.) use tmux-style key names and MUST be sent as a keystroke by themselves with NO trailing newline. Examples:
  - {{"keystrokes": "C-c", "duration": 0.1}} sends Ctrl+C
  - {{"keystrokes": "C-d", "duration": 0.1}} sends Ctrl+D
  - {{"keystrokes": "Tab", "duration": 0.1}} sends Tab
  - {{"keystrokes": "Up", "duration": 0.1}} sends the up-arrow key
  - {{"keystrokes": "Enter", "duration": 0.1}} sends a bare Enter key
  - DO NOT write "C-c\\n" - that types the literal text "C-c" followed by Enter, which does NOT send a Ctrl+C signal.

The "duration" attribute specifies the number of seconds to wait for the command to complete (default: 1.0) before the next command will be executed. On immediate tasks (e.g., cd, ls, echo, cat) set a duration of 0.1 seconds. On slow commands set an appropriate duration as you determine necessary.

It is better to set a smaller duration than a longer duration. It is always possible to wait again if the prior output has not finished, by running {{"keystrokes": "", "duration": 10.0}} on subsequent requests to wait longer. Never wait longer than 60 seconds; prefer to poll to see intermediate result status.

Important notes:
- Each command's keystrokes are sent exactly as written to the terminal
- Do not include extra whitespace before or after the keystrokes unless it's part of the intended command
- Extra text before or after the JSON will generate warnings but be tolerated
- The JSON must be valid - use proper escaping for quotes and special characters within strings
- Commands array can be empty if you want to wait without taking action

Task Description:
{instruction}

Current terminal state:
{terminal_state}
"""

XML_PROMPT_TEMPLATE = """\
You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as XML with the following structure:

<response>
<analysis>
Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?
</analysis>
<plan>
Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.
</plan>
<commands>
<keystrokes duration="0.1">ls -la
</keystrokes>
</commands>
<task_complete>true</task_complete>
</response>

Required sections:
- <analysis>: Your analysis of the current situation
- <plan>: Your plan for the next steps
- <commands>: XML structure containing commands to execute

The `duration` attribute of <keystrokes> specifies the number of seconds to wait for the command to complete (default: 1.0) before the next command will be executed. Never wait longer than 60 seconds; prefer to poll to see intermediate result status.

Optional sections:
- <task_complete>: Include this tag if the task is complete. If not present, task is assumed not complete.

IMPORTANT: The text inside each <keystrokes></keystrokes> tag will be used completely verbatim as keystrokes. DO NOT XML-encode special characters - write them directly:
- Use < and > directly, NOT &lt; and &gt;
- Use & directly, NOT &amp;
- Use quotes directly, NOT &quot;
- You must end every command with a newline (\\n) or it will not execute.

Special key sequences (use tmux-style escape sequences):
- C-c for Ctrl+C. MUST be sent as a keystroke by itself, e.g., <keystrokes>C-c</keystrokes>
- C-d for Ctrl+D. MUST be sent as a keystroke by itself, e.g., <keystrokes>C-d</keystrokes>
- For Enter/newline: simply add a newline in the XML.

Important notes:
- Each command's text content is sent exactly as keystrokes to the terminal
- Do not include extra whitespace before or after the command text unless it's part of the intended command
- Avoid extra text before or after the <response> tags
- Avoid additional XML tags outside of analysis/plan/commands/task_complete

Task Description:
{instruction}

Current terminal state:
{terminal_state}
"""

TIMEOUT_TEMPLATE = """\
Previous command:
{command}

The previous command timed out after {timeout_sec} seconds

It is possible that the command is not yet finished executing. If that is the case, then do nothing. It is also possible that you have entered an interactive shell and should continue sending keystrokes as normal.

Here is the current state of the terminal:

{terminal_state}
"""

COMPLETION_CONFIRMATION_JSON = (
    "Current terminal state:\n{terminal_output}\n\n"
    "Are you sure you want to mark the task as complete? "
    "This will trigger your solution to be graded and you won't be able to "
    'make any further corrections. If so, include "task_complete": true '
    "in your JSON response again."
)

COMPLETION_CONFIRMATION_XML = (
    "Current terminal state:\n{terminal_output}\n\n"
    "Are you sure you want to mark the task as complete? "
    "This will trigger your solution to be graded and you won't be able to "
    "make any further corrections. If so, include "
    "<task_complete>true</task_complete> again."
)
