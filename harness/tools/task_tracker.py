"""
Task tracker tool implementation.
"""

from typing import Optional

from .base import BaseTool, ToolParameter, ToolResult

_DETAILED_TASK_TRACKER_DESCRIPTION = """This tool provides structured task management capabilities for development workflows.
It enables systematic tracking of work items, progress monitoring, and efficient
organization of complex development activities.

The tool maintains visibility into project status and helps communicate
progress effectively to users.

## Application Guidelines

Utilize this tool in the following situations:

1. Multi-phase development work - When projects involve multiple sequential or
   parallel activities
2. Complex implementation tasks - Work requiring systematic planning and
   coordination across multiple components
3. Explicit user request for task organization - When users specifically ask
   for structured task management
4. Multiple concurrent requirements - When users present several work items
   that need coordination
5. Project initiation - Capture and organize user requirements at project start
6. Work commencement - Update task status to in_progress before beginning
   implementation. Maintain focus by limiting active work to one task
7. Task completion - Update status to done and identify any additional work
   that emerged during implementation

## Situations Where Tool Usage Is Unnecessary

Avoid using this tool when:

1. Single atomic tasks that require no decomposition
2. Trivial operations where tracking adds no organizational value
3. Simple activities completable in minimal steps
4. Pure information exchange or discussion

Note: For single straightforward tasks, proceed with direct implementation
rather than creating tracking overhead.

## Usage Scenarios

**Scenario A: Feature Development with Validation**
User request: "Build a user authentication system with login/logout functionality.
Don't forget to include input validation and error handling!"

Response approach: I'll implement a user authentication system with comprehensive
validation. Let me organize this work systematically.

*Task breakdown includes:*
1. Design authentication API endpoints and data models
2. Implement user login/logout controllers with session management
3. Add client-side form validation and error display
4. Create backend input sanitization
5. Implement comprehensive error handling for edge cases
6. Perform integration testing and fix any issues

**Scenario B: Codebase Refactoring**
User request: "I need to update all instances of the method 'fetchData' to
'retrieveInformation' throughout the entire project"

Response approach: Let me first analyze your codebase to identify all locations
where 'fetchData' appears.

*After code analysis*

I've located 12 occurrences of 'fetchData' across 6 files in your project.
I'll create a systematic plan to update these references.

*Organized task list includes specific file-by-file updates*

**Scenario C: Multi-feature Development**
User request: "Set up a blog platform with these components: admin dashboard,
article management, comment system, and user profiles."

Response approach: I'll help build your blog platform. Let me break down these
components into manageable implementation phases.

*Creates structured plan with each feature decomposed into specific development tasks*

## Counter-examples - Direct Implementation Preferred

**Counter-example A: Simple inquiry**
User request: "What's the syntax for a for loop in JavaScript?"

Response approach: Here's how to write a for loop in JavaScript:

```javascript
for (let i = 0; i < array.length; i++) {
    console.log(array[i]);
}
```

*No task tracking needed - single informational response*

**Counter-example B: Minor modification**
User request: "Please add a docstring to the processPayment function explaining
its parameters."

Response approach: I'll add a docstring to explain the processPayment function.

*Direct edit without task tracking for simple modification*

## Status Management and Workflow

1. **Status Values**: Track work using these states:
   - todo: Not yet initiated
   - in_progress: Currently active (maintain single focus)
   - done: Successfully completed

2. **Workflow Practices**:
   - Update status dynamically as work progresses
   - Mark completion immediately upon task finish
   - Limit active work to ONE task at any given time
   - Complete current activities before initiating new ones
   - Remove obsolete tasks from tracking entirely

3. **Completion Criteria**:
   - Mark tasks as done only when fully achieved
   - Keep status as in_progress if errors, blocks, or partial completion exist
   - Create new tasks for discovered issues or dependencies
   - Never mark done when:
       - Test suites are failing
       - Implementation remains incomplete
       - Unresolved errors persist
       - Required resources are unavailable

4. **Task Organization**:
   - Write precise, actionable descriptions
   - Decompose complex work into manageable units
   - Use descriptive, clear naming conventions

When uncertain, favor using this tool. Proactive task management demonstrates
systematic approach and ensures comprehensive requirement fulfillment.
"""

_SHORT_TASK_TRACKER_DESCRIPTION = """Provides structured task management for development workflows, enabling progress
tracking and systematic organization of complex coding activities.

* Apply to multi-phase projects (3+ distinct steps) or when managing multiple user requirements
* Update status (todo/in_progress/done) dynamically throughout work
* Maintain single active task focus at any time
* Mark completion immediately upon task finish
* Decompose complex work into manageable, actionable units
"""


class TaskTrackerTool(BaseTool):
    """Tool for tracking tasks and progress."""

    def __init__(self):
        self._task_list: list[dict] = []

    @property
    def name(self) -> str:
        return "task_tracker"

    def get_description(self, **kwargs) -> str:
        return (
            _SHORT_TASK_TRACKER_DESCRIPTION
            if kwargs.get("use_short_description", False)
            else _DETAILED_TASK_TRACKER_DESCRIPTION
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description='The command to execute. `view` shows the current task list. `plan` creates or updates the task list based on provided requirements and progress. Always `view` the current list before making changes.',
                required=True,
                enum=["view", "plan"],
            ),
            ToolParameter(
                name="task_list",
                type="array",
                description="The full task list. Required parameter of `plan` command.",
                required=False,
                items={
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique task identifier",
                        },
                        "title": {
                            "type": "string",
                            "description": "Brief task description",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["todo", "in_progress", "done"],
                            "description": "Current task status",
                        },
                        "notes": {
                            "type": "string",
                            "description": "Optional additional context or details",
                        },
                    },
                    "required": ["id", "title", "status"],
                    "additionalProperties": False,
                },
            ),
        ]

    def to_openai_schema(self, **kwargs) -> dict:
        schema = super().to_openai_schema(**kwargs)
        schema["function"]["parameters"]["additionalProperties"] = False
        return schema

    def execute(
        self,
        environment,
        command: str,
        task_list: Optional[list[dict]] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute task tracker command."""
        if command == "view":
            return self._view()
        elif command == "plan":
            return self._plan(task_list)
        else:
            return ToolResult(
                output=f"Error: Unknown command '{command}'",
                success=False,
            )

    def _view(self) -> ToolResult:
        """View current task list."""
        if not self._task_list:
            return ToolResult(
                output='No task list found. Use the "plan" command to create one.',
                success=True,
            )

        lines = ["# Task List", ""]
        for i, task in enumerate(self._task_list, 1):
            status_icon = {
                "todo": "[ ]",
                "in_progress": "[~]",
                "done": "[x]",
            }.get(task.get("status", "todo"), "[ ]")
            line = f"{i}. {status_icon} {task.get('title', 'Untitled')}"
            lines.append(line)
            notes = task.get("notes", "")
            if notes:
                lines.append(notes)

        return ToolResult(
            output="\n".join(lines),
            success=True,
            metadata={"task_list": self._task_list},
        )

    def _plan(self, task_list: Optional[list[dict]]) -> ToolResult:
        """Update task list."""
        if task_list is None:
            return ToolResult(
                output="Error: task_list is required for plan command",
                success=False,
            )

        # Normalize task list
        normalized = []
        for i, task in enumerate(task_list):
            if isinstance(task, dict):
                normalized.append({
                    "id": task.get("id", f"task-{i + 1}"),
                    "title": task.get("title", "Untitled task"),
                    "status": task.get("status", "todo"),
                    "notes": task.get("notes", ""),
                })
            else:
                return ToolResult(
                    output=f"Error: Invalid task format at index {i}",
                    success=False,
                )

        self._task_list = normalized

        # Count by status
        counts = {"todo": 0, "in_progress": 0, "done": 0}
        for task in self._task_list:
            status = task.get("status", "todo")
            if status in counts:
                counts[status] += 1

        return ToolResult(
            output=f"Task list has been updated with {len(self._task_list)} items.",
            success=True,
            metadata={"task_list": self._task_list, "counts": counts},
        )