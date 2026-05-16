"""UDA env — Universal Digital Agent execution environment.

Mirrors the file layout of :mod:`nanorollout.envs.cocoa_env`: top-level
``TaskExecutor`` here, sandbox HTTP clients in ``base.py``, runtime
backends in ``docker.py`` / ``modal.py``, tool schemas in ``tools.py``,
encryption helpers in ``decrypt.py``, logging in ``logger.py``, and
generic helpers in ``utils.py``. The task corpus ships beside the code
under ``adapter/<benchmark>/``.

The body of the executor + sandbox stack is currently a direct
duplicate of cocoa_env's implementation because UDA tasks share its
HTTP surface, encryption scheme, action vocabulary, and feedback loop.
Future UDA-specific divergence (SWE-bench-MM's ``npm test`` verifier,
OSWorld a11y-tree predicates, the Anthropic Computer-Use tool class
that uda-desktop exposes on ``/v1/computer-use/*``) lands here as
inline edits rather than as cross-package re-exports.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict

from .adapter import ADAPTER_ROOT
from .logger import setup_logging, get_logger
from nanorollout.harness.agents.uda.controller import (
    OpenAILLM, QwenLLM, BaseLLM, Controller, Human, GeminiLLM, ClaudeLLM,
    GLMLLM, KimiLLM, DeepSeekLLM,
    MODEL_PRICING_REGISTRY,
)
from .base import (
    ComputerUseSandboxClient,
    UnifiedSandboxClient,
)
from .utils import colorize, extract_config_info, measure_execution_time

# Decryption + grading is now driver-resident; see uda_env.driver.cocoa_v1
# (host-side test.py.enc) and uda_env.driver.wildclaw_v1 (in-container
# grade.py).

logger = get_logger("uda.executor")

__all__ = [
    "ADAPTER_ROOT",
    "TaskExecutor",
    "ComputerUseSandboxClient",
    "UnifiedSandboxClient",
    "setup_logging",
    "get_logger",
]

# Anthropic Computer Tool action vocabulary (Action_20251124). The
# tool-call layer prefixes each with ``computer_use_`` to disambiguate
# from generic words like ``type`` / ``key``; see envs/uda_env/tools.py.
_COMPUTER_USE_ACTIONS = frozenset({
    "screenshot", "cursor_position", "mouse_move",
    "left_click", "right_click", "middle_click", "double_click", "triple_click",
    "left_click_drag", "left_mouse_down", "left_mouse_up",
    "key", "type", "hold_key", "scroll", "wait", "zoom",
})


def is_computer_use_action(action: Dict[str, Any]) -> bool:
    """Return True if ``action`` targets uda-desktop's /v1/computer-use/* surface.

    UDA's pixel-level GUI control replaces cocoa_env's CDP-based
    ``browser_*`` / ``dom_*`` family. Used by the TaskExecutor loop to
    decide when to capture a post-action screenshot.
    """
    if not isinstance(action, dict):
        return False
    action_type = action.get("action_type", "")
    if not isinstance(action_type, str) or not action_type.startswith("computer_use_"):
        return False
    return action_type[len("computer_use_"):] in _COMPUTER_USE_ACTIONS

def normalize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize action format by flattening parameters if present.
    
    Handles both formats:
    - {"action_type": "file_list", "path": "/home/gem/"} (correct)
    - {"action_type": "file_list", "parameters": {"path": "/home/gem/"}} (needs normalization)
    
    Args:
        action: Action dictionary that may have nested parameters
        
    Returns:
        Normalized action with parameters flattened to top level
    """
    if not isinstance(action, dict):
        return action
    
    # If action has "parameters" field, flatten it
    if "parameters" in action and isinstance(action.get("parameters"), dict):
        normalized = {"action_type": action.get("action_type")}
        normalized.update(action["parameters"])
        # Preserve other top-level fields (like tool_call_id)
        for key, value in action.items():
            if key not in ["parameters", "action_type"]:
                normalized[key] = value
        return normalized
    
    return action


def _format_model_output_for_log(value: Any, max_chars: int = 8000) -> str:
    """Serialize model output for readable logging with clipping."""
    if value is None:
        return ""

    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, indent=2, ensure_ascii=True)
        except Exception:
            text = str(value)

    if len(text) <= max_chars:
        return text

    remaining = len(text) - max_chars
    return f"{text[:max_chars]}\n... <truncated {remaining} chars>"

class TaskExecutor:
    """Executes tasks using a controller with agent feedback loop."""

    def __init__(self, config: dict, controller: Controller | None = None):
        """Initialize TaskExecutor.

        Args:
            config: Configuration dictionary with optional 'controller' section
            controller: Controller instance (LLM or Human). If None, creates controller from config.
        """
        self.config = config
        
        logger.info(f"Config: {config}")

        sandbox_config = config.get("sandbox", {})
        
        # Determine which sandbox client to use based on config
        client_type = sandbox_config.get("client_type", "shell").lower()
        
        if controller is None:
            # Get controller type and config from config dict
            controller_config = config.get("controller", {})
            controller_type = controller_config.get("type", "llm").lower()

            if controller_type == "human":
                controller = Human()
            elif controller_type == "gemini":
                llm_config = controller_config.get("args", {})
                controller = GeminiLLM(llm_config=llm_config, client_type=client_type)
            elif controller_type == "qwen":
                llm_config = controller_config.get("args", {})
                controller = QwenLLM(llm_config=llm_config, client_type=client_type)
            elif controller_type == "claude":
                llm_config = controller_config.get("args", {})
                controller = ClaudeLLM(llm_config=llm_config, client_type=client_type)
            elif controller_type == "glm":
                llm_config = controller_config.get("args", {})
                controller = GLMLLM(llm_config=llm_config, client_type=client_type)
            elif controller_type == "kimi":
                llm_config = controller_config.get("args", {})
                controller = KimiLLM(llm_config=llm_config, client_type=client_type)
            elif controller_type == "deepseek":
                llm_config = controller_config.get("args", {})
                controller = DeepSeekLLM(llm_config=llm_config, client_type=client_type)
            else:  # Default to OpenAILLM (handles "gpt", "openai", "llm", etc.)
                llm_config = controller_config.get("args", {})
                controller = OpenAILLM(llm_config=llm_config, client_type=client_type)

            logger.info(f"Controller initialized: {controller_type} (Model: {llm_config.get('model', 'unknown')})")

        self.controller = controller
        if "llm_provider" not in sandbox_config:
            # Check subclasses before parent classes (isinstance matches parents too)
            if isinstance(controller, ClaudeLLM):
                sandbox_config["llm_provider"] = "claude"
            elif isinstance(controller, GeminiLLM):
                sandbox_config["llm_provider"] = "gemini"
            elif isinstance(controller, DeepSeekLLM):
                sandbox_config["llm_provider"] = "deepseek"
            elif isinstance(controller, GLMLLM):
                sandbox_config["llm_provider"] = "glm"
            elif isinstance(controller, KimiLLM):
                sandbox_config["llm_provider"] = "kimi"
            elif isinstance(controller, QwenLLM):
                sandbox_config["llm_provider"] = "qwen"
            elif isinstance(controller, OpenAILLM):
                sandbox_config["llm_provider"] = "openai"
            else:
                sandbox_config["llm_provider"] = controller_type if "controller_type" in locals() else "llm"
        if "llm_model" not in sandbox_config:
            sandbox_config["llm_model"] = getattr(controller, "model", None)
        if client_type == "unified":
            self.sandbox_client = UnifiedSandboxClient(sandbox_config=sandbox_config)
            logger.info("Using UnifiedSandboxClient (computer-use + file + code + shell)")
        elif client_type in ("computer_use", "computer-use", "browser"):
            # ``browser`` accepted as a legacy alias for cocoa configs.
            self.sandbox_client = ComputerUseSandboxClient(sandbox_config=sandbox_config)
            logger.info("Using ComputerUseSandboxClient (GUI-only via /v1/computer-use/*)")
        else:
            # Default to UnifiedSandboxClient for "shell" or unknown types
            # UnifiedSandboxClient supports all tools including shell
            self.sandbox_client = UnifiedSandboxClient(sandbox_config=sandbox_config)
            logger.info(f"Using UnifiedSandboxClient as fallback for client_type='{client_type}'")
        self._environment_timing: Dict[str, Any] = {}
        self._last_task_timing: Dict[str, Any] = {}

    def setup_environment(self, task: dict, wait_time: int = 30) -> None:
        """Initialize the sandbox environment for task execution.

        Args:
            task: Task object containing task_dir and other task metadata
            wait_time: Time to wait for server to be ready (default: 30 seconds)
        """
        startup_started_at = time.perf_counter()
        self._environment_timing = {
            "task_name": task.get("task_name", "unknown"),
            "startup_started_at": startup_started_at,
        }

        if self.sandbox_client.create_environment(task, wait_time):
            startup_time_s = time.perf_counter() - startup_started_at
            self._environment_timing["sandbox_startup_s"] = startup_time_s
            runtime_label = self.sandbox_client.runtime_type
            runtime_id = self.sandbox_client.runtime_id or self.sandbox_client.container_id
            logger.info(
                "Sandbox environment ready (runtime=%s, id=%s)",
                runtime_label,
                runtime_id or "n/a",
            )
            logger.info("Sandbox startup time: %.3fs", startup_time_s)
        else:
            self._environment_timing["sandbox_startup_s"] = time.perf_counter() - startup_started_at
            raise RuntimeError("Sandbox environment failed to become ready")

        # Driver-specific in-container setup: push exec/ into the container,
        # run task-local warmup, etc. Cocoa drivers no-op; wildclaw stages
        # /tmp_workspace/<files> from the adapter's exec/.
        driver_name = task.get("driver")
        if driver_name:
            try:
                from .driver import load_driver
                driver = load_driver(driver_name)
                driver.setup_workspace(self.sandbox_client.runtime, task)
                driver.run_warmup(self.sandbox_client.runtime, task)
            except Exception as exc:
                logger.exception(
                    "Driver %s setup_workspace/run_warmup failed: %s",
                    driver_name,
                    exc,
                )
                raise

        self.controller.clear_history()
        if hasattr(self.controller, "reset_cost_tracking"):
            self.controller.reset_cost_tracking()

    def cleanup_environment(self) -> None:
        """Clean up the sandbox environment after execution."""
        self.sandbox_client.cleanup_environment()
        self.controller.clear_history()

    def get_last_timing_stats(self) -> Dict[str, Any]:
        """Return the latest collected timing stats for the current or previous task."""
        return dict(self._last_task_timing) if isinstance(self._last_task_timing, dict) else {}

    @measure_execution_time
    def run_task(self, task: dict) -> dict:
        """Run inference on the given task with agent loop.

        Args:
            task: Task dictionary containing task metadata including instruction

        Returns:
            Dictionary with results including status, model info, conversation history, and execution_time
        """
        task_desc = task.get("instruction", str(task))
        max_iterations = self.config.get("sandbox", {}).get("max_iterations", 10)
        # max_conversation_turns = self.config.get("sandbox", {}).get("max_conversation_turns", 5)  # Commented out - using self.messages for context instead

        logger.debug(f"Task description: {colorize(task_desc, 'YELLOW')}")

        def add_progress_note(base_message: str, current_iteration: int) -> str:
            """Append iteration progress context to controller prompts."""
            remaining = max(max_iterations - current_iteration, 0)
            note = (
                f"\n\n[Progress update: iteration {current_iteration}/{max_iterations}. "
                f"Remaining iterations: {remaining}.]"
            )
            if remaining <= 2:
                note += " You are near the maximum iteration budget. Prioritize finishing steps and produce the final boxed answer soon."
            return f"{base_message}{note}"

        # Store task description for initial prompt only
        self.task_description = task_desc

        def record_tool_feedback(action_dict: dict, feedback_dict: dict) -> None:
            """Append tool call outputs to controller history for OpenAI compliance."""
            if not isinstance(action_dict, dict):
                return
            tool_call_id = action_dict.get("tool_call_id")
            if not tool_call_id:
                return
            content = feedback_dict.get("message", "")
            if hasattr(self.controller, "add_tool_message"):
                self.controller.add_tool_message(tool_call_id, content if isinstance(content, str) else str(content))

        def finalize_iteration_timing(iteration_timing: Dict[str, Any], iteration_started_at: float) -> None:
            """Finalize and persist iteration-level timing once."""
            if iteration_timing.get("_finalized"):
                return
            iteration_timing["total_s"] = time.perf_counter() - iteration_started_at
            iteration_timing["other_overhead_s"] = max(
                iteration_timing["total_s"]
                - iteration_timing["llm_call_s"]
                - iteration_timing["tool_execution_s"]
                - iteration_timing["post_action_screenshot_s"],
                0.0,
            )
            timing_stats["iteration_total_s"] += iteration_timing["total_s"]
            timing_stats["other_overhead_total_s"] += iteration_timing["other_overhead_s"]
            iteration_timing["_finalized"] = True
            timing_stats["iterations"].append({
                key: value for key, value in iteration_timing.items() if key != "_finalized"
            })

        # Build initial prompt (only for first iteration)
        prompt = self.controller.build_prompt(task_description=task_desc)

        action = None
        final_iteration = 0
        last_feedback_with_image = None
        images_from_last_iteration = []  # Store images from the previous iteration only
        task_result = None  # Store task result if provided in task_complete
        timing_stats: Dict[str, Any] = {
            "sandbox_startup_s": self._environment_timing.get("sandbox_startup_s"),
            "llm_call_total_s": 0.0,
            "tool_execution_total_s": 0.0,
            "post_action_screenshot_total_s": 0.0,
            "iteration_total_s": 0.0,
            "other_overhead_total_s": 0.0,
            "iterations": [],
        }
        self._last_task_timing = {
            "task_name": task.get("task_name", "unknown"),
            "timing_stats": timing_stats,
        }
        
        # Initialize visualization data structure
        visualization_data = {
            "task_description": task_desc,
            "iterations": []
        }

        # Agent loop
        for iteration in range(1, max_iterations + 1):
            final_iteration = iteration
            logger.info(f"Iteration {iteration}/{max_iterations}")
            iteration_started_at = time.perf_counter()

            # Get controller response (already parsed into action dict)
            # Only include images from the previous iteration (i-1), not all historical images
            images_base64 = images_from_last_iteration.copy() if images_from_last_iteration else None
            if images_base64:
                logger.debug(f"Including {len(images_base64)} image(s) from previous iteration in next prompt")
            
            prompt_with_progress = add_progress_note(prompt, iteration)
            iteration_timing = {
                "iteration": iteration,
                "llm_call_s": 0.0,
                "tool_execution_s": 0.0,
                "post_action_screenshot_s": 0.0,
                "total_s": 0.0,
                "other_overhead_s": 0.0,
                "action_count": 0,
                "completed": False,
                "error_action": False,
            }
            # Pass list of images (only from previous iteration)
            llm_started_at = time.perf_counter()
            try:
                action = self.controller.call(prompt_with_progress, images_base64=images_base64)
            except Exception:
                llm_elapsed_s = time.perf_counter() - llm_started_at
                iteration_timing["llm_call_s"] = llm_elapsed_s
                timing_stats["llm_call_total_s"] += llm_elapsed_s
                finalize_iteration_timing(iteration_timing, iteration_started_at)
                raise
            llm_elapsed_s = time.perf_counter() - llm_started_at
            iteration_timing["llm_call_s"] = llm_elapsed_s
            timing_stats["llm_call_total_s"] += llm_elapsed_s
            
            # Extract think content from controller
            think_content = None
            if hasattr(self.controller, 'get_last_think'):
                think_content = self.controller.get_last_think()

            if think_content:
                logger.info(
                    "Model reasoning [iteration %s]:\n%s",
                    iteration,
                    _format_model_output_for_log(think_content),
                )
            logger.info(
                "Model output [iteration %s]:\n%s",
                iteration,
                _format_model_output_for_log(action),
            )

            # Handle error action (parsing errors from tool calls)
            if isinstance(action, dict) and action.get("action_type") == "error":
                error_message = action.get("error_message", "Unknown error occurred while parsing tool calls")
                logger.warning(f"Tool call parsing error: {error_message}")
                # Create feedback with error message to send back to model
                feedback = {
                    "done": False,
                    "message": f"Error: {error_message}\nPlease correct the tool call parameters and try again."
                }
                # Prepare next prompt with error feedback
                prompt = self.controller.build_prompt(
                    feedback=feedback.get("message", "Continue with the task.")
                )
                iteration_timing["error_action"] = True
                finalize_iteration_timing(iteration_timing, iteration_started_at)
                continue

            # Normalize action format
            action = normalize_action(action)

            # Handle multiple actions (from tool calling)
            if "actions" in action:
                # Execute multiple actions sequentially
                feedbacks = []
                done = False
                images_from_current_iteration = []  # Collect all images from current iteration
                computer_use_screenshots = []  # Collect only computer-use screenshots
                image_read_contents = [] # Collect only image_read contents
                iteration_actions = []  # Store actions for visualization
                
                for single_action in action["actions"]:
                    # Normalize action format
                    single_action = normalize_action(single_action)
                    tool_started_at = time.perf_counter()
                    try:
                        single_feedback = self.sandbox_client.get_feedback(single_action)
                    except Exception:
                        tool_elapsed_s = time.perf_counter() - tool_started_at
                        iteration_timing["tool_execution_s"] += tool_elapsed_s
                        timing_stats["tool_execution_total_s"] += tool_elapsed_s
                        iteration_timing["action_count"] += 1
                        finalize_iteration_timing(iteration_timing, iteration_started_at)
                        raise
                    tool_elapsed_s = time.perf_counter() - tool_started_at
                    iteration_timing["tool_execution_s"] += tool_elapsed_s
                    timing_stats["tool_execution_total_s"] += tool_elapsed_s
                    iteration_timing["action_count"] += 1
                    record_tool_feedback(single_action, single_feedback) # TODO: optimize OpenAI Tool Calling format to avoid extra messages in the conversation history
                    feedbacks.append(single_feedback.get("message", ""))
                    
                    # For computer-use actions, take a screenshot after execution (unless it's already a screenshot action)
                    screenshot_base64 = None
                    if is_computer_use_action(single_action) and single_action.get("action_type") != "computer_use_screenshot":
                        if hasattr(self.sandbox_client, 'take_screenshot'):
                            try:
                                screenshot_started_at = time.perf_counter()
                                screenshot_base64, _ = self.sandbox_client.take_screenshot()
                                screenshot_elapsed_s = time.perf_counter() - screenshot_started_at
                                iteration_timing["post_action_screenshot_s"] += screenshot_elapsed_s
                                timing_stats["post_action_screenshot_total_s"] += screenshot_elapsed_s
                                if screenshot_base64:
                                    computer_use_screenshots.append(screenshot_base64)
                            except Exception as e:
                                logger.warning(f"Failed to take screenshot after computer-use action: {e}")
                    
                    # Check if this action was a screenshot or image_read and has image_base64
                    if single_action.get("action_type") == "computer_use_screenshot" and "image_base64" in single_feedback:
                        image_base64 = single_feedback["image_base64"]
                        computer_use_screenshots.append(image_base64)
                    
                    if single_action.get("action_type") == "image_read" and "image_base64" in single_feedback:
                        image_base64 = single_feedback["image_base64"]
                        if image_base64 not in image_read_contents:
                            image_read_contents.append(image_base64)
                    
                    # Store action data for visualization
                    action_data = {
                        "action": single_action,
                        "observation": single_feedback.get("message", ""),
                        "screenshot": screenshot_base64 if screenshot_base64 else (single_feedback.get("image_base64") if single_action.get("action_type") in ["computer_use_screenshot", "image_read"] else None)
                    }
                    iteration_actions.append(action_data)
                    
                    if single_feedback.get("done"):
                        done = True
                        iteration_timing["completed"] = True
                        break
                
                # Combine all feedbacks
                combined_feedback = {
                    "done": done,
                    "message": "\n".join(feedbacks) # '/n' is used to separate each feedback
                }
                
                # Construct images list for next iteration
                # 1. Take ONLY the last computer-use screenshot if available
                # 2. Add ALL image_read contents
                images_from_last_iteration = []
                
                if computer_use_screenshots:
                    images_from_last_iteration.append(computer_use_screenshots[-1])
                
                # Append all manually read images
                images_from_last_iteration.extend(image_read_contents)
                
                # For backward compatibility in feedback dict (though mostly unused if images_from_last_iteration is set)
                if images_from_last_iteration:
                    combined_feedback["image_base64"] = images_from_last_iteration[-1]
                
                feedback = combined_feedback
                
                # Store iteration data for visualization
                visualization_data["iterations"].append({
                    "iteration": iteration,
                    "think": think_content,
                    "actions": iteration_actions
                })
            else:
                # Single action
                tool_started_at = time.perf_counter()
                try:
                    feedback = self.sandbox_client.get_feedback(action)
                except Exception:
                    tool_elapsed_s = time.perf_counter() - tool_started_at
                    iteration_timing["tool_execution_s"] += tool_elapsed_s
                    timing_stats["tool_execution_total_s"] += tool_elapsed_s
                    iteration_timing["action_count"] = 1
                    finalize_iteration_timing(iteration_timing, iteration_started_at)
                    raise
                tool_elapsed_s = time.perf_counter() - tool_started_at
                iteration_timing["tool_execution_s"] += tool_elapsed_s
                timing_stats["tool_execution_total_s"] += tool_elapsed_s
                iteration_timing["action_count"] = 1
                record_tool_feedback(action, feedback)
                
                # For computer-use actions, take a screenshot after execution (unless it's already a screenshot action)
                screenshot_base64 = None
                if is_computer_use_action(action) and action.get("action_type") != "computer_use_screenshot":
                    if hasattr(self.sandbox_client, 'take_screenshot'):
                        try:
                            screenshot_started_at = time.perf_counter()
                            screenshot_base64, _ = self.sandbox_client.take_screenshot()
                            screenshot_elapsed_s = time.perf_counter() - screenshot_started_at
                            iteration_timing["post_action_screenshot_s"] += screenshot_elapsed_s
                            timing_stats["post_action_screenshot_total_s"] += screenshot_elapsed_s
                        except Exception as e:
                            logger.warning(f"Failed to take screenshot after computer-use action: {e}")
                
                # Store images from this iteration for next iteration
                images_from_last_iteration = []  # Reset for current iteration
                if action.get("action_type") in ["computer_use_screenshot", "image_read"] and "image_base64" in feedback:
                    image_base64 = feedback["image_base64"]
                    images_from_last_iteration = [image_base64]  # Store single image for next iteration
                elif screenshot_base64:
                    images_from_last_iteration = [screenshot_base64]
                
                # Store iteration data for visualization
                visualization_data["iterations"].append({
                    "iteration": iteration,
                    "think": think_content,
                    "actions": [{
                        "action": action,
                        "observation": feedback.get("message", ""),
                        "screenshot": screenshot_base64 if screenshot_base64 else (feedback.get("image_base64") if action.get("action_type") in ["computer_use_screenshot", "image_read"] else None)
                    }]
                })

            # Store the full feedback (including image_base64) for next iteration
            last_feedback_with_image = feedback

            # Check if task is complete
            if feedback.get("done"):
                logger.info(f"Task completed at iteration {iteration}")
                iteration_timing["completed"] = True
                # Extract task_result if present
                if "task_result" in feedback:
                    task_result = feedback.get("task_result")
                finalize_iteration_timing(iteration_timing, iteration_started_at)
                break

            # Prepare next prompt - only feedback, context is maintained in self.messages
            prompt = self.controller.build_prompt(
                feedback=feedback.get("message", "Continue with the task.")
            )
            finalize_iteration_timing(iteration_timing, iteration_started_at)

        result_dict = task | extract_config_info(self.config) | {
            "status": "success",
            "iterations": final_iteration,
            "conversation": self.controller.get_history(),
            "execution_trace": self.sandbox_client.get_history(),
            "visualization_data": visualization_data,  # Add visualization data
            "sandbox_runtime": self.sandbox_client.get_runtime_metadata(),
            "timing_stats": timing_stats,
        }
        self._last_task_timing = {
            "task_name": task.get("task_name", "unknown"),
            "timing_stats": timing_stats,
        }
        
        # Add task_result if it was provided in task_complete
        if task_result:
            result_dict["task_result"] = task_result
        
        # Add API cost statistics if controller supports it
        if hasattr(self.controller, "get_cost_stats"):
            api_cost_stats = self.controller.get_cost_stats()
            result_dict["api_cost_stats"] = api_cost_stats

            try:
                model = api_cost_stats.get("model", "unknown")
                total_cost = float(api_cost_stats.get("total_cost_usd", 0.0) or 0.0)
                input_tokens = api_cost_stats.get("total_input_tokens", 0)
                cached_tokens = api_cost_stats.get("total_cached_tokens", 0)
                output_tokens = api_cost_stats.get("total_output_tokens", 0)
                reasoning_tokens = api_cost_stats.get("total_reasoning_tokens", 0)
                api_calls = api_cost_stats.get("api_calls", 0)
                pricing = MODEL_PRICING_REGISTRY.get(str(model).lower(), {})

                per_call = api_cost_stats.get("per_call_costs", [])
                for idx, call in enumerate(per_call, 1):
                    c = float(call.get("total_cost_usd", 0))
                    toks = call.get("tokens", {})
                    tier = call.get("pricing_tier", "")
                    logger.debug(
                        "  call #%d: $%.6f tokens=%s tier=%s", idx, c, toks, tier,
                    )

                logger.info(
                    "Task cost summary: model=%s calls=%d total_cost_usd=%.6f "
                    "input=%s cached=%s output=%s reasoning=%s",
                    model, api_calls, total_cost,
                    input_tokens, cached_tokens, output_tokens, reasoning_tokens,
                )
                print(
                    f"[Cost] {model} | calls={api_calls} | "
                    f"${total_cost:.6f} total | "
                    f"in={input_tokens} cached={cached_tokens} out={output_tokens} "
                    f"reasoning={reasoning_tokens} | pricing={pricing}"
                )
            except Exception:
                pass

        logger.info(
            "Task timing summary: startup=%.3fs llm=%.3fs tool=%.3fs screenshot=%.3fs other=%.3fs total_iterations=%.3fs",
            float(timing_stats.get("sandbox_startup_s") or 0.0),
            float(timing_stats.get("llm_call_total_s") or 0.0),
            float(timing_stats.get("tool_execution_total_s") or 0.0),
            float(timing_stats.get("post_action_screenshot_total_s") or 0.0),
            float(timing_stats.get("other_overhead_total_s") or 0.0),
            float(timing_stats.get("iteration_total_s") or 0.0),
        )
        
        return result_dict

    @measure_execution_time
    def run_eval(self, task: dict, result: dict) -> dict:
        """Score the rollout via the per-bench driver.

        cocoa-v1: decrypts test.py.enc host-side and calls ``test(result)``.
        wildclaw-v1: pushes grade.py + gt/ into the container, runs
        ``grade()`` inside the sandbox, parses the JSON float dict from
        stdout.

        Returns the driver-specific score dict (or None if the task has
        no bundled grader).
        """
        driver_name = task.get("driver")
        task_name = task.get("task_name", "unknown")
        if not driver_name:
            logger.debug(
                "No driver attached to task '%s'; skipping evaluation. "
                "Hint: ensure the runner uses driver.load_task() which "
                "stamps the driver field.",
                task_name,
            )
            return None

        try:
            from .driver import load_driver
            driver = load_driver(driver_name)
            scores = driver.score(self.sandbox_client.runtime, task, result)
            logger.info(
                "Eval done for '%s' (driver=%s): %s",
                task_name,
                driver_name,
                colorize(scores, "YELLOW"),
            )
            return scores
        except Exception as exc:
            logger.error("Eval failed for '%s': %s", task_name, exc)
            raise
