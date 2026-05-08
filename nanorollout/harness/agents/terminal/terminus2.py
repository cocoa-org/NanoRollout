"""
Terminus-2 agent.
"""

import logging
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from litellm import completion
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    from litellm import completion_cost
except Exception:  # pragma: no cover
    completion_cost = None

try:
    from litellm.utils import token_counter
except Exception:  # pragma: no cover
    token_counter = None

try:
    from litellm import get_model_info
except ImportError:  # pragma: no cover
    get_model_info = None

from .prompts import (
    COMPLETION_CONFIRMATION_JSON,
    COMPLETION_CONFIRMATION_XML,
    JSON_PROMPT_TEMPLATE,
    TIMEOUT_TEMPLATE,
    XML_PROMPT_TEMPLATE,
)
from .terminus_json_parser import TerminusJSONPlainParser
from .terminus_xml_parser import TerminusXMLPlainParser
from .tmux_adapter import TmuxAdapter

if TYPE_CHECKING:
    from nanorollout.envs.shell_env.base import ShellEnvironment as BaseEnvironment

logger = logging.getLogger(__name__)


class _ContextLengthError(Exception):
    """Internal signal for context length exceeded, triggering summarization."""
    pass


@dataclass
class Command:
    keystrokes: str
    duration_sec: float


class Terminus2Agent:
    """Terminus-2 agent adapted for TinyFlow.

    Uses a tmux session inside the container (via TmuxAdapter) and an LLM
    to iteratively solve terminal tasks.
    """

    def __init__(
        self,
        environment: "BaseEnvironment",
        model: str,
        parser_name: str = "json",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.6,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_iterations: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        enable_asciinema: bool = False,
        output_dir: Optional[str] = None,
        enable_summarize: bool = True,
        proactive_summarization_threshold: int = 8000,
    ) -> None:
        self._env = environment
        self._model = model
        self._parser_name = parser_name
        self._api_key = api_key
        self._api_base = api_base
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._max_iterations = max_iterations
        self._extra_body = extra_body or {}
        self._enable_asciinema = enable_asciinema
        self._output_dir = output_dir
        self._enable_summarize = enable_summarize
        self._proactive_summarization_threshold = proactive_summarization_threshold
        self._summarization_count = 0

        self._parser = self._make_parser()
        self._prompt_template = self._get_prompt_template()

        self._messages: List[Dict[str, Any]] = []
        self.llm_metrics: List[Dict[str, Any]] = []
        self._pending_completion = False

    # ------------------------------------------------------------------
    # Parser / prompt selection
    # ------------------------------------------------------------------

    def _make_parser(self):
        if self._parser_name == "json":
            return TerminusJSONPlainParser()
        if self._parser_name == "xml":
            return TerminusXMLPlainParser()
        raise ValueError(f"Unknown parser_name: {self._parser_name}")

    def _get_prompt_template(self) -> str:
        if self._parser_name == "json":
            return JSON_PROMPT_TEMPLATE
        if self._parser_name == "xml":
            return XML_PROMPT_TEMPLATE
        raise ValueError(f"Unknown parser_name: {self._parser_name}")

    def _get_error_response_type(self) -> str:
        if self._parser_name == "json":
            return "JSON response"
        return "response"

    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        if self._parser_name == "json":
            return COMPLETION_CONFIRMATION_JSON.format(
                terminal_output=terminal_output
            )
        return COMPLETION_CONFIRMATION_XML.format(
            terminal_output=terminal_output
        )

    # ------------------------------------------------------------------
    # Output limiting
    # ------------------------------------------------------------------

    @staticmethod
    def _limit_output_length(output: str, max_bytes: int = 10000) -> str:
        if len(output.encode("utf-8")) <= max_bytes:
            return output
        portion_size = max_bytes // 2
        output_bytes = output.encode("utf-8")
        first_portion = output_bytes[:portion_size].decode("utf-8", errors="ignore")
        last_portion = output_bytes[-portion_size:].decode("utf-8", errors="ignore")
        omitted_bytes = (
            len(output_bytes)
            - len(first_portion.encode("utf-8"))
            - len(last_portion.encode("utf-8"))
        )
        return (
            f"{first_portion}\n[... output limited to {max_bytes} bytes; "
            f"{omitted_bytes} interior bytes omitted ...]\n{last_portion}"
        )

    # ------------------------------------------------------------------
    # LLM calling
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=(
            retry_if_not_exception_type(_ContextLengthError)
            & retry_if_exception_type(Exception)
        ),
        reraise=True,
    )
    def _completion_with_retry(self, **kwargs) -> Any:
        """LLM completion call with tenacity retry on transient errors.

        Context-length errors are never retried — they propagate
        immediately for the summarization fallback to handle.
        """
        try:
            return completion(**kwargs, drop_params=True)
        except Exception as e:
            err = str(e).lower()
            if self._enable_summarize and any(p in err for p in (
                "context length", "context_length_exceeded",
                "maximum context length", "too many tokens",
            )):
                logger.warning("Context length exceeded, triggering reactive summarization")
                raise _ContextLengthError(str(e)) from e
            logger.warning("LLM call failed (will retry): %s", e)
            raise

    def _call_llm(self, prompt: str, _output_length_retries: int = 0) -> str:
        """Call the LLM and return the assistant response text."""
        self._messages.append({"role": "user", "content": prompt})
        # Log the prompt eagerly so the trial.log captures the terminal pane
        # state (which is embedded in the prompt) even if the agent loop is
        # killed before _write_artifacts has a chance to dump trajectory.jsonl.
        logger.info("LLM prompt (last user message):\n%s", prompt)

        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": self._messages,
            "temperature": self._temperature,
            "extra_body": self._extra_body,
        }
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
            kwargs["custom_llm_provider"] = "openai"

        if "gpt-5" in self._model.lower():
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
            mt = kwargs.pop("max_tokens", None) or 16384
            kwargs["max_completion_tokens"] = max(16384, mt)

        try:
            response = self._completion_with_retry(**kwargs)
        except _ContextLengthError:
            self._messages.pop()  # remove user message before raising
            raise
        except Exception:
            self._messages.pop()  # remove user message on final failure
            raise
        self._record_llm_metrics(response)

        choice = response.choices[0]
        raw_content = choice.message.content or ""
        # TODO: support API-level reasoning_content and <think> tag stripping
        reasoning = getattr(choice.message, "reasoning_content", None) or ""
        # content = re.sub(r"<think>[\s\S]*?</think>", "", raw_content).strip()
        content = raw_content

        # Handle output length exceeded (finish_reason="length")
        if choice.finish_reason == "length" and content:
            logger.warning("Output truncated (finish_reason=length)")
            # Try to parse the truncated response — if valid, use it
            try:
                parse_result = self._parser.parse_response(content)
                if not parse_result.error:
                    logger.info("Truncated response is parseable, using it")
                    if reasoning:
                        self._messages.append({"role": "assistant", "reasoning_content": reasoning, "content": content})
                    else:
                        self._messages.append({"role": "assistant", "content": content})
                    return content
            except Exception:
                pass
            # Can't salvage — ask LLM to chunk its response
            if _output_length_retries < 2:
                if reasoning:
                    self._messages.append({"role": "assistant", "reasoning_content": reasoning, "content": content})
                else:
                    self._messages.append({"role": "assistant", "content": content})
                error_msg = (
                    "ERROR!! NONE of the actions you just requested were performed "
                    "because your output was truncated. "
                    "Re-issue this request, breaking it into smaller chunks."
                )
                return self._call_llm(error_msg, _output_length_retries + 1)
            else:
                logger.warning("Output length retry limit reached, using truncated response")

        if not content:
            logger.warning(
                "Empty LLM response: finish_reason=%s, refusal=%s, tool_calls=%s",
                choice.finish_reason,
                getattr(choice.message, "refusal", None),
                getattr(choice.message, "tool_calls", None),
            )
        logger.info("LLM response:\n%s", raw_content)
        # Store stripped content in messages (no thinking text),
        # matching Harbor's default behavior (interleaved_thinking=False).
        if reasoning:
            self._messages.append({"role": "assistant", "reasoning_content": reasoning, "content": content})
        else:
            self._messages.append({"role": "assistant", "content": content})
        return content

    def _record_llm_metrics(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        usage_dict = None
        if usage is not None:
            if hasattr(usage, "model_dump"):
                usage_dict = usage.model_dump()
            elif isinstance(usage, dict):
                usage_dict = dict(usage)
            else:
                usage_dict = {"raw": str(usage)}

        cost = getattr(response, "cost", None)
        hidden = getattr(response, "_hidden_params", None)
        if cost is None and isinstance(hidden, dict):
            for key in ("cost", "response_cost", "total_cost"):
                if key in hidden:
                    cost = hidden[key]
                    break
        if cost is None and completion_cost is not None:
            try:
                cost = completion_cost(response=response)
            except Exception:
                cost = None

        self.llm_metrics.append(
            {
                "model": getattr(response, "model", None),
                "usage": usage_dict,
                "cost": cost,
            }
        )

    # ------------------------------------------------------------------
    # Context summarization (matches Harbor's 3-step process)
    # ------------------------------------------------------------------

    def _get_model_context_limit(self) -> int:
        if get_model_info is None:
            return 1_000_000
        try:
            info = get_model_info(self._model)
            return int(
                info.get("max_input_tokens")
                or info.get("max_tokens")
                or 1_000_000
            )
        except Exception:
            return 1_000_000

    def _count_total_tokens(self) -> int:
        if token_counter is None:
            return 0
        try:
            return token_counter(model=self._model, messages=self._messages)
        except Exception:
            return 0

    def _unwind_messages_to_free_tokens(self, target: int = 4000) -> None:
        limit = self._get_model_context_limit()
        while len(self._messages) > 1:
            if limit - self._count_total_tokens() >= target:
                break
            if len(self._messages) >= 3:
                self._messages = self._messages[:-2]
            else:
                break

    def _subagent_call(self, messages: List[Dict[str, Any]]) -> str:
        """Make a standalone LLM call for summarization subagents."""
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "top_p": self._top_p,
            "extra_body": self._extra_body,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
            kwargs["custom_llm_provider"] = "openai"
        resp = completion(**kwargs, drop_params=True)
        self._record_llm_metrics(resp)
        content = resp.choices[0].message.content or ""
        return re.sub(r"<think>[\s\S]*?</think>", "", content).strip()

    def _summarize(self, instruction: str, tmux: "TmuxAdapter") -> str:
        """Three-step context summarization matching Harbor's implementation.

        Step 1: LLM summarizes work done (full history)
        Step 2: Fresh LLM asks questions about gaps (empty history)
        Step 3: LLM answers questions (full history + summary)

        Returns a handoff prompt to inject as the next user message.
        """
        self._summarization_count += 1
        logger.info(
            "Starting context summarization #%d (%d messages)",
            self._summarization_count, len(self._messages),
        )

        # Step 1: Summary generation
        summary_prompt = (
            "You are about to hand off your work to another AI agent.\n"
            "Provide a comprehensive summary of what you accomplished on this task:\n\n"
            f"Original Task: {instruction}\n\n"
            "Cover:\n"
            "1. **Major Actions Completed** - Each significant command and outcome\n"
            "2. **Important Findings** - File locations, configs, errors discovered\n"
            "3. **Problems Addressed** - Issues encountered and resolutions\n"
            "4. **Current Status** - Where you are in task completion\n\n"
            "Be comprehensive. The next agent needs everything to continue."
        )
        summary = self._subagent_call(
            list(self._messages) + [{"role": "user", "content": summary_prompt}]
        )

        # Step 2: Questions (fresh context - identifies gaps)
        screen = tmux.capture_pane(capture_entire=False)
        q_prompt = (
            "You are picking up work from a previous AI agent.\n\n"
            f"**Original Task:** {instruction}\n\n"
            f"**Summary from Previous Agent:**\n{summary}\n\n"
            f"**Current Terminal:**\n{screen}\n\n"
            "Ask at least 5 questions about things NOT answered in the summary."
        )
        questions = self._subagent_call([{"role": "user", "content": q_prompt}])

        # Step 3: Answers (full history + summary - fills gaps)
        a_prompt = (
            "The next agent has a few questions for you, "
            "please answer each of them one by one in detail:\n\n" + questions
        )
        answers = self._subagent_call(
            list(self._messages) + [
                {"role": "user", "content": summary_prompt},
                {"role": "assistant", "content": summary},
                {"role": "user", "content": a_prompt},
            ]
        )

        # Reset messages to compressed form
        self._messages = [
            self._messages[0],  # system/first message
            {"role": "user", "content": q_prompt},
            {"role": "assistant", "content": questions},
        ]

        logger.info(
            "Summarization #%d complete. Messages reduced to %d.",
            self._summarization_count, len(self._messages),
        )

        return (
            "Here are the answers the other agent provided.\n\n"
            + answers
            + "\n\nContinue working on this task from where the previous agent left off."
            " You can no longer ask questions. Please follow the spec to interact"
            " with the terminal."
        )

    def _check_proactive_summarization(
        self, instruction: str, tmux: "TmuxAdapter",
    ) -> Optional[str]:
        """Check if proactive summarization is needed. Returns handoff prompt or None."""
        if not self._enable_summarize or token_counter is None:
            return None
        free = self._get_model_context_limit() - self._count_total_tokens()
        if free < self._proactive_summarization_threshold:
            logger.info(
                "Proactive summarization: ~%d free tokens < %d threshold",
                free, self._proactive_summarization_threshold,
            )
            try:
                self._unwind_messages_to_free_tokens(4000)
                return self._summarize(instruction, tmux)
            except Exception:
                logger.exception("Proactive summarization failed")
        return None

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def _execute_commands(
        self,
        commands: List[Command],
        tmux: TmuxAdapter,
    ) -> Tuple[bool, str]:
        for command in commands:
            try:
                tmux.send_keys(
                    command.keystrokes,
                    block=False,
                    min_timeout_sec=min(command.duration_sec, 60),
                )
            except TimeoutError:
                return True, TIMEOUT_TEMPLATE.format(
                    timeout_sec=command.duration_sec,
                    command=command.keystrokes,
                    terminal_state=self._limit_output_length(
                        tmux.get_incremental_output()
                    ),
                )
        return False, self._limit_output_length(tmux.get_incremental_output())

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, instruction: str, total_timeout: Optional[float] = None) -> "Terminus2Result":
        """Run the terminus-2 agent on the given instruction.

        Sets up a tmux session, runs the agent loop, and returns a result
        compatible with TinyFlow's reward pipeline.
        """
        local_cast = (
            Path(self._output_dir) / "recording.cast"
            if self._enable_asciinema and self._output_dir
            else None
        )
        tmux = TmuxAdapter(
            self._env,
            enable_asciinema=self._enable_asciinema,
            local_cast_path=local_cast,
        )
        tmux.start()
        # Small delay for tmux to initialise
        time.sleep(0.5)

        initial_output = tmux.get_incremental_output()
        prompt = self._prompt_template.format(
            instruction=instruction,
            terminal_state=self._limit_output_length(initial_output),
        )

        finished = False
        iterations = 0
        error: Optional[str] = None
        start_time = time.time()
        original_instruction = instruction

        try:
            while self._max_iterations is None or iterations < self._max_iterations:
                if total_timeout and (time.time() - start_time) > total_timeout:
                    logger.warning("Agent total timeout (%.0fs) exceeded", total_timeout)
                    break
                iterations += 1

                if not tmux.is_session_alive():
                    logger.info("Tmux session ended, breaking out of agent loop")
                    break

                # Proactive summarization check
                if self._enable_summarize:
                    handoff = self._check_proactive_summarization(
                        original_instruction, tmux
                    )
                    if handoff is not None:
                        prompt = handoff

                # Call LLM with reactive context-length fallback
                try:
                    response_text = self._call_llm(prompt)
                except _ContextLengthError:
                    logger.info("Reactive summarization triggered")
                    try:
                        # Fallback 1: Full 3-step summarization
                        self._unwind_messages_to_free_tokens(4000)
                        prompt = self._summarize(original_instruction, tmux)
                    except Exception:
                        try:
                            # Fallback 2: Short LLM-assisted summary
                            logger.warning("Full summarization failed, trying short summary")
                            screen = tmux.capture_pane(capture_entire=False)
                            limited_screen = screen[-1000:] if screen else ""
                            short_prompt = (
                                f"Briefly continue this task: {original_instruction}\n\n"
                                f"Current state: {limited_screen}\n\n"
                                f"Next steps (2-3 sentences):"
                            )
                            short_response = self._subagent_call(
                                [{"role": "user", "content": short_prompt}]
                            )
                            self._messages = [self._messages[0]]
                            prompt = f"{original_instruction}\n\nSummary: {short_response}"
                        except Exception:
                            # Fallback 3: Ultimate fallback (no LLM calls)
                            logger.warning("Short summary failed, using ultimate fallback")
                            screen = tmux.capture_pane(capture_entire=False)
                            self._messages = [self._messages[0]]
                            prompt = f"{original_instruction}\n\nCurrent terminal:\n{screen[-1000:] if screen else ''}"
                    # Safety net: if even the post-summarization LLM call fails,
                    # return a synthetic no-op response to keep the agent alive.
                    try:
                        response_text = self._call_llm(prompt)
                    except Exception:
                        logger.error("Even fallback LLM call failed, using synthetic response")
                        response_text = '{"commands": [], "task_complete": false}'
                        self._messages.append({"role": "assistant", "content": response_text})

                result = self._parser.parse_response(response_text)

                feedback = ""
                if result.error:
                    feedback += f"ERROR: {result.error}"
                    if result.warning:
                        feedback += f"\nWARNINGS: {result.warning}"
                elif result.warning:
                    feedback += f"WARNINGS: {result.warning}"

                if result.warning:
                    logger.info("Parser warnings: %s", result.warning)

                # Parse errors -> retry
                if feedback and "ERROR:" in feedback:
                    prompt = (
                        f"Previous response had parsing errors:\n{feedback}\n\n"
                        f"Please fix these issues and provide a proper "
                        f"{self._get_error_response_type()}."
                    )
                    continue

                commands = [
                    Command(keystrokes=pc.keystrokes, duration_sec=min(pc.duration, 60))
                    for pc in result.commands
                ]
                timeout_occurred, terminal_output = self._execute_commands(
                    commands, tmux
                )

                # Double-confirmation for task completion
                if result.is_task_complete:
                    if self._pending_completion:
                        finished = True
                        break
                    else:
                        self._pending_completion = True
                        prompt = self._get_completion_confirmation_message(
                            terminal_output
                        )
                        continue
                else:
                    self._pending_completion = False

                if feedback and "WARNINGS:" in feedback:
                    prompt = (
                        f"Previous response had warnings:\n{feedback}\n\n"
                        f"{self._limit_output_length(terminal_output)}"
                    )
                else:
                    prompt = self._limit_output_length(terminal_output)

        except Exception as exc:
            logger.exception("Terminus-2 agent loop failed: %s", exc)
            error = str(exc)
        finally:
            tmux.stop()

        return Terminus2Result(
            success=finished,
            message="Task completed" if finished else (error or "Max iterations reached"),
            history=self._messages,
            llm_metrics=self.llm_metrics,
            llm_cost_total=sum(
                m.get("cost", 0.0)
                for m in self.llm_metrics
                if isinstance(m.get("cost"), (int, float))
            ),
            patch="",
            iterations=iterations,
            error=error,
        )


@dataclass
class Terminus2Result:
    """Result compatible with TinyFlow's AgentResult interface."""

    success: bool
    message: str
    history: List[Dict[str, Any]]
    llm_metrics: List[Dict[str, Any]]
    llm_cost_total: float
    patch: str
    iterations: int
    error: Optional[str]
