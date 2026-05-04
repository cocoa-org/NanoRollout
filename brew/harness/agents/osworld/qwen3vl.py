"""Qwen3-VL agent for OSWorld.

Faithfully follows OSWorld's mm_agents/qwen3vl_agent.py logic:
  - tool_call protocol with <tool_call> XML tags
  - relative 0-999 coordinate system (scaled to actual pixels)
  - smart_resize image preprocessing
  - multi-turn history (last N screenshots + responses)
  - action→pyautogui translation

Adapted for TinyFlow: uses OpenAI-compatible API (vLLM endpoint).
"""
import base64
import json
import logging
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import backoff
import openai
from PIL import Image

from brew.harness.agents.osworld.qwen_vl_utils import smart_resize

logger = logging.getLogger(__name__)

MAX_RETRY_TIMES = 5


def _process_image(image_bytes: bytes) -> str:
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height=height, width=width, factor=32, max_pixels=16 * 16 * 4 * 12800,
    )
    image = image.resize((resized_width, resized_height))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class Qwen3VLAgent:

    def __init__(
        self,
        model: str = "qwen3-vl-30b",
        max_tokens: int = 32768,
        top_p: float = 0.9,
        temperature: float = 0.0,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        history_n: int = 4,
        coordinate_type: str = "relative",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.history_n = history_n
        self.coordinate_type = coordinate_type
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

        self.thoughts: List[str] = []
        self.actions: List[str] = []
        self.observations: List[Dict] = []
        self.responses: List[str] = []
        self.screenshots: List[str] = []

    def reset(self, _logger=None):
        global logger
        if _logger is not None:
            logger = _logger
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.responses = []
        self.screenshots = []

    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        screenshot_bytes = obs["screenshot"]
        image = Image.open(BytesIO(screenshot_bytes))
        original_width, original_height = image.size

        processed_image = _process_image(screenshot_bytes)
        processed_img = Image.open(BytesIO(base64.b64decode(processed_image)))
        processed_width, processed_height = processed_img.size

        self.screenshots.append(processed_image)
        current_step = len(self.actions)

        # Build previous actions summary (for steps before the history window)
        history_start_idx = max(0, current_step - self.history_n)
        previous_actions = []
        for i in range(history_start_idx):
            if i < len(self.actions):
                previous_actions.append(f"Step {i+1}: {self.actions[i]}")
        previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

        # System prompt with tool definition — identical to OSWorld qwen3vl_agent.py
        screen_res = (
            f"{processed_width}x{processed_height}"
            if self.coordinate_type == "absolute"
            else "1000x1000"
        )
        description_prompt = "\n".join([
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.",
            f"* The screen's resolution is {screen_res}.",
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ])

        action_description_prompt = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
"""

        tools_def = {
            "type": "function",
            "function": {
                "name_for_human": "computer_use",
                "name": "computer_use",
                "description": description_prompt,
                "parameters": {
                    "properties": {
                        "action": {
                            "description": action_description_prompt,
                            "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag",
                                     "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"],
                            "type": "string"
                        },
                        "keys": {"description": "Required only by `action=key`.", "type": "array"},
                        "text": {"description": "Required only by `action=type`.", "type": "string"},
                        "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                        "pixels": {"description": "The amount of scrolling.", "type": "number"},
                        "time": {"description": "The seconds to wait.", "type": "number"},
                        "status": {
                            "description": "The status of the task.",
                            "type": "string",
                            "enum": ["success", "failure"]
                        }
                    },
                    "required": ["action"],
                    "type": "object"
                },
                "args_format": "Format the arguments as a JSON object."
            }
        }

        system_prompt = (
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
            + json.dumps(tools_def)
            + "\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within "
            "<tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>\n\n"
            "# Response format\n\n"
            "Response format for every step:\n"
            "1) Action: a short imperative describing what to do in the UI.\n"
            "2) A single <tool_call>...</tool_call> block containing only the JSON: "
            '{"name": <function-name>, "arguments": <args-json-object>}.\n\n'
            "Rules:\n"
            "- Output exactly in the order: Action, <tool_call>.\n"
            "- Be brief: one sentence for Action.\n"
            "- Do not output anything else outside those parts.\n"
            "- If finishing, use action=terminate in the tool call."
        )

        instruction_prompt = (
            f"Please generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
            f"Instruction: {instruction}\n\n"
            f"Previous actions:\n{previous_actions_str}"
        )

        # Build messages with history — identical to OSWorld qwen3vl_agent.py
        messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]

        history_len = min(self.history_n, len(self.responses))
        if history_len > 0:
            history_responses = self.responses[-history_len:]
            history_screenshots = self.screenshots[-history_len - 1:-1]

            for idx in range(history_len):
                if idx < len(history_screenshots):
                    img_url = f"data:image/png;base64,{history_screenshots[idx]}"
                    if idx == 0:
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": img_url}},
                                {"type": "text", "text": instruction_prompt},
                            ],
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {"url": img_url}}],
                        })
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": history_responses[idx]}],
                })

            messages.append({
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{processed_image}"}}],
            })
        else:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{processed_image}"}},
                    {"type": "text", "text": instruction_prompt},
                ],
            })

        response = self._call_llm(messages)
        logger.info("Qwen3VL output: %s", response[:300] if response else "")

        self.responses.append(response)
        low_level_instruction, pyautogui_code = self._parse_response(
            response, original_width, original_height, processed_width, processed_height,
        )
        logger.info("Low level instruction: %s", low_level_instruction)
        logger.info("Pyautogui code: %s", pyautogui_code)
        self.actions.append(low_level_instruction)

        return response, pyautogui_code

    # ---- LLM call (OpenAI-compatible, for vLLM) ----

    @backoff.on_exception(backoff.constant, (openai.RateLimitError, openai.InternalServerError), interval=30, max_tries=5)
    def _call_llm(self, messages: List[Dict]) -> str:
        client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        for attempt in range(1, MAX_RETRY_TIMES + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                logger.error("LLM call attempt %d/%d failed: %s", attempt, MAX_RETRY_TIMES, e)
                if attempt < MAX_RETRY_TIMES:
                    time.sleep(5)
        return ""

    # ---- Response parsing (identical to OSWorld qwen3vl_agent.py) ----

    def _parse_response(
        self, response: str,
        original_width: int, original_height: int,
        processed_width: int, processed_height: int,
    ) -> Tuple[str, List[str]]:
        low_level_instruction = ""
        pyautogui_code: List[str] = []

        if not response or not response.strip():
            return low_level_instruction, pyautogui_code

        def adjust_coordinates(x: float, y: float) -> Tuple[int, int]:
            if self.coordinate_type == "absolute":
                if processed_width and processed_height:
                    return int(x * original_width / processed_width), int(y * original_height / processed_height)
                return int(x), int(y)
            return int(x * original_width / 999), int(y * original_height / 999)

        def process_tool_call(json_str: str) -> None:
            nonlocal low_level_instruction
            try:
                tool_call = json.loads(json_str)
                if tool_call.get("name") != "computer_use":
                    return
                args = tool_call["arguments"]
                action = args["action"]

                if action == "left_click":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        ax, ay = adjust_coordinates(x, y)
                        pyautogui_code.append(f"pyautogui.click({ax}, {ay})")
                    else:
                        pyautogui_code.append("pyautogui.click()")
                elif action == "right_click":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        ax, ay = adjust_coordinates(x, y)
                        pyautogui_code.append(f"pyautogui.rightClick({ax}, {ay})")
                    else:
                        pyautogui_code.append("pyautogui.rightClick()")
                elif action == "middle_click":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        ax, ay = adjust_coordinates(x, y)
                        pyautogui_code.append(f"pyautogui.middleClick({ax}, {ay})")
                    else:
                        pyautogui_code.append("pyautogui.middleClick()")
                elif action == "double_click":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        ax, ay = adjust_coordinates(x, y)
                        pyautogui_code.append(f"pyautogui.doubleClick({ax}, {ay})")
                    else:
                        pyautogui_code.append("pyautogui.doubleClick()")
                elif action == "type":
                    text = args.get("text", "")
                    pyautogui_code.append(f"pyautogui.typewrite('{text}')")
                elif action == "key":
                    keys = args.get("keys", [])
                    if isinstance(keys, list):
                        cleaned = []
                        for k in keys:
                            if isinstance(k, str):
                                k = k.strip().strip("[]'\"")
                                if k.startswith("keys="):
                                    k = k[5:]
                                cleaned.append(k)
                        keys = cleaned
                    keys_str = ", ".join(f"'{k}'" for k in keys)
                    if len(keys) > 1:
                        pyautogui_code.append(f"pyautogui.hotkey({keys_str})")
                    else:
                        pyautogui_code.append(f"pyautogui.press({keys_str})")
                elif action == "scroll":
                    pixels = args.get("pixels", 0)
                    pyautogui_code.append(f"pyautogui.scroll({pixels})")
                elif action == "wait":
                    pyautogui_code.append("WAIT")
                elif action == "terminate":
                    pyautogui_code.append("DONE")
                elif action == "mouse_move":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        ax, ay = adjust_coordinates(x, y)
                        pyautogui_code.append(f"pyautogui.moveTo({ax}, {ay})")
                elif action == "left_click_drag":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        ax, ay = adjust_coordinates(x, y)
                        duration = args.get("duration", 0.5)
                        pyautogui_code.append(f"pyautogui.dragTo({ax}, {ay}, duration={duration})")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error("Failed to parse tool call: %s", e)

        # Parse response lines for <tool_call> blocks
        lines = response.split("\n")
        inside_tool_call = False
        current_tool_call: List[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("action:"):
                if not low_level_instruction:
                    low_level_instruction = line.split(":", 1)[-1].strip()
                continue
            if line.startswith("<tool_call>"):
                inside_tool_call = True
                continue
            elif line.startswith("</tool_call>"):
                if current_tool_call:
                    process_tool_call("\n".join(current_tool_call))
                    current_tool_call = []
                inside_tool_call = False
                continue
            if inside_tool_call:
                current_tool_call.append(line)
                continue
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    if "name" in obj and "arguments" in obj:
                        process_tool_call(line)
                except json.JSONDecodeError:
                    pass

        if current_tool_call:
            process_tool_call("\n".join(current_tool_call))

        if not low_level_instruction and pyautogui_code:
            action_type = pyautogui_code[0].split(".", 1)[1].split("(", 1)[0] if "." in pyautogui_code[0] else pyautogui_code[0]
            low_level_instruction = f"Performing {action_type} action"

        return low_level_instruction, pyautogui_code
