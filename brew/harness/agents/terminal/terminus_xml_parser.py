"""Parser for terminus XML plain response format.

Ported from harbor terminus_2/terminus_xml_plain_parser.py.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ParsedCommand:
    keystrokes: str
    duration: float


@dataclass
class ParseResult:
    commands: List[ParsedCommand]
    is_task_complete: bool
    error: str
    warning: str


class TerminusXMLPlainParser:
    """Parser for terminus XML plain response format."""

    def __init__(self):
        self.required_sections = ["analysis", "plan", "commands"]

    def parse_response(self, response: str) -> ParseResult:
        result = self._try_parse_response(response)

        if result.error:
            for fix_name, fix_function in self._get_auto_fixes():
                corrected_response, was_fixed = fix_function(response, result.error)
                if was_fixed:
                    corrected_result = self._try_parse_response(corrected_response)
                    if corrected_result.error == "":
                        auto_warning = (
                            f"AUTO-CORRECTED: {fix_name} - "
                            "please fix this in future responses"
                        )
                        corrected_result.warning = self._combine_warnings(
                            auto_warning, corrected_result.warning
                        )
                        return corrected_result

        return result

    def _try_parse_response(self, response: str) -> ParseResult:
        warnings = []
        self._check_extra_text(response, warnings)

        response_content = self._extract_response_content(response)
        if not response_content:
            return ParseResult(
                [],
                False,
                "No <response> tag found",
                "- " + "\n- ".join(warnings) if warnings else "",
            )

        is_complete = self._check_task_complete(response_content)
        sections = self._extract_sections(response_content, warnings)

        commands_content = sections.get("commands", "")
        if not commands_content:
            if "commands" in sections:
                if not is_complete:
                    warnings.append(
                        "Commands section is empty; not taking any action."
                    )
                return ParseResult(
                    [],
                    is_complete,
                    "",
                    "- " + "\n- ".join(warnings) if warnings else "",
                )
            else:
                if is_complete:
                    return ParseResult(
                        [], True, "", "- " + "\n- ".join(warnings) if warnings else ""
                    )
                return ParseResult(
                    [],
                    False,
                    "Missing <commands> section",
                    "- " + "\n- ".join(warnings) if warnings else "",
                )

        commands, parse_error = self._parse_xml_commands(commands_content, warnings)
        if parse_error:
            if is_complete:
                warnings.append(parse_error)
                return ParseResult(
                    [], True, "", "- " + "\n- ".join(warnings) if warnings else ""
                )
            return ParseResult(
                [], False, parse_error, "- " + "\n- ".join(warnings) if warnings else ""
            )

        return ParseResult(
            commands, is_complete, "", "- " + "\n- ".join(warnings) if warnings else ""
        )

    def _get_auto_fixes(self):
        return [
            (
                "Missing </response> tag was automatically inserted",
                self._fix_missing_response_tag,
            ),
        ]

    def _combine_warnings(self, auto_warning: str, existing_warning: str) -> str:
        if existing_warning:
            return f"- {auto_warning}\n{existing_warning}"
        return f"- {auto_warning}"

    def _fix_missing_response_tag(self, response: str, error: str) -> tuple[str, bool]:
        if "No <response> tag found" not in error:
            return response, False
        corrected = response.rstrip() + "\n</response>"
        return corrected, True

    def _check_extra_text(self, response: str, warnings: List[str]) -> None:
        start_pos = response.find("<response>")
        end_pos = response.find("</response>")
        if start_pos == -1:
            return
        before_text = response[:start_pos].strip()
        if before_text:
            warnings.append("Extra text detected before <response> tag")
        if end_pos != -1:
            after_text = response[end_pos + len("</response>"):].strip()
            if after_text:
                warnings.append("Extra text detected after </response> tag")
                total_response_count = response.count("<response>")
                if total_response_count > 1:
                    warnings.append(
                        f"IMPORTANT: Only issue one <response> block at a time. "
                        f"You issued {total_response_count} and only the first "
                        f"was executed."
                    )

    def _extract_response_content(self, response: str) -> str:
        start_pos = response.find("<response>")
        if start_pos == -1:
            return ""
        end_pos = response.find("</response>", start_pos)
        if end_pos == -1:
            return response[start_pos + len("<response>"):].strip()
        return response[start_pos + len("<response>"):end_pos].strip()

    def _extract_sections(self, content: str, warnings: List[str]) -> dict:
        sections = {}
        found_sections = set()

        section_patterns = {
            "analysis": (
                r"<analysis>(.*?)</analysis>",
                r"<analysis\s*/>",
                r"<analysis></analysis>",
            ),
            "plan": (r"<plan>(.*?)</plan>", r"<plan\s*/>", r"<plan></plan>"),
            "commands": (
                r"<commands>(.*?)</commands>",
                r"<commands\s*/>",
                r"<commands></commands>",
            ),
            "task_complete": (
                r"<task_complete>(.*?)</task_complete>",
                r"<task_complete\s*/>",
                r"<task_complete></task_complete>",
            ),
        }

        for section_name, patterns in section_patterns.items():
            full_pattern, self_closing_pattern, empty_pattern = patterns
            match = re.search(full_pattern, content, re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()
                found_sections.add(section_name)
                continue
            if re.search(self_closing_pattern, content):
                sections[section_name] = ""
                found_sections.add(section_name)
                continue
            if re.search(empty_pattern, content):
                sections[section_name] = ""
                found_sections.add(section_name)
                continue

        required = set(self.required_sections)
        missing = required - found_sections
        for section in missing:
            if section != "task_complete":
                warnings.append(f"Missing <{section}> section")

        self._check_section_order(content, warnings)
        return sections

    def _parse_xml_commands(
        self, xml_content: str, warnings: List[str]
    ) -> tuple[List[ParsedCommand], str]:
        commands = []
        keystrokes_pattern = re.compile(
            r"<keystrokes([^>]*)>(.*?)</keystrokes>", re.DOTALL
        )
        matches = keystrokes_pattern.findall(xml_content)
        for i, (attributes_str, keystrokes_content) in enumerate(matches):
            duration = 1.0
            duration_match = re.search(
                r'duration\s*=\s*["\']([^"\']*)["\']', attributes_str
            )
            if duration_match:
                try:
                    duration = float(duration_match.group(1))
                except ValueError:
                    warnings.append(
                        f"Command {i + 1}: Invalid duration value "
                        f"'{duration_match.group(1)}', using default 1.0"
                    )
            else:
                warnings.append(
                    f"Command {i + 1}: Missing duration attribute, using default 1.0"
                )

            if i < len(matches) - 1 and not keystrokes_content.endswith("\n"):
                warnings.append(
                    f"Command {i + 1} should end with newline when followed "
                    f"by another command."
                )

            commands.append(
                ParsedCommand(keystrokes=keystrokes_content, duration=duration)
            )

        return commands, ""

    def _check_section_order(self, content: str, warnings: List[str]) -> None:
        expected_order = ["analysis", "plan", "commands"]
        positions = {}
        for section in expected_order:
            match = re.search(f"<{section}(?:\\s|>|/>)", content)
            if match:
                positions[section] = match.start()

        if len(positions) < 2:
            return

        present_sections = [
            (s, positions[s]) for s in expected_order if s in positions
        ]
        actual_order = [s for s, _ in sorted(present_sections, key=lambda x: x[1])]
        expected_present = [s for s in expected_order if s in positions]

        if actual_order != expected_present:
            actual_str = " → ".join(actual_order)
            expected_str = " → ".join(expected_present)
            warnings.append(
                f"Sections appear in wrong order. Found: {actual_str}, "
                f"expected: {expected_str}"
            )

    def _check_task_complete(self, response_content: str) -> bool:
        true_match = re.search(
            r"<task_complete>\s*true\s*</task_complete>",
            response_content,
            re.IGNORECASE,
        )
        return bool(true_match)

    def salvage_truncated_response(
        self, truncated_response: str
    ) -> Tuple[Optional[str], bool]:
        commands_end = truncated_response.find("</commands>")
        if commands_end == -1:
            return None, False
        response_end = truncated_response.find("</response>", commands_end)
        if response_end == -1:
            return None, False
        clean_response = truncated_response[:response_end + len("</response>")]
        try:
            parse_result = self.parse_response(clean_response)
            has_multiple_blocks = False
            if parse_result.warning:
                warning_lower = parse_result.warning.lower()
                has_multiple_blocks = (
                    "only issue one" in warning_lower
                    and "block at a time" in warning_lower
                )
            if not parse_result.error and not has_multiple_blocks:
                return clean_response, False
            return None, has_multiple_blocks
        except Exception:
            return None, False
