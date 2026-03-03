"""
Parser — extracts structured data from raw LLM output.
"""

import re
import json
import logging

logger = logging.getLogger(__name__)


class ParsedResponse:
    def __init__(self):
        self.raw: str = ""
        self.thoughts: str = ""
        self.code: str | None = None           # python RUN block
        self.latex: str | None = None          # latex PAPER block
        self.action: str | None = None         # DONE | ITERATE | DEBUG
        self.action_arg: str | None = None
        self.hypothesis: str | None = None
        self.experiment_spec: str | None = None  # researcher → coder handoff


def parse_llm_response(text: str) -> ParsedResponse:
    result = ParsedResponse()
    result.raw = text

    # Extract python RUN block
    py_match = re.search(r"```python\s+RUN\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if py_match:
        result.code = py_match.group(1).strip()

    # Extract latex PAPER block
    latex_match = re.search(r"```latex\s+PAPER\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if latex_match:
        result.latex = latex_match.group(1).strip()

    # Extract action
    action_match = re.search(r"^ACTION:\s*(\w+)\s*(.*)$", text, re.MULTILINE)
    if action_match:
        result.action = action_match.group(1).upper()
        result.action_arg = action_match.group(2).strip() or None

    # Extract hypothesis (optional)
    hyp_match = re.search(r"(?:Hypothesis|hypothesis):\s*(.+)", text)
    if hyp_match:
        result.hypothesis = hyp_match.group(1).strip()

    # Extract experiment spec (researcher → coder handoff)
    # Accepts: ```spec\n...\n```  OR  SPEC:\n...\nEND SPEC
    spec_match = re.search(
        r"```spec\s*\n(.*?)```|SPEC:\s*\n(.*?)END SPEC",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if spec_match:
        result.experiment_spec = (spec_match.group(1) or spec_match.group(2)).strip()

    # Everything before code blocks = thoughts
    thoughts = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    thoughts = re.sub(r"^ACTION:.*$", "", thoughts, flags=re.MULTILINE)
    result.thoughts = thoughts.strip()

    return result


def extract_metrics(output: str) -> dict:
    """Try to parse JSON metrics from the last print() in code output."""
    # Look for JSON object in output (LLM is prompted to print metrics as JSON)
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    return {}


def extract_error(output: str, exit_code: int) -> str | None:
    """Extract traceback/error from output if exit code != 0."""
    if exit_code == 0:
        return None
    # Find Traceback block
    tb_match = re.search(r"(Traceback \(most recent call last\):.*)", output, re.DOTALL)
    if tb_match:
        return tb_match.group(1)
    # Return last 20 lines as fallback
    lines = output.strip().splitlines()
    return "\n".join(lines[-20:]) if lines else output
