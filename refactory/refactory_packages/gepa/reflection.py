"""GEPA's reflective dataset, reflection prompt and instruction extraction.

The prompt and the extractor are ported verbatim from ``gepa`` 0.1.4
``strategies/instruction_proposal.py:13-153`` (``InstructionProposalSignature``).
The reflective dataset follows the DSPy adapter
(``adapters/dspy_adapter/dspy_adapter.py:341-474``): per component, one record per
minibatch example, ``{"Inputs", "Generated Outputs", "Feedback"}``, from one call of
the component's predictor in that example's trace, picked with the adapter's own
``random.Random(0)`` (``dspy_adapter.py:111``) when there are several.

In refactory a component is a text knob and its "predictor" the agent node(s) that
render it; a call is one of the item record's ``exchanges`` (``{node, prompt,
response}``). DSPy's inputs are the signature's input fields, without the
instruction; here they are the exchange's rendered prompt with the component's own
text taken out, since the reflection prompt shows that text already.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from typing import Any

#: ``instruction_proposal.py:13-29``, verbatim.
DEFAULT_PROMPT_TEMPLATE = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_param>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<side_info>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""

#: Where the component's own text stood in an exchange's prompt.
INSTRUCTION_PLACEHOLDER = "[the instructions above]"


def check_template(template: str) -> None:
    """``instruction_proposal.py:35-42``: a template needs both placeholders."""
    missing = [p for p in ("<curr_param>", "<side_info>") if p not in template]
    if missing:
        raise ValueError(f"gepa: the reflection prompt template lacks {', '.join(missing)}")


def render_value(value: Any, level: int = 3) -> str:
    """``instruction_proposal.py:65-87`` (images aside: refactory records hold none)."""
    if isinstance(value, dict):
        s = "".join(f"{'#' * level} {k}\n" + render_value(v, min(level + 1, 6)) for k, v in value.items())
        return s + ("\n" if not value else "")
    if isinstance(value, (list, tuple)):
        s = "".join(f"{'#' * level} Item {i + 1}\n" + render_value(v, min(level + 1, 6))
                    for i, v in enumerate(value))
        return s + ("\n" if not value else "")
    return f"{str(value).strip()}\n\n"


def format_samples(samples: Sequence[Mapping[str, Any]]) -> str:
    """``instruction_proposal.py:89-96``: each record as a markdown ``# Example n``."""
    def sample_md(sample: Mapping[str, Any], n: int) -> str:
        return f"# Example {n}\n" + "".join(f"## {k}\n" + render_value(v, 3) for k, v in sample.items())

    return "\n\n".join(sample_md(sample, i + 1) for i, sample in enumerate(samples))


def reflection_prompt(current: str, records: Sequence[Mapping[str, Any]],
                      template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    """``instruction_proposal.py:98-113``: the template with ``<curr_param>`` then
    ``<side_info>`` replaced, in that order."""
    return template.replace("<curr_param>", current).replace("<side_info>", format_samples(records))


def extract_instruction(lm_out: str) -> str:
    """``instruction_proposal.py:124-153``: the text between the first and the last
    fence, its language tag dropped; an unfenced or half-fenced reply as it stands.
    Called on the stripped reply (``reflection_lm.py:172``)."""
    start = lm_out.find("```") + 3
    end = lm_out.rfind("```")
    if start >= end:
        stripped = lm_out.strip()
        if stripped.startswith("```"):
            match = re.match(r"^```\S*\n?", lm_out)
            if match:
                return lm_out[match.end():].strip()
        elif stripped.endswith("```"):
            return stripped[:-3].strip()
        return stripped
    content = lm_out[start:end]
    match = re.match(r"^\S*\n", content)
    if match:
        content = content[match.end():]
    return content.strip()


def feedback_text(record: Mapping[str, Any], mode: str) -> str:
    """The feedback for one example.

    DSPy's feedback is the metric's own text, and ``"This trajectory got a score of
    {score}."`` when the metric gives a number only. Refactory's grader text is the
    record's ``fail_reason`` (``VerifyResult.details`` of a failed item) or, for an
    item that errored, ``error_text``. ``mode="details"`` (the default) gives that
    text where there is one and the score sentence otherwise; ``mode="score"`` always
    gives the score sentence (the plan's score-only ablation).
    """
    score = record.get("score")
    plain = f"This trajectory got a score of {score}."
    if mode == "score":
        return plain
    details = record.get("fail_reason") or record.get("error_text")
    return str(details).strip() if details else plain


def reflective_dataset(
    records: Sequence[Mapping[str, Any]],
    nodes: Sequence[str],
    component_text: str,
    *,
    feedback: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """One component's records (``dspy_adapter.py:341-474``): per minibatch example,
    one of its ``nodes``' calls (one ``rng.choice``), as ``Inputs``,
    ``Generated Outputs`` and ``Feedback``. An example with no such call is skipped
    (``:376-377``), as is one whose run failed with no call to show (``:385-386``)."""
    node_set = set(nodes)
    out: list[dict[str, Any]] = []
    for record in records:
        calls = [x for x in record.get("exchanges") or () if str(x.get("node")) in node_set]
        if not calls:
            continue
        # Drawn even from one call, as the adapter does (``:387``).
        call = rng.choice(calls)
        prompt = str(call.get("prompt") or "")
        if component_text.strip():
            prompt = prompt.replace(component_text, INSTRUCTION_PLACEHOLDER)
        out.append({
            "Inputs": prompt,
            "Generated Outputs": str(call.get("response") or ""),
            "Feedback": feedback_text(record, feedback),
        })
    return out


def outcome_dataset(records: Sequence[Mapping[str, Any]], *, feedback: str) -> list[dict[str, Any]]:
    """For a component no agent node renders (an op reads it): one record per example
    from the outcome alone, as GEPA's ``DefaultAdapter`` makes them
    (``adapters/default_adapter/default_adapter.py:176-206``)."""
    return [{
        "Inputs": str(record.get("id", "")),
        "Generated Outputs": str(record.get("prediction") or ""),
        "Feedback": feedback_text(record, feedback),
    } for record in records]
